import os, re, html, joblib
import numpy as np, pandas as pd, streamlit as st
from features import clean, style_vector, STYLE_NAMES
from sources import CRED, SATIRE
from extractor import fetch_article, find_feed, domain_age_years, root_domain

st.set_page_config(page_title="TruthLens - Fake News Detector", page_icon="🔍", layout="wide")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


@st.cache_data(ttl=86400, show_spinner=False)
def cached_age(d):
    return domain_age_years(d)


def prob_fake(texts):
    return load().predict_proba([clean(t) for t in texts])[:, 1]


def verdict(p):
    if p >= 0.65: return "🔴 Likely FAKE"
    if p <= 0.35: return "🟢 Likely REAL"
    return "🟡 Uncertain"


def source_report(domain):
    """Show credibility score, satire flag and domain age."""
    score = CRED.get(domain)
    if domain in SATIRE:
        st.error(f"⚠️ {domain} is a known SATIRE site - content is not meant to be factual.")
    if score is not None:
        (st.success if score >= 75 else st.warning if score >= 40 else st.error)(
            f"Source credibility: **{score}/100** ({domain})")
    else:
        st.info(f"'{domain}' is not in the credibility database - verify the outlet independently.")
    age = cached_age(domain)
    if age is not None:
        msg = f"Domain age: {age:.1f} years"
        (st.warning if age < 1 else st.caption)(msg + (" - very new site, be cautious." if age < 1 else ""))


def explain_sentences(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s) > 20][:40]
    if len(sents) < 2:
        return sents, np.zeros(len(sents))
    base = prob_fake([" ".join(sents)])[0]
    without = prob_fake([" ".join(sents[:i] + sents[i + 1:]) for i in range(len(sents))])
    return sents, base - without


def top_words(text, k=8):
    m = load()
    vec = m.named_steps["feats"].transformer_list[0][1]
    names = vec.get_feature_names_out()
    coef = m.named_steps["clf"].named_estimators_["lr"].coef_[0][:len(names)]
    c = vec.transform([clean(text)]).multiply(coef).tocsr()
    order = np.argsort(c.data)
    fake = [names[c.indices[i]] for i in order[::-1][:k] if c.data[i] > 0]
    real = [names[c.indices[i]] for i in order[:k] if c.data[i] < 0]
    return fake, real


def llm_factcheck(text):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic
    r = anthropic.Anthropic(api_key=key).messages.create(
        model="claude-sonnet-4-6", max_tokens=900,
        messages=[{"role": "user", "content":
                   "You are a fact-checking assistant. For the article below: 1) list the 3 most "
                   "checkable factual claims, 2) say which look plausible/dubious and why, 3) list "
                   "manipulation techniques used, 4) suggest what to verify and where. Be concise.\n\n"
                   "ARTICLE:\n" + text[:6000]}])
    return r.content[0].text


def render_analysis(text, domain=None, use_llm=False):
    p = float(prob_fake([text])[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Verdict", verdict(p))
    c2.metric("Fake probability", f"{p:.1%}")
    c3.metric("Confidence", f"{abs(p - .5) * 200:.0f}%")
    st.progress(p)
    if domain:
        source_report(domain)
    st.subheader("Why? Sentence-level explanation")
    st.caption("Red pushes toward fake, green toward real (occlusion analysis)")
    sents, contrib = explain_sentences(clean(text))
    mx = max(np.abs(contrib).max(), 1e-9) if len(contrib) else 1
    out = ""
    for s, c in zip(sents, contrib):
        a = min(abs(c) / mx, 1) * 0.6
        col = f"rgba(220,50,50,{a})" if c > 0 else f"rgba(40,170,80,{a})"
        out += f"<span style='background:{col};padding:2px'>{html.escape(s)}</span> "
    st.markdown(out, unsafe_allow_html=True)
    fk, rl = top_words(text)
    a, b = st.columns(2)
    a.write("**Words signalling FAKE**"); a.write(", ".join(fk) or "-")
    b.write("**Words signalling REAL**"); b.write(", ".join(rl) or "-")
    with st.expander("Writing-style profile"):
        st.dataframe(pd.DataFrame({"feature": STYLE_NAMES,
                                   "value": np.round(style_vector(clean(text)), 3)}), hide_index=True)
    if use_llm:
        with st.spinner("AI fact-checking..."):
            try:
                st.subheader("🤖 AI fact-check")
                st.write(llm_factcheck(text) or "Set the ANTHROPIC_API_KEY environment variable.")
            except Exception as e:
                st.error(e)
    st.session_state.history.append({"source": domain or "pasted text", "snippet": text[:70],
                                     "p_fake": round(p, 3), "verdict": verdict(p)})
    st.warning("Models detect writing patterns, not truth. Always verify with trusted sources.")


if "history" not in st.session_state:
    st.session_state.history = []

st.title("🔍 TruthLens - AI Fake News Detector")
st.caption("Stacked ML ensemble + explainability + source credibility + LLM fact-check")

if not os.path.exists("model.joblib"):
    st.error("model.joblib not found. Run `python train.py` first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Analyze", "Batch CSV", "History & About"])

with tab1:
    mode = st.radio("Input", ["Article URL", "News website (latest articles)", "Paste text"], horizontal=True)
    use_llm = st.checkbox("Add AI fact-check (needs ANTHROPIC_API_KEY)")

    if mode == "Article URL":
        url = st.text_input("Direct link to a news article")
        if st.button("Analyze article", type="primary") and url:
            try:
                with st.spinner("Fetching article..."):
                    title, body = fetch_article(url)
                st.success(f"Fetched: {title[:100]} ({len(body.split())} words)")
                render_analysis(title + ". " + body, root_domain(url), use_llm)
            except ValueError as e:
                st.error(str(e))

    elif mode == "News website (latest articles)":
        site = st.text_input("Website address (e.g. https://www.bbc.com)")
        n = st.slider("Articles to scan", 3, 15, 6)
        if st.button("Scan website", type="primary") and site:
            if "//" not in site:
                site = "https://" + site
            dom = root_domain(site)
            source_report(dom)
            with st.spinner("Finding RSS feed..."):
                feed = find_feed(site)
            if not feed:
                st.error("No RSS feed found for this site. Try an Article URL instead.")
            else:
                rows, bar = [], st.progress(0.0)
                entries = feed.entries[:n]
                for i, e in enumerate(entries):
                    try:
                        t, b = fetch_article(e.link)
                        rows.append({"title": t or e.get("title", ""), "text": t + ". " + b, "link": e.link})
                    except ValueError:
                        pass
                    bar.progress((i + 1) / len(entries))
                if not rows:
                    st.error("Could not extract any articles (site may block bots or use paywalls).")
                else:
                    d = pd.DataFrame(rows)
                    d["p_fake"] = prob_fake(d.text.tolist()).round(3)
                    d["verdict"] = d.p_fake.map(verdict)
                    st.metric("Site average fake probability", f"{d.p_fake.mean():.1%}")
                    st.dataframe(d[["title", "p_fake", "verdict", "link"]], hide_index=True)
                    st.bar_chart(d.set_index("title").p_fake)

    else:
        text = st.text_area("Headline + article", height=220)
        if st.button("Analyze text", type="primary"):
            if len(text.split()) < 15:
                st.error("Please paste at least 15 words.")
            else:
                render_analysis(text, None, use_llm)

with tab2:
    f = st.file_uploader("CSV with a 'text' column", type="csv")
    if f:
        d = pd.read_csv(f)
        d["p_fake"] = prob_fake(d["text"].astype(str).tolist()).round(3)
        d["verdict"] = d.p_fake.map(verdict)
        st.dataframe(d)
        st.download_button("Download results", d.to_csv(index=False), "results.csv")

with tab3:
    if st.session_state.history:
        h = pd.DataFrame(st.session_state.history)
        st.dataframe(h); st.line_chart(h.p_fake)
    st.markdown("""**Pipeline:** leakage-free cleaning → TF-IDF word + char n-grams + 12 stylometric
features → stacking ensemble (LogReg, calibrated SVM, Complement NB) → calibrated probability.
**Extras:** trafilatura extraction, RSS site scanning, domain age (WHOIS), credibility database, LLM claim analysis.""")
