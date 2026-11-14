import re
from urllib.parse import urlparse, urljoin
import requests, feedparser, trafilatura
from bs4 import BeautifulSoup
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0 (TruthLens research project)"}
SLD = {"co", "com", "org", "gov", "ac", "net"}


def root_domain(url_or_domain):
    d = urlparse(url_or_domain).netloc if "//" in url_or_domain else url_or_domain
    p = d.lower().replace("www.", "").split(".")
    return ".".join(p[-3:]) if len(p) >= 3 and p[-2] in SLD else ".".join(p[-2:])


def fetch_html(url):
    h = trafilatura.fetch_url(url)
    if h:
        return h
    r = requests.get(url, timeout=12, headers=UA)
    r.raise_for_status()
    return r.text


def fetch_article(url):
    """Return (title, text). Raises ValueError with a friendly message."""
    try:
        h = fetch_html(url)
    except Exception as e:
        raise ValueError(f"Could not open the page ({e}). The site may block bots.")
    txt = trafilatura.extract(h, include_comments=False, include_tables=False) or ""
    meta = trafilatura.extract_metadata(h)
    title = (meta.title if meta and meta.title else "") or ""
    if len(txt.split()) < 80:
        raise ValueError("Too little article text found. It may be a homepage/section page, "
                         "a paywall, a login wall or a JavaScript-only site. Paste a direct article "
                         "link or use the text box.")
    return title, txt


def find_feed(site_url):
    base = "{u.scheme}://{u.netloc}".format(u=urlparse(site_url))
    cands = []
    try:
        soup = BeautifulSoup(requests.get(site_url, timeout=12, headers=UA).text, "html.parser")
        for l in soup.find_all("link", rel="alternate"):
            if "rss" in (l.get("type") or "") or "atom" in (l.get("type") or ""):
                cands.append(urljoin(site_url, l.get("href")))
    except Exception:
        pass
    cands += [base + p for p in ("/feed", "/feed/", "/rss", "/rss.xml", "/rss/", "/feeds/all.rss.xml")]
    for c in cands:
        f = feedparser.parse(c, request_headers=UA)
        if f.entries:
            return f
    return None


def domain_age_years(domain):
    try:
        import whois
        cd = whois.whois(domain).creation_date
        cd = cd[0] if isinstance(cd, list) else cd
        if cd.tzinfo is None:
            cd = cd.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - cd).days / 365.25
    except Exception:
        return None
