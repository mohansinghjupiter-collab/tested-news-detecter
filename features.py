import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

CLICKBAIT = ["shocking", "unbelievable", "you won't believe", "breaking", "exposed", "secret",
             "miracle", "destroyed", "slams", "bombshell", "outrage", "hoax", "wake up",
             "must see", "share this", "they don't want", "banned", "cover-up", "total"]
ATTRIB = ["said", "according", "reported", "told", "stated", "announced", "confirmed",
          "spokesman", "spokeswoman", "statement", "officials"]
STYLE_NAMES = ["caps_ratio", "exclamations", "questions", "avg_word_len", "avg_sent_len",
               "clickbait_terms", "attribution_terms", "first_second_person", "quotes",
               "digit_ratio", "lexical_diversity", "title_style_caps_words"]


def clean(t):
    """Remove dataset leakage (agency tags, URLs, image credits)."""
    t = str(t)
    t = re.sub(r"^.{0,60}?\((Reuters|AP)\)\s*-\s*", "", t)
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"(featured image|image via|getty images|21st century wire).*", "", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def style_vector(text):
    words = re.findall(r"[A-Za-z']+", text)
    n = max(len(words), 1)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    low = text.lower()
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    return [
        sum(c.isupper() for c in text) / max(len(text), 1),
        text.count("!") / n * 100,
        text.count("?") / n * 100,
        sum(len(w) for w in words) / n,
        n / max(len(sents), 1),
        sum(low.count(k) for k in CLICKBAIT),
        sum(low.count(" " + k) for k in ATTRIB) / n * 100,
        sum(1 for w in words if w.lower() in {"i", "we", "you", "my", "our", "your"}) / n * 100,
        (text.count('"') + text.count("“")) / 2,
        sum(c.isdigit() for c in text) / max(len(text), 1),
        len(set(w.lower() for w in words)) / n,
        caps_words / n * 100,
    ]


class StyleFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([style_vector(t) for t in X])
