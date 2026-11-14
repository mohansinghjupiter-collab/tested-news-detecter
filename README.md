# TruthLens - Fake News Detector
## Setup
1. `pip install -r requirements.txt`
2. Download "Fake and Real News Dataset" (ISOT) from Kaggle -> put `Fake.csv`, `True.csv` in `data/`
3. `python train.py`  (prints accuracy, ROC-AUC, confusion matrix; saves model.joblib)
4. `streamlit run app.py`
5. Optional AI fact-check: set `ANTHROPIC_API_KEY`

## Advanced concepts to mention in viva
- Stacking ensemble (meta-learning) with probability calibration
- Word + character n-gram TF-IDF + stylometric feature engineering
- Data-leakage removal ("(Reuters)" tags) - ISOT gives ~99% otherwise misleadingly
- Explainable AI: sentence-level occlusion + coefficient-based word attribution
- Source credibility, URL scraping, batch scoring, LLM claim analysis
- Limitation: cross-domain generalisation; detects style, not truth

## Extensions
Fine-tune DistilBERT/RoBERTa (HuggingFace) and add as 4th ensemble member; test on LIAR / FakeNewsNet for cross-dataset evaluation.

## Deploy: GitHub -> Railway
1. Train locally first (`python train.py`) so `model.joblib` exists. Check it is < 100 MB
   (`ls -lh model.joblib`). If larger, lower `max_features` in train.py or use Git LFS.
2. `git init && git add . && git commit -m "TruthLens"` then push to a new GitHub repo.
   (`data/*.csv` is git-ignored; `model.joblib` IS committed so the server needs no dataset.)
3. railway.app -> New Project -> Deploy from GitHub repo -> select it. Railway detects
   requirements.txt and the Procfile automatically.
4. Variables tab: add `ANTHROPIC_API_KEY` (optional, enables AI fact-check).
5. Settings -> Networking -> Generate Domain. Open the URL.

Notes: first build takes a few minutes. WHOIS/RSS need outbound internet (Railway allows it).
Some news sites block cloud-server IPs, so scraping may fail on those; the app shows a friendly error.
