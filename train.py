"""Train: python train.py   (needs data/True.csv and data/Fake.csv - ISOT / Kaggle 'Fake and Real News')"""
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from features import clean, StyleFeatures

fake = pd.read_csv("data/Fake.csv").assign(label=1)
real = pd.read_csv("data/True.csv").assign(label=0)
df = pd.concat([fake, real]).drop_duplicates(subset="text")
df["x"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).map(clean)
df = df[df.x.str.len() > 100]
Xtr, Xte, ytr, yte = train_test_split(df.x, df.label, test_size=0.2, stratify=df.label, random_state=42)

feats = FeatureUnion([
    ("w", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True, stop_words="english")),
    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=30000, sublinear_tf=True)),
    ("s", Pipeline([("st", StyleFeatures()), ("sc", MinMaxScaler(clip=True))])),
])
stack = StackingClassifier(
    estimators=[("lr", LogisticRegression(C=5, max_iter=1000)),
                ("svm", CalibratedClassifierCV(LinearSVC(C=0.5), cv=3)),
                ("nb", ComplementNB(alpha=0.3))],
    final_estimator=LogisticRegression(), cv=3, n_jobs=-1)
model = Pipeline([("feats", feats), ("clf", stack)])
model.fit(Xtr, ytr)

p = model.predict_proba(Xte)[:, 1]
print(classification_report(yte, p > 0.5, target_names=["real", "fake"]))
print("ROC-AUC:", roc_auc_score(yte, p))
print(confusion_matrix(yte, p > 0.5))
joblib.dump(model, "model.joblib", compress=3)
print("saved model.joblib")
