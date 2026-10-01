"""Ek deneyler: metin temsili (kelime n-gram vs. virgülle ayrılmış nota) ve gradient boosting.

Ana notebook ile aynı veri hazırlığı, aynı bölme (random_state=42) ve aynı pipeline'ı kullanır.
Çalıştırma: python extra_experiments.py   (veri: data/fra_cleaned.csv)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # tek çekirdekli ortamlarda HistGradientBoosting'in takılmasını önler
import sys, tempfile, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RS = 42
data = pd.read_csv("data/fra_cleaned.csv", sep=";", encoding="latin-1")
top = data["Brand"].value_counts().head(10).index.tolist()
df = data[data["Brand"].isin(top)].copy()
df["Rating Value"] = pd.to_numeric(df["Rating Value"].astype(str).str.replace(",", "."), errors="coerce")
for c in ["Top", "Middle", "Base"]:
    df[c] = df[c].fillna("")
df["Notes"] = (df["Top"] + ", " + df["Middle"] + ", " + df["Base"]).str.lower().str.strip()
df = df.dropna(subset=["Year"]).reset_index(drop=True)
df["Year"] = df["Year"].clip(lower=1950)
df["Log Rating Count"] = np.log1p(df["Rating Count"])
num_cols = ["Rating Value", "Log Rating Count", "Year"]; cat_cols = ["Gender"]

X = df[["Notes", "Gender", "Rating Value", "Log Rating Count", "Year"]]; y = df["Brand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
print(len(X_train), len(X_test), flush=True)

def note_tokenizer(text):
    return [t.strip() for t in text.split(",") if t.strip()]

def to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X

def tfidf(mode):
    if mode == "word":
        return TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    return TfidfVectorizer(tokenizer=note_tokenizer, token_pattern=None, lowercase=False, max_features=500)

def make_pipe(clf, mode, memory):
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ("text", tfidf(mode), "Notes")])
    rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=RS, n_jobs=1), n_features_to_select=50, step=0.1)
    return Pipeline([("prep", prep), ("rfe", rfe), ("dense", FunctionTransformer(to_dense, accept_sparse=True)),
                     ("clf", clf)], memory=memory)

MODELS = {
    "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RS), {"clf__C": [0.1, 1, 10]}),
    "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=RS, n_jobs=1),
                      {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]}),
    "k-NN": (KNeighborsClassifier(), {"clf__n_neighbors": [3, 5, 7], "clf__weights": ["uniform", "distance"]}),
    "HistGradientBoosting": (HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False, random_state=RS),
                             {"clf__learning_rate": [0.05, 0.1], "clf__max_iter": [100, 200], "clf__max_depth": [3, None]}),
}
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

plan = [("word", ["Random Forest", "HistGradientBoosting"]),
        ("note", ["Random Forest", "Logistic Regression", "k-NN", "HistGradientBoosting"])]
out = []
for mode, names in plan:
    mem = tempfile.mkdtemp()
    for n in names:
        t0 = time.time()
        clf, params = MODELS[n]
        g = GridSearchCV(make_pipe(clf, mode, mem), params, cv=cv, scoring="f1_macro", n_jobs=1)
        g.fit(X_train, y_train)
        p = g.predict(X_test)
        row = dict(tok=mode, model=n, cv_f1=round(g.best_score_, 3), test_acc=round(accuracy_score(y_test, p), 3),
                   test_f1=round(f1_score(y_test, p, average="macro"), 3),
                   params={k.replace("clf__", ""): v for k, v in g.best_params_.items()}, sec=round(time.time() - t0))
        print(row, flush=True)
        out.append(row)
json.dump(out, open("extra_experiments_results.json", "w"), default=str)
print("\nÖzet:")
print(pd.DataFrame(out).drop(columns=["params", "sec"]).to_string(index=False))
