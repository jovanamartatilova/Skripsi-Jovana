# -*- coding: utf-8 -*-
"""
ProtokolLain.py - 4 skenario proposal dengan protokol paper acuan lain yang TIDAK memakai TB anak
(PEMBANDING untuk dosen). Setiap protokol dijalankan SEKALI; aturannya ditetapkan sebelum melihat hasil.

  --protokol ndagijimana : Ndagijimana dkk. (2023, Rwanda). SMOTE pada seluruh data -> split 80:20 acak bertingkat,
                           threshold 0,5, seleksi IG + Gini. Model tambahan dari paper: Gradient Boosting.
  --protokol shen        : Shen dkk. (2023, Papua Nugini). Split acak 90:10 (latih:uji), seleksi fitur LASSO hanya di
                           data latih, threshold "closest top-left" pada kurva ROC dari 10-fold CV data latih.
                           Model tambahan dari paper: LASSO + Gradient Boosting (pengganti XGBoost).
  --protokol mkungudza   : Mkungudza dkk. (2024, Malawi). Split acak 80:20, seleksi fitur LASSO di data latih,
                           threshold SpEqualSe (sensitivitas = spesifisitas) dari data latih.
Skenario dipilih dari CV data LATIH (F1 > recall > AUC), bukan dari data uji.
Jalankan: python ProtokolLain.py --protokol ndagijimana   (lalu shen, lalu mkungudza)
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # custom_transformer.py (Winsorizer)

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
ap = argparse.ArgumentParser()
ap.add_argument("--protokol", required=True, choices=["ndagijimana", "shen", "mkungudza"])
ap.add_argument("--csv", default=str(BASE / "output" / "BuildDataset" / "balita_ifls5.csv"))
ap.add_argument("--pre", default=str(BASE / "output" / "2_Preprocessing" / "preprocessor.joblib"))
ap.add_argument("--out", default=None)
ap.add_argument("--k", type=int, default=12, help="jumlah variabel untuk seleksi IG + Gini (ndagijimana)")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
P = args.protokol
OUT = Path(args.out or BASE / "output" / f"10_Protokol_{P.capitalize()}")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
CONF = {
    "ndagijimana": dict(judul="Ndagijimana dkk. (2023): SMOTE seluruh data -> split 80:20, threshold 0,5",
                        smote=True, uji=0.20, seleksi="iggini", thr="tetap", cv=5, tambahan="GB"),
    "shen": dict(judul="Shen dkk. (2023): split 90:10, seleksi LASSO, threshold closest top-left (10-fold CV latih)",
                 smote=False, uji=0.10, seleksi="lasso", thr="topleft", cv=10, tambahan="GB"),
    "mkungudza": dict(judul="Mkungudza dkk. (2024): split 80:20, seleksi LASSO, threshold SpEqualSe (data latih)",
                      smote=False, uji=0.20, seleksi="lasso", thr="speqse", cv=5, tambahan=None),
}[P]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def smote(X, y, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    Xm = X[y == 1]
    n = int((y == 0).sum() - len(Xm))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    i = rng.integers(0, len(Xm), n)
    j = nn[i, rng.integers(0, k, n)]
    S = Xm[i] + rng.random((n, 1)) * (Xm[j] - Xm[i])
    return np.vstack([X, S]), np.r_[y, np.ones(n, int)], np.r_[np.zeros(len(X), bool), np.ones(n, bool)]


def model(algo):
    if algo == "LR":
        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
    if algo == "GB":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                              class_weight="balanced", random_state=SEED)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


def seleksi(Z, y, asal):
    """Dipanggil pada data latih saja."""
    if CONF["seleksi"] == "lasso":
        las = LogisticRegressionCV(Cs=10, cv=5, penalty="l1", solver="liblinear", scoring="roc_auc",
                                   class_weight="balanced", max_iter=5000, random_state=SEED).fit(Z, y)
        t = pd.DataFrame({"v": asal, "b": np.abs(las.coef_[0])}).groupby("v").b.max()
        top = list(t[t > 0].sort_values(ascending=False).index) or list(t.sort_values(ascending=False).index[:5])
    else:
        dt = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", min_samples_leaf=20,
                                    random_state=SEED).fit(Z, y)
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=20,
                                    n_jobs=-1, random_state=SEED).fit(Z, y)
        t = pd.DataFrame({"v": asal, "ig": dt.feature_importances_, "g": rf.feature_importances_}).groupby("v").sum()
        t["r"] = t.ig.rank(ascending=False) + t.g.rank(ascending=False)
        top = list(t.sort_values(["r", "g"], ascending=[True, False]).index[:args.k])
    return [i for i, v in enumerate(asal) if v in set(top)], top


def cari_thr(y, p):
    if CONF["thr"] == "tetap":
        return 0.5
    fpr, tpr, t = roc_curve(y, p)
    ok = np.isfinite(t)
    fpr, tpr, t = fpr[ok], tpr[ok], t[ok]
    if CONF["thr"] == "topleft":
        i = np.argmin((1 - tpr) ** 2 + fpr ** 2)          # titik ROC terdekat ke pojok kiri atas
    else:
        i = np.argmin(np.abs(tpr - (1 - fpr)))           # sensitivitas = spesifisitas
    return float(np.clip(t[i], 0.01, 0.99))


def metrik(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp = int(((yh == 0) & (y == 0)).sum()), int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    return {"threshold": thr, "accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0), "f1": f1_score(y, yh, zero_division=0),
            "roc_auc": roc_auc_score(y, p), "spesifisitas": tn / max(tn + fp, 1), "NPV": tn / max(tn + fn, 1),
            "f1_tertimbang": f1_score(y, yh, average="weighted", zero_division=0)}


log(f"=== 4 SKENARIO DENGAN PROTOKOL {CONF['judul']} ===")
df = pd.read_csv(args.csv)
pre = joblib.load(args.pre)
fitur = list(pre.feature_names_in_)
X = df[fitur].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
y = df["stunting"].values.astype(int)
Z = np.asarray(pre.transform(X), float)
nama = [n.replace("num__", "").replace("kat__", "").replace("spl__", "") for n in pre.get_feature_names_out()]
kat = [c for c in fitur if c not in nama]
asal = [next((c for c in kat if n.startswith(c + "_")), n) for n in nama]
assert "tb_anak" not in fitur, "protokol ini tanpa TB anak"
log(f"[data] {len(df)} balita, {len(fitur)} kandidat fitur, stunting {y.mean() * 100:.2f}%")

sint = np.zeros(len(y), bool)
if CONF["smote"]:
    Z, y, sint = smote(Z, y)
    log(f"[SMOTE] {len(df)} -> {len(y)} baris ({sint.sum()} sintetis)")
i_tr, i_te = train_test_split(np.arange(len(y)), test_size=CONF["uji"], stratify=y, random_state=SEED)
log(f"[split] latih {len(i_tr)}, uji {len(i_te)} ({CONF['uji']:.0%})"
    + (f"; data sintetis di data uji: {sint[i_te].sum()} ({sint[i_te].mean() * 100:.1f}%)  <- KEBOCORAN" if CONF["smote"] else
       "; acak per balita, tanpa pengelompokan keluarga"))
Ztr, ytr, Zte, yte = Z[i_tr], y[i_tr], Z[i_te], y[i_te]

SKEN = {"S1": ("LR", True), "S2": ("LR", False), "S3": ("RF", True), "S4": ("RF", False)}
if CONF["tambahan"]:
    SKEN["Paper"] = (CONF["tambahan"], CONF["seleksi"] == "lasso")
NAMA = {"LR": "Logistic Regression", "RF": "Random Forest", "GB": "Gradient Boosting"}

oof = {s: np.zeros(len(ytr)) for s in SKEN}
for a, b in StratifiedKFold(CONF["cv"], shuffle=True, random_state=SEED).split(Ztr, ytr):
    cols_k, _ = seleksi(Ztr[a], ytr[a], asal)
    for s, (algo, fs) in SKEN.items():
        cols = cols_k if fs else list(range(Ztr.shape[1]))
        oof[s][b] = model(algo).fit(Ztr[a][:, cols], ytr[a]).predict_proba(Ztr[b][:, cols])[:, 1]
thr = {s: cari_thr(ytr, oof[s]) for s in SKEN}
cv = pd.DataFrame([{"skenario": s, "algoritma": NAMA[a], "seleksi_fitur": "ya" if f else "tidak", **metrik(ytr, oof[s], thr[s])}
                   for s, (a, f) in SKEN.items()])
terpilih = cv[cv.skenario.isin(["S1", "S2", "S3", "S4"])].sort_values(["f1", "recall", "roc_auc"], ascending=False).iloc[0].skenario
kol = ["skenario", "algoritma", "seleksi_fitur", "threshold", "accuracy", "precision", "recall", "f1", "roc_auc"]
log(f"\n  CV {CONF['cv']}-fold pada data LATIH (threshold dari data latih; dasar pemilihan):")
log(cv[kol].round(4).to_string(index=False))
log(f"  -> skenario terpilih (F1 > recall > AUC): {terpilih}")

cols_full, top_full = seleksi(Ztr, ytr, asal)
Pte, rows = {}, []
for s, (algo, fs) in SKEN.items():
    cols = cols_full if fs else list(range(Ztr.shape[1]))
    Pte[s] = model(algo).fit(Ztr[:, cols], ytr).predict_proba(Zte[:, cols])[:, 1]
    rows.append({"skenario": s, "algoritma": NAMA[algo], "seleksi_fitur": "ya" if fs else "tidak", **metrik(yte, Pte[s], thr[s])})
te = pd.DataFrame(rows)
log("\n  Data UJI (threshold dikunci dari data latih):")
log(te[kol].round(4).to_string(index=False))
log("\n  Metrik tambahan data UJI:")
log(te[["skenario", "spesifisitas", "NPV", "f1_tertimbang"]].round(4).to_string(index=False))
log(f"  Fitur terpilih ({CONF['seleksi']}): {len(top_full)} variabel -> {', '.join(top_full)}")

rng = np.random.default_rng(SEED)
bs = pd.DataFrame([metrik(yte[r], Pte[terpilih][r], thr[terpilih]) for r in (rng.integers(0, len(yte), len(yte)) for _ in range(args.boot))])
bs = bs.drop(columns="threshold")
nilai = te.set_index("skenario").loc[terpilih]
ci = pd.DataFrame({"metrik": bs.columns, "nilai": [nilai[m] for m in bs.columns],
                   "CI95_bawah": bs.quantile(0.025).values, "CI95_atas": bs.quantile(0.975).values})
log(f"\n  Skenario terpilih {terpilih} pada data UJI (CI 95% bootstrap {args.boot}x):")
for r in ci.itertuples():
    log(f"    {r.metrik:16s} {r.nilai:.3f} [{r.CI95_bawah:.3f}-{r.CI95_atas:.3f}]")

ring = pd.DataFrame([["Protokol", CONF["judul"]], ["Skenario terpilih", terpilih],
                     ["Data sintetis di data uji", int(sint[i_te].sum()) if CONF["smote"] else "tidak ada (tanpa SMOTE)"],
                     ["Keterbatasan", "split acak per balita (tanpa pengelompokan keluarga)"
                      + ("; data uji memuat data sintetis SMOTE" if CONF["smote"] else "")]],
                    columns=["indikator", "nilai"])
with pd.ExcelWriter(OUT / f"Hasil_Protokol_{P.capitalize()}.xlsx", engine="openpyxl") as xw:
    for nm, d in {"00_Ringkasan": ring, "01_CV_Latih": cv.round(4), "02_Uji": te.round(4), "03_CI_Terpilih": ci.round(4),
                  "04_Fitur_Terpilih": pd.DataFrame({"variabel": top_full})}.items():
        d.to_excel(xw, sheet_name=nm, index=False)
(OUT / f"laporan_protokol_{P}.txt").write_text("\n".join(LOG), encoding="utf-8")
log(f"\nDisimpan di: {OUT}")