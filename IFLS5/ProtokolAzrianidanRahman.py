# -*- coding: utf-8 -*-
"""
ProtokolCV.py - 4 skenario proposal dengan protokol evaluasi penelitian terdahulu berbasis 10-fold CV
(PEMBANDING untuk dosen; hasil utama tetap pipeline 1-6 / protokol Ethiopia sesuai keputusan dosen)

  --protokol azriani : Azriani dkk. (2024). 10-fold CV acak pada SELURUH data (tanpa data uji terpisah,
                       tanpa pengelompokan keluarga), threshold 0,5, seleksi fitur Information Gain + Gini,
                       metrik rata-rata tertimbang dilaporkan.
  --protokol rahman  : Rahman dkk. (2021). Sama, tetapi seleksi fitur = regresi logistik multivariat p < 0,05
                       (dihitung di tiap training fold); yang ditonjolkan accuracy dan AUC.
Aturan: dijalankan SEKALI per protokol; skenario dipilih dari hasil CV (F1 > recall > AUC).
Jalankan: python ProtokolCV.py --protokol azriani   lalu   python ProtokolCV.py --protokol rahman
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # custom_transformer.py (Winsorizer)

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
ap = argparse.ArgumentParser()
ap.add_argument("--protokol", required=True, choices=["azriani", "rahman"])
ap.add_argument("--csv", default=str(BASE / "output" / "BuildDataset" / "balita_ifls5.csv"))
ap.add_argument("--pre", default=str(BASE / "output" / "2_Preprocessing" / "preprocessor.joblib"))
ap.add_argument("--out", default=None)
ap.add_argument("--k", type=int, default=12, help="jumlah variabel untuk seleksi Information Gain + Gini")
ap.add_argument("--fold", type=int, default=10)
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
OUT = Path(args.out or BASE / "output" / f"9_Protokol_{args.protokol.capitalize()}")
OUT.mkdir(parents=True, exist_ok=True)
SEED, THR = 42, 0.5
LOG = []
JUDUL = {"azriani": "Azriani dkk. (2024): 10-fold CV acak, seleksi IG + Gini, metrik tertimbang",
         "rahman": "Rahman dkk. (2021): 10-fold CV acak, seleksi regresi logistik p < 0,05"}


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def model(algo):
    if algo == "LR":
        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


def pilih_iggini(Z, y, asal, k):
    dt = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", min_samples_leaf=20,
                                random_state=SEED).fit(Z, y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=20,
                                n_jobs=-1, random_state=SEED).fit(Z, y)
    t = pd.DataFrame({"v": asal, "ig": dt.feature_importances_, "gini": rf.feature_importances_}).groupby("v").sum()
    t["rank"] = t.ig.rank(ascending=False) + t.gini.rank(ascending=False)
    top = list(t.sort_values(["rank", "gini"], ascending=[True, False]).index[:k])
    return [i for i, v in enumerate(asal) if v in set(top)], top


def pilih_pvalue(Z, y, asal):
    X = np.column_stack([np.ones(len(Z)), Z])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(Z, y)
    b = np.r_[lr.intercept_, lr.coef_[0]]
    p = 1 / (1 + np.exp(-X @ b))
    cov = np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)
    pv = 2 * norm.sf(np.abs(b / np.sqrt(np.clip(np.diag(cov), 1e-12, None))))[1:]
    t = pd.DataFrame({"v": asal, "p": pv}).groupby("v").p.min()
    top = list(t[t < 0.05].sort_values().index) or list(t.sort_values().index[:5])
    return [i for i, v in enumerate(asal) if v in set(top)], top


def metrik(y, p):
    yh = (p >= THR).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0), "f1": f1_score(y, yh, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "precision_tertimbang": precision_score(y, yh, average="weighted", zero_division=0),
            "recall_tertimbang": recall_score(y, yh, average="weighted", zero_division=0),
            "f1_tertimbang": f1_score(y, yh, average="weighted", zero_division=0)}


log(f"=== 4 SKENARIO DENGAN PROTOKOL {JUDUL[args.protokol]} ===")
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
log(f"[data] {len(df)} balita, {len(fitur)} kandidat fitur, stunting {y.mean() * 100:.2f}% | {args.fold}-fold CV acak, "
    "tanpa pengelompokan keluarga, threshold 0,5")

SKEN = {"S1": ("LR", True), "S2": ("LR", False), "S3": ("RF", True), "S4": ("RF", False)}
NAMA = {"LR": "Logistic Regression", "RF": "Random Forest"}
oof = {s: np.zeros(len(y)) for s in SKEN}
fold_rows, pilih_rows = [], []
for f, (a, b) in enumerate(StratifiedKFold(args.fold, shuffle=True, random_state=SEED).split(Z, y)):
    cols_k, top = pilih_pvalue(Z[a], y[a], asal) if args.protokol == "rahman" else pilih_iggini(Z[a], y[a], asal, args.k)
    pilih_rows += [{"fold": f, "variabel": v} for v in top]
    for s, (algo, fs) in SKEN.items():
        cols = cols_k if fs else list(range(Z.shape[1]))
        oof[s][b] = model(algo).fit(Z[a][:, cols], y[a]).predict_proba(Z[b][:, cols])[:, 1]
        fold_rows.append({"skenario": s, "fold": f, **metrik(y[b], oof[s][b])})
    log(f"  [fold {f}] seleksi fitur: {len(top)} variabel ({', '.join(top[:5])}, ...)")

hasil = pd.DataFrame([{"skenario": s, "algoritma": NAMA[algo], "seleksi_fitur": "ya" if fs else "tidak",
                       **metrik(y, oof[s])} for s, (algo, fs) in SKEN.items()])
fold = pd.DataFrame(fold_rows)
ms = fold.groupby("skenario")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"]).round(4)
ms.columns = [f"{m}_{st}" for m, st in ms.columns]   # ratakan kolom bertingkat agar bisa ditulis ke Excel
terpilih = hasil.sort_values(["f1", "recall", "roc_auc"], ascending=False).iloc[0].skenario
log(f"\n  Hasil {args.fold}-fold CV (threshold 0,5):")
log(hasil[["skenario", "algoritma", "seleksi_fitur", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4).to_string(index=False))
log("\n  Metrik tertimbang:")
log(hasil[["skenario", "accuracy", "precision_tertimbang", "recall_tertimbang", "f1_tertimbang"]].round(4).to_string(index=False))
log(f"  -> skenario terpilih (F1 > recall > AUC): {terpilih}")

rng = np.random.default_rng(SEED)
bs = pd.DataFrame([metrik(y[r], oof[terpilih][r]) for r in (rng.integers(0, len(y), len(y)) for _ in range(args.boot))])
nilai = hasil.set_index("skenario").loc[terpilih]
ci = pd.DataFrame({"metrik": bs.columns, "nilai": [nilai[m] for m in bs.columns],
                   "CI95_bawah": bs.quantile(0.025).values, "CI95_atas": bs.quantile(0.975).values})
log(f"\n  Skenario terpilih {terpilih} (CI 95% bootstrap {args.boot}x):")
for r in ci.itertuples():
    log(f"    {r.metrik:22s} {r.nilai:.3f} [{r.CI95_bawah:.3f}-{r.CI95_atas:.3f}]")

ring = pd.DataFrame([["Protokol", JUDUL[args.protokol]], ["Data", f"{len(df)} balita (tanpa SMOTE, tanpa data uji terpisah)"],
                     ["Kandidat fitur", len(fitur)], ["Threshold", THR], ["Skenario terpilih", terpilih],
                     ["Keterbatasan", "CV acak tanpa pengelompokan keluarga; tidak ada data uji terpisah"]],
                    columns=["indikator", "nilai"])
stab = pd.DataFrame(pilih_rows).groupby("variabel").fold.nunique().rename("n_fold_terpilih").sort_values(ascending=False).reset_index()
with pd.ExcelWriter(OUT / f"Hasil_Protokol_{args.protokol.capitalize()}.xlsx", engine="openpyxl") as xw:
    for nm, d in {"00_Ringkasan": ring, "01_Hasil_CV": hasil.round(4), "02_Per_Fold_MeanSD": ms.reset_index(),
                  "03_CI_Terpilih": ci.round(4), "04_Fitur_Terpilih": stab}.items():
        d.to_excel(xw, sheet_name=nm, index=False)
(OUT / f"laporan_protokol_{args.protokol}.txt").write_text("\n".join(LOG), encoding="utf-8")
log(f"\nDisimpan di: {OUT}")