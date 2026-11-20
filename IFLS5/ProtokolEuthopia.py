# -*- coding: utf-8 -*-
"""
8_PROTOKOL_ETHIOPIA.py - 4 skenario proposal dengan protokol evaluasi penelitian Scientific Reports (2025, Ethiopia)

ATURAN (ditetapkan SEBELUM dijalankan; disetujui dosen; jalankan SEKALI, jangan diubah setelah melihat hasil):
  1. Preprocessing pada seluruh data (preprocessor dari 2_PREPROCESSING.py, tanpa target).
  2. SMOTE pada SELURUH data sampai kelas seimbang, LALU split 80:20 acak bertingkat (seed 42).
     -> data uji memuat data sintetis (kebocoran data); WAJIB ditulis di Bab 3 dan keterbatasan.
  3. Empat skenario proposal: S1 LR + seleksi, S2 LR, S3 RF + seleksi, S4 RF.
     Seleksi fitur = Information Gain (DT entropy) + Gini Decrease (RF), rank-sum, K variabel teratas.
  4. Threshold tetap 0,5 (bawaan, sesuai paper) untuk semua skenario.
  5. Skenario dipilih dari 5-fold CV pada data LATIH (F1 > recall > AUC), BUKAN dari data uji.
Output: folder 8_Protokol_Ethiopia -> Hasil_Protokol_Ethiopia.xlsx, laporan_protokol_ethiopia.txt
Jalankan: python 8_PROTOKOL_ETHIOPIA.py
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # custom_transformer.py (Winsorizer)

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(BASE / "output" / "BuildDataset" / "balita_ifls5.csv"))
ap.add_argument("--pre", default=str(BASE / "output" / "2_Preprocessing" / "preprocessor.joblib"))
ap.add_argument("--out", default=str(BASE / "output" / "8_Protokol_Ethiopia"))
ap.add_argument("--k", type=int, default=12, help="jumlah variabel pada skenario dengan seleksi fitur")
ap.add_argument("--boot", type=int, default=1000)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
SEED, THR = 42, 0.5
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def smote(X, y, k=5, seed=SEED):
    """SMOTE (Chawla dkk., 2002): sampel sintetis kelas minoritas sampai kedua kelas seimbang."""
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
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


def pilih_fitur(Z, y, asal, k):
    """Information Gain + Gini Decrease, rank-sum per variabel asal; dipanggil pada data latih saja."""
    dt = DecisionTreeClassifier(criterion="entropy", class_weight="balanced", min_samples_leaf=20,
                                random_state=SEED).fit(Z, y)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=20,
                                n_jobs=-1, random_state=SEED).fit(Z, y)
    t = pd.DataFrame({"v": asal, "ig": dt.feature_importances_, "gini": rf.feature_importances_}).groupby("v").sum()
    t["rank"] = t.ig.rank(ascending=False) + t.gini.rank(ascending=False)
    top = t.sort_values(["rank", "gini"], ascending=[True, False]).index[:k]
    return [i for i, v in enumerate(asal) if v in set(top)], list(top)


def metrik(y, p):
    yh = (p >= THR).astype(int)
    return {"accuracy": accuracy_score(y, yh), "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0), "f1": f1_score(y, yh, zero_division=0),
            "roc_auc": roc_auc_score(y, p),
            "precision_tertimbang": precision_score(y, yh, average="weighted", zero_division=0),
            "recall_tertimbang": recall_score(y, yh, average="weighted", zero_division=0),
            "f1_tertimbang": f1_score(y, yh, average="weighted", zero_division=0)}


# ---------------- data & preprocessing (seluruh data, tanpa target) ----------------
log("=== 4 SKENARIO DENGAN PROTOKOL SCIENTIFIC REPORTS (2025, ETHIOPIA) ===")
df = pd.read_csv(args.csv)
pre = joblib.load(args.pre)
fitur = list(pre.feature_names_in_)
X = df[fitur].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
y = df["stunting"].values.astype(int)
Z = np.asarray(pre.transform(X), float)
nama = [n.replace("num__", "").replace("kat__", "").replace("spl__", "") for n in pre.get_feature_names_out()]
kat = [c for c in fitur if any(n.startswith(c + "_") for n in nama) and c not in nama]
asal = [next((c for c in kat if n.startswith(c + "_")), n) for n in nama]
assert "tb_anak" not in fitur, "protokol ini tanpa TB anak"
log(f"[data] {len(df)} balita, {len(fitur)} kandidat fitur, stunting {y.mean() * 100:.2f}%")

# ---------------- SMOTE seluruh data, lalu split ----------------
Zs, ys, sint = smote(Z, y)
idx = np.arange(len(ys))
i_tr, i_te = train_test_split(idx, test_size=0.2, stratify=ys, random_state=SEED)
log(f"[SMOTE] {len(y)} -> {len(ys)} baris ({sint.sum()} sintetis); kelas 1 = {ys.mean() * 100:.1f}%")
log(f"[split] latih {len(i_tr)}, uji {len(i_te)}; data sintetis di data uji: {sint[i_te].sum()} "
    f"({sint[i_te].mean() * 100:.1f}%)  <- KEBOCORAN, tulis di keterbatasan")
Ztr, ytr, Zte, yte = Zs[i_tr], ys[i_tr], Zs[i_te], ys[i_te]

SKEN = {"S1": ("LR", True), "S2": ("LR", False), "S3": ("RF", True), "S4": ("RF", False)}
NAMA = {"LR": "Logistic Regression", "RF": "Random Forest"}

# ---------------- pemilihan skenario: 5-fold CV pada data latih ----------------
cv_rows = []
oof = {s: np.zeros(len(ytr)) for s in SKEN}
for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Ztr, ytr):
    cols_k, _ = pilih_fitur(Ztr[a], ytr[a], asal, args.k)
    for s, (algo, fs) in SKEN.items():
        cols = cols_k if fs else list(range(Ztr.shape[1]))
        oof[s][b] = model(algo).fit(Ztr[a][:, cols], ytr[a]).predict_proba(Ztr[b][:, cols])[:, 1]
for s, (algo, fs) in SKEN.items():
    cv_rows.append({"skenario": s, "algoritma": NAMA[algo], "seleksi_fitur": "ya" if fs else "tidak", **metrik(ytr, oof[s])})
cv = pd.DataFrame(cv_rows)
terpilih = cv.sort_values(["f1", "recall", "roc_auc"], ascending=False).iloc[0].skenario
log("\n  CV 5-fold pada data LATIH (threshold 0,5; dasar pemilihan):")
log(cv[["skenario", "algoritma", "seleksi_fitur", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4).to_string(index=False))
log(f"  -> skenario terpilih (F1 > recall > AUC pada CV latih): {terpilih}")

# ---------------- latih ulang pada seluruh data latih -> data uji ----------------
cols_full, top_full = pilih_fitur(Ztr, ytr, asal, args.k)
te_rows, P = [], {}
for s, (algo, fs) in SKEN.items():
    cols = cols_full if fs else list(range(Ztr.shape[1]))
    P[s] = model(algo).fit(Ztr[:, cols], ytr).predict_proba(Zte[:, cols])[:, 1]
    te_rows.append({"skenario": s, "algoritma": NAMA[algo], "seleksi_fitur": "ya" if fs else "tidak", **metrik(yte, P[s])})
te = pd.DataFrame(te_rows)
log("\n  Data UJI (threshold 0,5):")
log(te[["skenario", "algoritma", "seleksi_fitur", "accuracy", "precision", "recall", "f1", "roc_auc"]].round(4).to_string(index=False))
log("\n  Metrik tertimbang data UJI:")
log(te[["skenario", "accuracy", "precision_tertimbang", "recall_tertimbang", "f1_tertimbang"]].round(4).to_string(index=False))
log(f"  Fitur terpilih (S1, S3; K={args.k}): {', '.join(top_full)}")

# ---------------- CI 95% bootstrap untuk skenario terpilih ----------------
rng = np.random.default_rng(SEED)
bs = []
for _ in range(args.boot):
    r = rng.integers(0, len(yte), len(yte))
    bs.append(metrik(yte[r], P[terpilih][r]))
bs = pd.DataFrame(bs)
ci = pd.DataFrame({"metrik": bs.columns, "nilai": [te.set_index("skenario").loc[terpilih, m] for m in bs.columns],
                   "CI95_bawah": bs.quantile(0.025).values, "CI95_atas": bs.quantile(0.975).values})
log(f"\n  Skenario terpilih {terpilih} pada data UJI (CI 95% bootstrap {args.boot}x):")
for r in ci.itertuples():
    log(f"    {r.metrik:22s} {r.nilai:.3f} [{r.CI95_bawah:.3f}-{r.CI95_atas:.3f}]")

ring = pd.DataFrame([
    ["Protokol", "Scientific Reports (2025): SMOTE seluruh data -> split 80:20 acak bertingkat"],
    ["Data", f"{len(df)} balita -> {len(ys)} baris setelah SMOTE ({sint.sum()} sintetis)"],
    ["Data sintetis di data uji", f"{sint[i_te].sum()} dari {len(i_te)} ({sint[i_te].mean() * 100:.1f}%)"],
    ["Kandidat fitur", len(fitur)], ["K seleksi fitur (S1, S3)", args.k], ["Threshold", THR],
    ["Pemilihan skenario", "5-fold CV pada data latih: F1 > recall > AUC"], ["Skenario terpilih", terpilih],
    ["Keterbatasan", "Data uji memuat data sintetis hasil SMOTE -> kinerja cenderung terlalu tinggi"],
], columns=["indikator", "nilai"])
with pd.ExcelWriter(OUT / "Hasil_Protokol_Ethiopia.xlsx", engine="openpyxl") as xw:
    for nm, d in {"00_Ringkasan": ring, "01_CV_Latih": cv.round(4), "02_Uji": te.round(4), "03_CI_Terpilih": ci.round(4),
                  "04_Fitur_Terpilih": pd.DataFrame({"variabel": top_full})}.items():
        d.to_excel(xw, sheet_name=nm, index=False)
(OUT / "laporan_protokol_ethiopia.txt").write_text("\n".join(LOG), encoding="utf-8")
log(f"\nDisimpan di: {OUT}")
log("CATATAN: jalankan SEKALI. Tulis di keterbatasan bahwa data uji memuat data sintetis (kebocoran data).")