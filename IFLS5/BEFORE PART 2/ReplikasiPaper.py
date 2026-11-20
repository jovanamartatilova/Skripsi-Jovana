"""
7_REPLIKASI_PAPER.py - Replikasi protokol evaluasi penelitian terdahulu pada fitur model final (TANPA TB).

TUJUAN: PEMBANDING untuk Bab 4, BUKAN hasil final. Model, fitur, dan data sama; yang berbeda hanya cara evaluasi.
  P0 Penelitian ini : split per keluarga (dari 3_SPLITTING), threshold dari OOF train, metrik kelas stunting.
  P1 Azriani dkk.   : 10-fold CV acak pada seluruh data, threshold 0,5, metrik RATA-RATA TERTIMBANG.
  P2 Rahman dkk.    : sama dengan P1, tetapi yang ditonjolkan hanya ACCURACY.
  P3 Ethiopia (Sci Rep 2025): SMOTE pada SELURUH data SEBELUM split 80:20 acak
Jalankan: python 7_REPLIKASI_PAPER.py --fitur-file <fitur_tahap3.txt>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(BASE / "output" / "BuildDataset" / "balita_ifls5.csv"))
ap.add_argument("--train", default=str(BASE / "output" / "2_Splitting" / "train.csv"))
ap.add_argument("--test", default=str(BASE / "output" / "2_Splitting" / "test.csv"))
ap.add_argument("--fitur-file", default=str(BASE / "output" / "6_FeatureImportance" / "fitur_tahap3.txt"))
ap.add_argument("--out", default=str(BASE / "output" / "7_Replikasi_Paper"))
ap.add_argument("--set-azriani", action="store_true",
                help="pakai prediktor ala Azriani dkk. (faktor kelahiran, orang tua termasuk tinggi ibu/ayah, sanitasi)")
ap.add_argument("--tambah-waz", action="store_true", help="tambahkan umur dan WAZ ke set Azriani")
ap.add_argument("--knn", action="store_true", help="pakai kNN (algoritma terbaik Azriani dkk.) sebagai ganti Random Forest")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
GRID = np.round(np.arange(0.01, 0.9901, 0.01), 2)
AZRIANI = ["jk_laki", "bb_lahir_kg", "ukuran_bayi", "lama_hamil_minggu", "anc_total", "jml_tablet_besi",
           "tempat_lahir_grup", "tinggi_ibu", "tinggi_ayah", "pend_ibu", "pend_ayah", "umur_ibu", "umur_ayah",
           "kerja_ibu", "kerja_ayah", "air_layak", "toilet_layak", "sampah_layak", "ada_perokok_RT",
           "jumlah_anggota_RT"]
if args.set_azriani:
    FITUR = AZRIANI + (["umur_bln", "waz"] if args.tambah_waz else [])
else:
    FITUR = [l.strip() for l in Path(args.fitur_file).read_text(encoding="utf-8").splitlines() if l.strip()]
assert "tb_anak" not in FITUR, "replikasi ini harus tanpa TB"


def model():
    if args.knn:
        return KNeighborsClassifier(n_neighbors=15)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


def pra():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


def smote(X, y, k=5, seed=SEED):
    """SMOTE sederhana (Chawla dkk., 2002): sampel sintetis kelas minoritas sampai seimbang."""
    rng = np.random.default_rng(seed)
    Xm = X[y == 1]
    n = int((y == 0).sum() - len(Xm))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    i = rng.integers(0, len(Xm), n)
    j = nn[i, rng.integers(0, k, n)]
    S = Xm[i] + rng.random((n, 1)) * (Xm[j] - Xm[i])
    return np.vstack([X, S]), np.r_[y, np.ones(n, int)]


def metrik(y, p, t, protokol, catatan):
    yh = (p >= t).astype(int)
    return {"protokol": protokol, "threshold": t,
            "accuracy": accuracy_score(y, yh),
            "precision_kelas_stunting": precision_score(y, yh, zero_division=0),
            "recall_kelas_stunting": recall_score(y, yh, zero_division=0),
            "f1_kelas_stunting": f1_score(y, yh, zero_division=0),
            "precision_tertimbang": precision_score(y, yh, average="weighted", zero_division=0),
            "recall_tertimbang": recall_score(y, yh, average="weighted", zero_division=0),
            "f1_tertimbang": f1_score(y, yh, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y, p), "catatan": catatan}


hasil = []
tr, te = pd.read_csv(args.train), pd.read_csv(args.test)
semua = pd.read_csv(args.csv)
print(f"Fitur ({len(FITUR)}): {', '.join(FITUR)}")

# P0 penelitian ini
ytr, yte = tr.stunting.values, te.stunting.values
oof = np.zeros(len(tr))
for k in sorted(tr.fold_cv.unique()):
    m = tr.fold_cv == k
    oof[m.values] = make_pipeline(pra(), model()).fit(tr.loc[~m, FITUR], ytr[~m.values]).predict_proba(tr.loc[m, FITUR])[:, 1]
t0 = float(max(GRID, key=lambda t: f1_score(ytr, oof >= t)))
p0 = make_pipeline(pra(), model()).fit(tr[FITUR], ytr).predict_proba(te[FITUR])[:, 1]
hasil.append(metrik(yte, p0, t0, "P0 Penelitian ini",
                    "split per keluarga, threshold dari OOF, test terpisah; SAH (hasil final)"))

# P1/P2 Azriani & Rahman: 10-fold CV acak pada seluruh data, threshold 0,5
X, y = semua[FITUR], semua.stunting.values
p1 = np.zeros(len(semua))
for a, b in StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y):
    p1[b] = make_pipeline(pra(), model()).fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
hasil.append(metrik(y, p1, 0.5, "P1 Azriani dkk. (10-fold CV, metrik tertimbang)",
                    "saudara bisa terpisah antar fold; lihat kolom *_tertimbang"))
hasil.append(metrik(y, p1, 0.5, "P2 Rahman dkk. (yang dilaporkan hanya accuracy)",
                    "sama dengan P1; accuracy tinggi belum tentu kelas stunting terdeteksi"))

# P3 Ethiopia: SMOTE seluruh data sebelum split
Z = pra().fit_transform(X)
Zs, ys = smote(Z, y)
a, b, ya, yb = train_test_split(Zs, ys, test_size=0.2, stratify=ys, random_state=SEED)
p3 = model().fit(a, ya).predict_proba(b)[:, 1]
hasil.append(metrik(yb, p3, 0.5, "P3 Ethiopia (SMOTE sebelum split)",
                    f"data sintetis {len(ys) - len(y)} baris ikut masuk data uji -> KEBOCORAN; TIDAK SAH sebagai hasil model"))

h = pd.DataFrame(hasil)
pd.set_option("display.width", 250, "display.max_columns", 20)
print(h.drop(columns="catatan").round(3).to_string(index=False))
for r in h.itertuples():
    print(f"  {r.protokol}: {r.catatan}")
h.round(4).to_excel(OUT / "Replikasi_Protokol_Paper.xlsx", index=False)
print(f"\nDisimpan: {OUT / 'Replikasi_Protokol_Paper.xlsx'}")
print("CATATAN: hasil final skripsi tetap P0. P1-P3 hanya pembanding untuk menjelaskan perbedaan dengan paper terdahulu.")