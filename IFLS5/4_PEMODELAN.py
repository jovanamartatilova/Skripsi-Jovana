"""
4_PEMODELAN.py  -  Pemodelan 4 Skenario IFLS5 (Proposal Skripsi Bab III, Subbab 3.1.8 - 3.1.10)

Skenario:
  S1 Logistic Regression + seleksi fitur     S2 Logistic Regression tanpa seleksi fitur
  S3 Random Forest       + seleksi fitur     S4 Random Forest       tanpa seleksi fitur

Alur --protokol-paper (dipakai Model A-D; Ndagijimana dkk., 2023; Scientific Reports, 2025):
  1. Preprocessor dimuat dari 2_PREPROCESSING.py (sudah di-fit pada seluruh data, tanpa target).
  2. SMOTE pada seluruh data -> split 80:20 stratified acak (random_state 42).
  3. 5-fold stratified CV pada data latih. Di tiap fold: seleksi fitur IG + Gini (S1, S3) dan tuning
     GridSearchCV 3-fold HANYA pada training fold -> probabilitas Out-of-Fold (OOF).
  4. Threshold 0,5. Pemilihan skenario HANYA dari OOF (urutan F1 > recall > ROC-AUC).
  5. Seleksi fitur dan tuning diulang pada seluruh data latih -> keempat skenario dievaluasi sekali di data uji.
  Feature importance resmi dihitung terpisah di 5_FEATUREIMPORTANCE.py.

Input  : train.csv dan test.csv dari 3_SPLITTING.py (digabung lagi sebelum SMOTE), preprocessor.joblib
Output (folder --out, mis. 4_Modeling_B):
    Ringkasan_Pemodelan.xlsx, oof.csv, model_S1..S4.joblib, model_final.joblib, prediksi_risiko.py,
    pemodelan_kurva.png, pemodelan_importance.png, laporan_pemodelan.txt
Cara jalan : python 4_PEMODELAN.py --protokol-paper --out "...\\4_Modeling_B"
             (Model A: + --dengan-tb --pre ...\\2_Preprocessing_A\\preprocessor.joblib;
              Model C/D: + --fitur-file <txt> --pre <preprocessor C/D>)
"""
import argparse
import re
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = ImbPipeline = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custom_transformer import Winsorizer  # noqa: E402, F401  (dibutuhkan untuk membuka preprocessor.joblib)

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
SPLIT = BASE / "output" / "3_Splitting"

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=str(SPLIT / "train.csv"))
ap.add_argument("--test", default=str(SPLIT / "test.csv"))
ap.add_argument("--out", default=str(BASE / "output" / "4_Modeling_B"))

ap.add_argument(
    "--k",
    type=int,
    default=12,
    help="jumlah VARIABEL terpilih pada skenario dengan seleksi fitur",
)
ap.add_argument(
    "--wajib",
    nargs="*",
    default=[],
    help="variabel yang selalu dipilih pada skenario seleksi fitur",
)
ap.add_argument(
    "--kuota",
    type=int,
    default=1,
    help="jumlah wakil minimal tiap domain; 0 = tanpa kuota",
)
ap.add_argument(
    "--min-leaf-seleksi",
    type=int,
    default=20,
    help="min_samples_leaf pohon seleksi fitur",
)
ap.add_argument(
    "--trees", type=int, default=500, help="jumlah pohon Random Forest (model)"
)
ap.add_argument(
    "--rf-leaf",
    type=int,
    default=5,
    help="min_samples_leaf Random Forest; bandingkan hanya pada OOF",
)
ap.add_argument(
    "--trees-seleksi", type=int, default=300, help="jumlah pohon RF untuk Gini Decrease"
)
ap.add_argument("--fitur-file", default=None, help="txt daftar variabel dari 5_FEATUREIMPORTANCE.py")
ap.add_argument("--seleksi", default="iggini", choices=["iggini", "lr_pvalue"],
                help="iggini = Information Gain + Gini Decrease; lr_pvalue = regresi logistik p < 0,05 (Rahman dkk., 2021)")
ap.add_argument("--tanpa-tune", dest="tune", action="store_false", help="matikan GridSearchCV (bawaan: tuning aktif)")
ap.add_argument("--inner", type=int, default=3, help="jumlah fold inner CV untuk tuning")
ap.add_argument("--protokol-paper", action="store_true",
                help="protokol paper acuan (Ndagijimana dkk., 2023; Sci Rep, 2025): SMOTE seluruh data -> split 80:20 acak "
                     "bertingkat -> 5-fold CV acak -> threshold 0,5")
ap.add_argument("--kriteria", default="f1", choices=["f1", "youden", "accuracy"],
                help="dasar memilih threshold pada OOF: f1 (proposal), youden (recall + spesifisitas - 1), accuracy")
ap.add_argument("--smote", action="store_true", help="SMOTE di training fold (menggantikan class_weight)")
ap.add_argument("--resampler", default="smote", choices=["smote", "borderline", "adasyn", "smoteenn"],
                help="varian SMOTE di training fold (dipakai bersama --smote)")
ap.add_argument("--pre", default=str(BASE / "output" / "2_Preprocessing" / "preprocessor.joblib"))
ap.add_argument("--dengan-tb", action="store_true",
                help="Model A/C: TB anak ikut sebagai prediktor (kebocoran label; pembanding batas atas)")

args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

FITUR_PILIH = None
if args.fitur_file:
    FITUR_PILIH = [l.strip() for l in Path(args.fitur_file).read_text(encoding="utf-8").splitlines() if l.strip()]

SEED = 42
KET = (("dengan TB anak (kebocoran label, pembanding batas atas)" if args.dengan_tb else "tanpa TB, dengan WAZ")
       + (" | pemodelan ulang dengan fitur terpilih" if args.fitur_file else " | semua kandidat fitur"))
K_FITUR = args.k
THR_GRID = np.round(
    np.arange(0.01, 0.9901, 0.01), 2
)  # semua kemungkinan threshold (Leevy dkk., 2023)
ID_COLS = ["pidlink", "hhid14", "pid14"]
TARGET, GRUP = "stunting", "hhid14"
TIDAK_DIPAKAI = [
    "haz", "fold_cv", "usia_sapih_bln", "air_minum_kode", "toilet_kode", "sampah_kode",
    "bb_ayah", "imt_ayah", "umur_ayah", "pend_ayah", "kerja_ayah", "ada_ayah_di_RT",
    "bblr", "hamil_prematur_proxy", "periksa_hamil", "tablet_besi", "bb_ibu", "usia_air_bln", "air_dini",
    "usia_mpasi_bln", "imun_bcg", "imun_polio", "imun_dpt", "imun_hepb", "imun_campak", "imun_rota",
    "log_pengeluaran_pangan_perkapita", "total_hari_makan_7hr", "hari_terbaring_4mgg", "jenis_sayur_buah_7hr",
    "ukuran_bayi", "status_sehat_umum", "hanya_asi", "asi_pernah", "kerja_ibu", "ada_ibu_di_RT", "limbah_kode",
    "provinsi", "perkotaan",
    "bb_anak", "tempat_lahir_grup",
]
NOMINAL = ["provinsi", "tempat_lahir_grup", "limbah_kode"]

DOMAIN = {
    "Karakteristik anak": ["umur_bln", "jk_laki", "tb_anak", "bb_anak", "waz"],
    "Makan & kesehatan anak": [
        "frek_makan_per_hari",
        "hanya_asi",
        "jenis_makanan_7hr",
        "total_hari_makan_7hr",
        "status_sehat_umum",
        "hari_sakit_4mgg",
        "hari_terbaring_4mgg",
        "jumlah_gejala_4mgg",
        "hari_protein_hewani_7hr",
        "hari_susu_7hr",
        "jenis_sayur_buah_7hr",
        "hari_makanan_olahan_7hr",
    ],
        "Kelahiran": [
        "anak_kembar",
        "bb_lahir_kg",
        "bblr",
        "ukuran_bayi",
        "lama_hamil_minggu",
        "hamil_prematur_proxy",
        "tempat_lahir_grup",
    ],
    "Layanan kehamilan": [
        "periksa_hamil",
        "anc_total",
        "tablet_besi",
        "jml_tablet_besi",
    ],
    "ASI & MPASI": [
        "asi_pernah",
        "masih_asi",
        "usia_air_bln",
        "usia_mpasi_bln",
        "air_dini",
        "mpasi_dini",
        "asi_eksklusif_proxy",
    ],
        "Layanan anak": [
        "hb_anak",
        "frek_timbang_6bln",
        "posyandu_4mgg",
        "vit_a_6bln",
        "punya_kartu_imun",
        "imun_bcg",
        "imun_polio",
        "imun_dpt",
        "imun_hepb",
        "imun_campak",
        "imun_rota",
        "jumlah_jenis_imun",
    ],
        "Orang tua": [
        "hb_ibu",
        "tinggi_ibu", 
        "tinggi_ayah",
        "bb_ibu",
        "bb_ayah",
        "imt_ibu",
        "imt_ayah",
        "umur_ibu",
        "umur_ayah",
        "ada_ibu_di_RT",
        "ada_ayah_di_RT",
    ],
        "Air & sanitasi": [
        "garam_beryodium",
        "bahan_bakar_padat",
        "masalah_lingkungan_rumah",
        "ventilasi_cukup",
        "dapur_satu_kamar_tidur",
        "lantai_tanah",
        "air_layak",
        "air_direbus",
        "toilet_layak",
        "limbah_kode",
        "sampah_layak",
    ],
        "Sosial ekonomi": [
        "indeks_kekayaan_pca",
        "pend_ibu",
        "pend_ayah",
        "kerja_ibu",
        "kerja_ayah",
        "perkotaan",
        "provinsi",
        "kartu_sehat",
        "jumlah_anggota_RT",
        "ada_perokok_RT",
        "log_pengeluaran_perkapita",
        "log_pengeluaran_pangan_perkapita",
    ],
}
DOMAIN_VAR = {v: k for k, vs in DOMAIN.items() for v in vs}
RENTANG = {
    "tb_anak": (30, 140),
    "bb_anak": (2, 40),
    "bb_lahir_kg": (0.5, 6.5),
    "lama_hamil_minggu": (20, 50),
    "jml_tablet_besi": (0, 400),
    "tinggi_ibu": (120, 210),
    "tinggi_ayah": (120, 210),
    "bb_ibu": (25, 200),
    "bb_ayah": (25, 200),
    "imt_ibu": (10, 60),
    "imt_ayah": (10, 60),
    "umur_ibu": (10, 90),
    "umur_ayah": (10, 90),
    "anc_total": (0, 20),
    "waz": (-6, 5),
}

# skenario: id -> (algoritma, pakai seleksi fitur)
SKENARIO = {
    "S1": ("LR", True),
    "S2": ("LR", False),
    "S3": ("RF", True),
    "S4": ("RF", False),
}
if args.smote and SMOTE is None:
    raise SystemExit("SMOTE butuh: pip install imbalanced-learn")
NAMA_ALGO = {"LR": "Logistic Regression", "RF": "Random Forest",}
GRID = {
    "LR": {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10], "penalty": ["l2"]},
    "RF": {"max_depth": [4, 6, 8, 12, None], "min_samples_leaf": [1, 3, 5, 10, 20, 40],
           "max_features": ["sqrt", 0.3, 0.5, 0.8]},
}
KRITERIA_PILIH = [
    "f1",
    "recall",
    "roc_auc",
]  # urutan prioritas pemilihan skenario (dari OOF)
DT_KW = dict(
    criterion="entropy", class_weight="balanced", min_samples_leaf=args.min_leaf_seleksi
)  # Information Gain (Decision Tree)

LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------------
# DAFTAR FITUR DAN BATAS WAJAR (harus sama dengan 2_PREPROCESSING.py)
# ----------------------------------------------------------------------------
def daftar_fitur(kolom):
    buang = set(ID_COLS + [TARGET] + TIDAK_DIPAKAI + ([] if args.dengan_tb else ["tb_anak"]))
    fit = [c for c in kolom if c not in buang]
    if FITUR_PILIH is not None:
        fit = [c for c in fit if c in FITUR_PILIH]
    kat = [c for c in fit if c in NOMINAL]
    num = [c for c in fit if c not in NOMINAL]
    return num, kat


def seragamkan(d, num, kat):
    d = d[num + kat].copy()
    for c in num + kat:
        d[c] = (
            pd.to_numeric(d[c], errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
    for c, (lo, hi) in RENTANG.items():
        if c in d.columns:
            d.loc[d[c].notna() & ~d[c].between(lo, hi), c] = np.nan
    return d

def nama_kolom(pre):
    return [
        n.replace("num__", "").replace("kat__", "").replace("spl__", "")
        for n in pre.get_feature_names_out()
    ]


def variabel_asal(kol, kat):
    for c in kat:
        if kol.startswith(c + "_"):
            return c
    kol = re.sub(r"^missingindicator_", "", kol)  # kolom penanda kosong -> variabel asalnya
    return re.sub(r"_sp_\d+$", "", kol)  # kolom spline -> variabel asalnya


# ----------------------------------------------------------------------------
# SELEKSI FITUR, MODEL, THRESHOLD, METRIK
# ----------------------------------------------------------------------------
def p_value_lr(Z, y):
    """p-value uji Wald tiap kolom dari regresi logistik multivariat (tanpa penalti). Dipanggil pada data TRAIN saja."""
    from scipy.stats import norm
    X = np.column_stack([np.ones(len(Z)), Z])
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(Z, y)
    b = np.r_[lr.intercept_, lr.coef_[0]]
    p = 1 / (1 + np.exp(-X @ b))
    cov = np.linalg.pinv((X * (p * (1 - p))[:, None]).T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    return 2 * norm.sf(np.abs(b / se))[1:]


def peringkat_fitur(Z, y, nama, kat):
    """Information Gain (DT entropy) + Gini Decrease (RF gini) -> rank-sum PER VARIABEL. Dipanggil pada data TRAIN saja."""
    if args.seleksi == "lr_pvalue":
        k = pd.DataFrame({"variabel": [variabel_asal(c, kat) for c in nama], "p": p_value_lr(Z, y)})
        t = k.groupby("variabel", sort=False)["p"].min().reset_index()
        t["rank_InfoGain"] = t.p.rank(method="min")      # nama kolom dipertahankan agar laporan tetap jalan
        t["rank_Gini"] = t.rank_InfoGain
        t["rank_sum"] = t.rank_InfoGain
        t["imp_GiniDecrease"] = -t.p
        t["domain"] = t.variabel.map(DOMAIN_VAR).fillna("Lainnya")
        return t.sort_values("rank_sum").reset_index(drop=True)
    dt = DecisionTreeClassifier(random_state=SEED, **DT_KW).fit(Z, y)
    rf = RandomForestClassifier(
        n_estimators=args.trees_seleksi,
        criterion="gini",
        class_weight="balanced",
        min_samples_leaf=args.min_leaf_seleksi,
        n_jobs=-1,
        random_state=SEED,
    ).fit(Z, y)
    k = pd.DataFrame(
        {
            "variabel": [variabel_asal(c, kat) for c in nama],
            "imp_InfoGain": dt.feature_importances_,
            "imp_GiniDecrease": rf.feature_importances_,
        }
    )
    t = (
        k.groupby("variabel", sort=False)[["imp_InfoGain", "imp_GiniDecrease"]]
        .sum()
        .reset_index()
    )
    t["rank_InfoGain"] = t.imp_InfoGain.rank(ascending=False, method="min")
    t["rank_Gini"] = t.imp_GiniDecrease.rank(ascending=False, method="min")
    t["rank_sum"] = t.rank_InfoGain + t.rank_Gini
    t["domain"] = t.variabel.map(DOMAIN_VAR).fillna("Lainnya")
    # seri pada rank-sum dipecah dengan Gini Decrease terbesar (aturan tambahan agar deterministik)
    return t.sort_values(
        ["rank_sum", "imp_GiniDecrease"], ascending=[True, False]
    ).reset_index(drop=True)

def pilih_variabel(rank):
    """Wakil terbaik tiap domain (kuota) lebih dulu, lalu sisa K diisi peringkat keseluruhan."""
    if args.seleksi == "lr_pvalue":
        urut = rank[rank.p < 0.05].copy()
        if urut.empty:
            urut = rank.head(K_FITUR).copy()
        urut["alasan"] = "p < 0,05 (regresi logistik)"
        return urut
    pilih = [v for v in args.wajib if v in set(rank.variabel)]
    if args.kuota > 0:
        for _, g in rank.groupby("domain", sort=False):
            pilih += [v for v in g.variabel.head(args.kuota) if v not in pilih]
    for v in rank.variabel:
        if len(pilih) >= K_FITUR:
            break
        if v not in pilih:
            pilih.append(v)
    urut = rank[rank.variabel.isin(pilih)].copy()
    urut["alasan"] = np.where(
        urut.groupby("domain").cumcount() < args.kuota,
        "wakil domain",
        "peringkat keseluruhan",
    )
    return urut


def indeks_kolom(nama, kat, terpilih):
    s = set(terpilih)
    return [i for i, c in enumerate(nama) if variabel_asal(c, kat) in s]


def buat_model(algo):
    cw = None if args.smote else "balanced"  # kalau SMOTE, class_weight dimatikan
    if algo == "LR":
        return LogisticRegression(class_weight=cw, max_iter=5000, random_state=SEED)
    return RandomForestClassifier(
        n_estimators=args.trees,
        min_samples_leaf=args.rf_leaf,
        class_weight=cw,
        n_jobs=-1,
        random_state=SEED,
    )


def inti_model(m):
    """Ambil model murni (buang pembungkus imblearn) supaya dashboard tidak butuh imbalanced-learn."""
    return m.named_steps["m"] if hasattr(m, "named_steps") else m


def latih_model(algo, Xa, ya, ga):
    """Latih model. Opsional SMOTE dan GridSearchCV; semuanya hanya memakai data Xa (training fold)."""
    est, grid = buat_model(algo), GRID[algo]
    if args.smote:
        from imblearn.over_sampling import ADASYN, BorderlineSMOTE
        from imblearn.combine import SMOTEENN
        rs = {"smote": SMOTE(random_state=SEED), "borderline": BorderlineSMOTE(random_state=SEED),
              "adasyn": ADASYN(random_state=SEED), "smoteenn": SMOTEENN(random_state=SEED)}[args.resampler]
        est = ImbPipeline([("smote", rs), ("m", est)])
        grid = {f"m__{k}": v for k, v in grid.items()}
    if not args.tune:
        return est.fit(Xa, ya), {}
    gs = GridSearchCV(
        est, grid, scoring="roc_auc", refit=True,
        n_jobs=1 if algo == "RF" else 2,   # batasi agar memori tidak habis
        cv=StratifiedGroupKFold(args.inner, shuffle=True, random_state=SEED),
    )
    gs.fit(Xa, ya, groups=ga)
    return gs.best_estimator_, gs.best_params_


def metrik(y, p, t):
    yh = (p >= t).astype(int)
    return {
        "threshold": float(t),
        "accuracy": accuracy_score(y, yh),
        "precision": precision_score(y, yh, zero_division=0),
        "recall": recall_score(y, yh, zero_division=0),
        "f1": f1_score(y, yh, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }


def cari_threshold(y, p):
    """F1 maksimum; seri -> recall lebih besar; seri lagi -> threshold lebih tinggi."""
    rows = []
    for t in THR_GRID:
        yh = (p >= t).astype(int)
        rec = recall_score(y, yh, zero_division=0)
        spes = ((yh == 0) & (y == 0)).sum() / max((y == 0).sum(), 1)
        rows.append(
            {
                "threshold": t,
                "f1": f1_score(y, yh, zero_division=0),
                "recall": rec,
                "precision": precision_score(y, yh, zero_division=0),
                "accuracy": accuracy_score(y, yh),
                "spesifisitas": spes,
                "youden": rec + spes - 1,
            }
        )
    tab = pd.DataFrame(rows)
    best = max(
        rows, key=lambda r: (round(r[args.kriteria], 10), round(r["recall"], 10), r["threshold"])
    )
    return (0.5 if args.protokol_paper else float(best["threshold"])), tab


def tulis_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for nm, d in sheets.items():
            d.to_excel(xw, sheet_name=nm[:31], index=False)
            ws = xw.sheets[nm[:31]]
            for col in ws.columns:
                w = max(
                    len(str(c.value)) if c.value is not None else 0 for c in col[:60]
                )
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(10, w + 2), 70
                )
            ws.freeze_panes = "A2"


HELPER = '''# -*- coding: utf-8 -*-
"""Fungsi bantu dashboard: memuat model final dan menghitung probabilitas + kategori risiko stunting.
Pemakaian:
    import joblib, pandas as pd
    from prediksi_risiko import prediksi_risiko
    b = joblib.load("model_final.joblib")
    hasil = prediksi_risiko(b, pd.DataFrame([{...nilai predictor...}]))   # kolom yang tidak diisi = NaN (diimputasi)
Model & preprocessor TIDAK dilatih ulang (proposal 3.1.11.2)."""
import numpy as np
import pandas as pd


def prediksi_risiko(bundle, df_input):
    num, kat = bundle["fitur_numerik"], bundle["fitur_kategorik"]
    d = df_input.reindex(columns=num + kat).copy()
    for c in num + kat:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c, (lo, hi) in bundle["rentang"].items():
        if c in d.columns:
            d.loc[d[c].notna() & ~d[c].between(lo, hi), c] = np.nan
    Z = np.asarray(bundle["preprocessor"].transform(d), float)[:, bundle["indeks_terpilih"]]
    p = bundle["model"].predict_proba(Z)[:, 1]
    return pd.DataFrame({"probabilitas_stunting": p,
                         "kategori_risiko": np.where(p >= bundle["threshold"], "Risiko Lebih Tinggi", "Risiko Lebih Rendah")})
'''
(OUT / "prediksi_risiko.py").write_text(HELPER, encoding="utf-8")

# ============================================================================
# MUAT DATA
# ============================================================================
log(f"=== PEMODELAN {len(SKENARIO)} SKENARIO IFLS5 (Proposal 3.1.8 - 3.1.10)"
    + (" | PROTOKOL PAPER (SMOTE sebelum split, threshold 0,5)" if args.protokol_paper else "") + " ===")
tr = pd.read_csv(args.train)
te = pd.read_csv(args.test)
assert not set(tr.pidlink) & set(te.pidlink), "ada balita di train dan test"
if "fold_cv" not in tr.columns:
    log(
        "[peringatan] fold_cv tidak ada -> dibuat ulang dengan StratifiedGroupKFold (grup hhid14)"
    )
    tr["fold_cv"] = -1
    for k, (_, iv) in enumerate(
        StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            tr, tr[TARGET], groups=tr[GRUP]
        )
    ):
        tr.iloc[iv, tr.columns.get_loc("fold_cv")] = k
ytr, yte = tr[TARGET].values.astype(int), te[TARGET].values.astype(int)
fold_arr = tr.fold_cv.values
FOLDS = sorted(np.unique(fold_arr))
log(
    f"[data] train {tr.shape}, test {te.shape} | stunting train {ytr.mean() * 100:.2f}%, test {yte.mean() * 100:.2f}% "
    f"| fold CV: {len(FOLDS)} | K seleksi fitur = {K_FITUR}"
)

log(f"\n{'=' * 78}\nPEMODELAN: {KET}\n{'=' * 78}")
log(
    "  CATATAN: TB anak ikut sebagai prediktor. HAZ dihitung dari TB, sehingga hasil model ini adalah batas atas "
    "(kebocoran label), bukan model yang dapat dipakai." if args.dengan_tb else
    "  CATATAN: set ini memuat WAZ (tanpa TB). BB/U berkaitan erat dengan status gizi, "
    "sehingga sebagian kinerja berasal dari ukuran tubuh anak. Tulis di keterbatasan."
)
num, kat = daftar_fitur(tr.columns)
Xtr, Xte = seragamkan(tr, num, kat), seragamkan(te, num, kat)
# Preprocessor dimuat dari 2_PREPROCESSING.py (sudah di-fit pada seluruh data, tanpa target)
pre_all = joblib.load(args.pre)
assert set(pre_all.feature_names_in_) == set(num + kat), (
    "Fitur preprocessor beda dengan fitur 4_PEMODELAN. Jalankan ulang 2_PREPROCESSING.py "
    "dengan --fitur-file yang sama."
)
Ztr_all = np.asarray(pre_all.transform(Xtr), float)
Zte_all = np.asarray(pre_all.transform(Xte), float)
nama_all = nama_kolom(pre_all)
log(f"  fitur mentah: {len(num)} numerik/biner/ordinal + {len(kat)} nominal")
if args.protokol_paper:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    if SMOTE is None:
        raise SystemExit("Protokol paper butuh: pip install imbalanced-learn")
    Z_all, y_all = np.vstack([Ztr_all, Zte_all]), np.r_[ytr, yte]
    Z_sm, y_sm = SMOTE(random_state=SEED).fit_resample(Z_all, y_all)
    sint = np.r_[np.zeros(len(y_all), bool), np.ones(len(y_sm) - len(y_all), bool)]
    i_a, i_b = train_test_split(np.arange(len(y_sm)), test_size=0.2, stratify=y_sm, random_state=SEED)
    Ztr_all, Zte_all, ytr, yte = Z_sm[i_a], Z_sm[i_b], y_sm[i_a], y_sm[i_b]
    fold_arr = np.zeros(len(ytr), int)
    for k_, (_, iv_) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Ztr_all, ytr)):
        fold_arr[iv_] = k_
    FOLDS = sorted(np.unique(fold_arr))
    grup_pp = np.arange(len(ytr))   # tiap baris dianggap kelompok sendiri (split acak per baris)
    log(f"  [PROTOKOL PAPER] SMOTE seluruh data: {len(y_all)} -> {len(y_sm)} baris ({sint.sum()} sintetis); "
        f"split 80:20 acak -> latih {len(ytr)}, uji {len(yte)}; data sintetis di uji: {sint[i_b].sum()} "
        f"({sint[i_b].mean() * 100:.1f}%), dicatat di keterbatasan; threshold 0,5")

# ------------------------------------------------------------------
# 1. CROSS-VALIDATION -> OOF
# ------------------------------------------------------------------
oof = {s: np.full(len(ytr), np.nan) for s in SKENARIO}
grup_arr = grup_pp if args.protokol_paper else tr[GRUP].values
param_rows = []
sel_rows = []
for k in FOLDS:
    i_tr, i_va = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
    Zt, Zv = Ztr_all[i_tr], Ztr_all[i_va]  # preprocessing sudah di-fit di awal
    nama_k = nama_all
    rank = peringkat_fitur(
        Zt, ytr[i_tr], nama_k, kat
    )  # seleksi HANYA training fold (dipakai S1 & S3)
    pilih_k = pilih_variabel(rank)
    top = pilih_k.variabel.tolist()
    idx_top = indeks_kolom(nama_k, kat, top)
    for r in pilih_k.itertuples():
        sel_rows.append(
            {
                "fold": int(k),
                "peringkat": len([x for x in sel_rows if x["fold"] == int(k)]) + 1,
                "kolom": r.variabel,
                "domain": r.domain,
                "alasan": r.alasan,
                "rank_InfoGain": r.rank_InfoGain,
                "rank_Gini": r.rank_Gini,
                "rank_sum": r.rank_sum,
            }
        )
    for sid, (algo, fs) in SKENARIO.items():
        cols = idx_top if fs else list(range(len(nama_k)))
        m, bp = latih_model(algo, Zt[:, cols], ytr[i_tr], grup_arr[i_tr])
        oof[sid][i_va] = m.predict_proba(Zv[:, cols])[:, 1]
        param_rows.append({"fold": int(k), "skenario": sid, **bp})
    log(
        f"  [CV fold {int(k)}] train {len(i_tr)}, valid {len(i_va)}, kolom hasil {len(nama_k)}; "
        f"fitur terpilih: {', '.join(top[:5])}, ..."
    )
assert all(not np.isnan(v).any() for v in oof.values()), "OOF belum terisi penuh"

# ------------------------------------------------------------------
# 2. THRESHOLD (dari OOF) + METRIK OOF + METRIK PER FOLD
# ------------------------------------------------------------------
thr, thr_tab, oof_rows, fold_rows = {}, [], [], []
for sid, (algo, fs) in SKENARIO.items():
    t, tab = cari_threshold(ytr, oof[sid])
    thr[sid] = t
    thr_tab.append(tab.assign(skenario=sid))
    mo = metrik(ytr, oof[sid], t)
    oof_rows.append(
        {
            "skenario": sid,
            "algoritma": NAMA_ALGO[algo],
            "seleksi_fitur": "ya" if fs else "tidak",
            **mo,
        }
    )
    for k in FOLDS:
        msk = fold_arr == k
        fold_rows.append(
            {"skenario": sid, "fold": int(k), **metrik(ytr[msk], oof[sid][msk], t)}
        )
oof_tab = pd.DataFrame(oof_rows)
fold_tab = pd.DataFrame(fold_rows)
thr_tab = pd.concat(thr_tab, ignore_index=True)
log("\n  Hasil OOF pada TRAIN (threshold " + ("0,5" if args.protokol_paper else "terbaik dari OOF") + "):")
log(oof_tab.round(4).to_string(index=False))

# ------------------------------------------------------------------
# 3. PEMILIHAN SKENARIO (hanya dari OOF)
# ------------------------------------------------------------------
urut = oof_tab.sort_values(KRITERIA_PILIH, ascending=False).reset_index(drop=True)
urut.insert(0, "peringkat", range(1, len(urut) + 1))
terpilih = urut.loc[0, "skenario"]
algo_t, fs_t = SKENARIO[terpilih]
selisih = urut.loc[0, "f1"] - urut.loc[1, "f1"]
log(
    f"\n  Skenario terpilih (urut {' > '.join(KRITERIA_PILIH)} pada OOF): {terpilih} "
    f"({NAMA_ALGO[algo_t]}, {'dengan' if fs_t else 'tanpa'} seleksi fitur), threshold {thr[terpilih]:.2f}"
)
log(
    f"  Selisih F1 OOF dengan peringkat 2: {selisih:.4f}"
    + (
        "  -> sangat tipis; pertimbangkan model yang lebih sederhana/interpretabel"
        if selisih < 0.005
        else ""
    )
)

# ------------------------------------------------------------------
# 4. REFIT AKHIR pada SELURUH TRAIN -> EVALUASI TEST
# ------------------------------------------------------------------
pre_full = pre_all
Zt_full, Zte = Ztr_all, Zte_all
nama_full = nama_all
rank_full = peringkat_fitur(Zt_full, ytr, nama_full, kat)
pilih_full = pilih_variabel(rank_full)
top_full = pilih_full.variabel.tolist()
idx_full = indeks_kolom(nama_full, kat, top_full)

test_rows, cm_rows, proba_te, modeluji, param_final = [], [], {}, {}, {}
for sid, (algo, fs) in SKENARIO.items():
    cols = idx_full if fs else list(range(len(nama_full)))
    m, bp = latih_model(algo, Zt_full[:, cols], ytr, grup_arr)
    param_final[sid] = bp
    m = inti_model(m)
    p = m.predict_proba(Zte[:, cols])[:, 1]
    proba_te[sid], modeluji[sid] = p, (m, cols)
    mt = metrik(yte, p, thr[sid])
    test_rows.append(
        {
            "skenario": sid,
            "algoritma": NAMA_ALGO[algo],
            "seleksi_fitur": "ya" if fs else "tidak",
            **mt,
        }
    )
    tn, fp, fn, tp = confusion_matrix(yte, (p >= thr[sid]).astype(int)).ravel()
    cm_rows.append({"skenario": sid, "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    bundle = {
        "skenario": sid,
        "algoritma": NAMA_ALGO[algo],
        "seleksi_fitur": fs,
        "threshold": thr[sid],
        "preprocessor": pre_full,
        "kolom_hasil": nama_full,
        "kolom_terpilih": [nama_full[i] for i in cols],
        "indeks_terpilih": cols,
        "model": m,
        "fitur_numerik": num,
        "fitur_kategorik": kat,
        "rentang": RENTANG,
        "versi_sklearn": sklearn.__version__,
        "seed": SEED,
        "K_fitur": K_FITUR,
    }
    joblib.dump(bundle, OUT / f"model_{sid}.joblib")
    if sid == terpilih:
        joblib.dump(bundle, OUT / f"model_final.joblib")
test_tab = pd.DataFrame(test_rows)
cm_tab = pd.DataFrame(cm_rows)
log("\n  Hasil TEST (threshold " + ("0,5" if args.protokol_paper else "dikunci dari OOF")
    + "; tidak dipakai untuk memilih):")
log(test_tab.round(4).to_string(index=False))

gap = (
    (oof_tab.set_index("skenario").roc_auc - test_tab.set_index("skenario").roc_auc)
    .abs()
    .max()
)
log(
    f"  Selisih AUC OOF vs test terbesar: {gap:.4f} (kecil = generalisasi konsisten)"
)

# ------------------------------------------------------------------
# 5. FEATURE IMPORTANCE (model terpilih)
# ------------------------------------------------------------------
m_t, cols_t = modeluji[terpilih]
nama_t = [nama_full[i] for i in cols_t]
if algo_t == "LR":
    koef = m_t.coef_[0]
    fi = pd.DataFrame(
        {
            "kolom": nama_t,
            "koefisien": koef,
            "odds_ratio_per_SD": np.exp(koef),
            "importance": np.abs(koef),
            "jenis_importance": "|koefisien| (fitur terstandardisasi)",
        }
    )
elif hasattr(m_t, "feature_importances_"):
    fi = pd.DataFrame(
        {
            "kolom": nama_t,
            "importance": m_t.feature_importances_,
            "jenis_importance": "Gini importance (mean decrease impurity)",
        }
    )
fi["variabel_asal"] = [variabel_asal(c, kat) for c in fi.kolom]
fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
fi_var = (
    fi.groupby("variabel_asal")[["importance"]]
    .sum()
    .sort_values("importance", ascending=False)
    .reset_index()
)
log(f"\n  Feature importance model {terpilih} (10 teratas, tingkat kolom):")
log(
    fi[["kolom", "importance"]]
    .head(10)
    .round(4)
    .to_string(index=False)
)
log(
    "  CATATAN: importance bawaan model (deskriptif). Feature importance resmi (permutation, data latih asli) "
    "ada di 5_FEATUREIMPORTANCE.py; importance bukan hubungan sebab-akibat (proposal 2.6.6)."
)

# ------------------------------------------------------------------
# 6. SIMPAN
# ------------------------------------------------------------------
sel_tab = pd.DataFrame(sel_rows)
stab = (
    sel_tab.groupby("kolom")
    .agg(n_fold_terpilih=("fold", "nunique"), rata2_rank_sum=("rank_sum", "mean"))
    .reset_index()
    .sort_values(["n_fold_terpilih", "rata2_rank_sum"], ascending=[False, True])
)
stab["terpilih_pada_refit_penuh"] = stab.kolom.isin(top_full).map(
    {True: "ya", False: "tidak"}
)
final_sel = pilih_full.copy()
ringkas = pd.DataFrame(
    [
        ["Set prediktor", KET],
        ["Fitur mentah (numerik+nominal)", f"{len(num)} + {len(kat)}"],
        ["Kolom setelah preprocessing (train penuh)", len(nama_full)],
        [
            "Jumlah variabel terpilih (K) / kuota per domain",
            f"{K_FITUR} / {args.kuota}",
        ],
        ["Fold CV", len(FOLDS)],
        [
            "Skenario terpilih",
            f"{terpilih}: {NAMA_ALGO[algo_t]}, {'dengan' if fs_t else 'tanpa'} seleksi fitur",
        ],
        ["Threshold terkunci", thr[terpilih]],
        [
            "F1 / Recall / AUC OOF (terpilih)",
            " / ".join(
                f"{oof_tab.set_index('skenario').loc[terpilih, c]:.4f}"
                for c in ["f1", "recall", "roc_auc"]
            ),
        ],
        [
            "F1 / Recall / AUC TEST (terpilih)",
            " / ".join(
                f"{test_tab.set_index('skenario').loc[terpilih, c]:.4f}"
                for c in ["f1", "recall", "roc_auc"]
            ),
        ],
        ["Kriteria pemilihan (OOF)", " > ".join(KRITERIA_PILIH)],
        [
            "Hyperparameter",
            "hasil GridSearchCV di tiap training fold (lihat sheet 13_Param_Final)" if args.tune
            else f"LR: max_iter=5000 | RF: n_estimators={args.trees}, min_samples_leaf={args.rf_leaf}; class_weight=balanced",
        ],
        ["random_state", SEED],
    ],
    columns=["indikator", "nilai"],
)
tulis_excel(
    OUT / f"Ringkasan_Pemodelan.xlsx",
    {
        "00_Ringkasan": ringkas,
        "01_Metrik_OOF": oof_tab,
        "02_Pemilihan_Model": urut,
        "03_Metrik_Test": test_tab,
        "04_Confusion_Test": cm_tab,
        "05_Metrik_Per_Fold": fold_tab,
        "06_Threshold_Grid": thr_tab.round(4),
        "07_Fitur_Per_Fold": sel_tab,
        "08_Stabilitas_Fitur": stab,
        "09_Fitur_Refit_Penuh": final_sel,
        "10_Importance_Kolom": fi,
        "11_Importance_Variabel": fi_var,
        "12_Param_Per_Fold": pd.DataFrame(param_rows),
        "13_Param_Final": pd.DataFrame([{"skenario": s, **p} for s, p in param_final.items()]),
    },
)
oof_df = (pd.DataFrame({TARGET: ytr, "fold_cv": fold_arr}) if args.protokol_paper
          else tr[ID_COLS + [TARGET, "fold_cv"]].copy())
for sid in SKENARIO:
    oof_df[f"p_oof_{sid}"] = oof[sid]
oof_df.to_csv(OUT / f"oof.csv", index=False)

# grafik 1: ROC OOF + F1 vs threshold
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
warna = {"S1": "#4C78A8", "S2": "#72B7B2", "S3": "#E45756", "S4": "#F58518"}
for sid, (algo, fs) in SKENARIO.items():
    lab = f"{sid} {algo}{'+FS' if fs else ''}"
    fpr, tpr, _ = roc_curve(ytr, oof[sid])
    ax[0].plot(
        fpr,
        tpr,
        color=warna[sid],
        label=f"{lab} (AUC {roc_auc_score(ytr, oof[sid]):.3f})",
    )
    sub = thr_tab[thr_tab.skenario == sid]
    ax[1].plot(sub.threshold, sub.f1, color=warna[sid], label=lab)
    ax[1].axvline(thr[sid], color=warna[sid], ls=":", lw=0.8)
    fpr, tpr, _ = roc_curve(yte, proba_te[sid])
    ax[2].plot(
        fpr,
        tpr,
        color=warna[sid],
        label=f"{lab} (AUC {roc_auc_score(yte, proba_te[sid]):.3f})",
    )
for a, jd in zip(
    ax, ["ROC OOF (train)", "F1 OOF menurut threshold", "ROC data TEST"]
):
    a.set_title(jd)
    a.legend(fontsize=7)
for a in (ax[0], ax[2]):
    a.plot([0, 1], [0, 1], color="gray", ls=":")
    a.set(xlabel="False positive rate", ylabel="True positive rate")
ax[1].set(xlabel="Threshold", ylabel="F1-score")
fig.suptitle(KET, y=1.02)
fig.savefig(OUT / f"pemodelan_kurva.png")
plt.close(fig)

# grafik 2: feature importance model terpilih
top15 = fi.head(15).iloc[::-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
ax[0].barh(top15.kolom, top15.importance, color="#4C78A8")
ax[0].set(
    title=f"Importance model {terpilih} (kolom)", xlabel=fi.jenis_importance.iloc[0]
)
tv = fi_var.head(15).iloc[::-1]
ax[1].barh(tv.variabel_asal, tv.importance, color="#F58518")
ax[1].set(title="Dijumlah per variabel asal", xlabel="importance")
fig.savefig(OUT / f"pemodelan_importance.png")
plt.close(fig)
log(f"\nHasil disimpan di: {OUT}")
log("  Ringkasan_Pemodelan.xlsx, oof.csv, model_S1..S4.joblib, model_final.joblib,")
log("  prediksi_risiko.py, pemodelan_kurva.png, pemodelan_importance.png, laporan_pemodelan.txt")
log(
    "CATATAN: test hanya dipakai SEKALI untuk evaluasi akhir; pemilihan skenario murni dari OOF train"
    + ("; threshold 0,5." if args.protokol_paper else "; threshold dari OOF train.")
)
(OUT / "laporan_pemodelan.txt").write_text("\n".join(LOG), encoding="utf-8")
print("Selesai.")