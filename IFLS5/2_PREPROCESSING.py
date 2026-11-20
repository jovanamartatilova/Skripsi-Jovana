"""
2_PREPROCESSING.py - Pre-processing IFLS5 (Proposal 3.1.3.b), dijalankan SEBELUM splitting.
Parameter (median, mean/SD, persentil winsorizing, kategori one-hot) dipelajari dari SELURUH data tanpa memakai target.
Catatan keterbatasan: test ikut memengaruhi parameter preprocessing.
Seleksi fitur, SMOTE, dan tuning TIDAK di sini; semuanya di dalam training fold (4_PEMODELAN.py).

Output (folder 2_Preprocessing): preprocessor.joblib, fitur.json, balita_preprocessed.csv, Ringkasan_Preprocessing.xlsx,
        viz_outlier_sebelum_sesudah.png, viz_missing_sebelum_sesudah.png, viz_missing_distribusi.png
Cara jalan: python 2_PREPROCESSING.py            (tambahkan --spline bila ingin spline umur & waz)
"""
import argparse, json, sys, warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from custom_transformer import Winsorizer

warnings.filterwarnings("ignore")

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
BUILD = BASE / "output" / "BuildDataset"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(BUILD / "balita_ifls5.csv"))
ap.add_argument("--out", default=str(BASE / "output" / "2_Preprocessing"))
ap.add_argument("--spline", action="store_true", help="spline pada umur_bln dan waz")
ap.add_argument("--missing-indicator", action="store_true",
                help="tambahkan kolom 0/1 penanda nilai kosong untuk setiap variabel yang punya missing value")
ap.add_argument("--tanpa-winsor", action="store_true",
                help="matikan winsorizing (pembanding sebelum penanganan outlier)")
ap.add_argument("--fitur-file", default=None, help="txt daftar variabel dari 5_FEATUREIMPORTANCE.py")
ap.add_argument("--dengan-tb", action="store_true",
                help="Model A/C: TB anak ikut sebagai prediktor (kebocoran label; pembanding batas atas)")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

ID_COLS = ["pidlink", "hhid14", "pid14"]
TARGET = "stunting"
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
ORDINAL = ["ukuran_bayi", "pend_ibu", "pend_ayah", "jumlah_jenis_imun", "status_sehat_umum"]
WINS = ["bb_lahir_kg", "tinggi_ibu", "tinggi_ayah", "imt_ibu", "umur_ibu", "jml_tablet_besi"]
WINS_LO, WINS_HI = (0, 100) if args.tanpa_winsor else (1, 99)   # 0-100 = batas min-maks, tidak ada yang dipangkas
SPLINE_VARS = ["umur_bln", "waz"]
RENTANG = {
    "tb_anak": (30, 140), "bb_anak": (2, 40), "bb_lahir_kg": (0.5, 6.5), "lama_hamil_minggu": (20, 50),
    "jml_tablet_besi": (0, 400), "tinggi_ibu": (120, 210), "tinggi_ayah": (120, 210),
    "bb_ibu": (25, 200), "bb_ayah": (25, 200), "imt_ibu": (10, 60), "imt_ayah": (10, 60),
    "umur_ibu": (10, 90), "umur_ayah": (10, 90), "anc_total": (0, 20), "waz": (-6, 5),
}
FITUR_PILIH = None
if args.fitur_file:
    FITUR_PILIH = [l.strip() for l in Path(args.fitur_file).read_text(encoding="utf-8").splitlines() if l.strip()]


def daftar_fitur(kolom):
    buang = set(ID_COLS + [TARGET] + TIDAK_DIPAKAI + ([] if args.dengan_tb else ["tb_anak"]))
    fit = [c for c in kolom if c not in buang]
    if FITUR_PILIH is not None:
        fit = [c for c in fit if c in FITUR_PILIH]
    return [c for c in fit if c not in NOMINAL], [c for c in fit if c in NOMINAL]


def seragamkan(d, num, kat):
    """Langkah 1: ke float, inf -> NaN, nilai di luar rentang kewajaran -> NaN."""
    d = d[num + kat].copy()
    for c in num + kat:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    n_luar = {}
    for c, (lo, hi) in RENTANG.items():
        if c in d.columns:
            m = d[c].notna() & ~d[c].between(lo, hi)
            n_luar[c] = int(m.sum())
            d.loc[m, c] = np.nan
    return d, n_luar


def buat_preprocessor(num, kat):
    """Langkah 2-5: winsorizing -> imputasi median -> standardisasi (numerik); imputasi modus -> one-hot (kategorik)."""
    idx = [num.index(c) for c in WINS if c in num]
    num_pipe = Pipeline([("win", Winsorizer(idx=idx, lo=WINS_LO, hi=WINS_HI)),
                         ("imp", SimpleImputer(strategy="median", add_indicator=args.missing_indicator)),
                         ("sc", StandardScaler())])
    parts = [("num", num_pipe, num)]
    if kat:
        parts.append(("kat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), kat))
    spl = [c for c in SPLINE_VARS if c in num] if args.spline else []
    if spl:
        parts.append(("spl", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sp", SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False)),
                                       ("sc", StandardScaler())]), spl))
    return ColumnTransformer(parts, sparse_threshold=0)


def nama_bersih(n):
    return n.replace("num__", "").replace("kat__", "").replace("spl__", "")


print("=== PRE-PROCESSING IFLS5 (sebelum splitting) ===")
df = pd.read_csv(args.csv)
num, kat = daftar_fitur(df.columns)
X, n_luar = seragamkan(df, num, kat)
print(f"[data] {df.shape} | fitur: {len(num)} numerik/biner/ordinal + {len(kat)} nominal")
if FITUR_PILIH is None:
    harus = 46 if args.dengan_tb else 45
    assert len(num) + len(kat) == harus, f"jumlah fitur {len(num) + len(kat)}, seharusnya {harus}; cek TIDAK_DIPAKAI vs 1_EDA.py"

pre = buat_preprocessor(num, kat).fit(X)
Z = np.asarray(pre.transform(X), float)
nama = [nama_bersih(n) for n in pre.get_feature_names_out()]
npipe = pre.named_transformers_["num"]
win, imp_n, sc = npipe.named_steps["win"], npipe.named_steps["imp"], npipe.named_steps["sc"]
idx_win = list(win.idx or [])
var_win = [num[j] for j in idx_win]

# ---- simpan ----
o = pd.DataFrame(Z, columns=nama)
for j, c in enumerate(ID_COLS + [TARGET]):
    o.insert(j, c, df[c].values)
o.to_csv(OUT / "balita_preprocessed.csv", index=False)
joblib.dump(pre, OUT / "preprocessor.joblib")
(OUT / "fitur.json").write_text(json.dumps({
    "versi_sklearn": sklearn.__version__, "spline": bool(args.spline),
    "fitur_numerik": num, "fitur_kategorik": kat, "nama_kolom_hasil": nama,
}, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- tabel penjelas ----
def tipe_var(c):
    if c in NOMINAL: return "nominal"
    if c in ORDINAL: return "ordinal"
    if set(X[c].dropna().unique()) <= {0.0, 1.0}: return "biner"
    return "numerik"

t_tipe = pd.DataFrame({"variabel": num + kat, "tipe": [tipe_var(c) for c in num + kat],
                       "pct_missing": [round(X[c].isna().mean() * 100, 2) for c in num + kat]})
t_konv = pd.DataFrame({"variabel": list(n_luar), "di_luar_rentang_jadi_NaN": list(n_luar.values())})
t_imp = pd.DataFrame({"variabel": num, "metode": "median", "nilai_imputasi": imp_n.statistics_.round(4)})
if kat:
    ik = pre.named_transformers_["kat"].named_steps["imp"]
    t_imp = pd.concat([t_imp, pd.DataFrame({"variabel": kat, "metode": "modus", "nilai_imputasi": ik.statistics_})])
t_win = pd.DataFrame({"variabel": var_win, "batas_bawah_P1": [win.lo_[j] for j in idx_win],
                      "batas_atas_P99": [win.hi_[j] for j in idx_win]})
ind = ([f"missingindicator_{num[i]}" for i in imp_n.indicator_.features_]
       if getattr(imp_n, "indicator_", None) is not None else [])
t_std = pd.DataFrame({"variabel": num + ind, "mean": sc.mean_.round(4), "SD": sc.scale_.round(4)})
rows = []
if kat:
    oh = pre.named_transformers_["kat"].named_steps["oh"]
    rows = [{"variabel": c, "n_kategori": len(k), "kategori": ", ".join(f"{x:g}" for x in k)}
            for c, k in zip(kat, oh.categories_)]
t_oh = pd.DataFrame(rows, columns=["variabel", "n_kategori", "kategori"])

# ---- pemeriksaan ----
nn = [i for i, n in enumerate(nama) if n in num]
cek = pd.DataFrame([
    ["Tidak ada NaN setelah transformasi", int(np.isnan(Z).sum()) == 0, int(np.isnan(Z).sum())],
    ["Standardisasi: mean = 0 dan SD = 1 (seluruh data)",
     bool(np.all(np.abs(Z[:, nn].mean(0)) < 1e-8) and np.all((np.abs(Z[:, nn].std(0) - 1) < 1e-8) | (Z[:, nn].std(0) == 0))), "ok"],
    ["Jumlah baris hasil = jumlah baris input", Z.shape[0] == len(df), f"{Z.shape[0]} = {len(df)}"],
], columns=["pemeriksaan", "lulus", "nilai"])
cek["hasil"] = np.where(cek.lulus, "LULUS", "GAGAL")
print(cek[["pemeriksaan", "hasil", "nilai"]].to_string(index=False))

ringkas = pd.DataFrame([
    ["Urutan", "preprocessing (seluruh data, tanpa target) -> splitting -> seleksi fitur/model di training fold"],
    ["Fitur numerik / nominal", f"{len(num)} / {len(kat)}"], ["Kolom hasil", Z.shape[1]],
    ["Winsorizing", f"P{WINS_LO}-P{WINS_HI} pada {len(var_win)} variabel"],
    ["Spline umur & waz", "ya" if args.spline else "tidak"],
    ["Keterbatasan", "parameter preprocessing dihitung dari seluruh data (tidak memakai target)"],
], columns=["indikator", "nilai"])
with pd.ExcelWriter(OUT / "Ringkasan_Preprocessing.xlsx", engine="openpyxl") as xw:
    for nm, d in {"00_Ringkasan": ringkas, "01_Tipe_Variabel": t_tipe, "02_Konversi_NaN": t_konv,
                  "03_Imputasi": t_imp, "04_Winsorizing": t_win, "05_Standardisasi": t_std,
                  "06_OneHot": t_oh, "07_Pemeriksaan": cek}.items():
        d.to_excel(xw, sheet_name=nm, index=False)
# ---- visualisasi sebelum vs sesudah penanganan outlier dan missing value ----
# memakai batas winsorizing dan nilai imputasi yang BENAR-BENAR dipelajari preprocessor di atas
LABEL = {"bb_lahir_kg": "Berat lahir (kg)", "tinggi_ibu": "Tinggi ibu (cm)", "tinggi_ayah": "Tinggi ayah (cm)",
         "imt_ibu": "IMT ibu (kg/m²)", "umur_ibu": "Umur ibu (tahun)", "jml_tablet_besi": "Jumlah tablet besi"}
MERAH, BIRU, ORANYE, HIJAU = "#E45756", "#4C78A8", "#F58518", "#54A24B"
Xw = X[num].copy()                                          # sesudah winsorizing, sebelum imputasi
for j in idx_win:
    Xw[num[j]] = Xw[num[j]].clip(win.lo_[j], win.hi_[j])
med = pd.Series(imp_n.statistics_, index=num)
Xi = Xw.fillna(med)                                         # sesudah imputasi

if var_win:
    nr = int(np.ceil(len(var_win) / 3))
    fig, axs = plt.subplots(nr, 3, figsize=(15, 4.3 * nr), squeeze=False)
    for ax, j in zip(axs.ravel(), idx_win):
        c = num[j]
        n_ganti = int(((X[c] < win.lo_[j]) | (X[c] > win.hi_[j])).sum())
        b = ax.boxplot([X[c].dropna(), Xw[c].dropna()], tick_labels=["Sebelum", "Sesudah"], widths=0.55,
                       patch_artist=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
        for kotak, warna in zip(b["boxes"], [MERAH, BIRU]):
            kotak.set_facecolor(warna); kotak.set_alpha(0.5)
        for v in (win.lo_[j], win.hi_[j]):
            ax.axhline(v, color="gray", ls="--", lw=0.8)
        ax.set_title(f"{LABEL.get(c, c)}\nP{WINS_LO} = {win.lo_[j]:.2f} | P{WINS_HI} = {win.hi_[j]:.2f} | "
                     f"{n_ganti} nilai diganti", fontsize=9)
    for ax in axs.ravel()[len(var_win):]:
        ax.axis("off")
    fig.suptitle(f"Penanganan outlier: sebelum vs sesudah winsorizing (P{WINS_LO}-P{WINS_HI})\n"
                 "garis putus-putus = batas winsorizing", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "viz_outlier_sebelum_sesudah.png", dpi=150); plt.close(fig)

tipe = pd.Series({c: tipe_var(c) for c in num})
tab = pd.DataFrame({"pct": Xw.isna().mean() * 100, "tipe": tipe, "median": med})
tab = tab[tab.pct > 0].sort_values("pct")
WARNA = {"numerik": BIRU, "biner": ORANYE, "ordinal": HIJAU}
fig, ax = plt.subplots(1, 2, figsize=(14, 0.28 * len(tab) + 2), sharey=True)
ax[0].barh(tab.index, tab.pct, color=tab.tipe.map(WARNA))
for i, v in enumerate(tab.pct):
    ax[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
ax[0].set(title=f"SEBELUM imputasi ({len(tab)} dari {len(num)} fitur ada yang kosong)", xlabel="% data kosong")
ax[1].barh(tab.index, Xi[tab.index].isna().mean() * 100, color=tab.tipe.map(WARNA))
ax[1].set(title="SESUDAH imputasi", xlabel="% data kosong", xlim=ax[0].get_xlim())
ax[1].text(0.5, 0.5, f"0% kosong pada\nseluruh {len(num)} fitur", transform=ax[1].transAxes,
           ha="center", va="center", fontsize=13, color="gray")
ax[0].legend(handles=[Patch(color=v, label=f"{k} (median)") for k, v in WARNA.items()], loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "viz_missing_sebelum_sesudah.png", dpi=150); plt.close(fig)

atas = tab.sort_values("pct", ascending=False)
top = atas[atas.tipe == "numerik"].index[:6].tolist() + atas[atas.tipe == "biner"].index[:2].tolist()
fig, axs = plt.subplots(2, 4, figsize=(17, 7.5))
for ax, c in zip(axs.ravel(), top):
    if tab.tipe[c] == "numerik":
        bins = np.histogram_bin_edges(Xi[c], bins=30)
        ax.hist(Xi[c], bins=bins, color=BIRU, alpha=0.45, label="sesudah (diimputasi)")
        ax.hist(Xw[c].dropna(), bins=bins, color=MERAH, alpha=0.55, label="sebelum (data terisi)")
        ax.axvline(med[c], color="k", ls="--", lw=1)
        ax.set_title(f"{c}\n{tab.pct[c]:.1f}% kosong -> diisi median {med[c]:.2f}", fontsize=9)
    else:
        sb, ss = Xw[c].value_counts(normalize=True), Xi[c].value_counts(normalize=True)
        x = np.array([0, 1])
        ax.bar(x - 0.2, [sb.get(0, 0) * 100, sb.get(1, 0) * 100], 0.4, color=MERAH, alpha=0.7, label="sebelum (data terisi)")
        ax.bar(x + 0.2, [ss.get(0, 0) * 100, ss.get(1, 0) * 100], 0.4, color=BIRU, alpha=0.7, label="sesudah (diimputasi)")
        ax.set_xticks(x, ["0 (tidak)", "1 (ya)"]); ax.set_ylabel("% balita")
        ax.set_title(f"{c} (biner)\n{tab.pct[c]:.1f}% kosong -> diisi {med[c]:.0f}", fontsize=9)
    ax.legend(fontsize=7)
for ax in axs.ravel()[len(top):]:
    ax.axis("off")
fig.suptitle("Penanganan missing value: distribusi sebelum vs sesudah imputasi (garis putus-putus = median)", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "viz_missing_distribusi.png", dpi=150); plt.close(fig)
print("Gambar: viz_outlier_sebelum_sesudah.png, viz_missing_sebelum_sesudah.png, viz_missing_distribusi.png")
print(f"Hasil disimpan di: {OUT}\nSelesai.")