"""
1_EDA.py  -  Eksplorasi Data IFLS5 (sesuai Proposal Skripsi Bab III, Subbab 3.1.2.b)

Urutan sheet/gambar mengikuti 8 butir di proposal:
  1. Struktur dan karakteristik dataset      -> 01_Audit_Katalog, 02_Struktur_Variabel, 03_Pemetaan_Paper
  2. Penggabungan sumber data                -> 04_Sumber_Data, 17_Laporan_Build
  3. Pemilihan subjek balita (0-59 bulan)    -> 05_Funnel_Subjek, 06_Umur_JK, gambar 01
  4. Pembentukan target stunting (HAZ)       -> 07_Target_HAZ, 07b_Prevalensi_Subgrup, gambar 02
  5. Missing value                           -> 08_Missing, 09_Missing_Struktural, 09b, 09c, gambar 03
  6. Duplikat dan pengenal individu          -> 10_Duplikat_Keluarga, 10b_Balita_per_Keluarga
  7. Nilai ekstrem (outlier)                 -> 11_Outlier, 12_Cek_Logika, gambar 04
  8. Hubungan antarvariabel (data latih)     -> 13_Korelasi_Tinggi, 13b, 13c_Korelasi_Fitur_Model, 14_Univariat,
                                                15_VIF, 18_Seleksi_Variabel, gambar 05-06

"""

import argparse
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
BUILD = BASE / "output" / "BuildDataset"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(BUILD / "balita_ifls5.csv"))
ap.add_argument("--ringkasan", default=str(BUILD / "Ringkasan_Dataset.xlsx"))
ap.add_argument("--laporan", default=str(BUILD / "laporan_build.txt"))
ap.add_argument("--kamus", default="kamus_variabel_IFLS5.xlsx")
ap.add_argument("--pemetaan", default="pemetaan_variabel_paper_ke_IFLS5.xlsx")
ap.add_argument("--out", default=str(BASE / "output" / "1_EDA"))
args = ap.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


def cari(path_str):
    """Kembalikan Path jika ada (cek juga beberapa folder umum), selain itu None."""
    p = Path(path_str)
    if p.exists():
        return p
    for d in [BUILD, BASE, BASE / "output", BASE.parent, Path.cwd()]:
        if (d / p.name).exists():
            return d / p.name
    return None


ID_COLS = ["pidlink", "hhid14", "pid14"]
TARGET = ["haz", "stunting"]
NOMINAL = [
    "provinsi",
    "tempat_lahir_grup",
    "air_minum_kode",
    "toilet_kode",
    "limbah_kode",
    "sampah_kode",
]
ORDINAL = [
    "ukuran_bayi",
    "pend_ibu",
    "pend_ayah",
    "jumlah_jenis_imun",
    "status_sehat_umum",
]

# Rentang yang dipakai saat build dataset (Ringkasan_Dataset.xlsx, kolom 'aturan')
RENTANG_BUILD = {
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
}

def _grup(alasan, daftar):
    return {v: alasan for v in daftar}

DIKELUARKAN = {
    **_grup("Komponen rumus HAZ (kebocoran label)", ["tb_anak"]),
    **_grup("Multikolinearitas (terbukti |r| >= 0,7 atau VIF > 10 di sheet 13/15)",
            ["bb_anak", "bb_ibu", "air_dini", "usia_air_bln", "usia_mpasi_bln", "log_pengeluaran_pangan_perkapita",
             "total_hari_makan_7hr", "jenis_sayur_buah_7hr", "imun_campak", "imun_hepb"]),
    **_grup("Turunan/komponen dari variabel yang sudah dipakai (informasi sama)",
            ["bblr", "hamil_prematur_proxy", "tablet_besi", "periksa_hamil", "hari_terbaring_4mgg",
             "imun_bcg", "imun_polio", "imun_dpt", "imun_rota"]),
    **_grup(
        "Variabel ayah: tidak bisa ditindaklanjuti kader, sulit diisi, tidak ada di kerangka faktor risiko 2.3.1",
        ["bb_ayah", "imt_ayah", "umur_ayah", "pend_ayah", "kerja_ayah", "ada_ayah_di_RT"],
    ),
    **_grup(
        "Subjektif atau lemah",
        ["ukuran_bayi", "status_sehat_umum", "hanya_asi", "asi_pernah", "kerja_ibu", "ada_ibu_di_RT",
         "tempat_lahir_grup", "limbah_kode"],
    ),
    **_grup("Konstan pada data Puskesmas (satu wilayah)", ["provinsi", "perkotaan"]),
    **_grup("Missing struktural terkait umur (hanya ada untuk anak yang sudah disapih)", ["usia_sapih_bln"]),
    **_grup("Kode asli nominal, sudah diringkas menjadi versi *_layak (0/1)", ["air_minum_kode", "toilet_kode", "sampah_kode"]),
}
KELOMPOK_UMUR_BINS = [0, 6, 12, 24, 36, 48, 60]
KELOMPOK_UMUR_LAB = ["0-5", "6-11", "12-23", "24-35", "36-47", "48-59"]

sheets = {}  
ringkas = [] 

def add_ringkas(k, v):
    ringkas.append((k, v))

def bh_adjust(p):
    """Koreksi Benjamini-Hochberg."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1)
    return out

# ----------------------------------------------------------------------------
# MUAT DATA
# ----------------------------------------------------------------------------
print("=== EDA IFLS5 (sesuai Proposal Bab III 3.1.2.b) ===")
df = pd.read_csv(args.csv)
print(f"[data] {args.csv}: {df.shape[0]} baris x {df.shape[1]} kolom")

FEATURES = [c for c in df.columns if c not in ID_COLS + TARGET]
TIPE = {}
for c in FEATURES:
    if c in NOMINAL:
        TIPE[c] = "nominal"
    elif c in ORDINAL:
        TIPE[c] = "ordinal"
    elif set(df[c].dropna().unique()) <= {0, 1}:
        TIPE[c] = "biner"
    else:
        TIPE[c] = "numerik"
NUM_COLS = [c for c in FEATURES if TIPE[c] == "numerik"]
df["kel_umur"] = pd.cut(
    df["umur_bln"], KELOMPOK_UMUR_BINS, labels=KELOMPOK_UMUR_LAB, right=False
)

# Fitur yang masuk pemodelan = kandidat - DIKELUARKAN
_tak_ada = [c for c in DIKELUARKAN if c not in df.columns]
if _tak_ada:
    print(f"  [peringatan] ada di DIKELUARKAN tetapi tidak ada di CSV: {_tak_ada}")
FITUR_MODEL = [c for c in FEATURES if c not in DIKELUARKAN]
print(
    f"[fitur] kandidat {len(FEATURES)} -> dipakai model {len(FITUR_MODEL)}, "
    f"dikeluarkan {len(FEATURES) - len(FITUR_MODEL)}"
)

# Indeks TRAIN (dipakai bagian 7 dan 8). Pakai split_assignment.csv bila sudah ada;
# kalau belum, replikasi aturan yang sama dengan 3_Splitting.py.
SPLIT_ASG = BASE / "output" / "3_Splitting" / "split_assignment.csv"
if SPLIT_ASG.exists():
    _asg = pd.read_csv(SPLIT_ASG)
    idx_train = np.where(df.pidlink.isin(_asg.loc[_asg.bagian == "train", "pidlink"]))[0]
    print(f"[split] indeks train dibaca dari {SPLIT_ASG.name}: {len(idx_train)} balita")
else:
    _lip = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(
            df, df.stunting, groups=df.hhid14
        )
    )
    idx_train, _ = min(
        _lip, key=lambda s: abs(df.stunting.iloc[s[1]].mean() - df.stunting.iloc[s[0]].mean())
    )
    print(f"[split] indeks train direplikasi dari aturan 3_Splitting.py: {len(idx_train)} balita")
ring_path = cari(args.ringkasan)
daftar_var = (
    pd.read_excel(ring_path, sheet_name="Daftar_Variabel") if ring_path else None
)
funnel = pd.read_excel(ring_path, sheet_name="Funnel") if ring_path else None

# ============================================================================
# 1. STRUKTUR DAN KARAKTERISTIK DATASET
# ============================================================================
print("\n[1] Struktur dan karakteristik dataset")

# 1a. Audit awal katalog (Tabel 3.4 proposal) dari kamus_variabel_IFLS5.xlsx
kamus_path = cari(args.kamus)
if kamus_path:
    kam = pd.read_excel(kamus_path, sheet_name="Kamus_Variabel")
    rf = pd.read_excel(kamus_path, sheet_name="Ringkasan_File")
    lab = kam["label_EN"].fillna("").str.lower()
    var = kam["variabel"].str.lower()
    audit = pd.DataFrame(
        [
            ["Jumlah berkas .dta", rf["file"].nunique(), "Ringkasan_File"],
            ["Jumlah baris dalam katalog variabel", len(kam), "Kamus_Variabel"],
            [
                "Berkas dengan kandidat variabel PIDLINK",
                kam[var.str.contains("pidlink")]["file"].nunique(),
                "nama variabel mengandung 'pidlink'",
            ],
            [
                "Berkas dengan kandidat variabel usia",
                kam[lab.str.contains(r"\bage\b") | var.str.contains("age")][
                    "file"
                ].nunique(),
                "label memuat kata 'age' atau nama variabel memuat 'age'",
            ],
            [
                "Berkas dengan kandidat variabel jenis kelamin",
                kam[lab.str.contains(r"\bsex\b|gender")]["file"].nunique(),
                "label memuat 'sex'/'gender'",
            ],
            [
                "Berkas kandidat sumber antropometri",
                kam[lab.str.contains("height|weight")]["file"].nunique(),
                "label memuat 'height'/'weight'",
            ],
        ],
        columns=["aspek_pemeriksaan", "hasil", "aturan_hitung"],
    )
    sheets["01_Audit_Katalog"] = audit
    print(audit.to_string(index=False))
else:
    print(
        "  [lewati] kamus_variabel_IFLS5.xlsx tidak ditemukan -> pakai --kamus <path>"
    )

# 1b. Struktur variabel dataset akhir
rows = []
for c in df.columns:
    if c == "kel_umur":
        continue
    s = df[c]
    peran = "ID" if c in ID_COLS else "TARGET" if c in TARGET else "kandidat prediktor"
    r = {
        "variabel": c,
        "peran": peran,
        "tipe": TIPE.get(c, "-"),
        "n_terisi": int(s.notna().sum()),
        "pct_missing": round(s.isna().mean() * 100, 2),
        "n_unik": int(s.nunique()),
    }
    if c not in ID_COLS or c == "pid14":
        r.update(
            {
                "min": s.min(),
                "q1": s.quantile(0.25),
                "median": s.median(),
                "mean": s.mean(),
                "q3": s.quantile(0.75),
                "max": s.max(),
                "sd": s.std(),
            }
        )
    rows.append(r)
struktur = pd.DataFrame(rows)
if daftar_var is not None:
    struktur = struktur.merge(
        daftar_var.rename(columns={"kolom": "variabel"}), on="variabel", how="left"
    )
sheets["02_Struktur_Variabel"] = struktur
print(
    f"  Dataset: {len(df)} balita, {len(FEATURES)} kandidat prediktor "
    f"(numerik {sum(t == 'numerik' for t in TIPE.values())}, biner {sum(t == 'biner' for t in TIPE.values())}, "
    f"ordinal {sum(t == 'ordinal' for t in TIPE.values())}, nominal {sum(t == 'nominal' for t in TIPE.values())})"
)
add_ringkas("Jumlah balita (baris)", len(df))
add_ringkas("Jumlah kandidat prediktor", len(FEATURES))

# 1c. Pemetaan variabel dari paper acuan (Tabel 3.3 proposal BELUM final: hanya urutan yang final)
pem_path = cari(args.pemetaan)
if pem_path:
    sheets["03_Pemetaan_Paper"] = pd.read_excel(pem_path, sheet_name="Pemetaan")
    try:
        sheets["03b_Tidak_Ada_Langsung"] = pd.read_excel(
            pem_path, sheet_name="Tidak_ada_langsung"
        )
    except Exception:
        pass
    print(
        f"  Pemetaan paper -> IFLS5 disalin dari {pem_path.name} ({len(sheets['03_Pemetaan_Paper'])} baris)"
    )
else:
    print(
        "  [lewati] pemetaan_variabel_paper_ke_IFLS5.xlsx tidak ditemukan -> pakai --pemetaan <path>"
    )


# ============================================================================
# 2. PENGGABUNGAN SUMBER DATA
# ============================================================================
print("\n[2] Penggabungan sumber data")
if daftar_var is not None:

    def ambil_file(s):
        m = re.match(r"\s*([a-z0-9]+_[a-z0-9]+)\.", str(s))
        return m.group(1) + ".dta" if m else "turunan/olahan"

    dv = daftar_var.copy()
    dv["berkas_sumber"] = dv["sumber_IFLS5"].apply(ambil_file)
    sumber = (
        dv.groupby("berkas_sumber")["kolom"]
        .agg(["count", lambda x: ", ".join(x)])
        .rename(columns={"count": "n_kolom", "<lambda_0>": "kolom"})
        .reset_index()
        .sort_values("n_kolom", ascending=False)
    )
    sheets["04_Sumber_Data"] = sumber
    print(sumber[["berkas_sumber", "n_kolom"]].to_string(index=False))

lap_path = cari(args.laporan)
if lap_path:
    txt = lap_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    sheets["17_Laporan_Build"] = pd.DataFrame({"baris_laporan_build": txt})
    for kw in ["Link ibu", "Link ayah", "riwayat kehamilan", "VERIFIKASI", "Cara ukur"]:
        for ln in txt:
            if kw in ln:
                print("  " + ln.strip())

# ============================================================================
# 3. PEMILIHAN SUBJEK BALITA
# ============================================================================
print("\n[3] Pemilihan subjek balita (0-59 bulan)")
if funnel is not None:
    f = funnel.copy()
    f["berkurang"] = f["n"].shift(1) - f["n"]
    sheets["05_Funnel_Subjek"] = f
    print(f.to_string(index=False))

grp = (
    df.groupby("kel_umur", observed=False)
    .agg(n=("pidlink", "size"), laki=("jk_laki", "sum"), stunting=("stunting", "sum"))
    .reset_index()
)
grp["perempuan"] = grp["n"] - grp["laki"]
grp["pct_dari_total"] = (grp["n"] / len(df) * 100).round(2)
grp["prevalensi_stunting_pct"] = (grp["stunting"] / grp["n"] * 100).round(2)
sheets["06_Umur_JK"] = grp
print(
    f"  Umur: rata-rata {df.umur_bln.mean():.1f} bln, rentang {df.umur_bln.min():.2f}-{df.umur_bln.max():.2f}; "
    f"laki-laki {df.jk_laki.mean() * 100:.1f}%"
)
add_ringkas("Umur min (bulan)", round(df.umur_bln.min(), 2))
add_ringkas("Umur maks (bulan)", round(df.umur_bln.max(), 2))
add_ringkas("Umur rata-rata (bulan)", round(df.umur_bln.mean(), 1))
add_ringkas("Proporsi laki-laki (%)", round(df.jk_laki.mean() * 100, 1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].hist(df.umur_bln, bins=30, color="#4C78A8", edgecolor="white")
ax[0].set(title="Distribusi umur balita", xlabel="Umur (bulan)", ylabel="Jumlah balita")
w = 0.4
x = np.arange(len(grp))
ax[1].bar(x - w / 2, grp["laki"], w, label="Laki-laki", color="#4C78A8")
ax[1].bar(x + w / 2, grp["perempuan"], w, label="Perempuan", color="#F58518")
ax[1].set_xticks(x)
ax[1].set_xticklabels(grp["kel_umur"])
ax[1].set(
    title="Jumlah balita per kelompok umur dan jenis kelamin",
    xlabel="Kelompok umur (bulan)",
)
ax[1].legend()
fig.savefig(OUT / "eda_01_umur_jk.png")
plt.close(fig)

# ============================================================================
# 4. TARGET STUNTING (HAZ)
# ============================================================================
print("\n[4] Target status stunting (HAZ < -2)")
n_st = int(df.stunting.sum())
prev = n_st / len(df) * 100
sev = int((df.haz < -3).sum())
mod = int(((df.haz >= -3) & (df.haz < -2)).sum())
konsisten = int(((df.haz < -2).astype(int) == df.stunting).sum())
tgt = pd.DataFrame(
    [
        ["Jumlah balita", len(df)],
        ["Stunting (HAZ < -2)", n_st],
        ["  - sangat pendek (HAZ < -3)", sev],
        ["  - pendek (-3 <= HAZ < -2)", mod],
        ["Tidak stunting", len(df) - n_st],
        ["Prevalensi stunting (%)", round(prev, 2)],
        ["Rasio kelas (tidak:stunting)", round((len(df) - n_st) / n_st, 2)],
        ["HAZ rata-rata", round(df.haz.mean(), 3)],
        ["HAZ SD", round(df.haz.std(), 3)],
        ["HAZ median", round(df.haz.median(), 3)],
        ["HAZ min", round(df.haz.min(), 3)],
        ["HAZ maks", round(df.haz.max(), 3)],
        ["Baris dengan label konsisten dengan HAZ", f"{konsisten} dari {len(df)}"],
        ["HAZ di luar [-6, 6]", int(((df.haz < -6) | (df.haz > 6)).sum())],
    ],
    columns=["indikator", "nilai"],
)
sheets["07_Target_HAZ"] = tgt
print(tgt.to_string(index=False))
add_ringkas("Prevalensi stunting (%)", round(prev, 2))
add_ringkas("Stunting / tidak", f"{n_st} / {len(df) - n_st}")
add_ringkas("Sangat pendek (HAZ<-3)", sev)
add_ringkas("Rasio kelas tidak:stunting", round((len(df) - n_st) / n_st, 2))

# prevalensi menurut subkelompok
sub = []
for nm, key in [
    ("kelompok umur", "kel_umur"),
    ("jenis kelamin", "jk_laki"),
    ("wilayah", "perkotaan"),
    ("lahir BBLR", "bblr"),
    ("MPASI dini", "mpasi_dini"),
]:
    t = (
        df.groupby(key, observed=False)["stunting"]
        .agg(["size", "sum", "mean"])
        .reset_index()
    )
    for _, r in t.iterrows():
        sub.append(
            [nm, r[key], int(r["size"]), int(r["sum"]), round(r["mean"] * 100, 2)]
        )
sub = pd.DataFrame(
    sub, columns=["kelompok", "kategori", "n", "n_stunting", "prevalensi_pct"]
)
sheets["07b_Prevalensi_Subgrup"] = sub

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(df.haz, bins=50, color="#4C78A8", edgecolor="white")
ax[0].axvline(-2, color="red", ls="--", label="Ambang stunting (-2)")
ax[0].axvline(-3, color="darkred", ls=":", label="Sangat pendek (-3)")
ax[0].set(title="Distribusi HAZ", xlabel="HAZ", ylabel="Jumlah balita")
ax[0].legend()
bulan = df.groupby(df.umur_bln.astype(int).clip(upper=59))["haz"].mean()
ax[1].plot(bulan.index, bulan.values, color="#4C78A8")
ax[1].axhline(-2, color="red", ls="--")
ax[1].axhline(0, color="gray", lw=0.6)
ax[1].set(
    title="Rata-rata HAZ menurut umur", xlabel="Umur (bulan)", ylabel="HAZ rata-rata"
)
ax[2].bar(grp["kel_umur"], grp["prevalensi_stunting_pct"], color="#E45756")
for i, v in enumerate(grp["prevalensi_stunting_pct"]):
    ax[2].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax[2].set(
    title="Prevalensi stunting per kelompok umur",
    xlabel="Kelompok umur (bulan)",
    ylabel="%",
)
fig.savefig(OUT / "eda_02_haz.png")
plt.close(fig)

# ============================================================================
# 5. MISSING VALUE
# ============================================================================
print("\n[5] Missing value")
miss = pd.DataFrame(
    {
        "variabel": FEATURES,
        "n_missing": [int(df[c].isna().sum()) for c in FEATURES],
        "pct_missing": [round(df[c].isna().mean() * 100, 2) for c in FEATURES],
    }
).sort_values("pct_missing", ascending=False)


def rekom(p):
    if p == 0:
        return "lengkap"
    if p < 5:
        return "imputasi (median/modus) wajar"
    if p < 30:
        return "imputasi; pertimbangkan flag missing"
    return "missing > 30%: pertimbangkan dikeluarkan atau pakai flag missing"


miss["rekomendasi_awal"] = miss["pct_missing"].apply(rekom)
miss.insert(1, "dipakai_model", np.where(miss.variabel.isin(FITUR_MODEL), "ya", "dikeluarkan"))
sheets["08_Missing"] = miss
n_gt10 = int((miss.pct_missing > 10).sum())
n_gt30 = int((miss.pct_missing > 30).sum())
print(
    f"  Variabel dengan missing > 10%: {n_gt10} dari {len(FEATURES)}; > 30%: {n_gt30}"
)
print(miss[miss.pct_missing > 10][["variabel", "pct_missing"]].to_string(index=False))
add_ringkas("Variabel missing > 10%", n_gt10)
add_ringkas(
    "Variabel missing > 30%", ", ".join(miss[miss.pct_missing > 30].variabel) or "-"
)
df["n_missing_baris"] = df[FEATURES].isna().sum(axis=1)
add_ringkas("Median jumlah missing per balita", int(df.n_missing_baris.median()))

# 5b. Missing struktural dan informativeness
rows = []
for c in [
    "usia_air_bln",
    "usia_mpasi_bln",
    "usia_sapih_bln",
    "air_dini",
    "mpasi_dini",
    "masih_asi",
    "imun_rota",
    "imun_campak",
    "jumlah_jenis_imun",
]:
    for g in KELOMPOK_UMUR_LAB:
        m = df[df.kel_umur == g][c].isna().mean() * 100
        rows.append({"variabel": c, "kelompok_umur": g, "pct_missing": round(m, 1)})
ms = pd.DataFrame(rows).pivot(
    index="variabel", columns="kelompok_umur", values="pct_missing"
)[KELOMPOK_UMUR_LAB]
ms.insert(0, "jenis", "missing terkait umur (belum waktunya / belum diberi)")
sheets["09_Missing_Struktural"] = ms.reset_index()

# missing karena orang tua tidak ada di roster / tidak terukur
rel = []
for par, idc in [("ayah", "ada_ayah_di_RT"), ("ibu", "ada_ibu_di_RT")]:
    for c in [f"tinggi_{par}", f"bb_{par}", f"pend_{par}", f"kerja_{par}"]:
        for v in (0, 1):
            sub_ = df[df[idc] == v]
            rel.append(
                {
                    "variabel": c,
                    f"{idc}": v,
                    "n": len(sub_),
                    "pct_missing": round(sub_[c].isna().mean() * 100, 1)
                    if len(sub_)
                    else np.nan,
                }
            )
sheets["09b_Missing_vs_OrangTua"] = pd.DataFrame(rel)

# apakah missing berasosiasi dengan stunting?
inf = []
for c in miss[miss.pct_missing >= 5].variabel:
    m = df[c].isna()
    tab = pd.crosstab(m, df.stunting)
    p = stats.chi2_contingency(tab)[1] if tab.shape == (2, 2) else np.nan
    inf.append(
        {
            "variabel": c,
            "pct_missing": round(m.mean() * 100, 1),
            "prev_stunting_jika_missing": round(df.loc[m, "stunting"].mean() * 100, 1),
            "prev_stunting_jika_terisi": round(df.loc[~m, "stunting"].mean() * 100, 1),
            "p_chi2": p,
        }
    )
inf = pd.DataFrame(inf)
inf["p_BH"] = bh_adjust(inf["p_chi2"].fillna(1)) if len(inf) else []
inf["missing_informatif"] = np.where(
    inf["p_BH"] < 0.05, "ya (flag missing berguna)", "tidak jelas"
)
sheets["09c_Missing_Informatif"] = inf
print(
    f"  Missing yang berasosiasi dengan status stunting (p_BH<0,05): {int((inf.p_BH < 0.05).sum())} dari {len(inf)} variabel"
)

top = miss.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 6))
ax.barh(
    top.variabel,
    top.pct_missing,
    color=np.where(top.pct_missing > 30, "#E45756", "#4C78A8"),
)
ax.axvline(10, color="gray", ls=":")
ax.axvline(30, color="red", ls="--")
ax.set(title="20 variabel dengan missing tertinggi", xlabel="% missing")
fig.savefig(OUT / "eda_03_missing.png")
plt.close(fig)

# ============================================================================
# 6. DUPLIKAT DAN PENGENAL INDIVIDU
# ============================================================================
print("\n[6] Duplikat dan pengenal individu")
dup_pid = int(df.pidlink.duplicated().sum())
dup_row = int(df.drop(columns=["kel_umur", "n_missing_baris"]).duplicated().sum())
dup_hh_pid = int(df.duplicated(["hhid14", "pid14"]).sum())
per_kel = df.groupby("hhid14").size()
dist_kel = per_kel.value_counts().sort_index()
multi = df[df.hhid14.isin(per_kel[per_kel > 1].index)]


def icc1(data, val, g):
    d = data[[g, val]].dropna()
    cnt = d.groupby(g)[val].size()
    d = d[d[g].isin(cnt[cnt > 1].index)]
    cnt = d.groupby(g)[val].size()
    N, k = len(d), len(cnt)
    if k < 2:
        return np.nan
    gm = d[val].mean()
    ssb = (cnt * (d.groupby(g)[val].mean() - gm) ** 2).sum()
    ssw = ((d[val] - d.groupby(g)[val].transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    k0 = (N - (cnt**2).sum() / N) / (k - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


icc_haz = icc1(df, "haz", "hhid14")
dup = pd.DataFrame(
    [
        ["Duplikat pidlink", dup_pid, "harus 0"],
        ["Duplikat (hhid14, pid14)", dup_hh_pid, "harus 0"],
        ["Baris duplikat penuh (semua kolom)", dup_row, "harus 0"],
        ["Jumlah keluarga (hhid14 unik)", int(per_kel.size), ""],
        ["Keluarga dengan > 1 balita", int((per_kel > 1).sum()), ""],
        ["Balita yang berasal dari keluarga > 1 balita", len(multi), ""],
        ["Maksimum balita per keluarga", int(per_kel.max()), ""],
        [
            "ICC(1) HAZ antar-saudara dalam keluarga",
            round(icc_haz, 3),
            "makin > 0 makin mirip antar-saudara dalam satu keluarga",
        ],
    ],
    columns=["pemeriksaan", "hasil", "keterangan"],
)
sheets["10_Duplikat_Keluarga"] = dup
sheets["10b_Balita_per_Keluarga"] = dist_kel.rename_axis(
    "balita_per_keluarga"
).reset_index(name="n_keluarga")
print(dup.to_string(index=False))
add_ringkas("Duplikat pidlink", dup_pid)
add_ringkas("Keluarga unik", int(per_kel.size))
add_ringkas("Keluarga dengan >1 balita", int((per_kel > 1).sum()))
add_ringkas("ICC(1) HAZ dalam keluarga", round(icc_haz, 3))

# ============================================================================
# 7. NILAI EKSTREM (OUTLIER)
# ============================================================================
print("\n[7] Nilai ekstrem (outlier)")
rows = []
for c in NUM_COLS:
    s = df[c].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (s - s.mean()) / s.std()
    p1, p99 = s.quantile(0.01), s.quantile(0.99)  # seluruh data, sama dengan winsorizing di 2_PREPROCESSING
    r = {
        "variabel": c,
        "n": len(s),
        "min": s.min(),
        "q1": q1,
        "median": s.median(),
        "q3": q3,
        "max": s.max(),
        "batas_bawah_IQR": lo,
        "batas_atas_IQR": hi,
        "n_outlier_IQR": int(((s < lo) | (s > hi)).sum()),
        "pct_outlier_IQR": round(((s < lo) | (s > hi)).mean() * 100, 2),
        "n_|z|>3": int((z.abs() > 3).sum()),
        "P1": p1,
        "P99": p99,
        "n_di_luar_P1_P99": int(((s < p1) | (s > p99)).sum()),
    }
    if c in RENTANG_BUILD:
        a, b = RENTANG_BUILD[c]
        r["rentang_build"] = f"{a}-{b}"
        r["n_tepat_di_batas_build"] = int(((s == a) | (s == b)).sum())
    rows.append(r)
outl = pd.DataFrame(rows).sort_values("pct_outlier_IQR", ascending=False)
outl.insert(1, "dipakai_model", np.where(outl.variabel.isin(FITUR_MODEL), "ya", "dikeluarkan"))
sheets["11_Outlier"] = outl
print(
    outl[["variabel", "min", "max", "n_outlier_IQR", "pct_outlier_IQR"]]
    .head(10)
    .to_string(index=False)
)
print(
    "  Outlier tidak dihapus (sesuai proposal 3.1.2.b.7): nilai di luar rentang biologis sudah dijadikan NaN "
    "saat build; sisanya dipertahankan."
)

# 7b. Pemeriksaan logika / konsistensi (rekomendasi: tinjau sebelum pemodelan)
cek = []


def tambah(nama, mask, dasar_mask=None, ket=""):
    dasar = int(dasar_mask.sum()) if dasar_mask is not None else len(mask)
    cek.append(
        {
            "pemeriksaan": nama,
            "n_melanggar": int(mask.sum()),
            "dari_n": dasar,
            "pct": round(mask.sum() / dasar * 100, 2) if dasar else np.nan,
            "keterangan": ket,
        }
    )


for c, nm in [
    ("usia_air_bln", "usia pertama diberi air"),
    ("usia_mpasi_bln", "usia pertama MPASI"),
    ("usia_sapih_bln", "usia disapih"),
]:
    ok = df[c].notna()
    tambah(
        f"{nm} > umur anak saat ini (+1 bln)",
        (df[c] > df.umur_bln + 1) & ok,
        ok,
        "tidak mungkin; kemungkinan salah satuan/kode (tahun vs bulan)",
    )
ok = df.masih_asi.notna() & df.usia_sapih_bln.notna()
tambah(
    "masih_asi=1 tetapi usia_sapih terisi",
    (df.masih_asi == 1) & df.usia_sapih_bln.notna(),
    ok,
)
tambah(
    "mpasi_dini terisi pada bayi < 6 bln (harus 0 setelah revisi build)",
    df.mpasi_dini.notna() & (df.umur_bln < 6),
    df.umur_bln < 6,
    "mpasi_dini hanya dinilai pada anak >= 6 bulan",
)
umur_ibu_lahir = df.umur_ibu - df.umur_bln / 12
okm = umur_ibu_lahir.notna()
tambah("umur ibu saat melahirkan < 14 tahun", (umur_ibu_lahir < 14) & okm, okm)
umur_ayah_lahir = df.umur_ayah - df.umur_bln / 12
oka = umur_ayah_lahir.notna()
tambah("umur ayah saat anak lahir < 14 tahun", (umur_ayah_lahir < 14) & oka, oka)
tambah("anc_total > 20 kunjungan", df.anc_total > 20, df.anc_total.notna())
tambah("jml_tablet_besi >= 365", df.jml_tablet_besi >= 365, df.jml_tablet_besi.notna())
tambah(
    "IMT ibu < 15 atau > 45", (df.imt_ibu < 15) | (df.imt_ibu > 45), df.imt_ibu.notna()
)
tambah(
    "IMT ayah < 15 atau > 45",
    (df.imt_ayah < 15) | (df.imt_ayah > 45),
    df.imt_ayah.notna(),
)
tambah(
    "jumlah_jenis_imun > 0 padahal semua imun inti = 0",
    (df.jumlah_jenis_imun > 0)
    & (
        df[["imun_bcg", "imun_polio", "imun_dpt", "imun_hepb", "imun_campak"]].sum(
            axis=1
        )
        == 0
    ),
    df.jumlah_jenis_imun.notna(),
)
tambah(
    "imun_campak=1 pada bayi < 6 bulan (cek validitas imunisasi)",
    (df.imun_campak == 1) & (df.umur_bln < 6),
    df.imun_campak.notna() & (df.umur_bln < 6),
    "campak biasanya baru diberikan sekitar 9 bulan",
)
tambah(
    "bblr=1 tetapi bb_lahir_kg >= 2,5 (inkonsisten)",
    (df.bblr == 1) & (df.bb_lahir_kg >= 2.5),
    df.bblr.notna(),
)
tambah(
    "hamil_prematur_proxy=1 tetapi lama_hamil >= 37",
    (df.hamil_prematur_proxy == 1) & (df.lama_hamil_minggu >= 37),
    df.hamil_prematur_proxy.notna(),
)
ceklog = pd.DataFrame(cek)
sheets["12_Cek_Logika"] = ceklog
print(ceklog[["pemeriksaan", "n_melanggar", "dari_n", "pct"]].to_string(index=False))

pilih = [c for c in NUM_COLS if c in FITUR_MODEL]  # boxplot hanya variabel numerik yang dipakai model

nc = 6
nr = int(np.ceil(len(pilih) / nc))
fig, axes = plt.subplots(nr, nc, figsize=(2.4 * nc, 2.6 * nr))
for a, c in zip(axes.ravel(), pilih):
    a.boxplot(df[c].dropna(), flierprops=dict(markersize=2, alpha=0.5))
    a.set_title(c, fontsize=8)
    a.set_xticks([])
for a in axes.ravel()[len(pilih) :]:
    a.axis("off")
fig.suptitle("Boxplot variabel numerik (titik = di luar 1,5 x IQR)", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "eda_04_outlier_boxplot.png")
plt.close(fig)

# ============================================================================
# 8. HUBUNGAN ANTARVARIABEL
# ============================================================================
print("\n[8] Hubungan antarvariabel (HANYA data TRAIN)")
df_all = df  # seluruh data, hanya untuk bagian 1-7
df = df_all.iloc[idx_train].reset_index(drop=True)  # mulai di sini df = TRAIN
print(f"  Bagian 8 memakai data train: {len(df)} dari {len(df_all)} balita")
add_ringkas("Balita train (dasar analisis seleksi fitur)", len(df))

# 8a. Korelasi Spearman antar-fitur (nominal dikecualikan: kode tidak berurutan)
KORR_COLS = [c for c in FEATURES if TIPE[c] != "nominal"]
corr = df[KORR_COLS].corr(method="spearman", min_periods=100)
pairs = []
for i, a in enumerate(KORR_COLS):
    for b in KORR_COLS[i + 1 :]:
        r = corr.loc[a, b]
        if pd.notna(r) and abs(r) >= 0.5:
            pairs.append(
                {
                    "var_1": a,
                    "var_2": b,
                    "spearman": round(r, 3),
                    "tingkat": "tinggi (>=0,7)"
                    if abs(r) >= 0.7
                    else "sedang (0,5-0,7)",
                    "catatan": "turunan/duplikat informasi"
                    if (
                        a.split("_")[0] == b.split("_")[0]
                        or {a, b}
                        in [
                            {"bb_ibu", "imt_ibu"},
                            {"bb_ayah", "imt_ayah"},
                            {"usia_mpasi_bln", "mpasi_dini"},
                            {"usia_air_bln", "air_dini"},
                            {"imun_rota", "jumlah_jenis_imun"},
                        ]
                    )
                    else "",
                }
            )
pairs = pd.DataFrame(pairs).sort_values("spearman", key=abs, ascending=False)
for k_ in ("var_1", "var_2"):
    pairs[f"{k_}_dipakai"] = np.where(pairs[k_].isin(FITUR_MODEL), "ya", "tidak")
sheets["13_Korelasi_Tinggi"] = pairs
sisa = pairs[(pairs.var_1_dipakai == "ya") & (pairs.var_2_dipakai == "ya")]
sheets["13c_Korelasi_Fitur_Model"] = sisa
print(
    f"  Pasangan |r| >= 0,5 yang MASIH ada di antara fitur model: {len(sisa)} "
    f"(tinggi >= 0,7: {int(sisa.tingkat.str.startswith('tinggi').sum())})"
)
print(
    f"  Pasangan fitur |Spearman| >= 0,7: {int((pairs.tingkat.str.startswith('tinggi')).sum())}"
)
print(
    pairs[pairs.tingkat.str.startswith("tinggi")][
        ["var_1", "var_2", "spearman"]
    ].to_string(index=False)
)

kor_target = pd.DataFrame(
    {
        "variabel": KORR_COLS,
        "spearman_dengan_HAZ": [
            round(df[[c, "haz"]].corr(method="spearman").iloc[0, 1], 3)
            for c in KORR_COLS
        ],
        "spearman_dengan_umur": [
            round(df[[c, "umur_bln"]].corr(method="spearman").iloc[0, 1], 3)
            for c in KORR_COLS
        ],
    }
)
kor_target["terkait_umur(|r|>=0,3)"] = np.where(
    kor_target.spearman_dengan_umur.abs() >= 0.3, "ya", ""
)
sheets["13b_Korelasi_HAZ_Umur"] = kor_target.sort_values(
    "spearman_dengan_HAZ", key=abs, ascending=False
)

KM = [c for c in FITUR_MODEL if TIPE[c] != "nominal"]
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(corr.loc[KM, KM].values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(KM)))
ax.set_yticks(range(len(KM)))
ax.set_xticklabels(KM, rotation=90, fontsize=7)
ax.set_yticklabels(KM, fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.7, label="Spearman")
ax.set_title("Korelasi Spearman antar fitur model (data train)")
fig.savefig(OUT / "eda_05_korelasi.png")
plt.close(fig)

# 8b. Uji univariat terhadap stunting
y = df.stunting
uni = []
for c in FEATURES:
    s = df[c]
    ok = s.notna()
    n = int(ok.sum())
    if TIPE[c] in ("numerik", "ordinal"):
        a, b = s[ok & (y == 1)], s[ok & (y == 0)]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = u / (len(a) * len(b))
        uni.append(
            {
                "variabel": c,
                "tipe": TIPE[c],
                "n": n,
                "uji": "Mann-Whitney",
                "median_stunting": a.median(),
                "median_tidak": b.median(),
                "ukuran_efek": round(auc, 3),
                "jenis_efek": "AUC (0,5 = tidak beda)",
                "kekuatan_efek": round(abs(auc - 0.5) * 2, 3),
                "p": p,
            }
        )
    else:
        tab = pd.crosstab(s[ok], y[ok])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        prev_k = (tab[1] / tab.sum(axis=1) * 100).round(1)
        uni.append(
            {
                "variabel": c,
                "tipe": TIPE[c],
                "n": n,
                "uji": "Chi-square",
                "median_stunting": np.nan,
                "median_tidak": np.nan,
                "ukuran_efek": round(v, 3),
                "jenis_efek": "Cramer's V",
                "kekuatan_efek": round(v, 3),
                "p": p,
                "prevalensi_stunting_per_kategori(%)": "; ".join(
                    f"{k}:{x}" for k, x in prev_k.items()
                )
                if len(prev_k) <= 12
                else f"{len(prev_k)} kategori",
            }
        )
uni = pd.DataFrame(uni)
uni["p_BH"] = bh_adjust(uni["p"])
uni["signifikan_BH<0,05"] = np.where(uni.p_BH < 0.05, "ya", "tidak")
uni = uni.sort_values("kekuatan_efek", ascending=False)
uni.insert(1, "dipakai_model", np.where(uni.variabel.isin(FITUR_MODEL), "ya", "dikeluarkan"))
sheets["14_Univariat"] = uni
nsig = int((uni.p_BH < 0.05).sum())
print(
    f"  Uji univariat: {nsig} dari {len(uni)} variabel berbeda nyata (p_BH < 0,05). 10 teratas:"
)
for _, r in uni.head(10).iterrows():
    print(f"    {r.variabel:22s} {r.jenis_efek}={r.ukuran_efek}  p_BH={r.p_BH:.2e}")
print(
    "  CATATAN: uji ini belum memperhitungkan klaster keluarga (deskriptif); seleksi fitur final lewat pemodelan."
)
add_ringkas("Variabel signifikan univariat (p_BH<0,05)", f"{nsig} dari {len(uni)}")

top = uni.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh(
    top.variabel,
    top.kekuatan_efek,
    color=np.where(top.jenis_efek.str.startswith("AUC"), "#4C78A8", "#F58518"),
)
ax.set(
    title="15 variabel dengan efek terkuat terhadap stunting", xlabel="Kekuatan efek"
)

ax.legend(
    handles=[
        Patch(color="#4C78A8", label="|AUC-0,5| x 2 (numerik/ordinal)"),
        Patch(color="#F58518", label="Cramer's V (biner/nominal)"),
    ],
    loc="lower right",
)
fig.savefig(OUT / "eda_06_univariat_top15.png")
plt.close(fig)

# 8c. Multikolinearitas (VIF) pada variabel numerik & ordinal
def hitung_vif(cols):
    """VIF dari matriks korelasi (data TRAIN, imputasi median hanya untuk keperluan VIF)."""
    cols = [c for c in cols if df[c].notna().any()]
    Xv = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df[cols]), columns=cols
    )
    Xv = Xv.loc[:, Xv.std() > 0]
    cm = np.corrcoef(Xv.values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(cm)), index=Xv.columns).round(2)


v_all = hitung_vif([c for c in FEATURES if TIPE[c] != "nominal"])
v_mod = hitung_vif([c for c in FITUR_MODEL if TIPE[c] != "nominal"])
vif = pd.DataFrame({"variabel": v_all.index, "VIF_semua_kandidat": v_all.values})
vif["dipakai_model"] = np.where(vif.variabel.isin(FITUR_MODEL), "ya", "dikeluarkan")
vif["VIF_fitur_model"] = vif.variabel.map(v_mod)
vif["interpretasi_fitur_model"] = pd.cut(
    vif.VIF_fitur_model,
    [0, 5, 10, np.inf],
    labels=["aman", "moderat (5-10)", "tinggi (>10)"],
)
sheets["15_VIF"] = vif.sort_values("VIF_semua_kandidat", ascending=False)
print(f"  VIF > 10 pada SEMUA kandidat: {', '.join(vif[vif.VIF_semua_kandidat > 10].variabel) or '-'}")
print(f"  VIF > 10 pada FITUR MODEL   : {', '.join(vif[vif.VIF_fitur_model > 10].variabel) or '-'}")
print("  (VIF penting untuk Logistic Regression; Random Forest tidak sensitif.)")

# 8d. Rekap seleksi variabel (dasar keputusan setelah bimbingan)
miss_tr = (df[FEATURES].isna().mean() * 100).round(2)
sel = pd.DataFrame({"variabel": FEATURES})
sel["status"] = np.where(sel.variabel.isin(FITUR_MODEL), "DIPAKAI", "DIKELUARKAN")
sel["alasan"] = sel.variabel.map(DIKELUARKAN).fillna(
    "Dipakai: faktor risiko (Subbab 2.3.1), realistis diisi, tidak terbukti multikolinear"
)
sel["tipe"] = sel.variabel.map(TIPE)
sel["pct_missing_train"] = sel.variabel.map(miss_tr)
sel["spearman_HAZ_train"] = sel.variabel.map(kor_target.set_index("variabel")["spearman_dengan_HAZ"])
sel["VIF_semua_kandidat"] = sel.variabel.map(v_all)
sel["VIF_fitur_model"] = sel.variabel.map(v_mod)
sel = sel.sort_values(["status", "variabel"], ascending=[False, True])
sheets["18_Seleksi_Variabel"] = sel
vif_tinggi = sel[(sel.status == "DIPAKAI") & (sel.VIF_fitur_model > 10)].variabel.tolist()
print(
    f"\n[8d] Fitur model: {len(FITUR_MODEL)} dari {len(FEATURES)} kandidat; "
    f"VIF > 10 pada fitur model: {', '.join(vif_tinggi) or '-'}"
)
add_ringkas("Fitur dipakai model / dikeluarkan", f"{len(FITUR_MODEL)} / {len(FEATURES) - len(FITUR_MODEL)}")
add_ringkas("Fitur model dengan VIF > 10", ", ".join(vif_tinggi) or "-")

# ============================================================================
# SIMPAN
# ============================================================================
sheets = {
    "00_Ringkasan": pd.DataFrame(ringkas, columns=["indikator", "nilai"]),
    **sheets,
}
xlsx = OUT / "EDA_IFLS5.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
    for nm, d in sheets.items():
        d.to_excel(xw, sheet_name=nm[:31], index=False)
        ws = xw.sheets[nm[:31]]
        for col in ws.columns:
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
        ws.freeze_panes = "A2"

print(f"\nHasil disimpan di: {OUT}")
print(
    "  EDA_IFLS5.xlsx, eda_01_umur_jk.png, eda_02_haz.png, eda_03_missing.png, eda_04_outlier_boxplot.png,"
)
print("  eda_05_korelasi.png, eda_06_univariat_top15.png")
print("Selesai.")
