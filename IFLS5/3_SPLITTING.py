"""
3_SPLITTING.py  -  Pembagian data pendukung per keluarga (data ASLI, tanpa SMOTE)

PERAN SEKARANG (protokol paper): split MODEL (SMOTE -> split 80:20 acak) dilakukan di 4_PEMODELAN.py
Skrip ini hanya menyediakan pembagian data asli per keluarga untuk:
  - 1_EDA.py bagian 8 (korelasi, VIF, univariat pada data latih)  -> split_assignment.csv
  - 5_FEATUREIMPORTANCE.py (permutation importance pada data latih asli) -> train.csv

  Split dikelompokkan per keluarga (hhid14), sehingga balita bersaudara berada di bagian yang sama.
  Untuk stratified biasa tanpa pengelompokan:   python 3_SPLITTING.py --tanpa-kelompok

Yang dihasilkan:
  1. Split 80:20  : StratifiedGroupKFold (grup = hhid14, stratifikasi = stunting); dari 5 kemungkinan pembagian,
                    dipilih yang prevalensi stunting train dan test-nya paling mirip (tanpa melihat hasil model).
  2. Fold CV      : 5 fold pada data train (kolom fold_cv), juga dikelompokkan + berstratifikasi.
                    Dipakai feature importance (CV per keluarga pada data latih asli).
  3. Pemeriksaan  : 11 pemeriksaan otomatis (ukuran, keluarga, prevalensi, fold, provinsi).
  4. Keseimbangan : standardized mean difference (SMD) train vs test untuk setiap fitur.
Output (folder 3_Splitting):
    train.csv  (semua kolom + fold_cv)   test.csv  (DIKUNCI: tidak dipakai untuk memilih fitur/parameter/threshold)
    split_assignment.csv  (pidlink -> train/test, fold_cv; untuk reproduksi)
    Ringkasan_Split.xlsx, eda_split_*.png, laporan_split.txt
Cara jalan : python 3_SPLITTING.py     (atau --csv <path> --out <folder>)
Kebutuhan  : pandas numpy scikit-learn matplotlib openpyxl
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
BUILD = BASE / "output" / "BuildDataset"

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=str(BUILD / "balita_ifls5.csv"))
ap.add_argument("--out", default=str(BASE / "output" / "3_Splitting"))
ap.add_argument(
    "--tanpa-kelompok",
    action="store_true",
    help="stratified biasa persis proposal 3.1.7 (saudara kandung bisa terpisah)",
)
ap.add_argument("--fold", type=int, default=5, help="jumlah fold CV pada data train (Azriani dkk.: 10)")
args = ap.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42  # proposal 3.1.7
TEST_FRAC = 0.20  # proposal 3.1.7 (80:20)
N_FOLD_CV = args.fold  # proposal 3.1.9: 5 fold; Azriani dkk.: 10 fold
KELOMPOK = not args.tanpa_kelompok
TARGET, GRUP = "stunting", "hhid14"
ID_COLS = ["pidlink", "hhid14", "pid14"]
NOMINAL = [
    "provinsi",
    "tempat_lahir_grup",
    "air_minum_kode",
    "toilet_kode",
    "limbah_kode",
    "sampah_kode",
]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------------------
# MUAT DATA
# ----------------------------------------------------------------------------
log("=== SPLITTING IFLS5 (Proposal 3.1.7) ===")
df = pd.read_csv(args.csv)
n, y = len(df), df[TARGET].values
log(
    f"[data] {args.csv}: {n} baris x {df.shape[1]} kolom | stunting {y.mean() * 100:.2f}% | keluarga {df[GRUP].nunique()}"
)
assert not df.duplicated(["hhid14", "pid14"]).any(), (
    "ada (hhid14, pid14) ganda; jalankan EDA dulu"
)
assert df[TARGET].isin([0, 1]).all(), "kolom target harus 0/1"
log(
    f"[mode] split dikelompokkan per keluarga: {KELOMPOK}  (seed {SEED}, test {TEST_FRAC:.0%}, CV {N_FOLD_CV}-fold)"
)

# ----------------------------------------------------------------------------
# 1. SPLIT 80:20
# ----------------------------------------------------------------------------
if KELOMPOK:
    sg = StratifiedGroupKFold(n_splits=int(round(1 / TEST_FRAC)), shuffle=True, random_state=SEED)
    i_tr, i_te = min(sg.split(df, y, groups=df[GRUP]), key=lambda s: abs(y[s[1]].mean() - y[s[0]].mean()))
else:
    i_tr, i_te = train_test_split(np.arange(n), test_size=TEST_FRAC, stratify=y, random_state=SEED)
tr = df.iloc[np.sort(i_tr)].copy()
te = df.iloc[np.sort(i_te)].copy()
log(
    f"\n[1] Split: train {len(tr)} ({len(tr) / n * 100:.1f}%), test {len(te)} ({len(te) / n * 100:.1f}%)"
)

# ----------------------------------------------------------------------------
# 3. FOLD CV pada data train (Subbab 3.1.9)
# ----------------------------------------------------------------------------
tr["fold_cv"] = -1
if KELOMPOK:

    def bagi_fold(seed):
        return list(
            StratifiedGroupKFold(
                n_splits=N_FOLD_CV, shuffle=True, random_state=seed
            ).split(tr, tr[TARGET], groups=tr[GRUP])
        )

    def sebaran(lipatan):
        p = [tr[TARGET].iloc[iv].mean() for _, iv in lipatan]
        return max(p) - min(p)

    kandidat = {s: bagi_fold(s) for s in range(SEED, SEED + 20)}
    seed_cv = min(kandidat, key=lambda s: sebaran(kandidat[s]))
    pembagi = kandidat[seed_cv]
    log(
        f"[2] Seed fold CV dipilih dari {SEED}-{SEED + 19} berdasarkan sebaran prevalensi terkecil "
        f"(tanpa melihat hasil model): {seed_cv}"
    )
else:
    pembagi = StratifiedKFold(
        n_splits=N_FOLD_CV, shuffle=True, random_state=SEED
    ).split(tr, tr[TARGET])
for k, (_, iv) in enumerate(pembagi):
    tr.iloc[iv, tr.columns.get_loc("fold_cv")] = k

fold = (
    tr.groupby("fold_cv")
    .agg(
        n=(TARGET, "size"),
        n_keluarga=(GRUP, "nunique"),
        n_stunting=(TARGET, "sum"),
        prevalensi_pct=(TARGET, lambda s: s.mean() * 100),
        umur_rata2_bln=("umur_bln", "mean"),
    )
    .round(2)
    .reset_index()
)
log(
    "[2] Fold CV pada train: "
    + " | ".join(
        f"f{int(r.fold_cv)}: n={int(r.n)}, stunting {r.prevalensi_pct:.1f}%"
        for r in fold.itertuples()
    )
)

# ----------------------------------------------------------------------------
# 4. UKURAN DAN PREVALENSI
# ----------------------------------------------------------------------------
ukuran = pd.DataFrame(
    [
        [
            "Seluruh data",
            n,
            df[GRUP].nunique(),
            int(y.sum()),
            round(y.mean() * 100, 2),
            round(df.umur_bln.mean(), 1),
            100.0,
        ],
        [
            "Train",
            len(tr),
            tr[GRUP].nunique(),
            int(tr[TARGET].sum()),
            round(tr[TARGET].mean() * 100, 2),
            round(tr.umur_bln.mean(), 1),
            round(len(tr) / n * 100, 1),
        ],
        [
            "Test",
            len(te),
            te[GRUP].nunique(),
            int(te[TARGET].sum()),
            round(te[TARGET].mean() * 100, 2),
            round(te.umur_bln.mean(), 1),
            round(len(te) / n * 100, 1),
        ],
    ],
    columns=[
        "bagian",
        "n_balita",
        "n_keluarga",
        "n_stunting",
        "prevalensi_pct",
        "umur_rata2_bln",
        "pct_dari_total",
    ],
)

# ----------------------------------------------------------------------------
# 5. KESEIMBANGAN FITUR (SMD train vs test)
# ----------------------------------------------------------------------------
fitur = [
    c
    for c in df.columns
    if c not in ID_COLS + [TARGET] + NOMINAL and pd.api.types.is_numeric_dtype(df[c])
]
rows = []
for c in fitur:
    a, b = tr[c].dropna(), te[c].dropna()
    sd = np.sqrt((a.var() + b.var()) / 2)
    smd = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
    rows.append(
        {
            "variabel": c,
            "mean_train": a.mean(),
            "mean_test": b.mean(),
            "SMD": smd,
            "|SMD|": abs(smd),
            "pct_kosong_train": tr[c].isna().mean() * 100,
            "pct_kosong_test": te[c].isna().mean() * 100,
            "seimbang(|SMD|<0,1)": "ya" if abs(smd) < 0.1 else "tidak",
        }
    )
smd = pd.DataFrame(rows).sort_values("|SMD|", ascending=False).round(4)
n_tdk = int((smd["|SMD|"] >= 0.1).sum())
log(
    f"[3] Keseimbangan fitur train vs test: {len(smd) - n_tdk} dari {len(smd)} fitur seimbang (|SMD| < 0,1); "
    f"terbesar: {smd.iloc[0].variabel} ({smd.iloc[0]['|SMD|']:.3f})"
)

prov = (
    pd.concat([tr.assign(bagian="train"), te.assign(bagian="test")])
    .groupby(["provinsi", "bagian"])[TARGET]
    .agg(["size", "mean"])
    .unstack("bagian")
    .fillna(0)
)
prov.columns = [f"{a}_{b}" for a, b in prov.columns]
prov = prov.rename(
    columns={
        "size_train": "n_train",
        "size_test": "n_test",
        "mean_train": "prev_train",
        "mean_test": "prev_test",
    }
)
prov[["prev_train", "prev_test"]] = (prov[["prev_train", "prev_test"]] * 100).round(1)
prov = prov.reset_index()

# ----------------------------------------------------------------------------
# 6. PEMERIKSAAN OTOMATIS
# ----------------------------------------------------------------------------
cek = []


def periksa(nama, lulus, nilai, kriteria):
    cek.append(
        {
            "no": len(cek) + 1,
            "pemeriksaan": nama,
            "hasil": "LULUS" if lulus else "GAGAL",
            "nilai": nilai,
            "kriteria": kriteria,
        }
    )


periksa(
    "Train + test = seluruh data",
    len(tr) + len(te) == n,
    f"{len(tr)} + {len(te)} = {len(tr) + len(te)}",
    f"= {n}",
)
periksa(
    "Tidak ada balita (pidlink) di train DAN test",
    len(set(tr.pidlink) & set(te.pidlink)) == 0,
    len(set(tr.pidlink) & set(te.pidlink)),
    "0",
)
fam_overlap = len(set(tr[GRUP]) & set(te[GRUP]))
periksa(
    "Keluarga yang ada di train DAN test",
    (fam_overlap == 0) if KELOMPOK else True,
    fam_overlap,
    "0 (mode kelompok)" if KELOMPOK else "informasi (mode stratified biasa)",
)
periksa(
    "Proporsi test mendekati 20%",
    abs(len(te) / n - TEST_FRAC) <= 0.01,
    f"{len(te) / n * 100:.2f}%",
    "19-21%",
)
beda = abs(tr[TARGET].mean() - te[TARGET].mean()) * 100
periksa(
    "Selisih prevalensi stunting train vs test",
    beda < 1.0,
    f"{beda:.2f} poin persen",
    "< 1 poin",
)
periksa(
    "Semua baris train punya fold_cv (0..4)",
    set(tr.fold_cv.unique()) == set(range(N_FOLD_CV)),
    sorted(tr.fold_cv.unique().tolist()),
    "0-4",
)
pisah_fold = 0
for k in range(N_FOLD_CV):
    pisah_fold += len(
        set(tr.loc[tr.fold_cv != k, GRUP]) & set(tr.loc[tr.fold_cv == k, GRUP])
    )
periksa(
    "Keluarga yang terpisah antar fold CV",
    (pisah_fold == 0) if KELOMPOK else True,
    pisah_fold,
    "0 (mode kelompok)" if KELOMPOK else "informasi",
)
selisih_fold = fold.prevalensi_pct.max() - fold.prevalensi_pct.min()
periksa(
    "Selisih prevalensi antar fold CV",
    selisih_fold < 2.0,
    f"{selisih_fold:.2f} poin persen",
    "< 2 poin",
)
rasio = fold.n.min() / fold.n.max()
periksa("Ukuran fold CV seimbang (min/maks)", rasio > 0.9, f"{rasio:.3f}", "> 0,90")
prov_hilang = sorted(set(te.provinsi.dropna()) - set(tr.provinsi.dropna()))
periksa(
    "Provinsi di test yang tidak ada di train",
    len(prov_hilang) == 0,
    prov_hilang or "-",
    "tidak ada",
)
periksa(
    "Ketidakseimbangan fitur berat (|SMD| >= 0,25)",
    int((smd["|SMD|"] >= 0.25).sum()) == 0,
    int((smd["|SMD|"] >= 0.25).sum()),
    "0",
)
cek = pd.DataFrame(cek)
cek["nilai"] = cek["nilai"].astype(str)
log("\n[4] Pemeriksaan otomatis:")
for r in cek.itertuples():
    log(
        f"    {r.no:>2}. [{r.hasil}] {r.pemeriksaan}: {r.nilai}  (kriteria {r.kriteria})"
    )
gagal = int((cek.hasil == "GAGAL").sum())
log(f"    -> {len(cek) - gagal} lulus, {gagal} gagal")

# ----------------------------------------------------------------------------
# SIMPAN
# ----------------------------------------------------------------------------
tr.to_csv(OUT / "train.csv", index=False, encoding="utf-8")
te.to_csv(OUT / "test.csv", index=False, encoding="utf-8")
asg = pd.concat(
    [
        tr[ID_COLS].assign(bagian="train", fold_cv=tr.fold_cv),
        te[ID_COLS].assign(bagian="test", fold_cv=np.nan),
    ]
)
asg.to_csv(OUT / "split_assignment.csv", index=False)

ringkas = pd.DataFrame(
    [
        [
            "Metode split",
            "StratifiedGroupKFold per keluarga (hhid14)"
            if KELOMPOK
            else "Stratified biasa (train_test_split)",
        ],
        ["random_state", SEED],
        ["Proporsi train : test", f"{(1 - TEST_FRAC):.0%} : {TEST_FRAC:.0%}"],
        ["Jumlah balita train / test", f"{len(tr)} / {len(te)}"],
        [
            "Prevalensi stunting train / test (%)",
            f"{tr[TARGET].mean() * 100:.2f} / {te[TARGET].mean() * 100:.2f}",
        ],
        ["Jumlah fold CV pada train", N_FOLD_CV],
        ["Pemeriksaan lulus", f"{len(cek) - gagal} dari {len(cek)}"],
        ["Fitur seimbang (|SMD| < 0,1)", f"{len(smd) - n_tdk} dari {len(smd)}"],
        [
            "Peran di pipeline",
            "split pendukung data asli untuk EDA bagian 8, feature importance; "
            "split model (setelah SMOTE) ada di 4_PEMODELAN.py",
        ],
    ],
    columns=["indikator", "nilai"],
)
sheets = {
    "00_Ringkasan": ringkas,
    "01_Ukuran_Prevalensi": ukuran,
    "02_Fold_CV": fold,
    "03_Pemeriksaan": cek,
    "04_Keseimbangan_Fitur": smd,
    "05_Provinsi": prov,
}
with pd.ExcelWriter(OUT / "Ringkasan_Split.xlsx", engine="openpyxl") as xw:
    for nm, d in sheets.items():
        d.to_excel(xw, sheet_name=nm[:31], index=False)
        ws = xw.sheets[nm[:31]]
        for col in ws.columns:
            w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 70)
        ws.freeze_panes = "A2"

# ----------------------------------------------------------------------------
# GRAFIK
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
lab = [f"fold {int(k)}" for k in fold.fold_cv] + ["TEST"]
ukur = list(fold.n) + [len(te)]
prevs = list(fold.prevalensi_pct) + [te[TARGET].mean() * 100]
ax[0].bar(lab, ukur, color=["#4C78A8"] * N_FOLD_CV + ["#E45756"])
for i, (u, p) in enumerate(zip(ukur, prevs)):
    ax[0].text(i, u + 8, f"{u}\n({p:.1f}%)", ha="center", fontsize=7)
ax[0].set(
    title="Ukuran bagian (prevalensi stunting)",
    ylabel="Jumlah balita",
    ylim=(0, max(ukur) * 1.18),
)
ax[1].plot(lab, prevs, marker="o", color="#4C78A8")
ax[1].axhline(
    y.mean() * 100, color="gray", ls="--", label=f"seluruh data {y.mean() * 100:.1f}%"
)
ax[1].set(
    title="Prevalensi stunting per bagian",
    ylabel="%",
    ylim=(y.mean() * 100 - 5, y.mean() * 100 + 5),
)
ax[1].legend()
top = smd.head(15).iloc[::-1]
ax[2].barh(
    top.variabel,
    top["|SMD|"],
    color=np.where(top["|SMD|"] >= 0.1, "#E45756", "#4C78A8"),
)
ax[2].axvline(0.1, color="red", ls="--", lw=1)
ax[2].set(
    title="Ketidakseimbangan fitur train vs test (15 terbesar)",
    xlabel="|SMD|  (< 0,1 = seimbang)",
)
fig.tight_layout()
fig.savefig(OUT / "eda_split_ringkasan.png")
plt.close(fig)

log(f"\nHasil disimpan di: {OUT}")
log(
    "  train.csv, test.csv, split_assignment.csv, Ringkasan_Split.xlsx, eda_split_ringkasan.png, laporan_split.txt"
)
log(
    "CATATAN: test.csv dikunci; jangan dipakai untuk memilih fitur, parameter, atau threshold."
)
(OUT / "laporan_split.txt").write_text("\n".join(LOG), encoding="utf-8")
print("Selesai.")
