# -*- coding: utf-8 -*-
"""
Visualisasi_Bab4_IFLS5.py - Semua gambar Bab 4 bagian data IFLS5, sekali run.

Simpan file ini di folder IFLS5 (sejajar 1_EDA.py dst.), lalu jalankan:
    python Visualisasi_Bab4_IFLS5.py

Input (hasil script sebelumnya, tidak ada yang dihitung ulang kecuali ROC data testing):
    output/1_EDA/EDA_IFLS5.xlsx
    output/BuildDataset/balita_ifls5.csv
    output/2_Preprocessing/Ringkasan_Preprocessing.xlsx
    output/3_Splitting/train.csv, test.csv           (untuk ROC data testing)
    output/4_Modeling_A/Ringkasan_Pemodelan.xlsx
    output/4_Modeling_B/Ringkasan_Pemodelan.xlsx, oof.csv, model_S1..S4.joblib
    output/5_FeatureImportance_B/Feature_Importance*.xlsx

Output: output/Visualisasi_Bab4_IFLS5/
    gambar_01 ... gambar_21 (.png, 300 dpi) + tabel_batas_wajar.csv + tabel_winsorizing.csv

Kalau salah satu file input tidak ada, gambar itu dilewati dan gambar lain tetap dibuat.
Warna dan huruf disamakan dengan grafik Puskesmas (biru #183B91, merah #E0553B, Arial).
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
args = ap.parse_args()

BASE = Path(args.base)
O = BASE / "output"
EDA = O / "1_EDA" / "EDA_IFLS5.xlsx"
BUILD_CSV = O / "BuildDataset" / "balita_ifls5.csv"
PRE_XLSX = O / "2_Preprocessing" / "Ringkasan_Preprocessing.xlsx"
SPLIT = O / "3_Splitting"
MODEL = {"A": O / "4_Modeling_A", "B": O / "4_Modeling_B"}
FI_DIR = O / "5_FeatureImportance_B"
OUT = O / "Visualisasi_Bab4_IFLS5"
OUT.mkdir(parents=True, exist_ok=True)

# ======================================================================
# GAYA (sama untuk semua gambar)
# ======================================================================
BIRU, MERAH, KUNING, HIJAU = "#183B91", "#E0553B", "#D9A300", "#2FAE7A"
BIRU_MUDA, ABU, TEKS, GRID = "#7F9BD6", "#B8BCC2", "#202020", "#E5E7EB"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titlepad": 14,
    "axes.labelsize": 12,
    "axes.edgecolor": "#9CA3AF",
    "axes.labelcolor": TEKS,
    "text.color": TEKS,
    "xtick.color": TEKS,
    "ytick.color": TEKS,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

SKENARIO = {
    "S1": "S1 LR + seleksi",
    "S2": "S2 LR tanpa seleksi",
    "S3": "S3 RF + seleksi",
    "S4": "S4 RF tanpa seleksi",
}
WARNA_SKENARIO = {"S1": BIRU_MUDA, "S2": BIRU, "S3": KUNING, "S4": MERAH}
METRIK = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
          "f1": "F1-score", "roc_auc": "ROC-AUC"}

LABEL = {
    "umur_bln": "Umur anak", "jk_laki": "Jenis kelamin", "tb_anak": "Tinggi badan anak",
    "bb_anak": "Berat badan anak", "waz": "WAZ (BB/U)", "bb_lahir_kg": "Berat lahir",
    "bblr": "BBLR", "ukuran_bayi": "Ukuran bayi saat lahir", "lama_hamil_minggu": "Lama kehamilan",
    "hamil_prematur_proxy": "Lahir prematur", "tempat_lahir_grup": "Tempat melahirkan",
    "anak_kembar": "Anak kembar", "periksa_hamil": "Periksa hamil", "anc_total": "Jumlah kunjungan ANC",
    "tablet_besi": "Dapat tablet besi", "jml_tablet_besi": "Jumlah tablet besi",
    "asi_pernah": "Pernah diberi ASI", "usia_air_bln": "Usia pertama diberi air",
    "usia_mpasi_bln": "Usia mulai MPASI", "usia_sapih_bln": "Usia disapih", "masih_asi": "Masih disusui",
    "air_dini": "Air sebelum 6 bulan", "mpasi_dini": "MPASI dini", "asi_eksklusif_proxy": "ASI eksklusif",
    "posyandu_4mgg": "Ke Posyandu 4 minggu", "vit_a_6bln": "Vitamin A 6 bulan",
    "punya_kartu_imun": "Punya KMS/KIA", "imun_bcg": "Imunisasi BCG", "imun_polio": "Imunisasi polio",
    "imun_dpt": "Imunisasi DPT", "imun_hepb": "Imunisasi hepatitis B", "imun_campak": "Imunisasi campak",
    "imun_rota": "Imunisasi rotavirus", "jumlah_jenis_imun": "Jumlah jenis imunisasi",
    "frek_timbang_6bln": "Ditimbang 6 bulan", "hb_anak": "Hemoglobin anak",
    "tinggi_ibu": "Tinggi badan ibu", "tinggi_ayah": "Tinggi badan ayah", "bb_ibu": "Berat badan ibu",
    "bb_ayah": "Berat badan ayah", "imt_ibu": "IMT ibu", "imt_ayah": "IMT ayah", "umur_ibu": "Umur ibu",
    "umur_ayah": "Umur ayah", "ada_ibu_di_RT": "Ibu serumah", "ada_ayah_di_RT": "Ayah serumah",
    "hb_ibu": "Hemoglobin ibu", "pend_ibu": "Pendidikan ibu", "pend_ayah": "Pendidikan ayah",
    "kerja_ibu": "Ibu bekerja", "kerja_ayah": "Ayah bekerja", "perkotaan": "Perkotaan",
    "provinsi": "Provinsi", "kartu_sehat": "Kartu kesehatan", "indeks_kekayaan_pca": "Indeks kekayaan",
    "jumlah_anggota_RT": "Jumlah anggota RT", "ada_perokok_RT": "Ada perokok di rumah",
    "log_pengeluaran_pangan_perkapita": "Pengeluaran pangan/kapita",
    "log_pengeluaran_perkapita": "Pengeluaran/kapita", "air_minum_kode": "Kode sumber air",
    "air_layak": "Air minum layak", "air_direbus": "Air minum direbus", "toilet_kode": "Kode tempat BAB",
    "toilet_layak": "Toilet layak", "limbah_kode": "Kode pembuangan limbah",
    "sampah_kode": "Kode pembuangan sampah", "sampah_layak": "Sampah dibuang layak",
    "garam_beryodium": "Garam beryodium", "bahan_bakar_padat": "Kayu bakar/arang",
    "masalah_lingkungan_rumah": "Masalah lingkungan rumah", "ventilasi_cukup": "Ventilasi cukup",
    "dapur_satu_kamar_tidur": "Dapur satu ruang kamar", "lantai_tanah": "Lantai tanah",
    "frek_makan_per_hari": "Frekuensi makan/hari", "hanya_asi": "Hanya ASI",
    "jenis_makanan_7hr": "Jenis makanan 7 hari", "total_hari_makan_7hr": "Total hari makan 7 hari",
    "status_sehat_umum": "Kesehatan umum anak", "hari_sakit_4mgg": "Hari sakit 4 minggu",
    "hari_terbaring_4mgg": "Hari terbaring 4 minggu", "jumlah_gejala_4mgg": "Jumlah gejala 4 minggu",
    "hari_protein_hewani_7hr": "Protein hewani 7 hari", "hari_susu_7hr": "Minum susu 7 hari",
    "jenis_sayur_buah_7hr": "Sayur dan buah 7 hari", "hari_makanan_olahan_7hr": "Makanan olahan 7 hari",
}

KATEGORI = {
    "air_layak": "Air, sanitasi, dan kebersihan lingkungan",
    "air_direbus": "Air, sanitasi, dan kebersihan lingkungan",
    "toilet_layak": "Air, sanitasi, dan kebersihan lingkungan",
    "sampah_layak": "Air, sanitasi, dan kebersihan lingkungan",
    "masalah_lingkungan_rumah": "Air, sanitasi, dan kebersihan lingkungan",
    "ventilasi_cukup": "Kondisi rumah dan paparan asap",
    "dapur_satu_kamar_tidur": "Kondisi rumah dan paparan asap",
    "ada_perokok_RT": "Kondisi rumah dan paparan asap",
    "masih_asi": "ASI dan MPASI",
    "asi_eksklusif_proxy": "ASI dan MPASI",
    "mpasi_dini": "ASI dan MPASI",
    "frek_makan_per_hari": "Asupan makan anak",
    "jenis_makanan_7hr": "Asupan makan anak",
    "hari_protein_hewani_7hr": "Asupan makan anak",
    "hari_susu_7hr": "Asupan makan anak",
    "hari_makanan_olahan_7hr": "Asupan makan anak",
    "garam_beryodium": "Asupan makan anak",
    "hb_anak": "Anemia (Hb) anak dan ibu",
    "hb_ibu": "Anemia (Hb) anak dan ibu",
    "frek_timbang_6bln": "Pemantauan pertumbuhan di Posyandu",
    "posyandu_4mgg": "Pemantauan pertumbuhan di Posyandu",
    "vit_a_6bln": "Pemantauan pertumbuhan di Posyandu",
    "punya_kartu_imun": "Imunisasi dan jaminan kesehatan",
    "jumlah_jenis_imun": "Imunisasi dan jaminan kesehatan",
    "kartu_sehat": "Imunisasi dan jaminan kesehatan",
}

# sama dengan 2_PREPROCESSING.py
RENTANG = {"bb_lahir_kg": (0.5, 6.5), "lama_hamil_minggu": (20, 50), "anc_total": (0, 20),
           "jml_tablet_besi": (0, 400), "tinggi_ibu": (120, 210), "tinggi_ayah": (120, 210),
           "imt_ibu": (10, 60), "umur_ibu": (10, 90), "waz": (-6, 5)}
WINS = ["bb_lahir_kg", "tinggi_ibu", "tinggi_ayah", "imt_ibu", "umur_ibu", "jml_tablet_besi"]
SATUAN = {"bb_lahir_kg": "kg", "tinggi_ibu": "cm", "tinggi_ayah": "cm", "imt_ibu": "kg/m²",
          "umur_ibu": "tahun", "jml_tablet_besi": "butir"}


# ======================================================================
# FUNGSI BANTU
# ======================================================================
def angka(x, d=0):
    """Format angka gaya Indonesia: 5.243 dan 34,56."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "#").replace(".", ",").replace("#", ".")


def persen_sumbu(ax, sumbu="y", d=0):
    f = FuncFormatter(lambda v, _: angka(v, d) + "%")
    (ax.yaxis if sumbu == "y" else ax.xaxis).set_major_formatter(f)


def desimal_sumbu(ax, sumbu="y", d=1):
    f = FuncFormatter(lambda v, _: angka(v, d))
    (ax.yaxis if sumbu == "y" else ax.xaxis).set_major_formatter(f)


def lbl(v):
    return LABEL.get(v, v)


def sheet(path, awalan):
    """Baca sheet yang namanya diawali awalan (mis. '01_Metrik')."""
    x = pd.ExcelFile(path)
    for s in x.sheet_names:
        if s.startswith(awalan):
            return x.parse(s)
    raise KeyError(f"sheet '{awalan}...' tidak ada di {path.name}")


def label_batang(ax, bars, teks, horizontal=False, ukuran=11, jarak=3):
    for b, t in zip(bars, teks):
        if horizontal:
            ax.annotate(t, (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(jarak, 0),
                        textcoords="offset points", va="center", ha="left", fontsize=ukuran)
        else:
            ax.annotate(t, (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, jarak),
                        textcoords="offset points", ha="center", va="bottom", fontsize=ukuran)


def simpan(fig, nama):
    path = OUT / f"{nama}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [ok] {path.name}")


DIBUAT, DILEWATI = [], []


def jalankan(nama, fungsi):
    try:
        fungsi()
        DIBUAT.append(nama)
    except Exception as e:  # satu gambar gagal tidak menghentikan yang lain
        plt.close("all")
        DILEWATI.append((nama, f"{type(e).__name__}: {e}"))
        print(f"  [lewati] {nama}: {type(e).__name__}: {e}")


def ringkasan_model(m):
    return MODEL[m] / "Ringkasan_Pemodelan.xlsx"


def hanya_s1_s4(d):
    return d[d.skenario.isin(SKENARIO)].copy()


# ======================================================================
# 4.2 EKSPLORASI DATA
# ======================================================================
def g01_alur_subjek():
    d = sheet(EDA, "05_Funnel")
    nama = ["Individu terukur\n(bus_us)", "Ditemukan di data\nanggota rumah tangga",
            "Umur 0–59 bulan", "HAZ dapat dihitung", "HAZ dalam rentang\n−6 sampai 6"]
    d = d.iloc[: len(nama)].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(d))[::-1]
    warna = [ABU, ABU, BIRU_MUDA, BIRU_MUDA, BIRU]
    bars = ax.barh(y, d.n, color=warna, height=0.6)
    teks = []
    for n, k in zip(d.n, d.berkurang):
        t = angka(n)
        if pd.notna(k) and k > 0:
            t += f"   (berkurang {angka(k)})"
        teks.append(t)
    label_batang(ax, bars, teks, horizontal=True)
    ax.set_yticks(y, nama[: len(d)])
    ax.set_xlim(0, d.n.max() * 1.35)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: angka(v)))
    ax.set_xlabel("Jumlah individu")
    ax.grid(axis="y", visible=False)
    ax.set_title("Alur Pemilihan Subjek Balita Data IFLS5")
    simpan(fig, "gambar_01_alur_pemilihan_subjek")


def _target():
    """Angka dari sheet 07_Target_HAZ (nama baris dicocokkan longgar)."""
    d = sheet(EDA, "07_Target")
    hasil = {}
    for ind, nilai in zip(d.indikator.astype(str), d.nilai):
        k = ind.strip().lstrip("-").strip().lower()
        for kunci in ["sangat pendek", "pendek", "tidak stunting", "stunting"]:
            if k.startswith(kunci) and kunci not in hasil:
                hasil[kunci] = float(nilai)
                break
    return hasil


def g02_status_tbu():
    d = _target()
    sp, p, n = d["sangat pendek"], d["pendek"], d["tidak stunting"]
    tot = sp + p + n
    fig, ax = plt.subplots(figsize=(8, 5))
    kat = ["Sangat pendek\n(HAZ < −3)", "Pendek\n(−3 ≤ HAZ < −2)", "Tidak stunting\n(HAZ ≥ −2)"]
    bars = ax.bar(kat, [sp, p, n], color=[KUNING, MERAH, BIRU], width=0.6)
    label_batang(ax, bars, [f"{angka(v)} ({angka(v / tot * 100, 2)}%)" for v in [sp, p, n]])
    ax.set_ylabel("Jumlah balita")
    ax.set_ylim(0, n * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: angka(v)))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Distribusi Status Tinggi Badan menurut Umur (n = {angka(tot)})")
    simpan(fig, "gambar_02_status_tbu")


def g03_prevalensi_umur():
    d = sheet(EDA, "06_Umur_JK")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = [f"{k} bln\n(n = {angka(n)})" for k, n in zip(d.kel_umur, d.n)]
    bars = ax.bar(x, d.prevalensi_stunting_pct, color=MERAH, width=0.6)
    label_batang(ax, bars, [f"{angka(v, 2)}%" for v in d.prevalensi_stunting_pct])
    tot = d.stunting.sum() / d.n.sum() * 100
    ax.axhline(tot, color=BIRU, ls="--", lw=1.2)
    ax.text(len(d) - 0.5, tot + 0.8, f"Rata-rata {angka(tot, 2)}%", color=BIRU, ha="right", fontsize=11)
    ax.set_ylim(0, d.prevalensi_stunting_pct.max() * 1.2)
    persen_sumbu(ax)
    ax.set_ylabel("Prevalensi stunting")
    ax.grid(axis="x", visible=False)
    ax.set_title("Prevalensi Stunting menurut Kelompok Umur")
    simpan(fig, "gambar_03_prevalensi_kelompok_umur")


def g04_prevalensi_subgrup():
    d = sheet(EDA, "07b_Prevalensi")
    d = d[d.kelompok != "kelompok umur"].copy()
    nama = {("jenis kelamin", 0): "Perempuan", ("jenis kelamin", 1): "Laki-laki",
            ("wilayah", 0): "Perdesaan", ("wilayah", 1): "Perkotaan",
            ("lahir BBLR", 0): "Tidak BBLR", ("lahir BBLR", 1): "BBLR",
            ("MPASI dini", 0): "Tidak MPASI dini", ("MPASI dini", 1): "MPASI dini"}
    judul = {"jenis kelamin": "Jenis kelamin", "wilayah": "Wilayah", "lahir BBLR": "Berat lahir",
             "MPASI dini": "MPASI dini"}
    grup = list(dict.fromkeys(d.kelompok))
    fig, axs = plt.subplots(1, len(grup), figsize=(3.3 * len(grup), 4.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, g in zip(axs, grup):
        s = d[d.kelompok == g]
        x = [f"{nama.get((g, int(float(k))), k)}\n(n = {angka(n)})" for k, n in zip(s.kategori, s.n)]
        bars = ax.bar(x, s.prevalensi_pct, color=[BIRU, MERAH][: len(s)], width=0.6)
        label_batang(ax, bars, [f"{angka(v, 2)}%" for v in s.prevalensi_pct], ukuran=10)
        ax.set_title(judul.get(g, g), fontsize=13)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=10)
    axs[0].set_ylim(0, d.prevalensi_pct.max() * 1.22)
    persen_sumbu(axs[0])
    axs[0].set_ylabel("Prevalensi stunting")
    fig.suptitle("Prevalensi Stunting menurut Karakteristik Balita", fontsize=15, y=1.02)
    fig.tight_layout()
    simpan(fig, "gambar_04_prevalensi_subgrup")


def g05_missing():
    d = sheet(EDA, "08_Missing")
    d = d[d.pct_missing > 10].sort_values("pct_missing")
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.6))
    warna = [BIRU if str(v).strip().lower() == "ya" else ABU for v in d.dipakai_model]
    bars = ax.barh([lbl(v) for v in d.variabel], d.pct_missing, color=warna, height=0.65)
    label_batang(ax, bars, [f"{angka(v, 2)}%" for v in d.pct_missing], horizontal=True, ukuran=10)
    ax.axvline(30, color=MERAH, ls="--", lw=1.2)
    ax.text(30.5, -0.9, "batas 30%", color=MERAH, fontsize=10)
    ax.set_xlim(0, d.pct_missing.max() * 1.18)
    persen_sumbu(ax, "x")
    ax.set_xlabel("Persentase missing value")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BIRU), plt.Rectangle((0, 0), 1, 1, color=ABU)],
              labels=["Dipakai di model", "Dikeluarkan"], loc="lower right")
    ax.set_title(f"Variabel dengan Missing Value di Atas 10% ({len(d)} variabel)")
    simpan(fig, "gambar_05_missing_value")


def g06_korelasi():
    d = sheet(EDA, "13_Korelasi_Tinggi")
    d = d[d.spearman.abs() >= 0.7].copy()
    d["pasangan"] = [f"{lbl(a)} – {lbl(b)}" for a, b in zip(d.var_1, d.var_2)]
    d = d.reindex(d.spearman.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(d) + 1.6))
    warna = [BIRU if v > 0 else MERAH for v in d.spearman]
    bars = ax.barh(d.pasangan, d.spearman.abs(), color=warna, height=0.65)
    label_batang(ax, bars, [angka(v, 3) for v in d.spearman], horizontal=True, ukuran=10)
    ax.axvline(0.7, color=TEKS, ls="--", lw=1)
    ax.set_xlim(0.6, 1.0)
    desimal_sumbu(ax, "x", 1)
    ax.set_xlabel("|r| Spearman")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BIRU), plt.Rectangle((0, 0), 1, 1, color=MERAH)],
              labels=["Korelasi positif", "Korelasi negatif"], loc="lower right")
    ax.set_title(f"Pasangan Variabel dengan Korelasi Kuat (|r| ≥ 0,7; {len(d)} pasangan)")
    simpan(fig, "gambar_06_korelasi_kuat")


def g07_univariat():
    d = sheet(EDA, "14_Univariat").sort_values("kekuatan_efek", ascending=False).head(15)
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    warna = [BIRU if u == "Mann-Whitney" else KUNING for u in d.uji]
    bars = ax.barh([lbl(v) for v in d.variabel], d.kekuatan_efek, color=warna, height=0.65)
    label_batang(ax, bars, [angka(v, 3) for v in d.kekuatan_efek], horizontal=True, ukuran=10)
    ax.set_xlim(0, d.kekuatan_efek.max() * 1.18)
    desimal_sumbu(ax, "x", 2)
    ax.set_xlabel("Kekuatan hubungan dengan status stunting")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BIRU), plt.Rectangle((0, 0), 1, 1, color=KUNING)],
              labels=["Mann-Whitney (|AUC − 0,5| × 2)", "Chi-square (Cramér's V)"], loc="lower right")
    ax.set_title("15 Variabel dengan Hubungan Univariat Terkuat")
    simpan(fig, "gambar_07_uji_univariat")


def g08_keluarga():
    d = sheet(EDA, "10b_Balita")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([f"{k} balita" for k in d.balita_per_keluarga], d.n_keluarga, color=BIRU, width=0.6)
    label_batang(ax, bars, [angka(v) for v in d.n_keluarga])
    ax.set_ylim(0, d.n_keluarga.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: angka(v)))
    ax.set_xlabel("Jumlah balita dalam satu keluarga")
    ax.set_ylabel("Jumlah keluarga")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Jumlah Balita per Keluarga ({angka(d.n_keluarga.sum())} keluarga)")
    simpan(fig, "gambar_08_balita_per_keluarga")


def g09_pemilihan_variabel():
    d = sheet(EDA, "18_Seleksi")

    def alasan(r):
        if str(r.status).upper() == "DIPAKAI":
            return "Dipakai di model"
        a = str(r.alasan).lower()
        for kunci, nama in [("kebocoran", "Kebocoran label"), ("multikolinear", "Korelasi kuat"),
                            ("turunan", "Turunan/komponen variabel lain"), ("kode asli", "Kode nominal"),
                            ("ayah", "Data ayah"), ("subjektif", "Subjektif atau lemah"),
                            ("konstan", "Tidak bervariasi di Puskesmas"), ("struktural", "Missing struktural")]:
            if kunci in a:
                return nama
        return "Lainnya"

    d["kel"] = d.apply(alasan, axis=1)
    hit = d.kel.value_counts()
    urut = ["Dipakai di model"] + [k for k in hit.index if k != "Dipakai di model"]
    hit = hit.reindex(urut).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    warna = [BIRU if k == "Dipakai di model" else MERAH for k in hit.index]
    bars = ax.barh(hit.index, hit.values, color=warna, height=0.62)
    label_batang(ax, bars, [angka(v) for v in hit.values], horizontal=True)
    ax.set_xlim(0, hit.max() * 1.15)
    ax.set_xlabel("Jumlah variabel")
    ax.grid(axis="y", visible=False)
    n_pakai = int(hit.get("Dipakai di model", 0))
    ax.set_title(f"Hasil Pemilihan Variabel ({n_pakai} dipakai, {len(d) - n_pakai} dikeluarkan)")
    simpan(fig, "gambar_09_pemilihan_variabel")


# ======================================================================
# 4.3 PRE-PROCESSING
# ======================================================================
def _data_mentah():
    df = pd.read_csv(BUILD_CSV)
    X = df[[c for c in RENTANG if c in df.columns] + [c for c in WINS if c in df.columns and c not in RENTANG]]
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X


def g10_winsorizing():
    X = _data_mentah()
    rows = []
    for c, (lo, hi) in RENTANG.items():
        if c in X.columns:
            m = X[c].notna() & ~X[c].between(lo, hi)
            rows.append({"variabel": c, "label": lbl(c), "batas_bawah": lo, "batas_atas": hi,
                         "jumlah_dijadikan_NaN": int(m.sum())})
            X.loc[m, c] = np.nan
    pd.DataFrame(rows).to_csv(OUT / "tabel_batas_wajar.csv", index=False, encoding="utf-8-sig", sep=";",
                              decimal=",")
    try:  # batas resmi dari 2_PREPROCESSING.py
        w = sheet(PRE_XLSX, "04_Winsorizing").set_index("variabel")
        p1, p99 = w["batas_bawah_P1"], w["batas_atas_P99"]
    except Exception:
        p1, p99 = X[WINS].quantile(0.01), X[WINS].quantile(0.99)
    wins = [c for c in WINS if c in X.columns]
    tw = []
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, c in zip(axs.ravel(), wins):
        s = X[c].dropna()
        t = s.clip(p1[c], p99[c])
        tw.append({"variabel": c, "label": lbl(c), "P1": round(float(p1[c]), 2), "P99": round(float(p99[c]), 2),
                   "n_di_bawah_P1": int((s < p1[c]).sum()), "n_di_atas_P99": int((s > p99[c]).sum()),
                   "min_sebelum": s.min(), "max_sebelum": s.max(), "min_sesudah": t.min(), "max_sesudah": t.max()})
        bp = ax.boxplot([s, t], widths=0.55, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=MERAH, markeredgecolor="none",
                                        alpha=0.6),
                        medianprops=dict(color=TEKS, lw=1.4))
        for patch, col in zip(bp["boxes"], [ABU, BIRU_MUDA]):
            patch.set_facecolor(col)
            patch.set_edgecolor("#6B7280")
        ax.set_xticks([1, 2], ["Sebelum", "Sesudah"])
        ax.set_title(f"{lbl(c)} ({SATUAN.get(c, '')})", fontsize=12)
        ax.grid(axis="x", visible=False)
    for ax in axs.ravel()[len(wins):]:
        ax.set_visible(False)
    fig.suptitle("Sebaran Data Sebelum dan Sesudah Winsorizing (P1–P99)", fontsize=15, y=1.0)
    fig.tight_layout()
    simpan(fig, "gambar_10_winsorizing")
    pd.DataFrame(tw).round(2).to_csv(OUT / "tabel_winsorizing.csv", index=False, encoding="utf-8-sig", sep=";",
                                     decimal=",")
    print("  [ok] tabel_batas_wajar.csv, tabel_winsorizing.csv (untuk isi [ISI] di Bab 4)")


def g11_missing_predictor():
    try:
        d = sheet(PRE_XLSX, "01_Tipe_Variabel")[["variabel", "pct_missing"]]
    except Exception:
        d = sheet(EDA, "08_Missing")
        d = d[d.dipakai_model.astype(str).str.lower().eq("ya")][["variabel", "pct_missing"]]
    d = d[d.pct_missing > 0].sort_values("pct_missing").tail(20)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.6))
    warna = [MERAH if v > 10 else BIRU for v in d.pct_missing]
    bars = ax.barh([lbl(v) for v in d.variabel], d.pct_missing, color=warna, height=0.65)
    label_batang(ax, bars, [f"{angka(v, 2)}%" for v in d.pct_missing], horizontal=True, ukuran=10)
    ax.set_xlim(0, d.pct_missing.max() * 1.18)
    persen_sumbu(ax, "x")
    ax.set_xlabel("Persentase missing value sebelum imputasi (sesudah imputasi: 0%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Missing Value pada 20 Predictor Model Sebelum Imputasi Median")
    simpan(fig, "gambar_11_missing_predictor")


# ======================================================================
# 4.7 SMOTE DAN SPLITTING
# ======================================================================
def g12_smote_split():
    t = _target()
    st0, ts0 = t["stunting"], t["tidak stunting"]
    cm = sheet(ringkasan_model("B"), "04_Confusion").set_index("skenario").loc["S4"]
    uji_st, uji_ts = cm.FN + cm.TP, cm.TN + cm.FP
    seimbang = max(st0, ts0)
    data = {"Sebelum SMOTE": (st0, ts0), "Setelah SMOTE": (seimbang, seimbang),
            "Data training (80%)": (seimbang - uji_st, seimbang - uji_ts),
            "Data testing (20%)": (uji_st, uji_ts)}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(data))
    lebar = 0.36
    b1 = ax.bar(x - lebar / 2, [v[0] for v in data.values()], lebar, color=MERAH, label="Stunting")
    b2 = ax.bar(x + lebar / 2, [v[1] for v in data.values()], lebar, color=BIRU, label="Tidak stunting")
    label_batang(ax, b1, [angka(v[0]) for v in data.values()], ukuran=10)
    label_batang(ax, b2, [angka(v[1]) for v in data.values()], ukuran=10)
    ax.set_xticks(x, [f"{k}\n(n = {angka(sum(v))})" for k, v in data.items()])
    ax.set_ylim(0, seimbang * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: angka(v)))
    ax.set_ylabel("Jumlah data")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Jumlah Kelas Sebelum dan Sesudah SMOTE serta Hasil Pembagian Data")
    simpan(fig, "gambar_12_smote_splitting")


# ======================================================================
# 4.8 - 4.9 PEMODELAN DAN EVALUASI
# ======================================================================
def _batang_metrik(ax, d, judul):
    x = np.arange(len(METRIK))
    n = len(d)
    lebar = 0.8 / n
    for i, r in enumerate(d.itertuples()):
        nilai = [getattr(r, k) for k in METRIK]
        bars = ax.bar(x - 0.4 + lebar * (i + 0.5), nilai, lebar, color=WARNA_SKENARIO[r.skenario],
                      label=SKENARIO[r.skenario])
        if r.skenario == "S4":  # angka hanya untuk skenario terpilih, angka lengkap ada di tabel
            label_batang(ax, bars, [angka(v, 3) for v in nilai], ukuran=9, jarak=2)
    ax.set_xticks(x, list(METRIK.values()))
    ax.set_ylim(0.5, 1.0)
    desimal_sumbu(ax, "y", 1)
    ax.grid(axis="x", visible=False)
    ax.set_title(judul, fontsize=13)


def g13_metrik_model_b():
    oof = hanya_s1_s4(sheet(ringkasan_model("B"), "01_Metrik_OOF"))
    tes = hanya_s1_s4(sheet(ringkasan_model("B"), "03_Metrik_Test"))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    _batang_metrik(axs[0], oof, "Validasi silang (OOF) data training")
    _batang_metrik(axs[1], tes, "Data testing")
    axs[0].set_ylabel("Nilai metrik")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Kinerja Empat Skenario pada Model B (threshold 0,5)", fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    simpan(fig, "gambar_13_metrik_skenario_model_b")


def g14_model_a_vs_b():
    hasil = []
    for m in ["A", "B"]:
        pil = sheet(ringkasan_model(m), "02_Pemilihan")
        pil = hanya_s1_s4(pil).sort_values("peringkat").iloc[0].skenario
        tes = sheet(ringkasan_model(m), "03_Metrik_Test").set_index("skenario").loc[pil]
        hasil.append((m, pil, tes))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(METRIK))
    lebar = 0.36
    for i, (m, s, r) in enumerate(hasil):
        nilai = [r[k] for k in METRIK]
        ket = "dengan tinggi badan anak" if m == "A" else "tanpa tinggi badan anak"
        bars = ax.bar(x - lebar / 2 + i * lebar, nilai, lebar, color=[ABU, BIRU][i],
                      label=f"Model {m} – {SKENARIO[s]} ({ket})")
        label_batang(ax, bars, [angka(v, 3) for v in nilai], ukuran=9, jarak=2)
    ax.set_xticks(x, list(METRIK.values()))
    ax.set_ylim(0.5, 1.05)
    desimal_sumbu(ax, "y", 1)
    ax.set_ylabel("Nilai metrik (data testing)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Perbandingan Skenario Terpilih Model A dan Model B")
    simpan(fig, "gambar_14_model_a_vs_b")


def g15_per_fold():
    d = sheet(ringkasan_model("B"), "05_Metrik_Per_Fold")
    d = d[d.skenario == "S4"].sort_values("fold")
    fig, ax = plt.subplots(figsize=(9, 5))
    fold = [f"Fold {int(f) + 1}" for f in d.fold]
    for kol, warna, nama, geser in [("roc_auc", BIRU, "ROC-AUC", 8), ("f1", MERAH, "F1-score", -15)]:
        ax.plot(fold, d[kol], marker="o", color=warna, lw=2, label=f"{nama} (rata-rata {angka(d[kol].mean(), 3)})")
        for xi, v in zip(fold, d[kol]):
            ax.annotate(angka(v, 3), (xi, v), xytext=(0, geser), textcoords="offset points", ha="center",
                        fontsize=10, color=warna)
    ax.set_ylim(0.75, 0.95)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    desimal_sumbu(ax, "y", 2)
    ax.set_ylabel("Nilai metrik")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Kinerja S4 Model B pada Setiap Fold Validasi Silang")
    simpan(fig, "gambar_15_kinerja_per_fold")


def g16_confusion():
    cm = sheet(ringkasan_model("B"), "04_Confusion").set_index("skenario").loc["S4"]
    m = np.array([[cm.TN, cm.FP], [cm.FN, cm.TP]], float)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    ax.imshow([[0, 1], [1, 0]], cmap=matplotlib.colors.ListedColormap(["#DCE6F7", "#FBE3DE"]))
    ket = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            pct = m[i, j] / m[i].sum() * 100
            ax.text(j, i, f"{angka(m[i, j])}\n{ket[i][j]} ({angka(pct, 1)}%)", ha="center", va="center",
                    fontsize=14, color=BIRU if i == j else MERAH, fontweight="bold")
    ax.set_xticks([0, 1], ["Tidak stunting", "Stunting"])
    ax.set_yticks([0, 1], ["Tidak stunting", "Stunting"])
    ax.set_xlabel("Hasil prediksi")
    ax.set_ylabel("Kondisi sebenarnya")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion Matrix S4 Model B pada Data Testing")
    simpan(fig, "gambar_16_confusion_matrix")


def _roc(ax, y, prob, judul):
    from sklearn.metrics import roc_auc_score, roc_curve
    for s in SKENARIO:
        if s not in prob:
            continue
        fpr, tpr, _ = roc_curve(y, prob[s])
        ax.plot(fpr, tpr, color=WARNA_SKENARIO[s], lw=2.2 if s == "S4" else 1.6,
                label=f"{SKENARIO[s]} (AUC {angka(roc_auc_score(y, prob[s]), 3)})")
    ax.plot([0, 1], [0, 1], color=ABU, ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    desimal_sumbu(ax, "x", 1)
    desimal_sumbu(ax, "y", 1)
    ax.set_xlabel("False positive rate (1 − spesifisitas)")
    ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title(judul, fontsize=13)


def _prob_testing():
    """Mengulang langkah 4_PEMODELAN.py --protokol-paper untuk mendapat probabilitas data testing."""
    import joblib
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    tr, te = pd.read_csv(SPLIT / "train.csv"), pd.read_csv(SPLIT / "test.csv")
    y_all = np.r_[tr.stunting.values, te.stunting.values].astype(int)
    prob, y_uji = {}, None
    for s in SKENARIO:
        f = MODEL["B"] / f"model_{s}.joblib"
        if not f.exists():
            continue
        b = joblib.load(f)
        fit = b["fitur_numerik"] + b["fitur_kategorik"]

        def siap(d):
            x = d[fit].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            for c, (lo, hi) in b["rentang"].items():
                if c in x.columns:
                    x.loc[x[c].notna() & ~x[c].between(lo, hi), c] = np.nan
            return x

        Z = np.vstack([b["preprocessor"].transform(siap(tr)), b["preprocessor"].transform(siap(te))]).astype(float)
        Zs, ys = SMOTE(random_state=42).fit_resample(Z, y_all)
        _, ib = train_test_split(np.arange(len(ys)), test_size=0.2, stratify=ys, random_state=42)
        prob[s] = b["model"].predict_proba(Zs[ib][:, b["indeks_terpilih"]])[:, 1]
        y_uji = ys[ib]
    if not prob:
        raise FileNotFoundError("model_S1..S4.joblib tidak ditemukan")
    return y_uji, prob


def g17_roc():
    oof = pd.read_csv(MODEL["B"] / "oof.csv")
    prob_oof = {s: oof[f"p_oof_{s}"].values for s in SKENARIO if f"p_oof_{s}" in oof.columns}
    try:
        y_uji, prob_uji = _prob_testing()
        ada_uji = True
    except Exception as e:
        print(f"    (ROC data testing dilewati: {type(e).__name__}: {e})")
        ada_uji = False
    fig, axs = plt.subplots(1, 2 if ada_uji else 1, figsize=(13 if ada_uji else 6.8, 6))
    axs = np.atleast_1d(axs)
    _roc(axs[0], oof.stunting.values, prob_oof, "Validasi silang (OOF) data training")
    if ada_uji:
        _roc(axs[1], y_uji, prob_uji, "Data testing")
    fig.suptitle("Kurva ROC Empat Skenario Model B", fontsize=15, y=1.0)
    fig.tight_layout()
    simpan(fig, "gambar_17_kurva_roc")


# ======================================================================
# 4.10 FEATURE IMPORTANCE
# ======================================================================
def _fi_file():
    f = sorted(FI_DIR.glob("Feature_Importance*.xlsx"))
    if not f:
        raise FileNotFoundError(f"Feature_Importance*.xlsx tidak ada di {FI_DIR}")
    return f[0]


def _warna_jenis(j):
    j = str(j).lower()
    if j.startswith("kontrol"):
        return BIRU
    if j.startswith("faktor latar"):
        return KUNING
    return HIJAU


def g18_fi_top15():
    d = sheet(_fi_file(), "01_Semua").sort_values("turun_AUC_rata2", ascending=False).head(15).iloc[::-1]
    nama = [lbl(v) for v in d.variabel]
    nilai = d.turun_AUC_rata2.values
    batas = np.sort(nilai)[-2] * 1.7  # WAZ jauh lebih besar, batangnya dipotong agar yang lain terbaca
    fig, ax = plt.subplots(figsize=(10, 6.8))
    tampil = np.minimum(nilai, batas)
    bars = ax.barh(nama, tampil, xerr=np.where(nilai > batas, 0, 2 * d.galat_baku.values),
                   color=[_warna_jenis(j) for j in d.jenis], height=0.65,
                   error_kw=dict(ecolor="#6B7280", lw=1, capsize=2))
    ujung = np.where(nilai > batas, tampil, tampil + 2 * d.galat_baku.values)
    for b, v, u in zip(bars, nilai, ujung):
        t = angka(v, 4) + ("  (batang dipotong)" if v > batas else "")
        ax.annotate(t, (u, b.get_y() + b.get_height() / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=10)
    ax.set_xlim(0, batas * 1.45)
    desimal_sumbu(ax, "x", 3)
    ax.set_xlabel("Rata-rata penurunan ROC-AUC (garis = 2 × standard error)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in [BIRU, KUNING, HIJAU]],
              labels=["Kontrol", "Latar (tidak dapat ditindaklanjuti)", "Dapat ditindaklanjuti kader"],
              loc="lower right")
    ax.set_title("15 Predictor dengan Permutation Importance Terbesar (S4 Model B)")
    simpan(fig, "gambar_18_permutation_importance_top15")


def g19_pareto():
    d = sheet(_fi_file(), "02_Dapat").sort_values("porsi_pct", ascending=False)
    d = d[d.porsi_pct > 0]
    pilih = d.dipilih.astype(str).str.lower().eq("ya").values
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(d))
    bars = ax.bar(x, d.porsi_pct, color=np.where(pilih, HIJAU, ABU), width=0.7)
    label_batang(ax, bars, [angka(v, 1) for v in d.porsi_pct], ukuran=8, jarak=2)
    ax.set_xticks(x, [lbl(v) for v in d.variabel], rotation=45, ha="right", fontsize=10)
    persen_sumbu(ax)
    ax.set_ylabel("Porsi kontribusi")
    ax.grid(axis="x", visible=False)
    ax2 = ax.twinx()
    ax2.plot(x, d.kumulatif_pct, color=MERAH, marker="o", ms=4, lw=1.8)
    ax2.axhline(80, color=MERAH, ls="--", lw=1)
    ax2.text(len(d) - 0.6, 81.5, "batas 80%", color=MERAH, ha="right", fontsize=10)
    ax2.set_ylim(0, 105)
    persen_sumbu(ax2)
    ax2.set_ylabel("Kontribusi kumulatif", color=MERAH)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    n = int(pilih.sum())
    kum = d.kumulatif_pct.values[n - 1] if n else 0
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HIJAU), plt.Rectangle((0, 0), 1, 1, color=ABU)],
              labels=[f"Dipilih untuk dashboard ({n} variabel, {angka(kum, 2)}%)", "Tidak dipilih"],
              loc="center right")
    ax.set_title("Kontribusi Variabel yang Dapat Ditindaklanjuti Kader")
    simpan(fig, "gambar_19_pareto_dapat_ditindaklanjuti")


def g20_kategori():
    d = sheet(_fi_file(), "02_Dapat").copy()
    d["kategori"] = d.variabel.map(KATEGORI).fillna("Lainnya")
    k = d.groupby("kategori").agg(porsi=("porsi_pct", "sum"), n=("variabel", "size")).sort_values("porsi")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.barh([f"{i} ({n} variabel)" for i, n in zip(k.index, k.n)], k.porsi, color=HIJAU, height=0.62)
    label_batang(ax, bars, [f"{angka(v, 1)}%" for v in k.porsi], horizontal=True)
    ax.set_xlim(0, k.porsi.max() * 1.18)
    persen_sumbu(ax, "x")
    ax.set_xlabel("Jumlah porsi kontribusi variabel dalam kategori")
    ax.grid(axis="y", visible=False)
    ax.set_title("Kontribusi Tujuh Kategori Faktor yang Dapat Ditindaklanjuti")
    simpan(fig, "gambar_20_kategori_faktor")


def g21_kelompok():
    d = sheet(_fi_file(), "01_Semua")

    def kel(j):
        j = str(j).lower()
        return "Kontrol" if j.startswith("kontrol") else ("Latar" if j.startswith("faktor latar")
                                                         else "Dapat ditindaklanjuti")

    d["kel"] = d.jenis.map(kel)
    g = d.groupby("kel").agg(kontribusi=("kontribusi_pct_total", "sum"), n=("variabel", "size"),
                             nyata=("nyata", lambda s: s.astype(str).str.lower().eq("ya").sum()))
    g = g.reindex(["Kontrol", "Latar", "Dapat ditindaklanjuti"])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar([f"{i}\n({n} variabel, {v} nyata)" for i, n, v in zip(g.index, g.n, g.nyata)], g.kontribusi,
                  color=[BIRU, KUNING, HIJAU], width=0.55)
    label_batang(ax, bars, [f"{angka(v, 1)}%" for v in g.kontribusi])
    ax.set_ylim(0, 100)
    persen_sumbu(ax)
    ax.set_ylabel("Porsi dari total kontribusi")
    ax.grid(axis="x", visible=False)
    ax.set_title("Kontribusi Predictor menurut Kelompok (S4 Model B)")
    simpan(fig, "gambar_21_kontribusi_kelompok")


# ======================================================================
# JALANKAN SEMUA
# ======================================================================
DAFTAR = [
    ("4.2 Gambar 01 alur pemilihan subjek", g01_alur_subjek),
    ("4.2 Gambar 02 status TB/U", g02_status_tbu),
    ("4.2 Gambar 03 prevalensi kelompok umur", g03_prevalensi_umur),
    ("4.2 Gambar 04 prevalensi subgrup", g04_prevalensi_subgrup),
    ("4.2 Gambar 05 missing value", g05_missing),
    ("4.2 Gambar 06 korelasi kuat", g06_korelasi),
    ("4.2 Gambar 07 uji univariat", g07_univariat),
    ("4.2 Gambar 08 balita per keluarga", g08_keluarga),
    ("4.2 Gambar 09 pemilihan variabel", g09_pemilihan_variabel),
    ("4.3 Gambar 10 winsorizing", g10_winsorizing),
    ("4.3 Gambar 11 missing predictor", g11_missing_predictor),
    ("4.7 Gambar 12 SMOTE dan splitting", g12_smote_split),
    ("4.9 Gambar 13 metrik skenario Model B", g13_metrik_model_b),
    ("4.9 Gambar 14 Model A vs B", g14_model_a_vs_b),
    ("4.9 Gambar 15 kinerja per fold", g15_per_fold),
    ("4.9 Gambar 16 confusion matrix", g16_confusion),
    ("4.9 Gambar 17 kurva ROC", g17_roc),
    ("4.10 Gambar 18 permutation importance", g18_fi_top15),
    ("4.10 Gambar 19 pareto dapat ditindaklanjuti", g19_pareto),
    ("4.10 Gambar 20 kategori faktor", g20_kategori),
    ("4.10 Gambar 21 kontribusi kelompok", g21_kelompok),
]

if __name__ == "__main__":
    print("=== VISUALISASI BAB 4 DATA IFLS5 ===")
    print(f"Folder output: {OUT}\n")
    for nama, f in DAFTAR:
        print(nama)
        jalankan(nama, f)
    print(f"\nSelesai: {len(DIBUAT)} gambar dibuat, {len(DILEWATI)} dilewati.")
    for nama, alasan in DILEWATI:
        print(f"  - {nama}: {alasan}")
    sys.exit(0)