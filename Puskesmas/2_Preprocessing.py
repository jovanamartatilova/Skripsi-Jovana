# ==========================================================
# 2_PREPROCESSING.PY
# INPUT  : Dataset/Puskesmas/*.xlsx
# OUTPUT : Output/data_preprocessed.xlsx
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================================
# 1. FOLDER INPUT DAN OUTPUT
# ==========================================================

input_folder = Path("Dataset/Puskesmas")
output_folder = Path("Output")

output_folder.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. FUNGSI MEMBERSIHKAN NAMA KOLOM
# ==========================================================


def clean_columns(df):

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    return df

# ==========================================================
# 4. LOAD SEMUA FILE PUSKESMAS
# ==========================================================

files = sorted(input_folder.glob("*.xlsx"))

if len(files) == 0:
    raise FileNotFoundError(
        f"Tidak ditemukan file .xlsx di dalam folder {input_folder}"
    )

print("=" * 70)
print("FILE DATASET YANG DIGUNAKAN")
print("=" * 70)

for file in files:
    print(file.name)

print()
print(f"Jumlah file : {len(files)}")


# ==========================================================
# 5. GABUNGKAN DATA
# ==========================================================

list_df = []

for file in files:
    temp = pd.read_excel(file, dtype={"NIK": str})
    temp = clean_columns(temp)

    # Simpan sumber data
    temp["Sumber Data"] = file.stem

    list_df.append(temp)

df = pd.concat(list_df, ignore_index=True)

jumlah_data_awal = len(df)

print()
print("=" * 70)
print("HASIL PENGGABUNGAN")
print("=" * 70)
print(f"Jumlah data awal : {jumlah_data_awal}")


# ==========================================================
# 6. RENAME KOLOM (Penyeragaman nama variabel)
# ==========================================================

rename_columns = {
    "No": "Nomor",
    "JK": "Jenis Kelamin",
    "Tgl Lahir": "Tanggal Lahir",
    "BB Lahir": "Berat Badan Lahir",
    "TB Lahir": "Tinggi Badan Lahir",
    "Nama Ortu": "Nama Orang Tua",
    "Kab/Kota": "Kabupaten/Kota",
    "Kec": "Kecamatan",
    "Pukesmas": "Puskesmas",
    "Desa/Kel": "Desa/Kelurahan",
    "Berat": "Berat Badan",
    "Tinggi": "Tinggi Badan",
    "ZS BB/U": "Z-Score BB/U",
    "ZS TB/U": "Z-Score TB/U",
    "ZS BB/TB": "Z-Score BB/TB",
    "Jml Vit A": "Jumlah Vitamin A",
}

df = df.rename(columns=rename_columns)


# ==========================================================
# 7. KONVERSI TANGGAL
# ==========================================================

for col in ["Tanggal Lahir", "Tanggal Pengukuran"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ==========================================================
# 8. KONVERSI VARIABEL NUMERIK
# ==========================================================

numeric_columns = [
    "Nomor",
    "Berat Badan Lahir",
    "Tinggi Badan Lahir",
    "Berat Badan",
    "Tinggi Badan",
    "LiLA",
    "Z-Score BB/U",
    "Z-Score TB/U",
    "Z-Score BB/TB",
    "Jumlah Vitamin A",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ==========================================================
# 8b. PENYERAGAMAN NILAI (Transformasi Data)
# ==========================================================

print()
print("=" * 70)
print("PENYERAGAMAN NILAI")
print("=" * 70)

# 1) Hapus spasi berlebih (termasuk non-breaking space pada NIK)
kolom_teks = [
    "NIK",
    "Nama",
    "Jenis Kelamin",
    "Cara Ukur",
    "BB/U",
    "TB/U",
    "BB/TB",
    "Naik Berat Badan",
    "KPSP",
    "KIA",
    "Kelas Ibu Balita",
    "MBG",
    "Posyandu",
]

for col in kolom_teks:
    if col in df.columns:
        df[col] = df[col].str.strip()

# 2) Tanda "-" dianggap sebagai nilai kosong
kolom_strip = ["Cara Ukur", "KPSP", "KIA", "MBG"]
print("Nilai '-' dijadikan missing:")
print(df[kolom_strip].eq("-").sum().to_string())
df[kolom_strip] = df[kolom_strip].replace("-", np.nan)

# 3) Satuan gram -> kilogram
mask_berat_gram = df["Berat Badan"] > 1000
mask_bblahir_gram = df["Berat Badan Lahir"] > 100

print(f"Berat badan dikonversi gram -> kg      : {mask_berat_gram.sum()}")
print(f"Berat badan lahir dikonversi gram -> kg: {mask_bblahir_gram.sum()}")

df.loc[mask_berat_gram, "Berat Badan"] /= 1000
df.loc[mask_bblahir_gram, "Berat Badan Lahir"] /= 1000

# Penyeragaman Jenis Kelamin dan Tanggal Lahir per NIK
# (menggunakan nilai yang paling sering tercatat pada NIK tersebut)
for col in ["Jenis Kelamin", "Tanggal Lahir"]:
    modus = df.groupby("NIK")[col].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else pd.NaT
    )
    awal = df[col].copy()
    df[col] = df["NIK"].map(modus).fillna(df[col])
    print(f"{col} diseragamkan per NIK : {(awal != df[col]).sum()} record")
# ==========================================================
# 9. PERHITUNGAN USIA
# ==========================================================

df["_Usia Bulan"] = (df["Tanggal Pengukuran"] - df["Tanggal Lahir"]).dt.days / 30.4375
# Usia dalam bulan penuh (kalender) untuk kriteria 0-59 bulan
df["_Usia Bulan Penuh"] = (
    (df["Tanggal Pengukuran"].dt.year - df["Tanggal Lahir"].dt.year) * 12
    + (df["Tanggal Pengukuran"].dt.month - df["Tanggal Lahir"].dt.month)
    - (df["Tanggal Pengukuran"].dt.day < df["Tanggal Lahir"].dt.day).astype(int)
)
# ==========================================================
# 10. PEMERIKSAAN KESESUAIAN DATA (Tabel 3.6)
# ==========================================================

tanggal_lahir_kosong = df["Tanggal Lahir"].isna()
tanggal_ukur_kosong = df["Tanggal Pengukuran"].isna()
pengukuran_sebelum_lahir = df["Tanggal Pengukuran"] < df["Tanggal Lahir"]

usia_tidak_terbaca = df["_Usia Bulan"].isna()
usia_di_bawah_0 = df["_Usia Bulan"] < 0
usia_60_atau_lebih = df["_Usia Bulan Penuh"] >= 60

jk_kosong = df["Jenis Kelamin"].isna()

bb_kosong = df["Berat Badan"].isna()
bb_tidak_valid = df["Berat Badan"] <= 0

tb_kosong = df["Tinggi Badan"].isna()
tb_tidak_valid = df["Tinggi Badan"] <= 0

lila_kosong = df["LiLA"].isna()
lila_tidak_valid = df["LiLA"] <= 0


# ==========================================================
# 11. PENANGANAN LiLA
# ==========================================================
# LiLA bukan variabel utama:
# - LiLA kosong tetap dipertahankan.
# - LiLA <= 0 dianggap tidak valid dan dijadikan missing value.

df.loc[lila_tidak_valid, "LiLA"] = np.nan


# ==========================================================
# 12. FILTER DATA VALID
# ==========================================================

invalid_data = (
    jk_kosong
    | tanggal_lahir_kosong
    | tanggal_ukur_kosong
    | pengukuran_sebelum_lahir
    | usia_tidak_terbaca
    | usia_di_bawah_0
    | usia_60_atau_lebih
    | bb_kosong
    | bb_tidak_valid
    | tb_kosong
    | tb_tidak_valid
)

df_clean = df.loc[~invalid_data].copy()

# 12b. HAPUS DUPLIKAT (NIK + Tanggal Pengukuran)
kunci = ["NIK", "Tanggal Pengukuran"]

dup_mask = df_clean["NIK"].notna() & df_clean.duplicated(subset=kunci, keep=False)
tidak_murni = (
    df_clean[dup_mask]
    .groupby(kunci)[["Berat Badan", "Tinggi Badan"]]
    .nunique()
    .gt(1)
    .any(axis=1)
    .sum()
)

# pertahankan record paling lengkap; jika sama, yang terakhir di file
df_clean["_kelengkapan"] = df_clean.notna().sum(axis=1)
df_clean["_urut"] = range(len(df_clean))
df_urut = df_clean.sort_values(["_kelengkapan", "_urut"])
df_urut = df_urut[df_urut["NIK"].notna()]
idx_buang = df_urut.index[df_urut.duplicated(subset=kunci, keep="last")]

df_clean = df_clean.drop(index=idx_buang).drop(columns=["_kelengkapan", "_urut"])


# ==========================================================
# 13. KELOMPOK UMUR (untuk visualisasi dashboard)
# ==========================================================

bins = [-0.001, 6, 12, 24, 36, 48, 60]

labels = [
    "0-5 bulan",
    "6-11 bulan",
    "12-23 bulan",
    "24-35 bulan",
    "36-47 bulan",
    "48-59 bulan",
]

df_clean["Kelompok Umur"] = pd.cut(
    df_clean["_Usia Bulan"], bins=bins, labels=labels, right=False
)
df_clean = df_clean.drop(columns=["_Usia Bulan", "_Usia Bulan Penuh"])

# ==========================================================
# 14. URUTKAN DATA
# ==========================================================

df_clean = df_clean.sort_values(
    by=["NIK", "Tanggal Pengukuran"], na_position="last"
).reset_index(drop=True)


# ==========================================================
# 15. RINGKASAN PREPROCESSING
# ==========================================================

jumlah_data_akhir = len(df_clean)

print()
print("=" * 70)
print("HASIL PREPROCESSING")
print("=" * 70)

print(f"Jumlah data awal             : {jumlah_data_awal}")
print(f"Jumlah data akhir            : {jumlah_data_akhir}")
print(f"Data tidak digunakan         : {jumlah_data_awal - jumlah_data_akhir}")

print(f"  - termasuk duplikat NIK+tanggal dihapus : {len(idx_buang)}")
print(f"  - grup duplikat dengan Berat/Tinggi beda : {tidak_murni}")


print()
print("Pemeriksaan kesesuaian data (sebelum filter):")
print(f"Jenis kelamin kosong         : {jk_kosong.sum()}")
print(f"Tanggal lahir kosong         : {tanggal_lahir_kosong.sum()}")
print(f"Tanggal pengukuran kosong    : {tanggal_ukur_kosong.sum()}")
print(f"Pengukuran sebelum lahir     : {pengukuran_sebelum_lahir.sum()}")
print(f"Usia tidak terbaca           : {usia_tidak_terbaca.sum()}")
print(f"Usia < 0 bulan               : {usia_di_bawah_0.sum()}")
print(f"Usia >= 60 bulan             : {usia_60_atau_lebih.sum()}")
print(f"Berat badan kosong           : {bb_kosong.sum()}")
print(f"Berat badan <= 0             : {bb_tidak_valid.sum()}")
print(f"Tinggi badan kosong          : {tb_kosong.sum()}")
print(f"Tinggi badan <= 0            : {tb_tidak_valid.sum()}")

print()
print("LiLA:")
print(f"LiLA kosong (asli)           : {lila_kosong.sum()}")
print(f"LiLA <= 0 (jadi missing)     : {lila_tidak_valid.sum()}")
print(
    f"LiLA missing pada data akhir : "
    f"{df_clean['LiLA'].isna().sum()} dari {jumlah_data_akhir}"
)

print()
print(f"Balita unik awal             : {df['NIK'].nunique()}")
print(f"Balita unik akhir            : {df_clean['NIK'].nunique()}")


# ==========================================================
# 16. SIMPAN DATA PREPROCESSING
# ==========================================================

output_file = output_folder / "data_preprocessed.xlsx"

df_clean.to_excel(output_file, index=False)

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)
print(f"✔ {output_file}")
print()
print("Preprocessing selesai.")
