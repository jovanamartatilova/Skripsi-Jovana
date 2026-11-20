from pathlib import Path

import pandas as pd

# ==========================================================
# 1. FOLDER INPUT
# ==========================================================

input_folder = Path("Dataset/Puskesmas")

# ==========================================================
# 2. LOAD DAN GABUNGKAN DATA
# ==========================================================

files = sorted(input_folder.glob("*.xlsx"))

if len(files) == 0:
    raise FileNotFoundError(
        f"Tidak ditemukan file .xlsx di dalam folder: {input_folder}"
    )

list_df = []

for file in files:
    temp = pd.read_excel(file, dtype={"NIK": str})

    # Kolom bantu untuk mengetahui asal file setiap record
    temp["Sumber Data"] = file.stem

    list_df.append(temp)

df = pd.concat(list_df, ignore_index=True)

# Pembersihan nama kolom
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

df["NIK"] = df["NIK"].str.strip()

# ==========================================================
# 3. DAFTAR VARIABEL
# ==========================================================

kolom_penting = [
    "NIK",
    "Nama",
    "JK",
    "Tgl Lahir",
    "BB Lahir",
    "Usia Saat Ukur",
    "Tanggal Pengukuran",
    "Berat",
    "Tinggi",
    "LiLA",
    "BB/U",
    "ZS BB/U",
    "TB/U",
    "ZS TB/U",
    "BB/TB",
    "ZS BB/TB",
]

kolom_tanggal = ["Tgl Lahir", "Tanggal Pengukuran"]

# Variabel numerik = variabel yang diperiksa outlier-nya
kolom_numerik = [
    "BB Lahir",
    "Berat",
    "Tinggi",
    "LiLA",
    "ZS BB/U",
    "ZS TB/U",
    "ZS BB/TB",
]

# ==========================================================
# 4. STRUKTUR DAN KARAKTERISTIK DATASET
# ==========================================================

print("=" * 80)
print("1. STRUKTUR DAN KARAKTERISTIK DATASET")
print("=" * 80)

print(f"Jumlah file     : {len(files)}")

print("\nJumlah record per file:")
print(df["Sumber Data"].value_counts().sort_index().to_string())

print(f"\nJumlah record   : {df.shape[0]}")
print(f"Jumlah variabel : {df.shape[1] - 1}  (tidak termasuk 'Sumber Data')")

# ----------------------------------------------------------
# Konversi tipe data (agar pemeriksaan outlier dapat dilakukan)
# ----------------------------------------------------------

for col in kolom_tanggal:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

for col in kolom_numerik:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


print("\nNama dan tipe data setiap variabel:")

tipe_data = pd.DataFrame(
    {"Variabel": df.columns, "Tipe Data": df.dtypes.astype(str).values}
)

tipe_data = tipe_data[tipe_data["Variabel"] != "Sumber Data"]

print(tipe_data.to_string(index=False))


print("\nKeberadaan variabel yang diperlukan:")

kolom_tidak_tersedia = [col for col in kolom_penting if col not in df.columns]

if kolom_tidak_tersedia:
    print("Variabel yang tidak ditemukan:")
    for col in kolom_tidak_tersedia:
        print(f"- {col}")
else:
    print("Seluruh variabel yang diperlukan tersedia.")
print("\nKesesuaian file dengan bulan Tanggal Pengukuran:")
print(pd.crosstab(df["Sumber Data"], df["Tanggal Pengukuran"].dt.month))

# ==========================================================
# 5. MISSING VALUE
# ==========================================================

print("\n")
print("=" * 80)
print("2. MISSING VALUE")
print("=" * 80)

missing = df.drop(columns=["Sumber Data"]).isna().sum()

missing_summary = pd.DataFrame(
    {
        "Variabel": missing.index,
        "Jumlah Missing": missing.values,
        "Persentase (%)": (missing / len(df) * 100).round(2).values,
    }
)

missing_summary = missing_summary[missing_summary["Jumlah Missing"] > 0].sort_values(
    by="Jumlah Missing", ascending=False
)

if len(missing_summary) == 0:
    print("Tidak terdapat missing value pada dataset.")
else:
    print(missing_summary.to_string(index=False))

print("\nJumlah nilai '-' (penanda kosong) per variabel:")
dash = (
    df.drop(columns=["Sumber Data"])
    .astype(str)
    .apply(lambda s: s.str.strip().eq("-"))
    .sum()
)
dash = dash[dash > 0]
print(dash.to_string() if len(dash) else "Tidak ada.")
print(f"\nLiLA bernilai <= 0 : {(df['LiLA'] <= 0).sum()}")

# ==========================================================
# 6. DATA DUPLIKAT
# Berdasarkan kombinasi NIK dan Tanggal Pengukuran
# ==========================================================

print("\n")
print("=" * 80)
print("3. DATA DUPLIKAT")
print("=" * 80)

if "NIK" in df.columns and "Tanggal Pengukuran" in df.columns:
    duplikat = df.duplicated(subset=["NIK", "Tanggal Pengukuran"], keep=False)

    print(
        "Jumlah record yang terlibat duplikasi "
        f"(NIK + Tanggal Pengukuran) : {duplikat.sum()}"
    )

    cek_nik = (
        df.dropna(subset=["NIK"])
        .groupby("NIK")
        .agg(
            n_lahir=("Tgl Lahir", "nunique"),
            n_jk=("JK", "nunique"),
            n_nama=("Nama", "nunique"),
        )
    )
    print(f"NIK kosong              : {df['NIK'].isna().sum()}")
    print(f"NIK dengan >1 Tgl Lahir : {(cek_nik.n_lahir > 1).sum()}")
    print(f"NIK dengan >1 JK        : {(cek_nik.n_jk > 1).sum()}")
    print(f"NIK dengan >1 Nama      : {(cek_nik.n_nama > 1).sum()}")

else:
    print(
        "Kolom NIK atau Tanggal Pengukuran tidak tersedia, "
        "pemeriksaan duplikat tidak dapat dilakukan."
    )

# ==========================================================
# 7. OUTLIER (METODE IQR)
# Hanya identifikasi, tidak ada data yang dihapus. karena menggambarkan keadaan yang sebenarnya.
# ==========================================================

print("\n")
print("=" * 80)
print("4. OUTLIER (METODE IQR)")
print("=" * 80)

outlier_summary = []

for col in kolom_numerik:
    if col not in df.columns:
        continue

    series = df[col].dropna()

    if len(series) < 4:
        continue

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    batas_bawah = Q1 - 1.5 * IQR
    batas_atas = Q3 + 1.5 * IQR

    jumlah_ekstrem = ((series < batas_bawah) | (series > batas_atas)).sum()

    outlier_summary.append(
        {
            "Variabel": col,
            "Min": series.min(),
            "Max": series.max(),
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Batas Bawah": batas_bawah,
            "Batas Atas": batas_atas,
            "Jumlah Outlier": jumlah_ekstrem,
            "Persentase (%)": jumlah_ekstrem / len(series) * 100,
        }
    )

if outlier_summary:
    print(pd.DataFrame(outlier_summary).round(2).to_string(index=False))
else:
    print("Tidak terdapat variabel yang dapat diperiksa outlier-nya.")


# ==========================================================
# 8. PENUTUP
# ==========================================================

print("\n")
print("=" * 80)
print("EDA selesai")
print("=" * 80)
