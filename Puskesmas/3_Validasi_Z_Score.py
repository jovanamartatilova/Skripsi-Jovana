# ============================================================
# 3. VALIDASI Z-SCORE PUSKESMAS MENGGUNAKAN PYGROWUP 0.8.2
# ============================================================

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pygrowup import Calculator

# ============================================================
# 1. FOLDER INPUT DAN OUTPUT
# ============================================================

OUTPUT_FOLDER = Path("Output")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_FOLDER / "data_preprocessed.xlsx"
OUTPUT_FILE = OUTPUT_FOLDER / "data_zscore_validated.xlsx"


# ============================================================
# 2. NAMA KOLOM DATASET
# ============================================================

COLUMN_NIK = "NIK"
COLUMN_DOB = "Tanggal Lahir"
COLUMN_MEASUREMENT_DATE = "Tanggal Pengukuran"
COLUMN_SEX = "Jenis Kelamin"
COLUMN_WEIGHT = "Berat Badan"
COLUMN_HEIGHT = "Tinggi Badan"
COLUMN_MEASUREMENT_METHOD = "Cara Ukur"

# Kolom Z-score asli dari Puskesmas.
COLUMN_WFA_ORIGINAL = "Z-Score BB/U"
COLUMN_LHFA_ORIGINAL = "Z-Score TB/U"
COLUMN_WFH_ORIGINAL = "Z-Score BB/TB"


# ============================================================
# 3. PENGATURAN LOGGING
# ============================================================
# Pesan seperti "too long for recumbent" berasal dari logger
# pygrowup. Pesan tersebut tetap dicatat, tetapi tidak membuat
# program berhenti.

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

logging.getLogger("pygrowup").setLevel(logging.WARNING)


# ============================================================
# 4. LOAD DATA
# ============================================================

print("=" * 70)
print("LOAD DATA HASIL PREPROCESSING")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

print(f"File input  : {INPUT_FILE}")
print(f"Jumlah data : {len(df)}")


# ============================================================
# 5. CEK KOLOM YANG DIBUTUHKAN
# ============================================================

required_columns = [
    COLUMN_NIK,
    COLUMN_DOB,
    COLUMN_MEASUREMENT_DATE,
    COLUMN_SEX,
    COLUMN_WEIGHT,
    COLUMN_HEIGHT,
    COLUMN_MEASUREMENT_METHOD,
    COLUMN_WFA_ORIGINAL,
    COLUMN_LHFA_ORIGINAL,
    COLUMN_WFH_ORIGINAL,
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise KeyError(
        "Kolom berikut tidak ditemukan dalam dataset:\n"
        + "\n".join(f"- {col}" for col in missing_columns)
    )


# ============================================================
# 6. KONVERSI TIPE DATA
# ============================================================

print()
print("=" * 70)
print("KONVERSI TIPE DATA")
print("=" * 70)

df[COLUMN_DOB] = pd.to_datetime(df[COLUMN_DOB], errors="coerce")

df[COLUMN_MEASUREMENT_DATE] = pd.to_datetime(
    df[COLUMN_MEASUREMENT_DATE], errors="coerce"
)

numeric_columns = [
    COLUMN_WEIGHT,
    COLUMN_HEIGHT,
    COLUMN_WFA_ORIGINAL,
    COLUMN_LHFA_ORIGINAL,
    COLUMN_WFH_ORIGINAL,
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 7. NORMALISASI JENIS KELAMIN
# ============================================================


def normalize_sex(value):
    """
    Mengubah berbagai format jenis kelamin menjadi:
    M = male/laki-laki
    F = female/perempuan
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    male_values = {"l", "lk", "laki-laki", "laki laki", "laki_laki", "male", "m", "1"}

    female_values = {"p", "pr", "perempuan", "female", "f", "2"}

    if value in male_values:
        return "M"

    if value in female_values:
        return "F"

    return np.nan


df["Sex_Pygrowup"] = df[COLUMN_SEX].apply(normalize_sex)


# ============================================================
# 8. HITUNG USIA DALAM BULAN
# ============================================================


def calculate_age_in_months(row):
    """
    Menghitung usia dalam bulan berdasarkan:
    tanggal pengukuran - tanggal lahir.

    pygrowup 0.8.2 menggunakan usia dalam bulan.
    """

    date_of_birth = row[COLUMN_DOB]
    measurement_date = row[COLUMN_MEASUREMENT_DATE]

    if pd.isna(date_of_birth):
        return np.nan

    if pd.isna(measurement_date):
        return np.nan

    if measurement_date < date_of_birth:
        return np.nan

    age_in_days = (measurement_date - date_of_birth).days

    age_in_months = age_in_days / 30.4375

    return age_in_months


df["Age_Months_Pygrowup"] = df.apply(calculate_age_in_months, axis=1)


# ============================================================
# 9. NORMALISASI CARA UKUR
# ============================================================


def normalize_measurement_method(value):
    """
    Mengubah berbagai bentuk cara ukur menjadi:
    - WFH untuk pengukuran berdiri
    - WFL untuk pengukuran terlentang
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    standing_values = {
        "berdiri",
        "berdiri tegak",
        "standing",
        "height",
        "tinggi",
        "wfh",
    }

    recumbent_values = {
        "terlentang",
        "tidur",
        "recumbent",
        "lying",
        "length",
        "panjang",
        "wfl",
    }

    if value in standing_values:
        return "WFH"

    if value in recumbent_values:
        return "WFL"

    return np.nan


df["Measurement_Method_Pygrowup"] = df[COLUMN_MEASUREMENT_METHOD].apply(
    normalize_measurement_method
)

kosong = df["Measurement_Method_Pygrowup"].isna() & df["Age_Months_Pygrowup"].notna()

# Hanya dilaporkan, tidak diubah
tidak_sesuai = (
    (df["Age_Months_Pygrowup"] < 24) & (df["Measurement_Method_Pygrowup"] == "WFH")
) | ((df["Age_Months_Pygrowup"] >= 24) & (df["Measurement_Method_Pygrowup"] == "WFL"))

# Aturan WHO: < 24 bulan terlentang (WFL), >= 24 bulan berdiri (WFH)
df.loc[kosong, "Measurement_Method_Pygrowup"] = np.where(
    df.loc[kosong, "Age_Months_Pygrowup"] < 24, "WFL", "WFH"
)
df["Cara_Ukur_Sumber"] = "Data Puskesmas"
df.loc[kosong, "Cara_Ukur_Sumber"] = "Diisi dari usia"

print(f"Cara ukur diisi dari usia           : {kosong.sum()}")
print(f"Cara ukur Puskesmas tidak sesuai usia: {tidak_sesuai.sum()}")
# Koreksi panjang/tinggi ±0,7 cm (WHO / Permenkes No. 2 Tahun 2020)
df["Tinggi_Koreksi"] = df[COLUMN_HEIGHT]
df.loc[(df["Age_Months_Pygrowup"] < 24) & (df["Measurement_Method_Pygrowup"] == "WFH"), "Tinggi_Koreksi"] += 0.7
df.loc[(df["Age_Months_Pygrowup"] >= 24) & (df["Measurement_Method_Pygrowup"] == "WFL"), "Tinggi_Koreksi"] -= 0.7
print(f"Tinggi dikoreksi ±0,7 cm             : {(df['Tinggi_Koreksi'] != df[COLUMN_HEIGHT]).sum()}")


# ============================================================
# 10. INISIALISASI CALCULATOR
# ============================================================

calculator = Calculator(
    adjust_height_data=False,
    adjust_weight_scores=False,
    include_cdc=False,
    logger_name="pygrowup",
    log_level="WARNING",
)


# ============================================================
# 11. FUNGSI KONVERSI KE FLOAT
# ============================================================


def to_float_or_nan(value):
    """
    Mengubah Decimal, integer, atau float menjadi float.
    Nilai kosong/tidak valid menjadi NaN.
    """

    if pd.isna(value):
        return np.nan

    try:
        return float(value)

    except (TypeError, ValueError):
        return np.nan


# ============================================================
# 12. FUNGSI PERHITUNGAN Z-SCORE
# ============================================================


def calculate_zscores(row):

    result = {
        "BB_U_Pygrowup": np.nan,
        "TB_U_Pygrowup": np.nan,
        "BB_TB_Pygrowup": np.nan,
        "Zscore_Status": "Tidak dapat dihitung",
        "Zscore_Error": np.nan,
    }

    sex = row["Sex_Pygrowup"]
    age_months = row["Age_Months_Pygrowup"]
    weight = row[COLUMN_WEIGHT]
    height = row["Tinggi_Koreksi"]
    measurement_method = row["Measurement_Method_Pygrowup"]

    # --------------------------------------------------------
    # Pemeriksaan data dasar
    # --------------------------------------------------------

    if pd.isna(sex):
        result["Zscore_Error"] = "Jenis kelamin tidak valid"
        return pd.Series(result)

    if pd.isna(age_months):
        result["Zscore_Error"] = "Usia tidak valid"
        return pd.Series(result)

    if pd.isna(weight) or weight <= 0:
        result["Zscore_Error"] = "Berat badan tidak valid"
        return pd.Series(result)

    if pd.isna(height) or height <= 0:
        result["Zscore_Error"] = "Tinggi/panjang badan tidak valid"
        return pd.Series(result)
    errors = []

    try:
        result["BB_U_Pygrowup"] = calculator.wfa(
            measurement=weight, age_in_months=age_months, sex=sex
        )
    except Exception as error:
        errors.append(f"BB/U: {type(error).__name__}: {error}")

    try:
        result["TB_U_Pygrowup"] = calculator.lhfa(
            measurement=height, age_in_months=age_months, sex=sex
        )
    except Exception as error:
        errors.append(f"TB/U: {type(error).__name__}: {error}")

    try:
        if pd.isna(measurement_method):
            raise ValueError("Cara ukur tidak valid")
        hitung = calculator.wfl if age_months < 24 else calculator.wfh
        result["BB_TB_Pygrowup"] = hitung(
            measurement=weight, age_in_months=age_months, sex=sex, height=height
        )
    except Exception as error:
        errors.append(f"BB/TB: {type(error).__name__}: {error}")

    n_ok = sum(
        pd.notna(result[k])
        for k in ["BB_U_Pygrowup", "TB_U_Pygrowup", "BB_TB_Pygrowup"]
    )
    if not errors:
        result["Zscore_Status"] = "Berhasil"
    elif n_ok > 0:
        result["Zscore_Status"] = "Sebagian berhasil"
    if errors:
        result["Zscore_Error"] = " | ".join(errors)

    return pd.Series(result)


# ============================================================
# 13. PERHITUNGAN Z-SCORE
# ============================================================

print()
print("=" * 70)
print("PERHITUNGAN Z-SCORE MENGGUNAKAN PYGROWUP")
print("=" * 70)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    zscore_result = df.apply(calculate_zscores, axis=1)

df = pd.concat([df, zscore_result], axis=1)

print(df["Zscore_Status"].value_counts(dropna=False))


# ============================================================
# 14. KONVERSI SEMUA Z-SCORE KE FLOAT
# ============================================================

zscore_columns = [
    "BB_U_Pygrowup",
    "TB_U_Pygrowup",
    "BB_TB_Pygrowup",
    COLUMN_WFA_ORIGINAL,
    COLUMN_LHFA_ORIGINAL,
    COLUMN_WFH_ORIGINAL,
]

for col in zscore_columns:
    df[col] = df[col].apply(to_float_or_nan)


# ============================================================
# 15. SIMPAN Z-SCORE ASLI PUSKESMAS
# ============================================================

df["BB_U_Asli_Puskesmas"] = df[COLUMN_WFA_ORIGINAL]

df["TB_U_Asli_Puskesmas"] = df[COLUMN_LHFA_ORIGINAL]

df["BB_TB_Asli_Puskesmas"] = df[COLUMN_WFH_ORIGINAL]


# ============================================================
# 16. HITUNG SELISIH Z-SCORE
# ============================================================

df["BB_U_Selisih"] = df["BB_U_Pygrowup"] - df["BB_U_Asli_Puskesmas"]

df["TB_U_Selisih"] = df["TB_U_Pygrowup"] - df["TB_U_Asli_Puskesmas"]

df["BB_TB_Selisih"] = df["BB_TB_Pygrowup"] - df["BB_TB_Asli_Puskesmas"]


# ============================================================
# 17. FUNGSI PEMBANDINGAN Z-SCORE
# ============================================================


def compare_zscore(original, calculated, tolerance=0.01):
    """
    Membandingkan Z-score asli dan hasil pygrowup.

    Toleransi 0.01 digunakan karena Z-score biasanya
    disajikan hingga dua angka desimal.
    """

    if pd.isna(original) or pd.isna(calculated):
        return np.nan

    # Selisih dibulatkan 2 desimal agar tidak terpengaruh
    # galat floating point (mis. 0.0100000001 > 0.01)
    selisih = round(abs(float(original) - float(calculated)), 2)

    return bool(selisih <= tolerance)


# ============================================================
# 18. CEK KESESUAIAN Z-SCORE
# ============================================================

TOLERANCE = 0.01

df["BB_U_Sesuai"] = df.apply(
    lambda row: compare_zscore(
        row["BB_U_Asli_Puskesmas"], row["BB_U_Pygrowup"], TOLERANCE
    ),
    axis=1,
)

df["TB_U_Sesuai"] = df.apply(
    lambda row: compare_zscore(
        row["TB_U_Asli_Puskesmas"], row["TB_U_Pygrowup"], TOLERANCE
    ),
    axis=1,
)

df["BB_TB_Sesuai"] = df.apply(
    lambda row: compare_zscore(
        row["BB_TB_Asli_Puskesmas"], row["BB_TB_Pygrowup"], TOLERANCE
    ),
    axis=1,
)


# ============================================================
# 19. TENTUKAN Z-SCORE FINAL
# ============================================================

# Hasil pygrowup digunakan sebagai nilai final apabila
# perhitungannya berhasil.
#
# Jika pygrowup gagal menghitung, nilai final dikosongkan
# agar tidak menggunakan nilai yang belum tervalidasi.

df["BB_U_Final"] = df["BB_U_Pygrowup"]

df["TB_U_Final"] = df["TB_U_Pygrowup"]

df["BB_TB_Final"] = df["BB_TB_Pygrowup"]


# ============================================================
# 20. RINGKASAN KESESUAIAN Z-SCORE
# ============================================================

print()
print("=" * 70)
print("RINGKASAN VALIDASI Z-SCORE")
print("=" * 70)

print()
print("--- BB/U ---")
print(df["BB_U_Sesuai"].value_counts(dropna=False))

print()
print("--- TB/U ---")
print(df["TB_U_Sesuai"].value_counts(dropna=False))

print()
print("--- BB/TB ---")
print(df["BB_TB_Sesuai"].value_counts(dropna=False))

for k in ["BB_U", "TB_U", "BB_TB"]:
    s = df[f"{k}_Selisih"].abs()
    print(f"{k}: rata-rata |selisih| = {s.mean():.3f}, median = {s.median():.3f}, "
          f"maks = {s.max():.3f}, |selisih| <= 0.1 : {(s <= 0.1).mean()*100:.2f}%")
    
# ============================================================
# 21. RINGKASAN Z-SCORE BERHASIL DIHITUNG
# ============================================================

print()
print("=" * 70)
print("JUMLAH Z-SCORE BERHASIL DIHITUNG")
print("=" * 70)

print(f"BB/U berhasil  : {df['BB_U_Pygrowup'].notna().sum()}")

print(f"TB/U berhasil  : {df['TB_U_Pygrowup'].notna().sum()}")

print(f"BB/TB berhasil : {df['BB_TB_Pygrowup'].notna().sum()}")


# ============================================================
# 22. RINGKASAN ERROR
# ============================================================

print()
print("=" * 70)
print("RINGKASAN ERROR PERHITUNGAN")
print("=" * 70)

error_distribution = df["Zscore_Error"].value_counts(dropna=False)

print(error_distribution)
print()
print("=" * 70)
print("5 RECORD DENGAN SELISIH TERBESAR")
print("=" * 70)

for k in ["TB_U", "BB_U"]:
    idx = df[f"{k}_Selisih"].abs().nlargest(5).index
    print(f"\n--- {k} ---")
    print(
        df.loc[idx, ["NIK", "Nama", "Tanggal Lahir", "Tanggal Pengukuran",
                     f"{k}_Asli_Puskesmas", f"{k}_Pygrowup", f"{k}_Selisih"]]
        .to_string(index=False)
    )


# ============================================================
# 23. SIMPAN DATASET HASIL VALIDASI
# ============================================================

df.to_excel(OUTPUT_FILE, index=False)


# ============================================================
# 24. OUTPUT
# ============================================================

print()
print("=" * 70)
print("OUTPUT")
print("=" * 70)

print(f"✔ File hasil validasi: {OUTPUT_FILE}")

print()
print("Validasi Z-score menggunakan pygrowup selesai.")
