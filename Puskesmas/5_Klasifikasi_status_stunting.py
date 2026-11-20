# ============================================================
# 05_IDENTIFIKASI_STUNTING.PY
# ============================================================
#
# IDENTIFIKASI STATUS STUNTING BERDASARKAN PENGUKURAN TERAKHIR
#
# INPUT:
#   Output/data_status_gizi.xlsx
#
# OUTPUT:
#   Output/data_stunting.xlsx
#
# CATATAN:
#   - Dataset status gizi tetap menggunakan seluruh pengukuran.
#   - Dataset stunting hanya menggunakan satu pengukuran terakhir
#     untuk setiap NIK.
#   - Stunting ditentukan berdasarkan TB/U < -2 SD.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 1. FOLDER INPUT DAN OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = BASE_DIR / "Output"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_FILE = OUTPUT_FOLDER / "data_status_gizi.xlsx"

OUTPUT_FILE = OUTPUT_FOLDER / "data_stunting.xlsx"


# ============================================================
# 2. NAMA KOLOM
# ============================================================

COLUMN_NIK = "NIK"
COLUMN_MEASUREMENT_DATE = "Tanggal Pengukuran"

# Kolom Z-score final dari tahap validasi.
COLUMN_TB_U_FINAL = "TB_U_Final"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOAD DATA STATUS GIZI FINAL")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

print(f"File input       : {INPUT_FILE}")
print(f"Jumlah observasi : {len(df)}")


# ============================================================
# 4. CEK KOLOM YANG DIBUTUHKAN
# ============================================================

required_columns = [COLUMN_NIK, COLUMN_MEASUREMENT_DATE, COLUMN_TB_U_FINAL]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise KeyError(
        "Kolom berikut tidak ditemukan dalam dataset:\n"
        + "\n".join(f"- {col}" for col in missing_columns)
    )

print("✔ Seluruh kolom yang dibutuhkan tersedia.")


# ============================================================
# 5. KONVERSI FORMAT DATA
# ============================================================

print()
print("=" * 70)
print("KONVERSI FORMAT DATA")
print("=" * 70)

df[COLUMN_MEASUREMENT_DATE] = pd.to_datetime(
    df[COLUMN_MEASUREMENT_DATE], errors="coerce"
)

df[COLUMN_TB_U_FINAL] = pd.to_numeric(df[COLUMN_TB_U_FINAL], errors="coerce")


# ============================================================
# 6. VALIDASI TANGGAL PENGUKURAN
# ============================================================

print()
print("=" * 70)
print("VALIDASI TANGGAL PENGUKURAN")
print("=" * 70)

jumlah_tanggal_invalid = df[COLUMN_MEASUREMENT_DATE].isna().sum()

print(f"Tanggal valid   : {len(df) - jumlah_tanggal_invalid}")

print(f"Tanggal invalid : {jumlah_tanggal_invalid}")

if jumlah_tanggal_invalid > 0:
    raise ValueError(
        "Terdapat tanggal pengukuran yang tidak valid. "
        "Perbaiki data sebelum seleksi pengukuran terakhir."
    )


# ============================================================
# 7. VALIDASI NIK
# ============================================================

print()
print("=" * 70)
print("VALIDASI NIK")
print("=" * 70)

jumlah_nik_kosong = df[COLUMN_NIK].isna().sum()

print(f"NIK kosong : {jumlah_nik_kosong}")

if jumlah_nik_kosong > 0:
    print(
        "PERINGATAN: data dengan NIK kosong tidak "
        "digunakan dalam seleksi pengukuran terakhir."
    )


# ============================================================
# 8. SELEKSI DATA DENGAN NIK
# ============================================================
# Data tanpa NIK tidak dapat digunakan untuk menentukan
# pengukuran terakhir dari balita tertentu.

df_with_nik = df[df[COLUMN_NIK].notna()].copy()

print()
print("=" * 70)
print("DATA DENGAN NIK VALID")
print("=" * 70)

print(f"Jumlah observasi dengan NIK : {len(df_with_nik)}")

print(f"Jumlah NIK unik             : {df_with_nik[COLUMN_NIK].nunique()}")


# ============================================================
# 9. URUTKAN DATA
# ============================================================

df_with_nik["_tbu_valid"] = df_with_nik[COLUMN_TB_U_FINAL].notna()

df_with_nik = df_with_nik.sort_values(
    by=[COLUMN_NIK, "_tbu_valid", COLUMN_MEASUREMENT_DATE], ascending=[True, True, True]
)


# ============================================================
# 10. AMBIL PENGUKURAN TERAKHIR SETIAP BALITA
# ============================================================
#
# Satu NIK dapat memiliki beberapa pengukuran selama Januari
# sampai Juli. Untuk dataset stunting, hanya pengukuran dengan
# tanggal paling akhir yang digunakan.
#
# Seluruh pengukuran sebelumnya tetap tersedia di:
#   Output/data_status_gizi.xlsx
# ============================================================

print()
print("=" * 70)
print("SELEKSI PENGUKURAN TERAKHIR")
print("=" * 70)

jumlah_observasi_awal = len(df_with_nik)
jumlah_nik_awal = df_with_nik[COLUMN_NIK].nunique()

df_latest = (
    df_with_nik.drop_duplicates(subset=[COLUMN_NIK], keep="last")
    .drop(columns=["_tbu_valid"])
    .reset_index(drop=True)
)

jumlah_observasi_akhir = len(df_latest)
jumlah_nik_akhir = df_latest[COLUMN_NIK].nunique()

print(f"Observasi sebelum seleksi : {jumlah_observasi_awal}")

print(f"NIK unik sebelum seleksi  : {jumlah_nik_awal}")

print(f"Observasi setelah seleksi : {jumlah_observasi_akhir}")

print(f"NIK unik setelah seleksi  : {jumlah_nik_akhir}")


# ============================================================
# 11. VALIDASI SATU OBSERVASI PER NIK
# ============================================================

jumlah_nik_duplikat = df_latest[COLUMN_NIK].duplicated().sum()

print(f"NIK duplikat setelah seleksi : {jumlah_nik_duplikat}")

if jumlah_nik_duplikat != 0:
    raise ValueError("Masih terdapat NIK duplikat setelah seleksi.")

if jumlah_observasi_akhir != jumlah_nik_akhir:
    raise ValueError("Jumlah observasi akhir tidak sama dengan jumlah NIK unik.")

print("✔ Setiap NIK memiliki tepat satu pengukuran terakhir.")


# ============================================================
# 12. KLASIFIKASI KATEGORI TB/U
# ============================================================
#
# Berdasarkan Tabel 2.1:
#
# Z-score < -3       : Sangat pendek
# -3 <= Z-score < -2 : Pendek
# -2 <= Z-score <= 3 : Normal
# Z-score > 3        : Tinggi
# ============================================================

print()
print("=" * 70)
print("KLASIFIKASI KATEGORI TB/U")
print("=" * 70)


def classify_tb_u(zscore):

    if pd.isna(zscore):
        return pd.NA

    if zscore < -3:
        return "Sangat pendek"

    elif zscore < -2:
        return "Pendek"

    elif zscore <= 3:
        return "Normal"

    else:
        return "Tinggi"


df_latest["Status TB/U"] = df_latest[COLUMN_TB_U_FINAL].apply(classify_tb_u)


# ============================================================
# 13. PEMBENTUKAN STATUS STUNTING
# ============================================================
#
# TB/U < -2 SD  : Stunting
# TB/U >= -2 SD : Tidak Stunting
#
# Z-score kosong:
#   Tidak diketahui
# ============================================================

print()
print("=" * 70)
print("PEMBENTUKAN STATUS STUNTING")
print("=" * 70)


def classify_stunting(zscore):

    if pd.isna(zscore):
        return pd.NA

    if zscore < -2:
        return "Stunting"

    else:
        return "Tidak Stunting"


df_latest["Status Stunting"] = df_latest[COLUMN_TB_U_FINAL].apply(classify_stunting)


# ============================================================
# 14. PEMBENTUKAN VARIABEL BINER STUNTING
# ============================================================
#
# 1 = Stunting
# 0 = Tidak Stunting
# NA = Tidak diketahui
# ============================================================


def create_stunting_binary(status):

    if pd.isna(status):
        return pd.NA

    if status == "Stunting":
        return 1

    if status == "Tidak Stunting":
        return 0

    return pd.NA


df_latest["Stunting"] = (
    df_latest["Status Stunting"].apply(create_stunting_binary).astype("Int64")
)


# ============================================================
# 15. INFORMASI PERIODE PENGUKURAN TERAKHIR
# ============================================================

df_latest["Tahun Pengukuran Terakhir"] = df_latest[COLUMN_MEASUREMENT_DATE].dt.year

df_latest["Bulan Pengukuran Terakhir"] = df_latest[COLUMN_MEASUREMENT_DATE].dt.month

nama_bulan = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
}

df_latest["Nama Bulan Pengukuran Terakhir"] = df_latest[
    "Bulan Pengukuran Terakhir"
].map(nama_bulan)


# ============================================================
# 16. RINGKASAN STATUS TB/U
# ============================================================

print()
print("=" * 70)
print("DISTRIBUSI STATUS TB/U")
print("=" * 70)

print(df_latest["Status TB/U"].value_counts(dropna=False))


# ============================================================
# 17. RINGKASAN STATUS STUNTING
# ============================================================

print()
print("=" * 70)
print("DISTRIBUSI STATUS STUNTING")
print("=" * 70)

print(df_latest["Status Stunting"].value_counts(dropna=False))


# ============================================================
# 18. RINGKASAN JUMLAH STUNTING
# ============================================================

jumlah_balita = len(df_latest)

jumlah_stunting = df_latest["Stunting"].eq(1).sum()

jumlah_tidak_stunting = df_latest["Stunting"].eq(0).sum()

jumlah_tidak_diketahui = df_latest["Stunting"].isna().sum()

jumlah_status_diketahui = jumlah_stunting + jumlah_tidak_stunting

if jumlah_status_diketahui > 0:
    prevalensi_stunting = jumlah_stunting / jumlah_status_diketahui * 100

else:
    prevalensi_stunting = np.nan


print()
print("=" * 70)
print("RINGKASAN STUNTING")
print("=" * 70)

print(f"Jumlah balita              : {jumlah_balita}")

print(f"Stunting                   : {jumlah_stunting}")

print(f"Tidak stunting             : {jumlah_tidak_stunting}")

print(f"Tidak diketahui            : {jumlah_tidak_diketahui}")

print(
    f"Prevalensi stunting        : {prevalensi_stunting:.2f}%"
    if not pd.isna(prevalensi_stunting)
    else "Prevalensi tidak dapat dihitung."
)


# ============================================================
# 19. KONTEKS DAN PERBANDINGAN DENGAN SELURUH OBSERVASI
# ============================================================

print()
print("=" * 70)
print("KONTEKS HASIL STUNTING")
print("=" * 70)

print("Status stunting = status pada pengukuran TERAKHIR setiap balita")
print("(tidak selalu bulan Juli). Denominator = jumlah balita unik.")

stunting_semua_observasi = df["Status TB/U"].isin(["Pendek", "Sangat pendek"]).sum()

print()
print("Perbandingan denominator:")
print(
    f"  Seluruh observasi Jan-Jul : "
    f"{stunting_semua_observasi} dari {len(df)} observasi "
    f"({stunting_semua_observasi / len(df) * 100:.2f}%)"
)
print(
    f"  Pengukuran terakhir       : "
    f"{jumlah_stunting} dari {jumlah_status_diketahui} balita "
    f"({prevalensi_stunting:.2f}%)"
)

print()
print("Tingkat keparahan (pengukuran terakhir):")

for kategori in ["Pendek", "Sangat pendek"]:
    n = (df_latest["Status TB/U"] == kategori).sum()
    print(
        f"  {kategori:<14}: {n} balita "
        f"({n / jumlah_balita * 100:.2f}% dari seluruh balita, "
        f"{n / jumlah_stunting * 100:.2f}% dari balita stunting)"
    )


# ============================================================
# 20. FUNGSI TABEL PREVALENSI PER KELOMPOK
# ============================================================


def tabel_prevalensi(kolom, urutan):

    tabel = pd.crosstab(
        df_latest[kolom].astype(str), df_latest["Status Stunting"]
    ).reindex(urutan, fill_value=0)

    for kolom_status in ["Stunting", "Tidak Stunting"]:
        if kolom_status not in tabel.columns:
            tabel[kolom_status] = 0

    tabel = tabel[["Stunting", "Tidak Stunting"]]
    tabel["Total balita"] = tabel.sum(axis=1)
    tabel.loc["Seluruh balita"] = tabel.sum(axis=0)

    tabel["Prevalensi stunting (%)"] = (
        tabel["Stunting"] / tabel["Total balita"] * 100
    ).round(2)

    return tabel


# ============================================================
# 21. STUNTING MENURUT KARAKTERISTIK BALITA
# ============================================================

if "Jenis Kelamin" in df_latest.columns:
    print()
    print("=" * 70)
    print("STUNTING MENURUT JENIS KELAMIN (L = laki-laki, P = perempuan)")
    print("=" * 70)

    print(tabel_prevalensi("Jenis Kelamin", ["L", "P"]).to_string())


if "Kelompok Umur" in df_latest.columns:
    print()
    print("=" * 70)
    print("STUNTING MENURUT KELOMPOK UMUR")
    print("(umur saat pengukuran terakhir)")
    print("=" * 70)

    urutan_umur = [
        "0-5 bulan",
        "6-11 bulan",
        "12-23 bulan",
        "24-35 bulan",
        "36-47 bulan",
        "48-59 bulan",
    ]

    print(tabel_prevalensi("Kelompok Umur", urutan_umur).to_string())


print()
print("=" * 70)
print("STUNTING MENURUT BULAN PENGUKURAN TERAKHIR")
print("=" * 70)

urutan_bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli"]

print(tabel_prevalensi("Nama Bulan Pengukuran Terakhir", urutan_bulan).to_string())

# ============================================================
# 21B. VISUALISASI STATUS STUNTING BERDASARKAN JENIS KELAMIN
# ============================================================

import plotly.express as px

# Folder penyimpanan grafik
VISUAL_DIR = OUTPUT_FOLDER / "Visualisasi_Bab4"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

# Salin data yang memiliki jenis kelamin dan status stunting
plot_df = df_latest.dropna(subset=["Jenis Kelamin", "Status Stunting"]).copy()

# Hitung jumlah balita berdasarkan jenis kelamin dan status
plot_df["Jenis Kelamin"] = pd.Categorical(
    plot_df["Jenis Kelamin"], categories=["L", "P"], ordered=True
)

plot_data = (
    plot_df.groupby(["Jenis Kelamin", "Status Stunting"], observed=False)
    .size()
    .reset_index(name="Jumlah")
)

# Ubah kode jenis kelamin menjadi label yang mudah dipahami
plot_data["Jenis Kelamin"] = plot_data["Jenis Kelamin"].map(
    {"L": "Laki-laki", "P": "Perempuan"}
)

# Urutan status dan warna
status_order = ["Tidak Stunting", "Stunting"]

warna_status = {"Tidak Stunting": "#183B91", "Stunting": "#E0553B"}

# Buat grafik batang berkelompok
fig = px.bar(
    plot_data,
    x="Jenis Kelamin",
    y="Jumlah",
    color="Status Stunting",
    barmode="group",
    category_orders={
        "Jenis Kelamin": ["Laki-laki", "Perempuan"],
        "Status Stunting": status_order,
    },
    color_discrete_map=warna_status,
    text="Jumlah",
    labels={
        "Jenis Kelamin": "Jenis kelamin",
        "Jumlah": "Jumlah balita",
        "Status Stunting": "Status stunting",
    },
)

# ============================================================
# TAMPILAN GRAFIK SUPER BESAR UNTUK WORD
# ============================================================

fig.update_traces(
    texttemplate="%{y}",
    textposition="outside",
    cliponaxis=False,
    textfont=dict(size=34, color="#111111", family="Arial Black"),
    marker_line_width=0,
    hovertemplate=(
        "<b>%{fullData.name}</b><br>"
        "Jenis kelamin: %{x}<br>"
        "Jumlah balita: %{y}"
        "<extra></extra>"
    ),
)

fig.update_layout(
    title=dict(
        text="Status Stunting Berdasarkan Jenis Kelamin",
        x=0.5,
        xanchor="center",
        font=dict(family="Arial", size=42, color="#111111"),
    ),
    width=1800,
    height=1200,
    autosize=False,
    template="plotly_white",
    font=dict(family="Arial", size=32, color="#111111"),
    xaxis=dict(title=None, tickfont=dict(size=36), showgrid=False, automargin=True),
    yaxis=dict(
        title=dict(text="Jumlah balita", font=dict(size=34)),
        tickfont=dict(size=30),
        rangemode="tozero",
        gridcolor="#D5D5D5",
        zeroline=False,
        automargin=True,
    ),
    legend=dict(
        title=None,
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5,
        font=dict(size=30),
        itemsizing="constant",
    ),
    bargap=0.40,
    bargroupgap=0.15,
    margin=dict(l=150, r=100, t=240, b=150),
)

# Simpan sebagai PNG resolusi tinggi
nama_file = VISUAL_DIR / "grafik_stunting_berdasarkan_jenis_kelamin.png"

fig.write_image(str(nama_file), width=1800, height=1200, scale=3)

print(f"Grafik tersimpan: {nama_file}")

# Atur tampilan agar terbaca di Word
fig.update_layout(
    title=dict(
        text="Distribusi Status Stunting Berdasarkan Jenis Kelamin",
        x=0,
        xanchor="left",
        font=dict(size=24, color="#202020"),
    ),
    width=1600,
    height=900,
    template="plotly_white",
    font=dict(family="Arial", size=17, color="#202020"),
    xaxis=dict(title=None, tickfont=dict(size=18), showgrid=False),
    yaxis=dict(
        title="Jumlah balita",
        title_font=dict(size=18),
        tickfont=dict(size=15),
        rangemode="tozero",
        gridcolor="#E5E7EB",
        zeroline=False,
    ),
    legend=dict(
        title=None,
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=16),
    ),
    bargap=0.30,
    bargroupgap=0.08,
    margin=dict(l=90, r=40, t=130, b=90),
)

# ============================================================
# 21C. VISUALISASI DISTRIBUSI STATUS STUNTING KESELURUHAN
# BERDASARKAN PENGUKURAN TERAKHIR
# ============================================================

import plotly.express as px

# Ambil data dengan status stunting yang diketahui
plot_df = df_latest.dropna(subset=["Status TB/U", "Status Stunting"]).copy()

# Urutan kategori TB/U
kategori_tbu = ["Sangat pendek", "Pendek", "Normal", "Tinggi"]

# Urutan status stunting
kategori_stunting = ["Stunting", "Tidak Stunting"]

# Hitung jumlah balita berdasarkan kategori TB/U
plot_data = (
    plot_df.groupby(["Status TB/U", "Status Stunting"], observed=False)
    .size()
    .reset_index(name="Jumlah")
)

# Pastikan seluruh kategori ditampilkan, termasuk kategori bernilai nol
semua_kombinasi = pd.MultiIndex.from_product(
    [kategori_tbu, kategori_stunting], names=["Status TB/U", "Status Stunting"]
).to_frame(index=False)

plot_data = semua_kombinasi.merge(
    plot_data, on=["Status TB/U", "Status Stunting"], how="left"
)

plot_data["Jumlah"] = plot_data["Jumlah"].fillna(0).astype(int)

# Atur urutan kategori
plot_data["Status TB/U"] = pd.Categorical(
    plot_data["Status TB/U"], categories=kategori_tbu, ordered=True
)

plot_data["Status Stunting"] = pd.Categorical(
    plot_data["Status Stunting"], categories=kategori_stunting, ordered=True
)

plot_data = plot_data.sort_values(["Status TB/U", "Status Stunting"])

# Warna status
warna_status = {"Stunting": "#E0553B", "Tidak Stunting": "#183B91"}

# Buat grafik batang berkelompok
fig = px.bar(
    plot_data,
    x="Status TB/U",
    y="Jumlah",
    color="Status Stunting",
    barmode="group",
    category_orders={"Status TB/U": kategori_tbu, "Status Stunting": kategori_stunting},
    color_discrete_map=warna_status,
    text="Jumlah",
    labels={
        "Status TB/U": "Kategori TB/U",
        "Jumlah": "Jumlah balita",
        "Status Stunting": "Status stunting",
    },
)

# Label angka dan tooltip
fig.update_traces(
    texttemplate="%{y}",
    textposition="outside",
    cliponaxis=False,
    textfont=dict(size=30, color="#111111", family="Arial Black"),
    hovertemplate=(
        "<b>%{x}</b><br>Status: %{fullData.name}<br>Jumlah balita: %{y}<extra></extra>"
    ),
)

# Tampilan besar agar terbaca di Word
fig.update_layout(
    title=dict(
        text="Distribusi Status Stunting Berdasarkan Pengukuran Terakhir",
        x=0.5,
        xanchor="center",
        font=dict(family="Arial", size=36, color="#202020"),
    ),
    width=1800,
    height=1200,
    autosize=False,
    template="plotly_white",
    font=dict(family="Arial", size=25, color="#202020"),
    xaxis=dict(title=None, tickfont=dict(size=27), showgrid=False, automargin=True),
    yaxis=dict(
        title=dict(text="Jumlah balita", font=dict(size=29)),
        tickfont=dict(size=24),
        rangemode="tozero",
        gridcolor="#D5D5D5",
        zeroline=False,
        automargin=True,
    ),
    legend=dict(
        title=None,
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
        font=dict(size=25),
    ),
    bargap=0.30,
    bargroupgap=0.10,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=130, r=90, t=220, b=130),
)

# Simpan grafik
nama_file = VISUAL_DIR / "grafik_distribusi_stunting_pengukuran_terakhir.png"

fig.write_image(str(nama_file), width=1800, height=1200, scale=3)

print(f"Grafik distribusi status stunting tersimpan: {nama_file}")
# ============================================================
# 22. PREVIEW DATA FINAL
# ============================================================

print()
print("=" * 70)
print("PREVIEW DATA STUNTING")
print("=" * 70)

preview_columns = [
    "NIK",
    "Jenis Kelamin",
    "Tanggal Pengukuran",
    COLUMN_TB_U_FINAL,
    "Status TB/U",
    "Status Stunting",
    "Stunting",
]

available_preview_columns = [col for col in preview_columns if col in df_latest.columns]

print(df_latest[available_preview_columns].head(10))


# ============================================================
# 23. SIMPAN DATASET STUNTING
# ============================================================

df_latest.to_excel(OUTPUT_FILE, index=False)


# ============================================================
# 24. VALIDASI OUTPUT
# ============================================================

print()
print("=" * 70)
print("VALIDASI OUTPUT")
print("=" * 70)

print(f"Jumlah baris output : {len(df_latest)}")

print(f"Jumlah NIK unik     : {df_latest[COLUMN_NIK].nunique()}")

print(f"NIK duplikat        : {df_latest[COLUMN_NIK].duplicated().sum()}")

print(f"File output         : {OUTPUT_FILE}")


# ============================================================
# 25. SELESAI
# ============================================================

print()
print("=" * 70)
print("SELESAI")
print("=" * 70)

print("Dataset stunting berisi satu pengukuran terakhir untuk setiap balita.")

print(
    "Seluruh riwayat pengukuran Januari-Juli tetap "
    "dipertahankan dalam data_status_gizi.xlsx."
)
