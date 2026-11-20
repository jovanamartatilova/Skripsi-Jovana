# ============================================================
# KLASIFIKASI STATUS GIZI BERDASARKAN Z-SCORE FINAL
# ============================================================
#
# Input:
#   Output/data_zscore_validated.xlsx
#
# Output:
#   Output/data_status_gizi.xlsx
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

INPUT_FILE = OUTPUT_FOLDER / "data_zscore_validated.xlsx"

OUTPUT_FILE = OUTPUT_FOLDER / "data_status_gizi.xlsx"


# ============================================================
# 2. NAMA KOLOM
# ============================================================

COLUMN_NIK = "NIK"
COLUMN_MEASUREMENT_DATE = "Tanggal Pengukuran"

COLUMN_BB_U_FINAL = "BB_U_Final"
COLUMN_TB_U_FINAL = "TB_U_Final"
COLUMN_BB_TB_FINAL = "BB_TB_Final"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOAD DATA HASIL VALIDASI Z-SCORE")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"File tidak ditemukan: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

print(f"File input       : {INPUT_FILE}")
print(f"Jumlah observasi : {len(df)}")


# ============================================================
# 4. CEK KOLOM YANG DIBUTUHKAN
# ============================================================

required_columns = [
    COLUMN_NIK,
    COLUMN_MEASUREMENT_DATE,
    COLUMN_BB_U_FINAL,
    COLUMN_TB_U_FINAL,
    COLUMN_BB_TB_FINAL,
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise KeyError(
        "Kolom berikut tidak ditemukan dalam dataset:\n"
        + "\n".join(f"- {col}" for col in missing_columns)
    )


# ============================================================
# 5. KONVERSI TIPE DATA
# ============================================================

print()
print("=" * 70)
print("KONVERSI TIPE DATA")
print("=" * 70)

df[COLUMN_MEASUREMENT_DATE] = pd.to_datetime(
    df[COLUMN_MEASUREMENT_DATE], errors="coerce"
)

zscore_columns = [COLUMN_BB_U_FINAL, COLUMN_TB_U_FINAL, COLUMN_BB_TB_FINAL]

for col in zscore_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 6. FUNGSI KLASIFIKASI BB/U
# ============================================================


def classify_bb_u(zscore):
    """
    BB/U:

    Z-score < -3       : Berat badan sangat kurang
    -3 <= Z-score < -2 : Berat badan kurang
    -2 <= Z-score <= 1 : Berat badan normal
    Z-score > 1        : Risiko berat badan lebih
    """

    if pd.isna(zscore):
        return np.nan

    if zscore < -3:
        return "Berat badan sangat kurang"

    elif zscore < -2:
        return "Berat badan kurang"

    elif zscore <= 1:
        return "Berat badan normal"

    else:
        return "Risiko berat badan lebih"


# ============================================================
# 7. FUNGSI KLASIFIKASI TB/U
# ============================================================


def classify_tb_u(zscore):
    """
    TB/U:

    Z-score < -3       : Sangat pendek
    -3 <= Z-score < -2 : Pendek
    -2 <= Z-score <= 3 : Normal
    Z-score > 3        : Tinggi
    """

    if pd.isna(zscore):
        return np.nan

    if zscore < -3:
        return "Sangat pendek"

    elif zscore < -2:
        return "Pendek"

    elif zscore <= 3:
        return "Normal"

    else:
        return "Tinggi"


# ============================================================
# 8. FUNGSI KLASIFIKASI BB/TB
# ============================================================


def classify_bb_tb(zscore):
    """
    BB/TB:

    Z-score < -3       : Gizi buruk
    -3 <= Z-score < -2 : Gizi kurang
    -2 <= Z-score <= 1 : Gizi baik
    1 < Z-score <= 2   : Berisiko gizi lebih
    2 < Z-score <= 3   : Gizi lebih
    Z-score > 3        : Obesitas
    """

    if pd.isna(zscore):
        return np.nan

    if zscore < -3:
        return "Gizi buruk"

    elif zscore < -2:
        return "Gizi kurang"

    elif zscore <= 1:
        return "Gizi baik"

    elif zscore <= 2:
        return "Berisiko gizi lebih"

    elif zscore <= 3:
        return "Gizi lebih"

    else:
        return "Obesitas"


# ============================================================
# 9. KLASIFIKASI STATUS GIZI
# ============================================================

df["Status BB/U"] = df[COLUMN_BB_U_FINAL].apply(classify_bb_u)
df["Status TB/U"] = df[COLUMN_TB_U_FINAL].apply(classify_tb_u)
df["Status BB/TB"] = df[COLUMN_BB_TB_FINAL].apply(classify_bb_tb)


# ============================================================
# 10. INFORMASI PERIODE PENGUKURAN (untuk filter dashboard)
# ============================================================

nama_bulan = {
    1: "Januari",
    2: "Februari",
    3: "Maret",
    4: "April",
    5: "Mei",
    6: "Juni",
    7: "Juli",
}

urutan_bulan = list(nama_bulan.values())

df["Tahun Pengukuran"] = df[COLUMN_MEASUREMENT_DATE].dt.year
df["Bulan Pengukuran"] = df[COLUMN_MEASUREMENT_DATE].dt.month
df["Nama Bulan Pengukuran"] = pd.Categorical(
    df["Bulan Pengukuran"].map(nama_bulan), categories=urutan_bulan, ordered=True
)


# ============================================================
# 11. URUTAN KATEGORI DAN DEFINISI (untuk output terminal)
# ============================================================

INDIKATOR = {
    "BB/U": {
        "kolom": "Status BB/U",
        "urutan": [
            "Berat badan sangat kurang",
            "Berat badan kurang",
            "Berat badan normal",
            "Risiko berat badan lebih",
        ],
        "asli": "BB_U_Asli_Puskesmas",
        "fungsi": classify_bb_u,
        "ambang": [
            "z < -3          : Berat badan sangat kurang",
            "-3 <= z < -2    : Berat badan kurang",
            "-2 <= z <= +1   : Berat badan normal",
            "z > +1          : Risiko berat badan lebih",
        ],
    },
    "TB/U": {
        "kolom": "Status TB/U",
        "urutan": ["Sangat pendek", "Pendek", "Normal", "Tinggi"],
        "asli": "TB_U_Asli_Puskesmas",
        "fungsi": classify_tb_u,
        "ambang": [
            "z < -3          : Sangat pendek",
            "-3 <= z < -2    : Pendek",
            "-2 <= z <= +3   : Normal",
            "z > +3          : Tinggi",
        ],
    },
    "BB/TB": {
        "kolom": "Status BB/TB",
        "urutan": [
            "Gizi buruk",
            "Gizi kurang",
            "Gizi baik",
            "Berisiko gizi lebih",
            "Gizi lebih",
            "Obesitas",
        ],
        "asli": "BB_TB_Asli_Puskesmas",
        "fungsi": classify_bb_tb,
        "ambang": [
            "z < -3          : Gizi buruk",
            "-3 <= z < -2    : Gizi kurang",
            "-2 <= z <= +1   : Gizi baik",
            "+1 < z <= +2    : Berisiko gizi lebih",
            "+2 < z <= +3    : Gizi lebih",
            "z > +3          : Obesitas",
        ],
    },
}

MASALAH_GIZI = {
    "Underweight": (
        "Status BB/U",
        ["Berat badan kurang", "Berat badan sangat kurang"],
        "BB/U kurang + sangat kurang (z < -2)",
    ),
    "Stunting": (
        "Status TB/U",
        ["Pendek", "Sangat pendek"],
        "TB/U pendek + sangat pendek (z < -2)",
    ),
    "Wasting": (
        "Status BB/TB",
        ["Gizi kurang", "Gizi buruk"],
        "BB/TB gizi kurang + gizi buruk (z < -2)",
    ),
}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def garis(judul):
    print()
    print("=" * 70)
    print(judul)
    print("=" * 70)


# ============================================================
# 12. KONTEKS DATA
# ============================================================

jumlah_observasi = len(df)
jumlah_balita_unik = df[COLUMN_NIK].nunique(dropna=True)

garis("A. KONTEKS DATA")

print("Periode pengukuran       : Januari - Juli 2026")
print(f"Jumlah observasi         : {jumlah_observasi}")
print(f"Jumlah balita unik (NIK) : {jumlah_balita_unik}")
print()
print("Catatan: satu observasi = satu catatan pengukuran seorang")
print("balita pada satu tanggal. Satu balita dapat memiliki")
print("beberapa observasi (maksimal satu per bulan).")

duplikat_nik_bulan = df.duplicated(subset=[COLUMN_NIK, "Bulan Pengukuran"]).sum()

print(f"Balita diukur lebih dari sekali dalam bulan yang sama : {duplikat_nik_bulan}")

print()
print("Jumlah observasi (= jumlah balita diukur) per bulan:")

obs_bulanan = (
    df["Nama Bulan Pengukuran"].value_counts().reindex(urutan_bulan, fill_value=0)
)

for bulan, jumlah in obs_bulanan.items():
    print(f"  {bulan:<10}: {jumlah}")

print()
print("Distribusi jumlah pengukuran per balita:")

dist_pengukuran = df.groupby(COLUMN_NIK).size().value_counts().sort_index()

for jumlah_ukur, jumlah_balita in dist_pengukuran.items():
    print(f"  {jumlah_ukur} kali diukur : {jumlah_balita} balita")


# ============================================================
# 13. ATURAN KLASIFIKASI
# ============================================================

garis("B. ATURAN KLASIFIKASI (berdasarkan z-score final)")

for nama, info in INDIKATOR.items():
    print()
    print(f"{nama}:")
    for baris in info["ambang"]:
        print(f"  {baris}")


# ============================================================
# 14. DISTRIBUSI STATUS GIZI KESELURUHAN
# ============================================================

garis(
    "C. DISTRIBUSI STATUS GIZI KESELURUHAN\n"
    f"   (denominator = {jumlah_observasi} observasi, Jan-Jul 2026)"
)

for nama, info in INDIKATOR.items():
    jumlah = df[info["kolom"]].value_counts().reindex(info["urutan"], fill_value=0)

    persen = (jumlah / jumlah_observasi * 100).round(2)

    tabel = pd.DataFrame(
        {
            "Jumlah observasi": jumlah,
            "Persentase (%)": persen,
        }
    )

    tabel.loc["TOTAL"] = [jumlah.sum(), round(jumlah.sum() / jumlah_observasi * 100, 2)]
    tabel["Jumlah observasi"] = tabel["Jumlah observasi"].astype(int)

    print()
    print(f"--- {nama} ---")
    print(tabel.to_string())


# ============================================================
# 15. DISTRIBUSI STATUS GIZI PER BULAN
# ============================================================

garis(
    "D. DISTRIBUSI STATUS GIZI PER BULAN\n"
    "   (persentase = % dari observasi pada bulan tersebut)"
)

for nama, info in INDIKATOR.items():
    jumlah = pd.crosstab(df["Nama Bulan Pengukuran"].astype(str), df[info["kolom"]])

    jumlah = jumlah.reindex(index=urutan_bulan, columns=info["urutan"], fill_value=0)

    jumlah["Total"] = jumlah.sum(axis=1)
    jumlah.loc["Seluruh periode"] = jumlah.sum(axis=0)
    jumlah = jumlah.astype(int)

    persen = (jumlah[info["urutan"]].div(jumlah["Total"], axis=0) * 100).round(2)

    print()
    print(f"--- {nama}: JUMLAH observasi ---")
    print(jumlah.to_string())

    print()
    print(f"--- {nama}: PERSENTASE (%) ---")
    print(persen.to_string())


# ============================================================
# 16. RINGKASAN MASALAH GIZI (UNDERWEIGHT, STUNTING, WASTING)
# ============================================================

garis("E. RINGKASAN MASALAH GIZI")

print("Definisi (sesuai ambang batas WHO, z-score < -2 SD):")

for nama, (_, _, definisi) in MASALAH_GIZI.items():
    print(f"  {nama:<12}: {definisi}")

print()
print("Denominator persentase = jumlah observasi pada bulan tersebut.")

ringkasan = pd.DataFrame(index=urutan_bulan + ["Seluruh periode"])

total_bulanan = obs_bulanan.copy()
total_bulanan["Seluruh periode"] = jumlah_observasi

ringkasan["Observasi"] = total_bulanan

for nama, (kolom, kategori, _) in MASALAH_GIZI.items():
    flag = df[kolom].isin(kategori)

    jumlah_bulanan = (
        flag.groupby(df["Nama Bulan Pengukuran"], observed=False)
        .sum()
        .reindex(urutan_bulan, fill_value=0)
    )

    jumlah_bulanan["Seluruh periode"] = flag.sum()

    ringkasan[nama] = jumlah_bulanan.astype(int)
    ringkasan[f"{nama} (%)"] = (jumlah_bulanan / ringkasan["Observasi"] * 100).round(2)

print()
print(ringkasan.to_string())

print()
print("Rincian tingkat keparahan (seluruh periode):")

for nama, (kolom, kategori, _) in MASALAH_GIZI.items():
    for kat in kategori:
        n = (df[kolom] == kat).sum()
        print(f"  {nama:<12}- {kat:<27}: {n} ({n / jumlah_observasi * 100:.2f}%)")


# ============================================================
# 17. DAMPAK VALIDASI Z-SCORE TERHADAP KLASIFIKASI
# ============================================================

garis(
    "F. DAMPAK VALIDASI Z-SCORE TERHADAP KLASIFIKASI\n"
    "   (z-score asli Puskesmas vs z-score final pygrowup,\n"
    "    diklasifikasikan dengan ambang batas yang sama)"
)

ringkasan_perubahan = []
detail_perubahan = {}

for nama, info in INDIKATOR.items():
    status_asli = df[info["asli"]].apply(info["fungsi"])
    status_final = df[info["kolom"]]

    berubah = status_asli.notna() & status_final.notna() & (status_asli != status_final)

    ringkasan_perubahan.append(
        {
            "Indikator": nama,
            "Observasi berubah kategori": int(berubah.sum()),
            "Persentase (%)": round(berubah.sum() / jumlah_observasi * 100, 2),
        }
    )

    detail_perubahan[nama] = (
        pd.DataFrame(
            {
                "Kategori z-score Puskesmas": status_asli[berubah],
                "Kategori z-score final": status_final[berubah],
            }
        )
        .value_counts()
        .rename("Jumlah")
        .reset_index()
    )

print()
print(pd.DataFrame(ringkasan_perubahan).to_string(index=False))

for nama, detail in detail_perubahan.items():
    print()
    print(f"--- Arah perubahan {nama} ---")
    if len(detail) == 0:
        print("Tidak ada perubahan kategori.")
    else:
        print(detail.to_string(index=False))

print()
print("Perubahan stunting (TB/U pendek + sangat pendek):")

stunting_asli = (
    df[INDIKATOR["TB/U"]["asli"]]
    .apply(classify_tb_u)
    .isin(["Pendek", "Sangat pendek"])
    .sum()
)

stunting_final = df["Status TB/U"].isin(["Pendek", "Sangat pendek"]).sum()

print(f"  Berdasarkan z-score Puskesmas : {stunting_asli} observasi")
print(f"  Berdasarkan z-score final     : {stunting_final} observasi")
# ============================================================
# 17B. VISUALISASI STATUS GIZI PER BULAN
# ============================================================

import plotly.express as px

VISUAL_DIR = OUTPUT_FOLDER / "Visualisasi_Bab4"
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

BULAN_SHORT = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Mei",
    "Jun",
    "Jul",
    "Agu",
    "Sep",
    "Okt",
    "Nov",
    "Des",
]

WARNA_STATUS = {
    "Berat badan normal": "#183b91",
    "Berat badan kurang": "#e0553b",
    "Berat badan sangat kurang": "#d9a300",
    "Risiko berat badan lebih": "#2fae7a",
    "Normal": "#183b91",
    "Pendek": "#e0553b",
    "Sangat pendek": "#d9a300",
    "Tinggi": "#2fae7a",
    "Gizi baik": "#183b91",
    "Gizi kurang": "#e0553b",
    "Gizi buruk": "#d9a300",
    "Berisiko gizi lebih": "#2fae7a",
    "Gizi lebih": "#7c5ac7",
    "Obesitas": "#a65e2e",
}

# Gunakan tanggal pengukuran sebagai sumber bulan
df["_tanggal_visual"] = pd.to_datetime(df[COLUMN_MEASUREMENT_DATE], errors="coerce")

for nama, info in INDIKATOR.items():
    kolom_status = info["kolom"]
    kategori = info["urutan"]

    trend_df = df.dropna(subset=["_tanggal_visual", kolom_status]).copy()

    trend_df["_bulan"] = trend_df["_tanggal_visual"].dt.to_period("M").astype(str)

    trend_df["_status"] = trend_df[kolom_status]

    # Jumlah observasi per bulan dan kategori
    trend = (
        trend_df.groupby(["_bulan", "_status"], observed=True)
        .size()
        .reset_index(name="Jumlah")
    )

    # Total observasi pada setiap bulan
    total_bulan = trend_df.groupby("_bulan").size().reset_index(name="Total")

    trend = trend.merge(total_bulan, on="_bulan", how="left")

    trend["Persentase"] = trend["Jumlah"] / trend["Total"] * 100

    # Label bulan dan urutan kronologis
    trend["_tanggal_bulan"] = pd.to_datetime(
        trend["_bulan"] + "-01", format="%Y-%m-%d", errors="coerce"
    )

    trend["Bulan"] = (
        trend["_tanggal_bulan"].dt.month.map(lambda m: BULAN_SHORT[m - 1])
        + " "
        + trend["_tanggal_bulan"].dt.year.astype(str)
    )

    bulan_order = (
        trend[["_tanggal_bulan", "Bulan"]]
        .drop_duplicates()
        .sort_values("_tanggal_bulan")["Bulan"]
        .tolist()
    )

    trend["_status"] = pd.Categorical(
        trend["_status"], categories=kategori, ordered=True
    )

    trend = trend.sort_values(["_tanggal_bulan", "_status"])

    # Grafik batang berkelompok
    fig = px.bar(
        trend,
        x="Bulan",
        y="Persentase",
        color="_status",
        barmode="group",
        category_orders={
            "Bulan": bulan_order,
            "_status": kategori,
        },
        color_discrete_map=WARNA_STATUS,
        custom_data=["Jumlah", "Total"],
    )

    # Label angka di atas batang dan tooltip
    fig.update_traces(
        texttemplate="%{y:.1f}%",
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=24, color="#202020", family="Arial"),
        marker_line_width=0,
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Bulan: %{x}<br>"
            "Persentase: %{y:.1f}%<br>"
            "Jumlah observasi: %{customdata[0]}<br>"
            "Total observasi bulan: %{customdata[1]}"
            "<extra></extra>"
        ),
    )

    # Pengaturan ukuran dan tulisan
    fig.update_layout(
        title=dict(
            text=f"Perkembangan Status Gizi per Bulan ({nama})",
            x=0.5,
            xanchor="center",
            font=dict(size=34, color="#202020", family="Arial"),
        ),
        width=2400,
        height=1500,
        autosize=False,
        template="plotly_white",
        font=dict(
            family="Arial",
            size=24,
            color="#202020",
        ),
        xaxis=dict(
            title=None,
            categoryorder="array",
            categoryarray=bulan_order,
            showgrid=False,
            tickfont=dict(size=25),
            automargin=True,
        ),
        yaxis=dict(
            title=dict(
                text="Persentase observasi (%)",
                font=dict(size=28),
            ),
            rangemode="tozero",
            ticksuffix="%",
            gridcolor="#e5e7eb",
            zeroline=False,
            tickfont=dict(size=22),
            automargin=True,
        ),
        legend=dict(
            title=None,
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5,
            font=dict(size=23, color="#202020"),
            itemsizing="constant",
        ),
        bargap=0.20,
        bargroupgap=0.06,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=150, r=100, t=240, b=150),
    )

    # Simpan satu kali dengan resolusi tinggi
    nama_file = nama.replace("/", "_").replace(" ", "_")

    output_grafik = VISUAL_DIR / f"grafik_status_gizi_{nama_file}.png"

    fig.write_image(str(output_grafik), width=2400, height=1500, scale=3)

    print(f"Grafik tersimpan: {output_grafik}")

# Hapus kolom bantu dari dataframe
df = df.drop(columns=["_tanggal_visual"], errors="ignore")

print(f"Seluruh grafik tersimpan di: {VISUAL_DIR}")

# ============================================================
# 18. VALIDASI HASIL KLASIFIKASI
# ============================================================

garis("G. VALIDASI HASIL KLASIFIKASI")

for nama, info in INDIKATOR.items():
    print(
        f"{nama:<6} terklasifikasi : "
        f"{df[info['kolom']].notna().sum()} dari {jumlah_observasi}"
    )

print()
print("Seluruh observasi pengukuran dipertahankan.")
print("Tidak ada observasi yang dihapus atau dipilih ulang.")


# ============================================================
# 19. SIMPAN DATASET STATUS GIZI
# ============================================================

df.to_excel(OUTPUT_FILE, index=False)

garis("OUTPUT")
print(f"✔ Dataset status gizi: {OUTPUT_FILE}")
print()
print("Klasifikasi status gizi selesai.")
