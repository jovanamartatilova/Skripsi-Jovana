"""
0_BuildDataset_IFLS5.py  -  Membentuk dataset analisis balita (1 baris = 1 balita) dari IFLS5
==========================================================================================
Input : file .dta IFLS5 (folder FOLDER_DTA, dicari rekursif)
Output (folder FOLDER_OUT):
    balita_ifls5.csv        dataset akhir (target + fitur)
    Ringkasan_Dataset.xlsx  Daftar_Variabel (sumber & aturan tiap kolom), Ringkasan (missing, statistik), Funnel
    laporan_build.txt       laporan proses + pemeriksaan (angka untuk ditulis di Bab 3/4)

REVISI:
  1. Imunisasi kartu: vaksin dihitung diterima hanya jika TAHUN pada kartu valid (antara tahun lahir dan tahun
     wawancara). Versi lama memakai .notna(), padahal kolom rja28* terisi untuk SEMUA pemegang kartu.
  2. Jumlah tablet besi = 0 jika tidak dapat tablet; ANC = 0 jika tidak pernah periksa hamil.
  3. air_dini & mpasi_dini hanya dinilai pada anak >= 6 bulan; "belum diberi" pada anak >= 6 bulan = 0.
     Usia pemberian yang melebihi umur anak (+1 bulan) dijadikan NaN.
  4. Fitur baru: asi_eksklusif_proxy, air_direbus, jumlah_anggota_RT, ada_perokok_RT,
     log_pengeluaran_perkapita, log_pengeluaran_pangan_perkapita.
  5. Fitur pola makan dan keluhan sakit anak (Buku 5 FMA dan MAA).
  6. HAZ dan WAZ dihitung dari tabel WHO per hari (folder FOLDER_WHO), dengan koreksi cara ukur 0,7 cm.

Library: pandas, numpy, openpyxl. Tabel WHO: lhfa- dan wfa-...-zscore-expanded-tables.xlsx
"""

import os
import sys

import numpy as np
import pandas as pd

# ============================= PENGATURAN =============================
FOLDER_DTA = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\Dataset\IFLS5"
FOLDER_OUT = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output\BuildDataset"
FOLDER_WHO = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\Dataset\WHO"

UMUR_MIN_BLN = 0  # batas bawah umur (bulan, inklusif)
UMUR_MAX_BLN = 59  # batas atas umur (bulan penuh, inklusif -> umur < 60 bulan)
PAKAI_TB_BB_ANAK = (
    True  # True: tb_anak & bb_anak ikut sebagai kolom fitur; False: dibuang dari CSV
)
KOREKSI_CARA_UKUR = True  # True: koreksi 0,7 cm (berdiri <24 bln: +0,7; berbaring >=24 bln: -0,7) sesuai WHO
STUNTING_CUTOFF = -2.0  # HAZ < -2  -> stunting
HAZ_MIN, HAZ_MAX = (
    -6.0,
    6.0,
)  # di luar rentang ini dianggap tidak masuk akal (standar WHO) -> dikeluarkan
# Periode pengeluaran: ks02/ks03 = seminggu, ks06 = sebulan, ks08/ks09a = setahun.
FAKTOR_BULANAN = {"ks1": 52.0 / 12.0, "ks2": 1.0, "ks3": 1.0 / 12.0}
# =======================================================================

# kode khusus IFLS untuk "tidak ada orang tua di RT / meninggal / tidak tahu / missing"
KODE_ORTU_TIDAK_ADA = [51, 52, 98, 99]
# ar01a: anggota rumah tangga saat ini (panduan pengguna IFLS5). 3 = sudah pindah; 0, 6 = bukan anggota saat ini
AR01A_AKTIF = [1, 2, 5, 11]

# pengelompokan pendidikan (ar16) -> 0..4 ; kode lain (14, 17, 95, 98, 99) -> NaN. Ini KEPUTUSAN peneliti.
PEND_MAP = {
    1: 0,
    90: 0,  # tidak sekolah / TK
    2: 1,
    72: 1,
    11: 1,  # SD, MI, Paket A
    3: 2,
    4: 2,
    73: 2,
    12: 2,  # SMP, SMP kejuruan, MTs, Paket B
    5: 3,
    6: 3,
    74: 3,
    15: 3,  # SMA, SMK, MA, Paket C
    60: 4,
    61: 4,
    62: 4,
    63: 4,
    13: 4,
}  # Diploma, S1-S3, Universitas Terbuka

# pengelompokan tempat lahir (b4_ch1.ch19) -> 1..5 ; kode 95/99 -> NaN. KEPUTUSAN peneliti.
TEMPAT_LAHIR_MAP = {
    1: 1,
    2: 1,
    3: 1,  # 1 = rumah sakit (umum/swasta/bersalin)
    4: 2,
    5: 2,  # 2 = puskesmas / polindes
    6: 3,
    7: 3,  # 3 = praktik dokter / bidan
    8: 4,  # 4 = dukun beranak
    9: 5,
    10: 5,
}  # 5 = rumah sendiri / keluarga

# definisi "layak" (KEPUTUSAN peneliti; kode asli tetap disimpan di kolom *_kode)
AIR_LAYAK = [1, 2, 10]  # kr13: 1 pipa, 2 sumur pompa, 10 air kemasan
TOILET_LAYAK = [1]  # kr20: 1 toilet sendiri + septic tank
SAMPAH_LAYAK = [1]  # kr22: 1 dibuang di tempat sampah

# ---------------------------------------------------------------------
LOG = []
FUNNEL = []  # (tahap, n)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def indeks_file():
    idx = {}
    for root, _, files in os.walk(FOLDER_DTA):
        for f in files:
            if f.lower().endswith(".dta"):
                idx[f.lower()] = os.path.join(root, f)
    return idx


FILES = {}


def baca(nama, kolom):
    p = FILES.get(nama.lower())
    if p is None:
        sys.exit(f"File {nama} tidak ditemukan di {FOLDER_DTA}")
    try:
        df = pd.read_stata(
            p,
            columns=kolom,
            convert_categoricals=False,
            convert_dates=False,
            convert_missing=False,
        )
    except ValueError as e:
        sys.exit(f"Kolom tidak cocok di {nama}: {e}")
    if "hhid14" in df.columns:
        df["hhid14"] = df["hhid14"].astype(str).str.strip()
    if "pid14" in df.columns:
        df["pid14"] = pd.to_numeric(df["pid14"], errors="coerce").astype("Int64")
    return df


def tanpa_duplikat(df, kunci, nama):
    n = int(df.duplicated(kunci).sum())
    if n:
        log(
            f"  [peringatan] {nama}: {n} baris duplikat pada kunci {kunci}; diambil baris pertama"
        )
        df = df.drop_duplicates(kunci, keep="first")
    return df


def kode_ke_nan(s, kode):
    return s.where(~s.isin(kode))


def biner_ya_tidak(s, ya=1, tidak=3):
    out = pd.Series(np.nan, index=s.index)
    out[s == ya] = 1.0
    out[s == tidak] = 0.0
    return out


def tanggal(th, bl, hr, th_min, th_max):
    """Susun tanggal; hari tidak valid -> 15; bulan/tahun tidak valid -> NaT."""
    th = th.where(th.between(th_min, th_max))
    bl = bl.where(bl.between(1, 12))
    hr = hr.where(hr.between(1, 31), 15)
    awal = pd.to_datetime(
        pd.DataFrame({"year": th, "month": bl, "day": 1}), errors="coerce"
    )
    hari = np.minimum(hr, awal.dt.days_in_month)
    return awal + pd.to_timedelta((hari - 1).fillna(0), unit="D")


def zscore_who_harian(nilai, umur_bln, jk_laki, indikator):
    """z-score WHO 2006 dari tabel resmi per hari (expanded tables).
    indikator: 'lhfa' = TB/U, 'wfa' = BB/U. BB/U memakai penyesuaian WHO untuk |z| > 3."""
    z = pd.Series(np.nan, index=nilai.index)
    hari = (umur_bln * 30.4375).round()
    for jk, nama in ((1, "boys"), (0, "girls")):
        path = os.path.join(
            FOLDER_WHO, f"{indikator}-{nama}-zscore-expanded-tables.xlsx"
        )
        if not os.path.exists(path):
            sys.exit(f"Tabel WHO tidak ditemukan: {path}")
        t = pd.read_excel(path)
        t.columns = [str(c).strip().lower() for c in t.columns]
        t = t.set_index("day")
        m = (jk_laki == jk) & nilai.notna() & hari.isin(t.index)
        if not m.any():
            continue
        r = t.loc[hari[m].astype(int)]
        L, M, S = r["l"].to_numpy(), r["m"].to_numpy(), r["s"].to_numpy()
        y = nilai[m].to_numpy()
        zz = np.where(L == 0, np.log(y / M) / S, ((y / M) ** L - 1) / (L * S))
        if indikator == "wfa":  # penyesuaian WHO di luar +-3 SD (berat badan)
            sd = lambda k: M * (1 + L * S * k) ** (1 / L)
            sd3p, sd2p, sd3n, sd2n = sd(3), sd(2), sd(-3), sd(-2)
            zz = np.where(zz > 3, 3 + (y - sd3p) / (sd3p - sd2p), zz)
            zz = np.where(zz < -3, -3 + (y - sd3n) / (sd2n - sd3n), zz)
        z[m] = zz
    return z


def pengeluaran_rumah_tangga():
    """Pengeluaran RT per bulan dari b1_ks1 (pangan), b1_ks2 dan b1_ks3 (non-pangan)."""

    def bersih(s):
        return pd.to_numeric(s, errors="coerce").where(lambda x: x.between(0, 5e8))

    ks1 = baca("b1_ks1.dta", ["hhid14", "ks02", "ks03"])
    ks2 = baca("b1_ks2.dta", ["hhid14", "ks06"])
    ks3 = baca("b1_ks3.dta", ["hhid14", "ks08", "ks09a"])
    ks1["n"] = (
        bersih(ks1["ks02"]).fillna(0) + bersih(ks1["ks03"]).fillna(0)
    ) * FAKTOR_BULANAN["ks1"]
    ks2["n"] = bersih(ks2["ks06"]).fillna(0) * FAKTOR_BULANAN["ks2"]
    ks3["n"] = (
        bersih(ks3["ks08"]).fillna(0) + bersih(ks3["ks09a"]).fillna(0)
    ) * FAKTOR_BULANAN["ks3"]
    pangan = ks1.groupby("hhid14")["n"].sum().rename("_pangan_bln")
    non = (
        pd.concat([ks2[["hhid14", "n"]], ks3[["hhid14", "n"]]])
        .groupby("hhid14")["n"]
        .sum()
        .rename("_nonpangan_bln")
    )
    return pd.concat([pangan, non], axis=1).reset_index()


FREK_MAKAN = {
    1: 3.0,
    2: 2.0,
    3: 1.0,
    4: 5.5 / 7,
    5: 3.5 / 7,
    6: 2 / 7,
}  # fma01 -> kali makan per hari

# Arti kode fmatype sesuai kuesioner IFLS5 Buku 5, tabel FMA02 (halaman 268).
JENIS_MAKANAN = {
    "A": "ubi",
    "B": "telur",
    "C": "ikan",
    "D": "daging",
    "E": "susu",
    "F": "sayur_hijau",
    "G": "pisang",
    "H": "pepaya",
    "I": "wortel",
    "J": "mangga",
    "K": "mi_instan",
    "L": "makanan_cepat_saji",
    "M": "minuman_bersoda",
    "N": "sambal",
    "O": "gorengan",
    "P": "nasi",
    "Q": "camilan_manis",   # makanan manis (wajik, donat, wafer, coklat)
}


def fitur_jenis_makanan(f2, umur):
    h = f2.pivot_table(
        index=["hhid14", "pid14"], columns="fmatype", values="hari", aggfunc="first"
    )
    h = h.rename(columns=JENIS_MAKANAN)
    log(
        "  CEK URUTAN KODE fmatype: % anak >= 12 bulan yang makan tiap jenis dalam 7 hari terakhir"
    )
    cek = h.reset_index().merge(umur, on=["hhid14", "pid14"], how="inner")
    cek = cek[cek["umur_bln"] >= 12]
    for k, v in JENIS_MAKANAN.items():
        if v in cek.columns:
            log(f"    {k} = {v:<20s} {(cek[v] > 0).mean() * 100:5.1f}%")
    out = pd.DataFrame(index=h.index)
    hewani = ["telur", "ikan", "daging"]
    out["hari_protein_hewani_7hr"] = h[hewani].sum(axis=1, min_count=len(hewani))
    out["hari_susu_7hr"] = h["susu"]
    sayur_buah = ["sayur_hijau", "pisang", "pepaya", "wortel", "mangga"]
    out["jenis_sayur_buah_7hr"] = (
        (h[sayur_buah] > 0).sum(axis=1).where(h[sayur_buah].notna().all(axis=1))
    )
    olahan = [            # kode K, L, M, N, O (sama dengan data latih)
        "mi_instan",
        "makanan_cepat_saji",
        "minuman_bersoda",
        "sambal",
        "gorengan",
    ]
    out["hari_makanan_olahan_7hr"] = h[olahan].sum(axis=1, min_count=len(olahan))
    return out.reset_index()


def fitur_makan_sakit(umur):
    """Pola makan (Buku 5 FMA) dan keluhan sakit 4 minggu terakhir (Buku 5 MAA), 1 baris per anak."""
    f1 = tanpa_duplikat(
        baca("b5_fma1.dta", ["hhid14", "pid14", "fma01"]),
        ["hhid14", "pid14"],
        "b5_fma1",
    )
    f1["frek_makan_per_hari"] = f1["fma01"].map(FREK_MAKAN)
    f1["hanya_asi"] = np.where(
        f1["fma01"].isin(list(FREK_MAKAN) + [96]),
        (f1["fma01"] == 96).astype(float),
        np.nan,
    )

    f2 = baca("b5_fma2.dta", ["hhid14", "pid14", "fmatype", "fma02", "fma03"])
    f2["makan"] = np.where(
        f2["fma02"] == 1, 1.0, np.where(f2["fma02"] == 3, 0.0, np.nan)
    )
    f2["hari"] = np.where(
        f2["fma02"] == 3, 0.0, f2["fma03"].where(f2["fma03"].between(1, 7))
    )
    g2 = (
        f2.groupby(["hhid14", "pid14"])
        .agg(
            jenis_makanan_7hr=(
                "makan",
                lambda s: s.sum() if s.notna().any() else np.nan,
            ),
            total_hari_makan_7hr=(
                "hari",
                lambda s: s.sum() if s.notna().all() else np.nan,
            ),
        )
        .reset_index()
    )
    g2 = g2.merge(fitur_jenis_makanan(f2, umur), on=["hhid14", "pid14"], how="left")

    m1 = tanpa_duplikat(
        baca(
            "b5_maa1.dta",
            ["hhid14", "pid14", "maa0a", "maa0bx", "maa0b", "maa0cx", "maa0c"],
        ),
        ["hhid14", "pid14"],
        "b5_maa1",
    )
    m1["status_sehat_umum"] = m1["maa0a"].where(m1["maa0a"].isin([1, 2, 3, 4]))
    m1["hari_sakit_4mgg"] = m1["maa0b"].where(
        (m1["maa0bx"] == 1) & m1["maa0b"].between(0, 28)
    )
    m1["hari_terbaring_4mgg"] = m1["maa0c"].where(
        (m1["maa0cx"] == 1) & m1["maa0c"].between(0, 28)
    )

    m2 = baca("b5_maa2.dta", ["hhid14", "pid14", "maatype", "maa01"])
    m2 = m2[
        m2["maatype"].astype(str).str.len() == 2
    ]  # gejala utama; kode 3 huruf = rincian lanjutan
    m2["gejala"] = np.where(
        m2["maa01"] == 1, 1.0, np.where(m2["maa01"] == 3, 0.0, np.nan)
    )
    g4 = (
        m2.groupby(["hhid14", "pid14"])["gejala"]
        .agg(lambda s: s.sum() if s.notna().any() else np.nan)
        .rename("jumlah_gejala_4mgg")
        .reset_index()
    )

    out = (
        f1[["hhid14", "pid14", "frek_makan_per_hari", "hanya_asi"]]
        .merge(g2, on=["hhid14", "pid14"], how="outer")
        .merge(
            m1[
                [
                    "hhid14",
                    "pid14",
                    "status_sehat_umum",
                    "hari_sakit_4mgg",
                    "hari_terbaring_4mgg",
                ]
            ],
            on=["hhid14", "pid14"],
            how="outer",
        )
        .merge(g4, on=["hhid14", "pid14"], how="outer")
    )
    log(
        f"Modul makan & sakit anak: {len(out)} anak | median jenis makanan 7 hari {out['jenis_makanan_7hr'].median()} "
        f"| median gejala 4 minggu {out['jumlah_gejala_4mgg'].median()}"
    )
    return out


# kode jenis aset b2_hr1.hrtype (dari nama variabel hr02_<kode> di kamus)
ASET = {"A": "rumah ditempati", "B": "rumah/bangunan lain", "C": "tanah non-pertanian", "D1": "ternak/kolam ikan",
        "D2": "unggas", "D3": "tanaman keras", "E": "kendaraan", "F": "alat elektronik (radio, TV)",
        "G": "tabungan/deposito/saham", "H": "piutang", "J": "perhiasan", "K1": "perabot rumah tangga", "K2": "aset lain"}


def fitur_aset():
    """Indeks kekayaan rumah tangga dari kepemilikan aset (b2_hr1), dibentuk dengan PCA seperti metode DHS (tanpa target)."""
    from sklearn.decomposition import PCA
    hr = baca("b2_hr1.dta", ["hhid14", "hrtype", "hr01"])
    hr["hrtype"] = hr["hrtype"].astype(str).str.strip().str.upper()
    hr["milik"] = np.where(hr["hr01"] == 1, 1.0, np.where(hr["hr01"] == 3, 0.0, np.nan))
    tab = hr.pivot_table(index="hhid14", columns="hrtype", values="milik", aggfunc="max")
    tab = tab[[c for c in ASET if c in tab.columns]]
    log(f"\nCEK ASET RT: {len(tab)} rumah tangga | % memiliki: "
        + ", ".join(f"{c} {tab[c].mean() * 100:.0f}%" for c in tab.columns))
    lengkap = tab.dropna()
    Z = (lengkap - lengkap.mean()) / lengkap.std(ddof=0).replace(0, 1)
    pca = PCA(n_components=1).fit(Z)
    arah = 1.0 if pca.components_[0].sum() >= 0 else -1.0          # makin besar = makin kaya
    out = pd.DataFrame(index=tab.index)
    out.loc[lengkap.index, "indeks_kekayaan_pca"] = arah * pca.transform(Z)[:, 0]
    log(f"  Indeks kekayaan PCA: komponen 1 menjelaskan {pca.explained_variance_ratio_[0] * 100:.1f}% variasi; "
        "bobot: " + ", ".join(f"{c} {w * arah:+.2f}" for c, w in zip(tab.columns, pca.components_[0])))
    return out.reset_index()


def fitur_rumah():
    """Garam beryodium (b1_cov.us21), bahan bakar memasak (b2_kr.kr24), dan observasi rumah (bk_krk)."""
    cov = tanpa_duplikat(baca("b1_cov.dta", ["hhid14", "us21"]), ["hhid14"], "b1_cov")
    cov["garam_beryodium"] = np.where(cov["us21"] == 3, 1.0, np.where(cov["us21"].isin([1, 5]), 0.0, np.nan))
    kr = tanpa_duplikat(baca("b2_kr.dta", ["hhid14", "kr24"]), ["hhid14"], "b2_kr")
    kr["bahan_bakar_padat"] = np.where(kr["kr24"].isin([4, 5]), 1.0, np.where(kr["kr24"].isin([1, 2, 3, 7]), 0.0, np.nan))
    krk = tanpa_duplikat(baca("bk_krk.dta", ["hhid14", "krk02a", "krk02b", "krk02c", "krk02e", "krk02i", "krk08"]),
                         ["hhid14"], "bk_krk")
    yt = lambda s: np.where(s == 1, 1.0, np.where(s == 3, 0.0, np.nan))
    masalah = pd.DataFrame({c: yt(krk[c]) for c in ("krk02a", "krk02b", "krk02c")}, index=krk.index)
    krk["masalah_lingkungan_rumah"] = masalah.sum(axis=1, min_count=1)      # 0-3: kotoran, sampah, air tergenang
    krk["ventilasi_cukup"] = yt(krk["krk02e"])
    krk["dapur_satu_kamar_tidur"] = yt(krk["krk02i"])
    krk["lantai_tanah"] = np.where(krk["krk08"] == 6, 1.0, np.where(krk["krk08"].isin([1, 2, 3, 4, 5, 95]), 0.0, np.nan))
    out = (cov[["hhid14", "garam_beryodium"]]
           .merge(kr[["hhid14", "bahan_bakar_padat"]], on="hhid14", how="outer")
           .merge(krk[["hhid14", "masalah_lingkungan_rumah", "ventilasi_cukup", "dapur_satu_kamar_tidur", "lantai_tanah"]],
                  on="hhid14", how="outer"))
    log(f"\nCEK RUMAH: garam beryodium {out.garam_beryodium.mean() * 100:.0f}% | bahan bakar padat "
        f"{out.bahan_bakar_padat.mean() * 100:.0f}% | lantai tanah {out.lantai_tanah.mean() * 100:.0f}% | "
        f"ventilasi cukup {out.ventilasi_cukup.mean() * 100:.0f}% | median masalah lingkungan {out.masalah_lingkungan_rumah.median():.0f}")
    return out


def fitur_kelahiran(ch):
    """Urutan lahir, jarak dari kelahiran hidup sebelumnya (bulan), dan kembar, dari riwayat kehamilan ibu (b4_ch1)."""
    k = ch[["hhid14", "pid14", "ch05", "ch06", "ch06a", "ch09yr", "ch09mth", "ch27"]].copy()
    k = k.rename(columns={"pid14": "_ibu"}).sort_values(["hhid14", "_ibu", "ch05"])
    th, bl = k["ch09yr"].where(k["ch09yr"].between(1970, 2016)), k["ch09mth"].where(k["ch09mth"].between(1, 12))
    k["_bln"] = th * 12 + bl
    hidup = k["ch06"] == 2
    k["_bln_hidup"] = k["_bln"].where(hidup)
    g = k.groupby(["hhid14", "_ibu"], sort=False)
    k["urutan_lahir"] = hidup.astype(int).groupby([k["hhid14"], k["_ibu"]]).cumsum().where(hidup)
    k["jarak_lahir_bln"] = k["_bln"] - g["_bln_hidup"].transform(lambda s: s.shift(1).ffill())
    k["jarak_lahir_bln"] = k["jarak_lahir_bln"].where(hidup & k["jarak_lahir_bln"].between(9, 360))
    k["anak_kembar"] = np.where(k["ch06a"] == 1, 1.0, np.where(k["ch06a"] == 3, 0.0, np.nan))
    k["pid14"] = pd.to_numeric(k["ch27"], errors="coerce")
    k = k[k["pid14"].between(1, 89)].copy()
    k["pid14"] = k["pid14"].astype("Int64")
    k = k.drop_duplicates(["hhid14", "pid14"])
    log(f"CEK KELAHIRAN: median urutan lahir {k.urutan_lahir.median():.0f} | median jarak lahir "
        f"{k.jarak_lahir_bln.median():.0f} bulan | kembar {k.anak_kembar.mean() * 100:.1f}%")
    return k[["hhid14", "pid14", "urutan_lahir", "jarak_lahir_bln", "anak_kembar"]]


# =========================================================================
def main():
    global FILES
    if not os.path.isdir(FOLDER_DTA):
        sys.exit(f"Folder tidak ditemukan: {FOLDER_DTA}")
    os.makedirs(FOLDER_OUT, exist_ok=True)
    FILES = indeks_file()
    log("=== BuildDataset_IFLS5 ===")
    log(f"Folder data : {FOLDER_DTA}  ({len(FILES)} file .dta)")
    log(
        f"Umur        : {UMUR_MIN_BLN}-{UMUR_MAX_BLN} bulan | TB/BB anak jadi fitur: {PAKAI_TB_BB_ANAK} "
        f"| koreksi cara ukur: {KOREKSI_CARA_UKUR}"
    )

    # ------------------------- 1. Data dasar -------------------------
    ros = baca(
        "bk_ar1.dta",
        [
            "hhid14",
            "pid14",
            "pidlink",
            "ar07",
            "ar08day",
            "ar08mth",
            "ar08yr",
            "ar09",
            "ar10",
            "ar11",
            "ar15a",
            "ar16",
            "ar01a",
        ],
    )
    ros = tanpa_duplikat(ros, ["hhid14", "pid14"], "bk_ar1")
    us = baca("bus_us.dta", ["hhid14", "pid14", "us04x", "us04", "us05", "us06x", "us06", "us13x", "us13"])
    us = tanpa_duplikat(us, ["hhid14", "pid14"], "bus_us")
    tm = baca(
        "bus_time.dta", ["hhid14", "pid14", "time_occ", "ivwday", "ivwmth", "ivwyr"]
    )
    n_multi = int(tm.duplicated(["hhid14", "pid14"]).sum())
    if n_multi:
        log(
            f"  [info] bus_time: {n_multi} baris tambahan (kunjungan ganda); dipakai kunjungan pertama (time_occ terkecil)"
        )
    tm = tm.sort_values(["hhid14", "pid14", "time_occ"]).drop_duplicates(
        ["hhid14", "pid14"], keep="first"
    )

    FUNNEL.append(("Individu terukur di bus_us (semua umur)", len(us)))
    d = us.merge(ros, on=["hhid14", "pid14"], how="left").merge(
        tm[["hhid14", "pid14", "ivwday", "ivwmth", "ivwyr"]],
        on=["hhid14", "pid14"],
        how="left",
    )
    FUNNEL.append(
        ("  ... yang ditemukan di roster (bk_ar1)", int(d["pidlink"].notna().sum()))
    )

    # ------------------------- 2. Umur & filter balita -------------------------
    dob = tanggal(d["ar08yr"], d["ar08mth"], d["ar08day"], 1990, 2016)
    iv = tanggal(d["ivwyr"], d["ivwmth"], d["ivwday"], 2014, 2016)
    d["umur_bln"] = (iv - dob).dt.days / 30.4375
    d["_thn_lahir"], d["_thn_ivw"] = dob.dt.year, iv.dt.year
    kand = (
        d["ar09"].le(6) | d["ar09"].isna()
    )  # kandidat balita (umur roster <= 6 tahun atau kosong)
    log(
        f"Tanggal lahir tidak valid/kosong pada kandidat balita: {int((dob.isna() & kand).sum())} dari {int(kand.sum())} "
        f"(anak ini tidak bisa dihitung umurnya -> keluar) | tanggal wawancara tidak valid: {int(iv.isna().sum())}"
    )
    d = d[
        d["umur_bln"].notna()
        & (d["umur_bln"] >= UMUR_MIN_BLN)
        & (d["umur_bln"] < UMUR_MAX_BLN + 1)
    ].copy()
    FUNNEL.append((f"Umur {UMUR_MIN_BLN}-{UMUR_MAX_BLN} bulan", len(d)))

    # jenis kelamin (konvensi IFLS: 1 = laki-laki, 3 = perempuan; diverifikasi silang dengan b4_ch1.ch08 di bawah)
    n_jk_aneh = int((~d["ar07"].isin([1, 3]) & d["ar07"].notna()).sum())
    if n_jk_aneh:
        log(f"  [peringatan] ar07 punya {n_jk_aneh} nilai selain 1/3 -> dijadikan NaN")
    d["jk_laki"] = np.where(d["ar07"] == 1, 1.0, np.where(d["ar07"] == 3, 0.0, np.nan))

    # ------------------------- 3. Antropometri anak & HAZ -------------------------
    d["tb_anak"] = d["us04"].where((d["us04x"] == 1) & d["us04"].between(30, 140))
    d["bb_anak"] = d["us06"].where((d["us06x"] == 1) & d["us06"].between(2, 40))
    log(
        f"TB tidak terukur/tidak wajar: {int(d['tb_anak'].isna().sum())} dari {len(d)} balita | "
        f"BB: {int(d['bb_anak'].isna().sum())}"
    )
    tb_haz = d["tb_anak"].copy()
    if KOREKSI_CARA_UKUR:
        tb_haz = tb_haz.where(~((d["umur_bln"] < 24) & (d["us05"] == 1)), tb_haz + 0.7)
        tb_haz = tb_haz.where(~((d["umur_bln"] >= 24) & (d["us05"] == 3)), tb_haz - 0.7)
    n_a = int(((d["umur_bln"] < 24) & (d["us05"] == 1)).sum())
    n_b = int(((d["umur_bln"] >= 24) & (d["us05"] == 3)).sum())
    log(
        f"Cara ukur tidak sesuai standar WHO (berdiri <24 bln: {n_a}; berbaring >=24 bln: {n_b}) "
        f"-> {'DIKOREKSI' if KOREKSI_CARA_UKUR else 'tidak dikoreksi'}"
    )

    d["haz"] = zscore_who_harian(tb_haz, d["umur_bln"], d["jk_laki"], "lhfa")
    FUNNEL.append(
        ("HAZ dapat dihitung (TB, umur, JK valid)", int(d["haz"].notna().sum()))
    )
    tdk_wajar = int(((d["haz"] < HAZ_MIN) | (d["haz"] > HAZ_MAX)).sum())
    log(
        f"HAZ di luar [{HAZ_MIN}, {HAZ_MAX}] (tidak masuk akal): {tdk_wajar} -> dikeluarkan"
    )
    d = d[d["haz"].between(HAZ_MIN, HAZ_MAX)].copy()
    FUNNEL.append(
        (f"HAZ dalam rentang wajar [{HAZ_MIN}, {HAZ_MAX}] = dataset akhir", len(d))
    )
    d["stunting"] = (d["haz"] < STUNTING_CUTOFF).astype(int)
    d["waz"] = zscore_who_harian(d["bb_anak"], d["umur_bln"], d["jk_laki"], "wfa")
    d["waz"] = d["waz"].where(d["waz"].between(-6, 5))
    log(
        f"WAZ (BB/U WHO) terhitung pada {int(d['waz'].notna().sum())} balita | median {d['waz'].median():.2f}"
    )
    d["hb_anak"] = d["us13"].where((d["us13x"] == 1) & d["us13"].between(4, 20))   # g/dL; umur < 1 tahun tidak diukur
    log(
        f"Hb anak terukur: {int(d['hb_anak'].notna().sum())} balita | median {d['hb_anak'].median():.1f} g/dL | "
        f"anemia (Hb < 11, umur >= 12 bln): {(d.loc[d.umur_bln >= 12, 'hb_anak'] < 11).mean() * 100:.1f}%"
    )
    log(
        f"Prevalensi stunting (HAZ < {STUNTING_CUTOFF}): {d['stunting'].mean() * 100:.2f}% dari {len(d)} balita"
    )

    d = d.copy()
    # ------------------------- 4. Orang tua -------------------------
    ros_p = ros[["hhid14", "pid14", "ar09", "ar16", "ar15a", "ar01a"]]
    us_p = us[["hhid14", "pid14", "us04x", "us04", "us06x", "us06", "us13x", "us13"]].copy()
    us_p["hb"] = us_p["us13"].where((us_p["us13x"] == 1) & us_p["us13"].between(4, 20))
    us_p["tb"] = us_p["us04"].where(
        (us_p["us04x"] == 1) & us_p["us04"].between(120, 210)
    )
    us_p["bb"] = us_p["us06"].where(
        (us_p["us06x"] == 1) & us_p["us06"].between(25, 200)
    )
    ortu = ros_p.merge(us_p[["hhid14", "pid14", "tb", "bb", "hb"]], on=["hhid14", "pid14"], how="left")

    for peran, kol_id, sfx in (("ibu", "ar11", "ibu"), ("ayah", "ar10", "ayah")):
        d[f"_id_{peran}"] = pd.to_numeric(d[kol_id], errors="coerce")
        d[f"_id_{peran}"] = d[f"_id_{peran}"].where(
            ~d[f"_id_{peran}"].isin(KODE_ORTU_TIDAK_ADA) & (d[f"_id_{peran}"] > 0)
        )
        p = ortu.rename(
            columns={
                "pid14": f"_id_{peran}",
                "ar09": f"umur_{sfx}",
                "ar16": f"_pend_{sfx}",
                "ar15a": f"_kerja_{sfx}",
                "tb": f"tinggi_{sfx}",
                "bb": f"bb_{sfx}",
                "ar01a": f"_ar01a_{sfx}", "hb": f"hb_{sfx}"
            }
        )
        p[f"_id_{peran}"] = p[f"_id_{peran}"].astype("float64")
        d = d.merge(p, on=["hhid14", f"_id_{peran}"], how="left")
        d = (
            d.drop_duplicates(["hhid14", "pid14"])
            if d.duplicated(["hhid14", "pid14"]).any()
            else d
        )
        d[f"ada_{sfx}_di_RT"] = (
            d[f"_ar01a_{sfx}"]
            .isin(AR01A_AKTIF)
            .astype(float)
            .where(d[f"_id_{peran}"].notna(), 0.0)
        )
        d[f"umur_{sfx}"] = d[f"umur_{sfx}"].where(d[f"umur_{sfx}"].between(10, 90))
        d[f"imt_{sfx}"] = d[f"bb_{sfx}"] / (d[f"tinggi_{sfx}"] / 100.0) ** 2
        d[f"imt_{sfx}"] = d[f"imt_{sfx}"].where(d[f"imt_{sfx}"].between(10, 60))
        d[f"pend_{sfx}"] = d[f"_pend_{sfx}"].map(PEND_MAP)
        d[f"kerja_{sfx}"] = biner_ya_tidak(d[f"_kerja_{sfx}"])
        log(
            f"Link {peran}: ID valid {int(d[f'_id_{peran}'].notna().sum())} | ditemukan di roster "
            f"{int(d[f'umur_{sfx}'].notna().sum())} | masih serumah {int(d[f'ada_{sfx}_di_RT'].sum())} "
            f"| TB terukur {int(d[f'tinggi_{sfx}'].notna().sum())}"
        )

    # ------------------------- 5. Riwayat kehamilan ibu (b4_ch1) -------------------------
    kol_ch = [
        "hhid14",
        "pid14",
        "ch08",
        "ch05",
        "ch06",
        "ch06a",
        "ch09yr",
        "ch09mth",
        "ch14",
        "ch16a",
        "ch16b",
        "ch16c",
        "ch16f",
        "ch16g",
        "ch17",
        "ch17x",
        "ch19",
        "ch22",
        "ch23",
        "ch24",
        "ch24a",
        "ch24cx",
        "ch24c",
        "ch24c1",
        "ch24dx",
        "ch24d",
        "ch24d1",
        "ch24ex",
        "ch24e",
        "ch27",
    ]
    ch = baca("b4_ch1.dta", kol_ch)
    kelahiran = fitur_kelahiran(ch)
    ch = ch.rename(columns={"pid14": "_pid_ibu_ch1", "ch27": "pid14"})
    ch["pid14"] = pd.to_numeric(ch["pid14"], errors="coerce")
    ch = ch[ch["pid14"].notna() & (ch["pid14"] > 0) & (ch["pid14"] < 90)].copy()
    ch["pid14"] = ch["pid14"].astype("Int64")
    # kalau satu anak muncul >1 kali, utamakan yang ibunya sama dengan ar11
    d = d.merge(ch, on=["hhid14", "pid14"], how="left")
    if d.duplicated(["hhid14", "pid14"]).any():
        d["_cocok"] = (d["_pid_ibu_ch1"] == d["_id_ibu"]).fillna(False).astype(int)
        d = d.sort_values(
            ["hhid14", "pid14", "_cocok"], ascending=[True, True, False]
        ).drop_duplicates(["hhid14", "pid14"], keep="first")
    d = d.merge(kelahiran, on=["hhid14", "pid14"], how="left")
    d["_ada_ch1"] = d["_pid_ibu_ch1"].notna()
    log(
        f"Balita ditemukan di riwayat kehamilan ibu (b4_ch1): {int(d['_ada_ch1'].sum())} dari {len(d)}"
    )
    m = d["_ada_ch1"] & d["_id_ibu"].notna()
    if m.any():
        log(
            f"  ID ibu di b4_ch1 sama dengan ar11 (roster): {(d.loc[m, '_pid_ibu_ch1'] == d.loc[m, '_id_ibu']).mean() * 100:.1f}%"
        )
    m = d["ch08"].isin([1, 3]) & d["jk_laki"].notna()
    if m.any():
        sama = (
            (d.loc[m, "ch08"] == 1).astype(float) == d.loc[m, "jk_laki"]
        ).mean() * 100
        log(
            f"  VERIFIKASI kode jenis kelamin: ar07 (1=laki) sama dengan ch08 pada {sama:.1f}% dari {int(m.sum())} anak "
            f"(harus ~100%; kalau ~0% berarti kode terbalik)"
        )

    # berat lahir & turunan
    bbl = d["ch24"].where((d["ch23"] == 1) & d["ch24"].between(0.5, 6.5))
    n_tdk = int(
        ((d["ch23"] == 1) & d["ch24"].notna() & ~d["ch24"].between(0.5, 6.5)).sum()
    )
    log(f"Berat lahir tidak wajar (di luar 0,5-6,5 kg) dijadikan NaN: {n_tdk}")
    d["bb_lahir_kg"] = bbl
    d["bblr"] = np.where(bbl.notna(), (bbl < 2.5).astype(float), np.nan)
    d["ukuran_bayi"] = d["ch22"].where(d["ch22"].isin([1, 2, 3, 4, 5]))
    minggu = np.where(
        d["ch17x"] == 4, d["ch17"], np.where(d["ch17x"] == 5, d["ch17"] * 4.345, np.nan)
    )
    minggu = pd.Series(minggu, index=d.index).where(lambda s: s.between(20, 50))
    d["lama_hamil_minggu"] = minggu
    d["hamil_prematur_proxy"] = np.where(
        minggu.notna(), (minggu < 37).astype(float), np.nan
    )
    d["periksa_hamil"] = biner_ya_tidak(d["ch14"])
    anc = pd.concat(
        [kode_ke_nan(d[c], [98, 99]) for c in ("ch16a", "ch16b", "ch16c")], axis=1
    )
    d["anc_total"] = anc.sum(axis=1, min_count=1).where(lambda s: s <= 20)
    d.loc[d["ch14"] == 3, "anc_total"] = (
        0.0  # REVISI: tidak pernah periksa hamil = 0 kunjungan
    )
    d["tablet_besi"] = biner_ya_tidak(d["ch16f"])
    d["jml_tablet_besi"] = d["ch16g"].where(d["ch16g"].between(0, 400))
    d.loc[d["ch16f"] == 3, "jml_tablet_besi"] = (
        0.0  # REVISI: tidak dapat tablet = 0 tablet
    )
    log(
        f"Nol struktural: anc_total=0 pada {int((d['ch14'] == 3).sum())} anak; jml_tablet_besi=0 pada {int((d['ch16f'] == 3).sum())} anak"
    )
    d["tempat_lahir_grup"] = d["ch19"].map(TEMPAT_LAHIR_MAP)

    # menyusui & MPASI
    d["asi_pernah"] = biner_ya_tidak(d["ch24a"])

    def usia_bln(nilai, unit, status):
        n = nilai.where(nilai.between(0, 90))
        f = unit.map({3: 1 / 30.4375, 4: 1 / 4.345, 5: 1.0})
        return (n * f).where(status == 1)

    def buang_melebihi_umur(s, nama):
        salah = s.notna() & (s > d["umur_bln"] + 1)
        log(f"  {nama}: {int(salah.sum())} nilai melebihi umur anak (+1 bulan) -> NaN")
        return s.where(~salah)

    def dini(usia, status):
        """Hanya dinilai pada anak >= 6 bulan. 1 = diberi < 6 bulan; 0 = diberi >= 6 bulan atau belum diberi."""
        out = pd.Series(np.nan, index=d.index)
        cukup_umur = d["umur_bln"] >= 6
        out[cukup_umur & usia.notna()] = (usia[cukup_umur & usia.notna()] < 6).astype(
            float
        )
        out[cukup_umur & (status == 96)] = 0.0
        return out

    d["usia_air_bln"] = buang_melebihi_umur(
        usia_bln(d["ch24c"], d["ch24c1"], d["ch24cx"]), "usia_air_bln"
    )
    d["usia_mpasi_bln"] = buang_melebihi_umur(
        usia_bln(d["ch24d"], d["ch24d1"], d["ch24dx"]), "usia_mpasi_bln"
    )
    d["usia_sapih_bln"] = buang_melebihi_umur(
        d["ch24e"].where((d["ch24ex"] == 5) & d["ch24e"].between(0, 90)),
        "usia_sapih_bln",
    )
    d["masih_asi"] = np.where(
        d["ch24ex"].notna(), (d["ch24ex"] == 96).astype(float), np.nan
    )
    d["air_dini"] = dini(d["usia_air_bln"], d["ch24cx"])
    d["mpasi_dini"] = dini(d["usia_mpasi_bln"], d["ch24dx"])
    # ASI eksklusif (proxy): pernah ASI, serta air dan makanan lain tidak diberikan sebelum 6 bulan
    eks = pd.Series(np.nan, index=d.index)
    eks[(d["asi_pernah"] == 1) & (d["air_dini"] == 0) & (d["mpasi_dini"] == 0)] = 1.0
    eks[(d["asi_pernah"] == 0) | (d["air_dini"] == 1) | (d["mpasi_dini"] == 1)] = 0.0
    d["asi_eksklusif_proxy"] = eks

    # ------------------------- 6. Layanan anak & imunisasi (Book 5) -------------------------
    r0 = tanpa_duplikat(
        baca("b5_rja0.dta", ["hhid14", "pid14", "rja0a"]),
        ["hhid14", "pid14"],
        "b5_rja0",
    )
    r2 = tanpa_duplikat(baca("b5_rja2.dta", ["hhid14", "pid14", "rja26", "rja32"]), ["hhid14", "pid14"], "b5_rja2")
    huruf = list("bcdefghijklmnop")
    kol_kartu = [f"rja28{t}_{h}" for h in huruf for t in ("ady", "amt", "ayr")]
    kol_recall = ["rja30a", "rja30b", "rja30c", "rja30d", "rja30e", "rja30g", "rja30h"]
    r3 = tanpa_duplikat(
        baca("b5_rja3.dta", ["hhid14", "pid14", "rja25a"] + kol_kartu + kol_recall),
        ["hhid14", "pid14"],
        "b5_rja3",
    )
    for r in (r0, r2, r3):
        d = d.merge(r, on=["hhid14", "pid14"], how="left")
    d["posyandu_4mgg"] = biner_ya_tidak(d["rja0a"])
    d["frek_timbang_6bln"] = d["rja32"].where(d["rja32"].between(0, 30))    # 98/99 -> NaN
    log(f"  frek_timbang_6bln: terisi {int(d['frek_timbang_6bln'].notna().sum())} balita | "
        f"median {d['frek_timbang_6bln'].median():.0f} kali")
    d["vit_a_6bln"] = biner_ya_tidak(d["rja25a"])
    d["punya_kartu_imun"] = np.where(
        d["rja26"].isin([1, 2]), 1.0, np.where(d["rja26"] == 3, 0.0, np.nan)
    )

    # huruf pada nama variabel kartu (dari label di kamus): b=BCG, c=Polio0, d=Polio1, e=Polio2, f=Polio3, n=Polio4,
    # g=DPT1, h=DPT2, i=DPT3, j=Campak, k=HepB1, l=HepB2, m=HepB3, o=Rota1, p=Rota2
    grup_kartu = {
        "bcg": "b",
        "polio": "cdefn",
        "dpt": "ghi",
        "hepb": "klm",
        "campak": "j",
        "rota": "op",
    }
    grup_recall = {
        "bcg": ["rja30a"],
        "polio": ["rja30b"],
        "dpt": ["rja30c"],
        "hepb": ["rja30e"],
        "campak": ["rja30d"],
        "rota": ["rja30g", "rja30h"],
    }

    # REVISI: kolom kartu terisi untuk semua pemegang kartu, jadi notna() bukan tanda vaksin diterima.
    # Vaksin dihitung diterima hanya jika tahun pada kartu berada di antara tahun lahir dan tahun wawancara.
    def tanggal_valid(h):
        th = pd.to_numeric(d[f"rja28ayr_{h}"], errors="coerce")
        return th.between(d["_thn_lahir"], d["_thn_ivw"])

    valid_h = pd.DataFrame({h: tanggal_valid(h) for h in huruf})
    ada_kartu_any = valid_h.any(axis=1)
    baris_kartu = d[kol_kartu].notna().any(axis=1)
    log(
        f"\nCEK KARTU IMUNISASI: {int(baris_kartu.sum())} balita punya baris kartu; "
        f"{int(ada_kartu_any.sum())} punya minimal satu tanggal valid"
    )
    log(
        "  Nilai tahun terbanyak pada kartu campak (cek kode 'tidak diberi' di sini): "
        + str(
            d.loc[baris_kartu, "rja28ayr_j"]
            .value_counts(dropna=False)
            .head(10)
            .to_dict()
        )
    )
    if baris_kartu.any() and ada_kartu_any.sum() < 0.5 * baris_kartu.sum():
        log(
            "  [PERINGATAN] Kurang dari separuh pemegang kartu punya tanggal valid. Cek format tahun rja28ayr_*."
        )

    kolom_imun = []
    imun_baru = {}
    for nm in grup_kartu:
        dari_kartu = valid_h[list(grup_kartu[nm])].any(axis=1)
        rc = pd.concat([biner_ya_tidak(d[c]) for c in grup_recall[nm]], axis=1)
        rc_val = rc.max(
            axis=1
        )  # 1 kalau ada yang 1; 0 kalau semua terjawab 0; NaN kalau semua NaN
        v = pd.Series(np.nan, index=d.index)
        v[rc_val.notna()] = rc_val[rc_val.notna()]
        v[dari_kartu] = 1.0
        v[v.isna() & ada_kartu_any] = (
            0.0  # ada kartu berisi tanggal, vaksin ini tidak tercatat -> 0
        )
        imun_baru[f"imun_{nm}"] = v
        kolom_imun.append(f"imun_{nm}")
    imun_baru["jumlah_jenis_imun"] = pd.DataFrame(imun_baru).sum(axis=1, min_count=1)
    d = pd.concat([d, pd.DataFrame(imun_baru)], axis=1)

    kel = pd.cut(d["umur_bln"], [0, 2, 4, 6, 9, 12, 60], right=False)
    log(
        "  Cakupan imunisasi menurut umur (bulan). Harus masuk akal: campak rendah sebelum 9 bulan:"
    )
    log((d.groupby(kel, observed=True)[kolom_imun].mean() * 100).round(1).to_string())

    d = d.copy()  # rapikan fragmentasi memori
    # ------------------------- 7. Rumah tangga -------------------------
    kr = tanpa_duplikat(
        baca("b2_kr.dta", ["hhid14", "kr13", "kr13a", "kr20", "kr21", "kr22", "kr26"]),
        ["hhid14"],
        "b2_kr",
    )
    sc = tanpa_duplikat(
        baca("bk_sc1.dta", ["hhid14", "sc05", "sc01_14_14"]), ["hhid14"], "bk_sc1"
    )
    d = d.merge(kr, on="hhid14", how="left").merge(sc, on="hhid14", how="left")
    d["perkotaan"] = np.where(
        d["sc05"] == 1, 1.0, np.where(d["sc05"] == 2, 0.0, np.nan)
    )
    d["provinsi"] = d["sc01_14_14"]
    for k, nm in (
        ("kr13", "air_minum_kode"),
        ("kr20", "toilet_kode"),
        ("kr21", "limbah_kode"),
        ("kr22", "sampah_kode"),
    ):
        d[nm] = kode_ke_nan(d[k], [99])
    d["air_layak"] = np.where(
        d["air_minum_kode"].notna(),
        d["air_minum_kode"].isin(AIR_LAYAK).astype(float),
        np.nan,
    )
    d["toilet_layak"] = np.where(
        d["toilet_kode"].notna(),
        d["toilet_kode"].isin(TOILET_LAYAK).astype(float),
        np.nan,
    )
    d["sampah_layak"] = np.where(
        d["sampah_kode"].notna(),
        d["sampah_kode"].isin(SAMPAH_LAYAK).astype(float),
        np.nan,
    )
    d["kartu_sehat"] = biner_ya_tidak(d["kr26"])
    d["air_direbus"] = biner_ya_tidak(d["kr13a"])

    # jumlah anggota RT: baris roster dengan umur terisi
    ang = (
        ros[ros["ar01a"].isin(AR01A_AKTIF)]
        .groupby("hhid14")
        .size()
        .rename("jumlah_anggota_RT")
        .reset_index()
    )
    d = d.merge(ang, on="hhid14", how="left")

    # ada perokok aktif di RT
    km = baca("b3b_km.dta", ["hhid14", "km01d", "km01e", "km04"])
    km["_rokok"] = ((km["km01d"] == 1) | (km["km01e"] == 1)) & (km["km04"] == 1)
    rk = (
        km.groupby("hhid14")["_rokok"]
        .max()
        .astype(float)
        .rename("ada_perokok_RT")
        .reset_index()
    )
    d = d.merge(rk, on="hhid14", how="left")

    # pengeluaran per kapita per bulan (ditransformasi log karena sangat menceng)
    pk = pengeluaran_rumah_tangga()
    d = d.merge(pk, on="hhid14", how="left")
    d["log_pengeluaran_pangan_perkapita"] = np.log1p(
        d["_pangan_bln"] / d["jumlah_anggota_RT"]
    )
    d["log_pengeluaran_perkapita"] = np.log1p(
        (d["_pangan_bln"].fillna(0) + d["_nonpangan_bln"].fillna(0)).where(
            d["_pangan_bln"].notna() | d["_nonpangan_bln"].notna()
        )
        / d["jumlah_anggota_RT"]
    )
    log(
        f"\nCEK FITUR BARU: median anggota RT {d['jumlah_anggota_RT'].median():.0f}; "
        f"ada perokok RT {d['ada_perokok_RT'].mean() * 100:.1f}%; "
        f"median pengeluaran per kapita/bulan Rp{np.expm1(d['log_pengeluaran_perkapita'].median()):,.0f}"
    )

    d = d.merge(fitur_aset(), on="hhid14", how="left")
    d = d.merge(fitur_rumah(), on="hhid14", how="left")
    d = d.merge(fitur_makan_sakit(d[["hhid14", "pid14", "umur_bln"]]), on=["hhid14", "pid14"], how="left")

    # ------------------------- 8. Susun kolom akhir + dokumentasi -------------------------
    kolom = [  # (kolom, sumber, aturan)
        ("pidlink", "bk_ar1.pidlink", "ID individu"),
        (
            "hhid14",
            "bk_ar1.hhid14",
            "ID rumah tangga (kunci pengelompokan saat split/CV)",
        ),
        ("pid14", "bk_ar1.pid14", "nomor urut individu dalam RT"),
        (
            "haz",
            "bus_us.us04, us05 + umur + jenis kelamin",
            "z-score TB/U standar WHO 2006 (tabel WHO per hari; koreksi cara ukur 0,7 cm). TARGET-TURUNAN: JANGAN dipakai sebagai fitur",
        ),
        (
            "stunting",
            "dari haz",
            f"1 jika HAZ < {STUNTING_CUTOFF}, selain itu 0 (TARGET)",
        ),
        (
            "umur_bln",
            "bk_ar1.ar08yr/mth/day, bus_time.ivwyr/mth/day",
            "(tanggal wawancara US - tanggal lahir)/30,4375; hari lahir tidak valid -> 15",
        ),
        (
            "jk_laki",
            "bk_ar1.ar07",
            "1 = laki-laki (kode 1), 0 = perempuan (kode 3); diverifikasi dengan b4_ch1.ch08",
        ),
    ]
    if PAKAI_TB_BB_ANAK:
        kolom += [
            (
                "tb_anak",
                "bus_us.us04 (+us04x=1)",
                "tinggi badan anak, cm; di luar 30-140 -> NaN",
            ),
            (
                "bb_anak",
                "bus_us.us06 (+us06x=1)",
                "berat badan anak, kg; di luar 2-40 -> NaN",
            ),
            (
                "waz",
                "dari bb_anak + umur + jenis kelamin",
                "z-score BB/U standar WHO 2006; di luar -6 s.d. 5 -> NaN",
            ),
        ]
    kolom += [
        (
            "bb_lahir_kg",
            "b4_ch1.ch24 (+ch23=1)",
            "berat lahir kg; di luar 0,5-6,5 -> NaN",
        ),
        ("bblr", "dari bb_lahir_kg", "1 jika < 2,5 kg"),
        (
            "ukuran_bayi",
            "b4_ch1.ch22",
            "persepsi ibu: 1 sangat besar ... 5 sangat kecil; 98/99 -> NaN",
        ),
        (
            "lama_hamil_minggu",
            "b4_ch1.ch17 + ch17x",
            "ch17x=4 minggu, 5 bulan (x4,345); di luar 20-50 minggu -> NaN",
        ),
        (
            "hamil_prematur_proxy",
            "dari lama_hamil_minggu",
            "1 jika < 37 minggu. PROXY kasar (IFLS tidak punya variabel prematur; bulan dibulatkan)",
        ),
        ("periksa_hamil", "b4_ch1.ch14", "1 ya, 0 tidak; 8/9 -> NaN"),
        (
            "anc_total",
            "b4_ch1.ch16a+ch16b+ch16c",
            "jumlah kunjungan ANC tri 1-3; 98/99 -> NaN; 0 jika ch14=3 (tidak pernah periksa)",
        ),
        ("tablet_besi", "b4_ch1.ch16f", "1 ya, 0 tidak"),
        (
            "jml_tablet_besi",
            "b4_ch1.ch16g",
            "jumlah tablet besi; di luar 0-400 -> NaN; 0 jika ch16f=3 (tidak dapat)",
        ),
        (
            "tempat_lahir_grup",
            "b4_ch1.ch19",
            "1 RS, 2 puskesmas/polindes, 3 praktik dokter/bidan, 4 dukun, 5 rumah; 95/99 -> NaN",
        ),
        ("asi_pernah", "b4_ch1.ch24a", "1 pernah disusui, 0 tidak"),
        (
            "usia_air_bln",
            "b4_ch1.ch24c, ch24c1, ch24cx",
            "usia pertama diberi air (bulan); hanya jika ch24cx=1; melebihi umur anak -> NaN",
        ),
        (
            "usia_mpasi_bln",
            "b4_ch1.ch24d, ch24d1, ch24dx",
            "usia pertama diberi makanan lain (bulan); melebihi umur anak -> NaN",
        ),
        (
            "usia_sapih_bln",
            "b4_ch1.ch24e, ch24ex",
            "usia disapih (bulan); hanya jika ch24ex=5; melebihi umur anak -> NaN",
        ),
        ("masih_asi", "b4_ch1.ch24ex", "1 jika masih menyusu (kode 96)"),
        (
            "air_dini",
            "dari usia_air_bln, ch24cx",
            "hanya anak >= 6 bulan: 1 jika air diberikan < 6 bulan; 0 jika >= 6 bulan atau belum diberi",
        ),
        (
            "mpasi_dini",
            "dari usia_mpasi_bln, ch24dx",
            "hanya anak >= 6 bulan: 1 jika makanan lain diberikan < 6 bulan; 0 jika >= 6 bulan atau belum diberi",
        ),
        (
            "asi_eksklusif_proxy",
            "dari asi_pernah, air_dini, mpasi_dini",
            "hanya anak >= 6 bulan: 1 jika pernah ASI, air_dini=0, mpasi_dini=0. PROXY",
        ),
        ("posyandu_4mgg", "b5_rja0.rja0a", "1 ke posyandu dalam 4 minggu terakhir"),
        ("vit_a_6bln", "b5_rja3.rja25a", "1 dapat vitamin A 6 bulan terakhir"),
        (
            "punya_kartu_imun",
            "b5_rja2.rja26",
            "1 punya kartu imunisasi (kode 1/2), 0 tidak (kode 3)",
        ),
        (
            "imun_bcg",
            "b5_rja3.rja28ayr_b / rja30a",
            "1 jika tahun di kartu valid (tahun lahir s.d. tahun wawancara) atau ibu menjawab ya; 0 jika tidak; NaN jika tidak ada info",
        ),
        ("imun_polio", "b5_rja3.rja28ayr_c,d,e,f,n / rja30b", "idem (Polio 0-4)"),
        ("imun_dpt", "b5_rja3.rja28ayr_g,h,i / rja30c", "idem (DPT 1-3)"),
        ("imun_hepb", "b5_rja3.rja28ayr_k,l,m / rja30e", "idem (Hepatitis B 1-3)"),
        ("imun_campak", "b5_rja3.rja28ayr_j / rja30d", "idem (campak)"),
        ("imun_rota", "b5_rja3.rja28ayr_o,p / rja30g,h", "idem (rotavirus 1-2)"),
        (
            "jumlah_jenis_imun",
            "dari imun_*",
            "jumlah jenis (dari 6) yang diterima; NaN jika semua NaN. Hati-hati: bayi muda belum waktunya campak",
        ),
        ("tinggi_ibu", "bk_ar1.ar11 -> bus_us.us04", "cm; di luar 120-210 -> NaN"),
        ("tinggi_ayah", "bk_ar1.ar10 -> bus_us.us04", "cm; di luar 120-210 -> NaN"),
        ("bb_ibu", "bus_us.us06", "kg; di luar 25-200 -> NaN"),
        ("bb_ayah", "bus_us.us06", "kg; di luar 25-200 -> NaN"),
        ("imt_ibu", "dari bb & tinggi", "kg/m2; di luar 10-60 -> NaN"),
        ("imt_ayah", "dari bb & tinggi", "kg/m2; di luar 10-60 -> NaN"),
        ("umur_ibu", "bk_ar1.ar09", "tahun; di luar 10-90 -> NaN"),
        ("umur_ayah", "bk_ar1.ar09", "tahun; di luar 10-90 -> NaN"),
        (
            "pend_ibu",
            "bk_ar1.ar16",
            "0 tidak sekolah/TK, 1 SD, 2 SMP, 3 SMA, 4 perguruan tinggi; kode lain -> NaN (lihat PEND_MAP)",
        ),
        ("pend_ayah", "bk_ar1.ar16", "idem"),
        (
            "kerja_ibu",
            "bk_ar1.ar15a",
            "1 bekerja tahun lalu, 0 tidak; kode lain -> NaN",
        ),
        ("kerja_ayah", "bk_ar1.ar15a", "idem"),
        (
            "ada_ibu_di_RT",
            "bk_ar1.ar11, ar01a",
            "1 jika ibu masih tinggal di rumah tangga ini (ar01a = 1, 2, 5, 11)",
        ),
        (
            "ada_ayah_di_RT",
            "bk_ar1.ar10, ar01a",
            "1 jika ayah masih tinggal di rumah tangga ini (ar01a = 1, 2, 5, 11)",
        ),
        ("perkotaan", "bk_sc1.sc05", "1 perkotaan, 0 perdesaan"),
        ("provinsi", "bk_sc1.sc01_14_14", "kode provinsi BPS 2014"),
        ("air_minum_kode", "b2_kr.kr13", "kode asli sumber air minum; 99 -> NaN"),
        ("air_layak", "dari kr13", f"1 jika kode in {AIR_LAYAK} (keputusan peneliti)"),
        (
            "air_direbus",
            "b2_kr.kr13a",
            "1 air direbus sebelum diminum, 0 tidak; NaN jika tidak ditanyakan",
        ),
        ("toilet_kode", "b2_kr.kr20", "kode asli; 99 -> NaN"),
        (
            "toilet_layak",
            "dari kr20",
            f"1 jika kode in {TOILET_LAYAK} (keputusan peneliti)",
        ),
        ("limbah_kode", "b2_kr.kr21", "kode asli pembuangan limbah cair; 99 -> NaN"),
        ("sampah_kode", "b2_kr.kr22", "kode asli pembuangan sampah; 99 -> NaN"),
        (
            "sampah_layak",
            "dari kr22",
            f"1 jika kode in {SAMPAH_LAYAK} (keputusan peneliti)",
        ),
        ("kartu_sehat", "b2_kr.kr26", "1 RT punya kartu sehat/BPJS/JKN"),
        ("frek_timbang_6bln", "b5_rja2.rja32", "berapa kali anak ditimbang dalam 6 bulan terakhir; 98/99 -> NaN"),
        ("indeks_kekayaan_pca", "b2_hr1.hrtype, hr01", "komponen utama pertama (PCA) dari kepemilikan 13 jenis aset; makin besar = makin kaya"),
        ("hb_anak", "bus_us.us13 (+us13x=1)", "hemoglobin anak (g/dL); umur < 1 tahun tidak diukur; di luar 4-20 -> NaN"),
        ("hb_ibu", "bk_ar1.ar11 -> bus_us.us13", "hemoglobin ibu (g/dL); di luar 4-20 -> NaN"),
        ("anak_kembar", "b4_ch1.ch06a", "1 jika kelahiran kembar"),
        ("garam_beryodium", "b1_cov.us21", "1 jika tes garam berwarna biru (beryodium); 0 putih atau sedikit biru"),
        ("bahan_bakar_padat", "b2_kr.kr24", "1 jika memasak dengan kayu bakar atau arang"),
        ("masalah_lingkungan_rumah", "bk_krk.krk02a-c", "jumlah masalah (0-3): kotoran, tumpukan sampah, air tergenang di sekitar rumah"),
        ("ventilasi_cukup", "bk_krk.krk02e", "1 jika ventilasi rumah cukup (observasi pewawancara)"),
        ("dapur_satu_kamar_tidur", "bk_krk.krk02i", "1 jika dapur dan kamar tidur dalam ruangan yang sama"),
                ("lantai_tanah", "bk_krk.krk08", "1 jika lantai rumah dari tanah"),
        (
            "frek_makan_per_hari",
            "b5_fma1.fma01",
            "kali makan per hari; 5-6/3-4/2 kali per minggu dibagi 7; 96/98/99 -> NaN",
        ),
        ("hanya_asi", "b5_fma1.fma01", "1 jika anak masih hanya menyusu (kode 96)"),
        (
            "jenis_makanan_7hr",
            "b5_fma2.fma02",
            "jumlah jenis makanan (dari 17) yang dimakan 7 hari terakhir",
        ),
        (
            "total_hari_makan_7hr",
            "b5_fma2.fma02, fma03",
            "jumlah hari makan semua jenis makanan 7 hari terakhir; NaN jika ada yang tidak diketahui",
        ),
        ("status_sehat_umum", "b5_maa1.maa0a", "1 sangat sehat ... 4 tidak sehat"),
        (
            "hari_sakit_4mgg",
            "b5_maa1.maa0b",
            "hari tidak beraktivitas karena sakit, 4 minggu terakhir",
        ),
        (
            "hari_terbaring_4mgg",
            "b5_maa1.maa0c",
            "hari terbaring karena sakit, 4 minggu terakhir",
        ),
        (
            "jumlah_gejala_4mgg",
            "b5_maa2.maa01",
            "jumlah gejala utama (kode 2 huruf) yang dialami 4 minggu terakhir",
        ),
        (
            "hari_protein_hewani_7hr",
            "b5_fma2 kode B, C, D",
            "total hari makan telur, ikan, daging 7 hari terakhir (0-21); arti kode dari literatur",
        ),
        ("hari_susu_7hr", "b5_fma2 kode E", "hari minum susu 7 hari terakhir (0-7)"),
        (
            "jenis_sayur_buah_7hr",
            "b5_fma2 kode F-J",
            "jumlah jenis sayur hijau dan buah yang dimakan 7 hari terakhir (0-5)",
        ),
        (
            "hari_makanan_olahan_7hr",
            "b5_fma2 kode K-O",
            "total hari makan mi instan, cepat saji, soda, sambal, gorengan (0-35); kode K-O Buku 5 FMA02",
        ),
        (
            "jumlah_anggota_RT",
            "bk_ar1.ar01a",
            "jumlah anggota rumah tangga saat ini (ar01a = 1, 2, 5, 11) per hhid14",
        ),
        (
            "ada_perokok_RT",
            "b3b_km.km01d, km01e, km04",
            "1 jika ada anggota RT yang masih merokok; NaN jika RT tanpa data km",
        ),
        (
            "log_pengeluaran_pangan_perkapita",
            "b1_ks1.ks02+ks03",
            "log(1 + pangan per kapita per bulan)",
        ),
        (
            "log_pengeluaran_perkapita",
            "b1_ks1, b1_ks2, b1_ks3",
            "log(1 + pangan + non-pangan per kapita per bulan); periode lihat FAKTOR_BULANAN",
        ),
    ]
    nama_kol = [k[0] for k in kolom]
    hilang = [k for k in nama_kol if k not in d.columns]
    if hilang:
        sys.exit(f"Kolom akhir belum terbentuk: {hilang}")
    out = d[nama_kol].copy()

    # ------------------------- 9. Simpan + laporan -------------------------
    p_csv = os.path.join(FOLDER_OUT, "balita_ifls5.csv")
    out.to_csv(p_csv, index=False, encoding="utf-8")
    log(f"\nDataset akhir: {out.shape[0]} baris x {out.shape[1]} kolom -> {p_csv}")
    log(
        f"Jumlah keluarga (hhid14 unik): {out['hhid14'].nunique()} | stunting: {int(out['stunting'].sum())} "
        f"({out['stunting'].mean() * 100:.2f}%)"
    )
    log("\nKolom dengan missing > 30% (pertimbangkan sebelum dipakai):")
    miss = out.isna().mean().sort_values(ascending=False) * 100
    for k, v in miss[miss > 30].items():
        log(f"  {k:<24s} {v:5.1f}% kosong")

    ring = pd.DataFrame(
        {
            "kolom": out.columns,
            "tipe": [str(t) for t in out.dtypes],
            "n_terisi": out.notna().sum().values,
            "pct_kosong": (out.isna().mean() * 100).round(2).values,
            "min": [
                out[c].min() if pd.api.types.is_numeric_dtype(out[c]) else None
                for c in out.columns
            ],
            "rata2": [
                round(out[c].mean(), 3)
                if pd.api.types.is_numeric_dtype(out[c])
                else None
                for c in out.columns
            ],
            "max": [
                out[c].max() if pd.api.types.is_numeric_dtype(out[c]) else None
                for c in out.columns
            ],
        }
    )
    dok = pd.DataFrame(kolom, columns=["kolom", "sumber_IFLS5", "aturan"])
    fun = pd.DataFrame(FUNNEL, columns=["tahap", "n"])
    p_xlsx = os.path.join(FOLDER_OUT, "Ringkasan_Dataset.xlsx")
    with pd.ExcelWriter(p_xlsx, engine="openpyxl") as w:
        dok.to_excel(w, sheet_name="Daftar_Variabel", index=False)
        ring.to_excel(w, sheet_name="Ringkasan", index=False)
        fun.to_excel(w, sheet_name="Funnel", index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                L = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(L + 2, 10), 80
                )
    log(f"Ringkasan: {p_xlsx}")
    log("\nFunnel:")
    for t, n in FUNNEL:
        log(f"  {n:>7}  {t}")
    with open(
        os.path.join(FOLDER_OUT, "laporan_build.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("\n".join(LOG))
    print("\nSelesai.")


if __name__ == "__main__":
    main()
