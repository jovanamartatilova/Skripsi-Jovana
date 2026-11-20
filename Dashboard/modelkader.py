# -*- coding: utf-8 -*-
"""Form Cek Risiko Stunting dan prediksi Model B (S4 Random Forest, 45 fitur) untuk app.py."""
import html
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Winsorizer harus bisa di-import agar joblib dapat memuat model_final.joblib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "IFLS5"))
import custom_transformer  # noqa: E402,F401

# Tabel WHO per hari; sama dengan FOLDER_WHO di 0_BuildDataset_IFLS5.py (ML\Dataset\WHO)
FOLDER_WHO = Path(__file__).resolve().parent.parent / "Dataset" / "WHO"
YA_TIDAK = {1: "Ya", 0: "Tidak"}
SUDAH = {1: "Sudah", 0: "Belum"}

# ------------------------------------------------------------------
# Pilihan jawaban sesuai kuesioner IFLS5. Bentuk list [(nilai, teks)]:
# nilai = kode yang dipakai model (0/1), aturan sama dengan 0_BuildDataset_IFLS5.py.
# ------------------------------------------------------------------
# Buku 2 KR13; layak = kode 01, 02, 10
AIR_OPSI = [(1, "Ledeng"), (1, "Sumur/pompa (listrik/tangan)"), (0, "Sumur timba/perigi"), (0, "Mata air"),
            (0, "Air hujan"), (0, "Air sungai/kali"), (0, "Kolam/balong/empang"), (0, "Bak penampungan"),
            (1, "Air mineral/Aqua"), (0, "Lainnya")]
# Buku 2 KR20; layak = kode 01
TOILET_OPSI = [(1, "Jamban sendiri dengan tangki septik"), (0, "Jamban sendiri tanpa tangki septik"),
               (0, "Jamban bersama"), (0, "Jamban umum"), (0, "Kali/sungai/parit"), (0, "Kebun/sawah"),
               (0, "Selokan"), (0, "Kolam/balong/empang"), (0, "Kandang ternak"), (0, "Laut/danau"),
               (0, "Lainnya")]
# Buku 2 KR22; layak = kode 01
SAMPAH_OPSI = [(1, "Dibuang di tempat/tong sampah, diangkut petugas"), (0, "Dibakar"),
               (0, "Dibuang di sungai/kali/selokan"), (0, "Dibuang di pekarangan rumah/kebun, dibiarkan"),
               (0, "Ditimbun di lubang"), (0, "Hutan/gunung"), (0, "Laut/danau/pantai"), (0, "Sawah/ladang"),
               (0, "Lainnya")]
# Buku 1 US21: 3 biru -> 1; 1 putih, 5 sedikit biru -> 0
GARAM_OPSI = [(1, "Biru"), (0, "Sedikit biru"), (0, "Putih")]
# Buku 5 FMA01: kode 01-06 -> kali makan per hari; 96 baru minum ASI -> kosong
FREK_MAKAN_OPSI = [(3.0, "3 kali atau lebih dalam sehari"), (2.0, "2 kali sehari"), (1.0, "1 kali sehari"),
                   (5.5 / 7, "5-6 kali seminggu"), (3.5 / 7, "3-4 kali seminggu"),
                   (2 / 7, "2 kali atau kurang dalam seminggu"),
                   (float("nan"), "Baru minum ASI (belum makan)")]
# Buku K KRK02 A-C (observasi): jumlah yang terlihat = 0-3
MASALAH_OPSI = ["Kotoran manusia atau hewan", "Tumpukan sampah", "Genangan air"]

# Buku 5 FMA02, kode A-Q: jumlah hari dimakan dalam 1 minggu terakhir (FMA03)
# Kode IFLS5 di komentar: A ubi, B telur, C ikan, D daging, E susu, F sayur hijau, G pisang, H pepaya,
# I wortel, J mangga, K mi instan, L fast food, M soda, N sambal, O gorengan, P nasi, Q makanan manis
MAKANAN = {
    "mkn_nasi": "Nasi", "mkn_ubi": "Ubi",
    "mkn_telur": "Telur", "mkn_ikan": "Ikan", "mkn_daging": "Daging (sapi, ayam, dll.)",
    "mkn_susu": "Susu, keju, mentega, dan sejenisnya",
    "mkn_sayur_hijau": "Sayuran hijau", "mkn_wortel": "Wortel",
    "mkn_pisang": "Pisang", "mkn_pepaya": "Pepaya", "mkn_mangga": "Mangga",
    "mkn_mi_instan": "Mi instan", "mkn_cepat_saji": "Makanan cepat saji (KFC, burger, dll.)",
    "mkn_gorengan": "Gorengan (tempe, tahu, bakwan, dll.)",
    "mkn_makanan_manis": "Makanan manis (donat, wafer, cokelat, dll.)",
    "mkn_soda": "Minuman bersoda (Coca-Cola, Sprite, dll.)", "mkn_sambal": "Sambal",
}
HEWANI = ["mkn_telur", "mkn_ikan", "mkn_daging"]                                      # B, C, D
OLAHAN = ["mkn_mi_instan", "mkn_cepat_saji", "mkn_soda", "mkn_sambal", "mkn_gorengan"]  # K, L, M, N, O (= data latih)

# nama input -> (label, opsi; None = angka, langkah angka)
INPUT = {
    # Data anak: diambil otomatis dari data Puskesmas (tidak ditampilkan di form)
    "umur_bln": ("Usia (bulan)", None, 1.0),
    "jk_laki": ("Jenis kelamin", {1: "Laki-laki", 0: "Perempuan"}, None),
    "bb_anak": ("Berat badan anak (kg)", None, 0.1),
    # Air, sanitasi, dan kebersihan lingkungan (Buku 2 KR, Buku K KRK02)
    "air_layak": ("Apa sumber air utama untuk minum rumah tangga ini?", AIR_OPSI, None),
    "air_direbus": ("Apakah air untuk minum dididihkan sebelum diminum?", YA_TIDAK, None),
    "toilet_layak": ("Di mana sebagian besar anggota rumah tangga buang air besar?", TOILET_OPSI, None),
    "sampah_layak": ("Bagaimana rumah tangga ini membuang sampah?", SAMPAH_OPSI, None),
    "masalah_lingkungan_rumah": ("Apa saja yang terlihat di sekitar rumah? (boleh pilih lebih dari satu)", "multi", None),
    # Kondisi rumah dan paparan asap (Buku K KRK02, Buku 3B KM)
    "ventilasi_cukup": ("Apakah rumah memiliki ventilasi (jendela/lubang angin) yang cukup?", YA_TIDAK, None),
    "dapur_satu_kamar_tidur": ("Apakah dapur dan kamar tidur berada dalam satu ruang?", YA_TIDAK, None),
    "ada_perokok_RT": ("Apakah ada anggota rumah tangga yang masih merokok (rokok/cerutu/tembakau linting)?",
                       YA_TIDAK, None),
    # ASI dan MPASI (Buku 4 CH24)
    "masih_asi": ("Apakah anak masih menyusu ASI?", {1: "Masih menyusui", 0: "Sudah berhenti"}, None),
    "asi_pernah": ("Apakah ibu pernah menyusui anak, meskipun hanya sebentar?", YA_TIDAK, None),
    "sudah_air": ("Apakah anak sudah pernah diberi minum air (tawar, dengan gula, madu, air beras, atau teh)?",
                  SUDAH, None),
    "usia_air_bln": ("Jika sudah: pada umur berapa pertama kali diberi minum air? (bulan)", None, 0.5),
    "sudah_mpasi": ("Apakah anak sudah secara teratur (setiap hari) diberi makanan/minuman lain selain ASI?",
                    SUDAH, None),
    "usia_mpasi_bln": ("Jika sudah: pada umur berapa mulai diberi secara teratur? (bulan)", None, 0.5),
    # Asupan makan anak (Buku 5 FMA, Buku 1 US21)
    "frek_makan_per_hari": ("Biasanya anak makan berapa kali?", FREK_MAKAN_OPSI, None),
    "garam_beryodium": ("Hasil tes kadar yodium garam rumah tangga", GARAM_OPSI, None),
    **{k: (f"{v} (hari)", None, 1.0) for k, v in MAKANAN.items()},
    # Anemia (Buku US13)
    "hb_anak": ("Kadar hemoglobin anak (g/dL)", None, 0.1),
    "hb_ibu": ("Kadar hemoglobin ibu (g/dL)", None, 0.1),
    # Pemantauan pertumbuhan di Posyandu (Buku 5 RJA)
    "frek_timbang_6bln": ("Berapa kali anak ditimbang dalam 6 bulan terakhir?", None, 1.0),
    "posyandu_4mgg": ("Apakah anak pernah dibawa ke posyandu dalam 4 minggu terakhir?", YA_TIDAK, None),
    "vit_a_6bln": ("Apakah anak diberi vitamin A dalam 6 bulan terakhir?", YA_TIDAK, None),
    # Imunisasi dan jaminan kesehatan (Buku 5 RJA26-RJA30, Buku 2 KR26)
    "punya_kartu_imun": ("Apakah anak mempunyai KMS atau Kartu KIA?", {1: "Punya", 0: "Tidak punya"}, None),
    "imun_bcg": ("Imunisasi BCG", SUDAH, None),
    "imun_polio": ("Imunisasi polio (minimal 1 kali)", SUDAH, None),
    "imun_dpt": ("Imunisasi DPT (minimal 1 kali)", SUDAH, None),
    "imun_hepb": ("Imunisasi hepatitis B (minimal 1 kali)", SUDAH, None),
    "imun_campak": ("Imunisasi campak", SUDAH, None),
    "imun_rota": ("Imunisasi rotavirus (dosis 1 atau 2)", SUDAH, None),
    "kartu_sehat": ("Apakah rumah tangga ini mempunyai kartu sehat/JAMKESMAS/BPJS/JKN?", YA_TIDAK, None),
}
MAKS = {**{k: 7.0 for k in MAKANAN}, "frek_timbang_6bln": 30.0, "hb_anak": 20.0, "hb_ibu": 20.0,
        "usia_air_bln": 60.0, "usia_mpasi_bln": 60.0}

IMUN = ["imun_bcg", "imun_polio", "imun_dpt", "imun_hepb", "imun_campak", "imun_rota"]
# variabel model yang dihitung otomatis -> input yang dibutuhkan
TURUNAN = {
    "waz": ["bb_anak", "umur_bln", "jk_laki"],
    "mpasi_dini": ["sudah_mpasi", "usia_mpasi_bln", "umur_bln"],
    "asi_eksklusif_proxy": ["asi_pernah", "sudah_air", "usia_air_bln", "sudah_mpasi", "usia_mpasi_bln", "umur_bln"],
    "jumlah_jenis_imun": IMUN,
    "jenis_makanan_7hr": list(MAKANAN),
    "hari_protein_hewani_7hr": HEWANI,
    "hari_susu_7hr": ["mkn_susu"],
    "hari_makanan_olahan_7hr": OLAHAN,
}
# data yang sudah ada di data Puskesmas: diisi otomatis saat nama balita dipilih, tidak ditanyakan lagi
DARI_PUSKESMAS = ["umur_bln", "jk_laki", "bb_anak", "bb_lahir_kg"]
TIDAK_DIISI = {"indeks_kekayaan_pca"}
KONTROL = ["umur_bln", "jk_laki", "waz"]


@st.cache_resource
def muat_model(path):
    if not path.exists():
        return None, f"Model tidak ditemukan: {path}"
    try:
        return joblib.load(path), None
    except Exception as e:  # noqa: BLE001
        return None, f"Gagal membaca model: {e}"


def _input_dari(vars_):
    hasil = []
    for v in vars_:
        if v in TIDAK_DIISI:
            continue
        for i in TURUNAN.get(v, [v]):
            if i not in hasil:
                hasil.append(i)
    return hasil


def daftar_input(bundle, path_importance=None):
    """(input wajib, {kategori: input tambahan}).
    Wajib = kontrol + variabel dashboard (dipilih = ya); tambahan = variabel dapat ditindaklanjuti lainnya.
    Faktor latar tidak ditampilkan -> dibiarkan kosong dan diisi median oleh preprocessor."""
    vars_model = bundle["kolom_terpilih"]
    if path_importance is not None and path_importance.exists():
        imp = pd.read_csv(path_importance)
    else:
        imp = pd.DataFrame(columns=["variabel", "dipilih"])
    imp = imp[imp["variabel"].isin(vars_model)].copy()
    if "kategori" not in imp.columns:
        imp["kategori"] = "Faktor lain yang bisa ditindaklanjuti"
    dipilih = imp.loc[imp["dipilih"].astype(str).str.lower().eq("ya"), "variabel"].tolist()
    wajib = _input_dari([v for v in KONTROL if v in vars_model] + dipilih)
    tambahan, sudah = {}, set(wajib)
    for kat, g in imp.sort_values("kategori", kind="stable").groupby("kategori", sort=False):
        isi = [k for k in _input_dari(g["variabel"].tolist()) if k not in sudah]
        sudah.update(isi)
        if isi:
            tambahan[kat] = isi
    return wajib, tambahan


@lru_cache(maxsize=2)
def _tabel_wfa(jk_laki):
    nama = "boys" if jk_laki == 1 else "girls"
    t = pd.read_excel(FOLDER_WHO / f"wfa-{nama}-zscore-expanded-tables.xlsx")
    t.columns = [str(c).strip().lower() for c in t.columns]
    return t.set_index("day")


def _hitung_waz(bb, umur_bln, jk_laki):
    """z-score BB/U WHO per hari, sama dengan zscore_who_harian() di 0_BuildDataset_IFLS5.py."""
    try:
        t = _tabel_wfa(int(jk_laki))
        hari = int(np.round(umur_bln * 30.4375))
        if hari not in t.index:
            return None
        r = t.loc[hari]
        L, M, S = float(r["l"]), float(r["m"]), float(r["s"])
        z = math.log(bb / M) / S if L == 0 else ((bb / M) ** L - 1) / (L * S)

        def sd(k):
            return M * (1 + L * S * k) ** (1 / L)

        if z > 3:
            z = 3 + (bb - sd(3)) / (sd(3) - sd(2))
        elif z < -3:
            z = -3 + (bb - sd(-3)) / (sd(-2) - sd(-3))
        return float(z)
    except Exception:  # noqa: BLE001
        return None


def _isi_turunan(x):
    """Hitung variabel turunan dengan aturan yang sama seperti 0_BuildDataset_IFLS5.py."""
    def ada(k):
        v = x.get(k)
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    if ada("bb_anak") and ada("umur_bln") and ada("jk_laki") and 2 <= x["bb_anak"] <= 40:
        z = _hitung_waz(x["bb_anak"], x["umur_bln"], x["jk_laki"])
        if z is not None:
            x["waz"] = z
    if any(ada(k) for k in MAKANAN):   # sama dengan build: dihitung jika minimal 1 jenis terjawab
        x["jenis_makanan_7hr"] = float(sum(x[k] > 0 for k in MAKANAN if ada(k)))
    if all(ada(k) for k in HEWANI):
        x["hari_protein_hewani_7hr"] = float(sum(x[k] for k in HEWANI))
    if ada("mkn_susu"):
        x["hari_susu_7hr"] = float(x["mkn_susu"])
    if all(ada(k) for k in OLAHAN):
        x["hari_makanan_olahan_7hr"] = float(sum(x[k] for k in OLAHAN))
    # hb anak tidak diukur pada umur < 1 tahun di IFLS5 (bus_us.us13)
    if ada("umur_bln") and x["umur_bln"] < 12:
        x["hb_anak"] = np.nan
    # usia pemberian yang melebihi umur anak dianggap tidak valid (sama dengan buang_melebihi_umur)
    for k in ("usia_air_bln", "usia_mpasi_bln"):
        if ada(k) and ada("umur_bln") and x[k] > x["umur_bln"]:
            x[k] = np.nan
    if ada("umur_bln") and x["umur_bln"] >= 6:
        # belum pernah diberi -> 0 (tidak dini), sama dengan 0_BuildDataset_IFLS5.py
        if x.get("sudah_air") == 0:
            air_dini = 0.0
        else:
            air_dini = float(x["usia_air_bln"] < 6) if ada("usia_air_bln") else None
        if x.get("sudah_mpasi") == 0:
            x["mpasi_dini"] = 0.0
        elif ada("usia_mpasi_bln"):
            x["mpasi_dini"] = float(x["usia_mpasi_bln"] < 6)
        if x.get("asi_pernah") == 0 or air_dini == 1 or x.get("mpasi_dini") == 1:
            x["asi_eksklusif_proxy"] = 0.0
        elif x.get("asi_pernah") == 1 and air_dini == 0 and x.get("mpasi_dini") == 0:
            x["asi_eksklusif_proxy"] = 1.0
    terisi = [x[k] for k in IMUN if ada(k)]
    if terisi:
        x["jumlah_jenis_imun"] = float(sum(terisi))
    return x


def prediksi(bundle, nilai):
    """Probabilitas stunting. Alurnya sama dengan prediksi_risiko.py dari tahap pemodelan."""
    num, kat = bundle["fitur_numerik"], bundle["fitur_kategorik"]
    d = pd.DataFrame([_isi_turunan(dict(nilai))]).reindex(columns=num + kat)
    for c in num + kat:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c, (lo, hi) in bundle["rentang"].items():
        if c in d.columns:
            d.loc[d[c].notna() & ~d[c].between(lo, hi), c] = np.nan
    Z = np.asarray(bundle["preprocessor"].transform(d), float)[:, bundle["indeks_terpilih"]]
    return float(bundle["model"].predict_proba(Z)[:, 1][0])
NAMA_IMUN = {"imun_bcg": "BCG", "imun_polio": "polio", "imun_dpt": "DPT", "imun_hepb": "hepatitis B",
             "imun_campak": "campak", "imun_rota": "rotavirus"}


def faktor_perlu_diperhatikan(nilai):
    """Daftar kondisi pada isian kader yang perlu diperhatikan (dipakai setelah prediksi).
    Hanya membaca isian yang terisi; isian kosong tidak dinilai."""
    def v(k):
        x = nilai.get(k)
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else x

    umur = v("umur_bln")
    hasil = []
    # data dari Puskesmas (WAZ = predictor terkuat model)
    if v("bb_anak") is not None and umur is not None and v("jk_laki") is not None:
        z = _hitung_waz(v("bb_anak"), umur, v("jk_laki"))
        if z is not None and z < -2:
            hasil.append("Berat badan anak kurang untuk usianya (BB/U di bawah −2 SD)")
        elif z is not None and z < -1:
            hasil.append("Berat badan anak cenderung rendah untuk usianya (BB/U di bawah −1 SD)")
    # faktor utama
    if v("hb_anak") is not None and (umur is None or umur >= 12) and v("hb_anak") < 11:
        hasil.append(f"Kadar Hb anak {v('hb_anak'):.1f} g/dL, di bawah 11 g/dL".replace(".", ","))
    if v("sampah_layak") == 0:
        hasil.append("Sampah belum dibuang ke tempat sampah yang diangkut petugas")
    if v("masih_asi") == 0 and umur is not None and umur < 24:
        hasil.append("Anak sudah berhenti menyusu sebelum usia 2 tahun")
    if v("air_direbus") == 0:
        hasil.append("Air minum belum dididihkan sebelum diminum")
    if v("frek_timbang_6bln") is not None and v("frek_timbang_6bln") < 6:
        hasil.append(f"Anak hanya ditimbang {v('frek_timbang_6bln'):.0f} kali dalam 6 bulan terakhir "
                     "(dianjurkan setiap bulan)")
    if v("hb_ibu") is not None and v("hb_ibu") < 12:
        hasil.append(f"Kadar Hb ibu {v('hb_ibu'):.1f} g/dL, di bawah 12 g/dL".replace(".", ","))
    if v("air_layak") == 0:
        hasil.append("Sumber air minum belum termasuk sumber yang layak")
    if v("kartu_sehat") == 0:
        hasil.append("Rumah tangga belum memiliki kartu kesehatan (JKN/BPJS/KIS)")
    # faktor lain (hanya jika diisi)
    if v("toilet_layak") == 0:
        hasil.append("Belum menggunakan jamban sendiri dengan tangki septik")
    if v("masalah_lingkungan_rumah") is not None and v("masalah_lingkungan_rumah") > 0:
        hasil.append(f"Terlihat {v('masalah_lingkungan_rumah'):.0f} masalah kebersihan di sekitar rumah")
    if v("ventilasi_cukup") == 0:
        hasil.append("Ventilasi rumah belum cukup")
    if v("dapur_satu_kamar_tidur") == 1:
        hasil.append("Dapur berada satu ruang dengan kamar tidur (paparan asap)")
    if v("ada_perokok_RT") == 1:
        hasil.append("Ada anggota rumah tangga yang masih merokok")
    if v("posyandu_4mgg") == 0:
        hasil.append("Anak tidak dibawa ke Posyandu dalam 4 minggu terakhir")
    if v("vit_a_6bln") == 0:
        hasil.append("Anak belum mendapat vitamin A dalam 6 bulan terakhir")
    if v("punya_kartu_imun") == 0:
        hasil.append("Anak belum memiliki KMS atau Kartu KIA")
    belum = [NAMA_IMUN[k] for k in IMUN if v(k) == 0]
    if belum:
        hasil.append("Imunisasi yang belum diterima: " + ", ".join(belum))
    if v("garam_beryodium") == 0:
        hasil.append("Garam rumah tangga belum cukup beryodium")
    if v("frek_makan_per_hari") is not None and umur is not None and umur >= 12 and v("frek_makan_per_hari") < 3:
        hasil.append("Anak makan kurang dari 3 kali sehari")
    if all(v(k) is not None for k in HEWANI) and sum(v(k) for k in HEWANI) == 0:
        hasil.append("Anak tidak makan telur, ikan, atau daging dalam 7 hari terakhir")
    return hasil

def kunci_form():
    return [f"f_{k}" for k in INPUT] + [f"anak_{k}" for k in DARI_PUSKESMAS]

def kosongkan_form():
    """Kosongkan semua isian form dan data anak (dipakai saat ganti balita atau prediksi balita lain)."""
    for k, (_, opsi, _) in INPUT.items():
        st.session_state[f"f_{k}"] = [] if opsi == "multi" else None
    for k in DARI_PUSKESMAS:
        st.session_state.pop(f"anak_{k}", None)

def isi_dari_baris(nilai):
    """Saat nama balita dipilih: kosongkan form lama, lalu simpan data Puskesmas balita itu."""
    kosongkan_form()
    for k, v in nilai.items():
        if k in DARI_PUSKESMAS and v is not None and not pd.isna(v):
            st.session_state[f"anak_{k}"] = float(v)

@st.cache_data
def _baca_importance(path_str):
    p = Path(path_str)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["variabel", "label", "dipilih"])


def _variabel_induk(k):
    """Isian -> variabel model (mis. imun_bcg -> jumlah_jenis_imun, mkn_telur -> hari_protein_hewani_7hr)."""
    if k in MAKANAN:
        if k in HEWANI:
            return "hari_protein_hewani_7hr"
        if k in OLAHAN:
            return "hari_makanan_olahan_7hr"
        return "hari_susu_7hr" if k == "mkn_susu" else "jenis_makanan_7hr"
    for v, isi in TURUNAN.items():
        if k in isi and v not in KONTROL:
            return v
    return k


def _peta_penjelasan(path_importance):
    if path_importance is None:
        return {}
    imp = _baca_importance(str(path_importance))
    if "penjelasan_dashboard" not in imp.columns:
        return {}
    return dict(zip(imp["variabel"], imp["penjelasan_dashboard"].fillna("").astype(str)))
NAMA_TAB = {
    "1. Air, sanitasi, dan kebersihan lingkungan": "Air & sanitasi",
    "2. Kondisi rumah dan paparan asap": "Rumah & asap",
    "3. ASI dan MPASI": "ASI & MPASI",
    "4. Asupan makan anak": "Makan anak",
    "5. Anemia (Hb) anak dan ibu": "Anemia",
    "6. Pemantauan pertumbuhan di Posyandu": "Posyandu",
    "7. Imunisasi dan jaminan kesehatan": "Imunisasi & BPJS",
}

GAYA_PENJELASAN = """
<style>
.st-key-kotak_penjelasan [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; padding: 4px 0 6px 0; }
.st-key-kotak_penjelasan [data-baseweb="tab-highlight"],
.st-key-kotak_penjelasan [data-baseweb="tab-border"] { display: none; }
.st-key-kotak_penjelasan button[data-baseweb="tab"] {
    background: #f5f7fb; border: 1px solid #dfe5f0; border-radius: 8px;
    padding: 7px 14px; height: auto; margin: 0;
}
.st-key-kotak_penjelasan button[data-baseweb="tab"] p { font-size: 13px; font-weight: 500; color: #3a4150 !important; }
.st-key-kotak_penjelasan button[data-baseweb="tab"]:hover { background: #eef2fa; }
.st-key-kotak_penjelasan button[data-baseweb="tab"][aria-selected="true"] { background: #183b91; border-color: #183b91; }
.st-key-kotak_penjelasan button[data-baseweb="tab"][aria-selected="true"] p { color: #ffffff !important; }
</style>
"""

PENJELASAN_KATEGORI = {
    "1. Air, sanitasi, dan kebersihan lingkungan": "Air dan lingkungan yang tidak bersih membuat anak sering diare dan cacingan. Infeksi berulang mengganggu penyerapan zat gizi di usus sehingga pertumbuhan tinggi badan terhambat.",
    "2. Kondisi rumah dan paparan asap": "Asap dapur, asap rokok, dan udara yang tidak berganti membuat anak mudah terkena ISPA. Anak yang sering sakit nafsu makannya turun dan energinya dipakai untuk melawan infeksi, bukan untuk tumbuh.",
    "3. ASI dan MPASI": "ASI memberi zat gizi dan antibodi. Pemberian MPASI yang terlalu dini atau ASI yang dihentikan terlalu cepat meningkatkan risiko infeksi dan kekurangan gizi pada masa awal pertumbuhan.",
    "4. Asupan makan anak": "Pertumbuhan tinggi badan membutuhkan energi, protein, dan zat gizi mikro (zat besi, seng, kalsium, yodium) yang cukup. Makan yang kurang sering, kurang beragam, atau didominasi makanan olahan membuat kebutuhan ini tidak terpenuhi.",
    "5. Anemia (Hb) anak dan ibu": "Kadar Hb rendah menandakan kekurangan zat besi dan gizi secara umum. Anemia dan stunting sering terjadi bersamaan karena penyebabnya mirip, yaitu asupan kurang dan infeksi berulang.",
    "6. Pemantauan pertumbuhan di Posyandu": "Anak yang rutin ditimbang dan datang ke Posyandu lebih cepat ketahuan jika pertumbuhannya melambat, dan keluarganya mendapat penyuluhan gizi, vitamin A, serta PMT.",
    "7. Imunisasi dan jaminan kesehatan": "Imunisasi mencegah penyakit infeksi yang dapat menghambat pertumbuhan, sedangkan jaminan kesehatan memudahkan keluarga berobat sehingga sakit anak tidak berlarut-larut.",
}


def tampilkan_penjelasan(path_importance):
    """Penjelasan per kategori untuk pengetahuan kader (dari importance_dashboard.csv)."""
    imp = _baca_importance(str(path_importance)) if path_importance is not None else pd.DataFrame()
    if imp.empty or "penjelasan_dashboard" not in imp.columns:
        return
    imp = imp[imp["penjelasan_dashboard"].fillna("").astype(str).str.strip() != ""].copy()
    if imp.empty:
        return
    if "kategori" not in imp.columns:
        imp["kategori"] = "Faktor yang bisa ditindaklanjuti"
    imp = imp.sort_values("kategori", kind="stable")
    kategori = list(dict.fromkeys(imp["kategori"]))
    nama_tab = [NAMA_TAB.get(k, re.sub(r"^\d+\.\s*", "", k)) for k in kategori]

    st.markdown(GAYA_PENJELASAN, unsafe_allow_html=True)
    with st.container(key="kotak_penjelasan"):
        with st.expander("Kenapa faktor-faktor ini berpengaruh terhadap stunting?"):
            for tab, kat in zip(st.tabs(nama_tab), kategori):
                with tab:
                    pengantar = PENJELASAN_KATEGORI.get(kat, "")
                    isi = (
                        '<div style="font-size:13.5px;line-height:1.6;color:#374151;background:#f6f8fb;'
                        f'border-radius:8px;padding:10px 14px;margin-bottom:6px;">{html.escape(pengantar)}</div>'
                        if pengantar else ""
                    )
                    for i, r in enumerate(imp[imp["kategori"] == kat].itertuples()):
                        utama = str(r.dipilih).lower() == "ya"
                        ket = ('<div style="font-size:12px;color:#6b7280;margin-top:2px;">Faktor utama</div>'
                               if utama else "")
                        garis = "border-top:1px solid #e5e7eb;" if i else ""
                        isi += (
                            f'<div style="display:flex;flex-wrap:wrap;gap:4px 24px;padding:12px 2px;{garis}">'
                            '<div style="flex:0 0 230px;min-width:180px;">'
                            f'<div style="font-size:14px;font-weight:600;color:#1f2937;">{html.escape(str(r.label))}</div>'
                            f'{ket}</div>'
                            '<div style="flex:1 1 320px;font-size:13.5px;line-height:1.6;color:#374151;">'
                            f'{html.escape(str(r.penjelasan_dashboard))}</div></div>'
                        )
                    # padding bawah: menahan margin negatif bawaan st.markdown
                    st.markdown(f'<div style="padding:4px 0 20px 0;">{isi}</div>', unsafe_allow_html=True)

def _pasangan(opsi):
    return list(opsi.items()) if isinstance(opsi, dict) else list(opsi)

def _widget(k, bantuan=None, wajib=False):
    label, opsi, langkah = INPUT[k]
    if wajib:
        label = f"{label} :red[\\*]"
    bantuan = bantuan or None
    if opsi == "multi":
        pilih = st.multiselect(label, MASALAH_OPSI + ["Tidak ada yang terlihat"], placeholder="Pilih...", key=f"f_{k}", help=bantuan)
        if not pilih:
            return None
        return float(len([p for p in pilih if p != "Tidak ada yang terlihat"]))
    if opsi:
        pasangan = _pasangan(opsi)
        pilih = st.selectbox(label, [t for _, t in pasangan], index=None, placeholder="Pilih...", key=f"f_{k}", help=bantuan)
        return None if pilih is None else next(n for n, t in pasangan if t == pilih)
    fmt = "%.0f" if langkah >= 1 else "%.1f"
    v = st.number_input(label, step=langkah, format=fmt,
                        value=None, placeholder="Isi angka", key=f"f_{k}", help=bantuan)
    batas = MAKS.get(k)
    if v is not None and v < 0:
        st.markdown('<div class="input-error">Angka tidak boleh kurang dari 0.</div>',
                    unsafe_allow_html=True)
        return None
    if v is not None and batas is not None and v > batas:
        st.markdown(f'<div class="input-error">Angka maksimal {batas:g}. Periksa kembali isian ini.</div>',
                    unsafe_allow_html=True)
        return None
    return v


def _ringkas_anak():
    """Kartu kecil berisi data anak dari Puskesmas."""
    u, jk, bb = (st.session_state.get(f"anak_{k}") for k in ("umur_bln", "jk_laki", "bb_anak"))
    if u is None or jk is None or bb is None:
        return False
    teks = f"Usia {u:.0f} bulan · {'Laki-laki' if jk == 1 else 'Perempuan'} · BB {bb:.1f} kg".replace(".", ",", 1)
    st.markdown(f'<div class="info-box">Data anak dari Puskesmas: <b>{teks}</b></div>', unsafe_allow_html=True)
    return True


def render_form(bundle, path_importance=None):
    """Tampilkan form. Mengembalikan (nilai input, daftar isian wajib yang masih kosong)."""
    wajib, tambahan = daftar_input(bundle, path_importance)
    semua_tambahan = [k for isi in tambahan.values() for k in isi]
    tanya = [k for k in wajib + semua_tambahan if k not in DARI_PUSKESMAS]
    tak_dikenal = [k for k in tanya if k not in INPUT]
    if tak_dikenal:
        st.error("Variabel model ini belum punya input di modelkader.py: " + ", ".join(tak_dikenal))
        st.stop()

    jelas = _peta_penjelasan(path_importance)
    nilai = {k: st.session_state[f"anak_{k}"] for k in DARI_PUSKESMAS if f"anak_{k}" in st.session_state}
    kosong = [] if _ringkas_anak() else ["pilih nama balita terlebih dahulu"]

    # hb anak tidak ditanyakan untuk balita < 12 bulan (di IFLS5 tidak diukur)
    umur = st.session_state.get("anak_umur_bln")
    lewati = {"hb_anak"} if umur is not None and umur < 12 else set()
    wajib_tanya = [k for k in wajib if k not in DARI_PUSKESMAS and k not in lewati]
    tambahan = {kat: [k for k in isi if k not in lewati] for kat, isi in tambahan.items()}
    tambahan = {kat: isi for kat, isi in tambahan.items() if isi}
    if wajib_tanya:
        st.markdown(
            '<div class="form-heading">Faktor utama '
            '<span style="font-weight:400;font-size:0.85em;color:#6b7280">'
            '(tanda <span style="color:#d93025">*</span> wajib diisi)</span></div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        for i, k in enumerate(wajib_tanya):
            with (c1 if i % 2 == 0 else c2):
                nilai[k] = _widget(k, jelas.get(_variabel_induk(k)), wajib=True)
        kosong += [INPUT[k][0] for k in wajib_tanya if nilai.get(k) is None]

    if tambahan:
        st.markdown('<div class="form-heading">Faktor lain yang bisa ditindaklanjuti (isi jika diketahui)</div>',
                    unsafe_allow_html=True)
        for kat, isi in tambahan.items():
            with st.expander(kat):
                biasa = [k for k in isi if k not in MAKANAN]
                c1, c2 = st.columns(2)
                for i, k in enumerate(biasa):
                    with (c1 if i % 2 == 0 else c2):
                        nilai[k] = _widget(k, jelas.get(_variabel_induk(k)))
                makan = [k for k in MAKANAN if k in isi]
                if makan:
                    st.caption("Berapa hari anak makan/minum makanan berikut dalam 7 hari terakhir? (0-7)")
                    c1, c2 = st.columns(2)
                    for i, k in enumerate(makan):
                        with (c1 if i % 2 == 0 else c2):
                            nilai[k] = _widget(k)
    return nilai, kosong