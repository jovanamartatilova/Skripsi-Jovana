"""
5_FEATUREIMPORTANCE.py  -  Feature importance & pemilihan fitur untuk pemodelan ulang
(Proposal Skripsi Bab III, Subbab 3.1.10; bimbingan 08/11/2026 butir 1, 3, 4)

Alur penelitian:
  4_PEMODELAN.py (Model A/B) -> SKRIP INI -> 2_PREPROCESSING.py & 4_PEMODELAN.py --fitur-file (Model C/D)

Data: data latih ASLI (tanpa data sintetis) hasil 3_SPLITTING.py, dibagi per keluarga lewat kolom fold_cv.
Data uji tidak dipakai di skrip ini.

Yang dihitung (model = algoritma skenario terpilih di tahap 4, dilatih ulang pada SEMUA kandidat fitur per fold):
  1. Permutation importance per VARIABEL ASAL pada fold validasi (satu variabel diacak sebelum preprocessing, jadi
     dummy one-hot ikut teracak bersama) -> penurunan ROC-AUC, diulang --repeats kali per fold. METODE UTAMA.
  2. Importance bawaan model (RF: Gini/MDI, LR: |koefisien|), dijumlahkan per variabel asal. Pendukung.
  3. Arah hubungan menurut model (dari probabilitas OOF), untuk penjelasan di dashboard. Bukan sebab-akibat.

Aturan pemilihan (dapat diubah lewat argumen):
  - Variabel dikelompokkan: DAPAT DITINDAKLANJUTI (kader dapat melakukan sesuatu), KONTROL (umur, jenis kelamin, WAZ:
    tetap di model, tidak ditampilkan sebagai faktor intervensi), FAKTOR LATAR (sisanya, tidak dapat ditindaklanjuti).
  - Kontribusi kumulatif --batas persen dihitung DI DALAM kelompok dapat-ditindaklanjuti (bukan dari total).
    Alasan: bila dihitung dari total, daftar didominasi ukuran tubuh anak (WAZ) dan variabel latar.
  - Variabel dipilih bila kontribusinya nyata (rata-rata penurunan AUC > 2 x galat baku) dan belum melewati batas
    kumulatif; minimal --min-var variabel. Tambahkan --tanpa-uji untuk memilih murni dari kumulatif.
  - Fitur pemodelan ulang = KONTROL + variabel dapat-ditindaklanjuti terpilih  -> fitur_tahap3.txt

Input  : train.csv (+fold_cv) dari 3_SPLITTING.py dan model_final.joblib dari 4_PEMODELAN.py
         custom_transformer.py (satu folder dengan skrip ini)
Output (folder 5_FeatureImportance):
    Feature_Importance.xlsx, fitur_tahap3.txt, importance_dashboard.csv, fi_peringkat.png, fi_kumulatif.png,
    prediksi_risiko.py + jelaskan_risiko.py (untuk dashboard), laporan_fi.txt
Cara jalan : python 5_FEATUREIMPORTANCE.py
             python 5_FEATUREIMPORTANCE.py --batas 80 --min-var 5 --repeats 5
Kebutuhan  : pandas numpy scikit-learn matplotlib openpyxl joblib
"""

import argparse
import sys
import re
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # supaya custom_transformer.py ketemu saat memuat model

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9})

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
SPLIT = BASE / "output" / "3_Splitting"
MODEL = BASE / "output" / "4_Modeling_B"

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=str(SPLIT / "train.csv"))
ap.add_argument("--model", default=str(MODEL))
ap.add_argument("--out", default=str(BASE / "output" / "5_FeatureImportance_B"))
ap.add_argument("--repeats", type=int, default=5, help="ulangan permutasi per variabel per fold")
ap.add_argument("--batas", type=float, default=80.0, help="batas kontribusi kumulatif (%%) pada variabel dapat-ditindaklanjuti")
ap.add_argument("--min-var", type=int, default=5, help="jumlah minimal variabel dapat-ditindaklanjuti yang dipilih")
ap.add_argument("--tanpa-uji", action="store_true", help="pilih murni dari kumulatif (abaikan syarat kontribusi nyata)")
ap.add_argument(
    "--kontrol",
    nargs="*",
    default=["umur_bln", "jk_laki", "waz"],
    help="variabel kontrol: tetap di model, tidak ditampilkan sebagai faktor intervensi",
)
args = ap.parse_args()
MODEL = Path(args.model)
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET = "stunting"
JENIS_AKT = "Dapat ditindaklanjuti"
JENIS_KONTROL = "Kontrol (tetap di model)"
JENIS_LATAR = "Faktor latar (tidak dapat ditindaklanjuti)"

TINDAK_LANJUT = {
    "garam_beryodium",
    "masalah_lingkungan_rumah",
    "ventilasi_cukup",
    "dapur_satu_kamar_tidur",
    "hb_anak",
    "hb_ibu",
    "frek_timbang_6bln",
    "frek_makan_per_hari",
    "jenis_makanan_7hr",
    "total_hari_makan_7hr",
    "hari_protein_hewani_7hr",
    "hari_susu_7hr",
    "jenis_sayur_buah_7hr",
    "hari_makanan_olahan_7hr",
    "air_layak",
    "air_direbus",
    "toilet_layak",
    "limbah_kode",
    "sampah_layak",
    "ada_perokok_RT",
    "asi_pernah",
    "asi_eksklusif_proxy",
    "masih_asi",
    "usia_air_bln",
    "usia_mpasi_bln",
    "air_dini",
    "mpasi_dini",
    "posyandu_4mgg",
    "vit_a_6bln",
    "punya_kartu_imun",
    "kartu_sehat",
    "jumlah_jenis_imun",
}

LABEL = {
    "umur_bln": "Umur anak (bulan)",
    "jk_laki": "Jenis kelamin laki-laki",
    "bb_anak": "Berat badan anak (kg)",
    "tb_anak": "Tinggi badan anak (cm)",
    "waz": "Skor BB menurut umur (WAZ)",
    "bb_lahir_kg": "Berat badan lahir (kg)",
    "lama_hamil_minggu": "Lama kehamilan (minggu)",
    "anc_total": "Jumlah kunjungan ANC",
    "jml_tablet_besi": "Jumlah tablet besi",
    "tempat_lahir_grup": "Tempat melahirkan",
    "masih_asi": "Masih disusui",
    "mpasi_dini": "MPASI dini (<6 bulan)",
    "asi_eksklusif_proxy": "ASI eksklusif (perkiraan)",
    "posyandu_4mgg": "Ke Posyandu 4 minggu terakhir",
    "vit_a_6bln": "Vitamin A 6 bulan terakhir",
    "punya_kartu_imun": "Punya kartu imunisasi",
    "jumlah_jenis_imun": "Jumlah jenis imunisasi",
    "imt_ibu": "IMT ibu",
    "umur_ibu": "Umur ibu (tahun)",
    "pend_ibu": "Pendidikan ibu",
    "air_layak": "Sumber air minum layak",
    "air_direbus": "Air minum direbus",
    "toilet_layak": "Toilet layak",
    "sampah_layak": "Pembuangan sampah layak",
    "kartu_sehat": "Punya kartu kesehatan",
    "frek_makan_per_hari": "Frekuensi makan per hari",
    "jenis_makanan_7hr": "Jenis makanan 7 hari terakhir",
    "hari_sakit_4mgg": "Hari sakit 4 minggu terakhir",
    "jumlah_gejala_4mgg": "Jumlah gejala sakit 4 minggu terakhir",
    "hari_protein_hewani_7hr": "Hari makan protein hewani (7 hari)",
    "hari_susu_7hr": "Hari minum susu (7 hari)",
    "jenis_sayur_buah_7hr": "Jenis sayur dan buah (7 hari)",
    "hari_makanan_olahan_7hr": "Hari makan makanan olahan (7 hari)",
    "jumlah_anggota_RT": "Jumlah anggota rumah tangga",
    "ada_perokok_RT": "Ada perokok di rumah",
        "log_pengeluaran_perkapita": "Pengeluaran per kapita (log)",
    "frek_timbang_6bln": "Frekuensi ditimbang 6 bulan terakhir",
    "indeks_kekayaan_pca": "Indeks kekayaan rumah tangga (PCA)",
    "hb_anak": "Kadar hemoglobin anak (g/dL)",
    "hb_ibu": "Kadar hemoglobin ibu (g/dL)",
    "garam_beryodium": "Garam beryodium",
    "masalah_lingkungan_rumah": "Masalah kebersihan sekitar rumah",
    "ventilasi_cukup": "Ventilasi rumah cukup",
    "dapur_satu_kamar_tidur": "Dapur menyatu dengan kamar tidur",
    "urutan_lahir": "Urutan kelahiran anak",
    "jarak_lahir_bln": "Jarak dari kelahiran sebelumnya (bulan)",
    "anak_kembar": "Anak kembar",
    "bahan_bakar_padat": "Memasak dengan kayu bakar/arang",
    "lantai_tanah": "Lantai rumah dari tanah",
}
KATEGORI = {
    "air_layak": "1. Air, sanitasi, dan kebersihan lingkungan",
    "air_direbus": "1. Air, sanitasi, dan kebersihan lingkungan",
    "toilet_layak": "1. Air, sanitasi, dan kebersihan lingkungan",
    "sampah_layak": "1. Air, sanitasi, dan kebersihan lingkungan",
    "masalah_lingkungan_rumah": "1. Air, sanitasi, dan kebersihan lingkungan",
    "ventilasi_cukup": "2. Kondisi rumah dan paparan asap",
    "dapur_satu_kamar_tidur": "2. Kondisi rumah dan paparan asap",
    "ada_perokok_RT": "2. Kondisi rumah dan paparan asap",
    "masih_asi": "3. ASI dan MPASI",
    "asi_eksklusif_proxy": "3. ASI dan MPASI",
    "mpasi_dini": "3. ASI dan MPASI",
    "frek_makan_per_hari": "4. Asupan makan anak",
    "jenis_makanan_7hr": "4. Asupan makan anak",
    "hari_protein_hewani_7hr": "4. Asupan makan anak",
    "hari_susu_7hr": "4. Asupan makan anak",
    "hari_makanan_olahan_7hr": "4. Asupan makan anak",
    "garam_beryodium": "4. Asupan makan anak",
    "hb_anak": "5. Anemia (Hb) anak dan ibu",
    "hb_ibu": "5. Anemia (Hb) anak dan ibu",
    "frek_timbang_6bln": "6. Pemantauan pertumbuhan di Posyandu",
    "posyandu_4mgg": "6. Pemantauan pertumbuhan di Posyandu",
    "vit_a_6bln": "6. Pemantauan pertumbuhan di Posyandu",
    "punya_kartu_imun": "7. Imunisasi dan jaminan kesehatan",
    "jumlah_jenis_imun": "7. Imunisasi dan jaminan kesehatan",
    "kartu_sehat": "7. Imunisasi dan jaminan kesehatan",
}

PENJELASAN_KATEGORI = {
    "1. Air, sanitasi, dan kebersihan lingkungan": "Air dan lingkungan yang tidak bersih membuat anak sering diare dan cacingan. Infeksi berulang mengganggu penyerapan zat gizi di usus sehingga pertumbuhan tinggi badan terhambat.",
    "2. Kondisi rumah dan paparan asap": "Asap dapur, asap rokok, dan udara yang tidak berganti membuat anak mudah terkena ISPA. Anak yang sering sakit nafsu makannya turun dan energinya dipakai untuk melawan infeksi, bukan untuk tumbuh.",
    "3. ASI dan MPASI": "ASI memberi zat gizi dan antibodi. Pemberian MPASI yang terlalu dini atau ASI yang dihentikan terlalu cepat meningkatkan risiko infeksi dan kekurangan gizi pada masa awal pertumbuhan.",
    "4. Asupan makan anak": "Pertumbuhan tinggi badan membutuhkan energi, protein, dan zat gizi mikro (zat besi, seng, kalsium, yodium) yang cukup. Makan yang kurang sering, kurang beragam, atau didominasi makanan olahan membuat kebutuhan ini tidak terpenuhi.",
    "5. Anemia (Hb) anak dan ibu": "Kadar Hb rendah menandakan kekurangan zat besi dan gizi secara umum. Anemia dan stunting sering terjadi bersamaan karena penyebabnya mirip, yaitu asupan kurang dan infeksi berulang.",
    "6. Pemantauan pertumbuhan di Posyandu": "Anak yang rutin ditimbang dan datang ke Posyandu lebih cepat ketahuan jika pertumbuhannya melambat, dan keluarganya mendapat penyuluhan gizi, vitamin A, serta PMT.",
    "7. Imunisasi dan jaminan kesehatan": "Imunisasi mencegah penyakit infeksi yang dapat menghambat pertumbuhan, sedangkan jaminan kesehatan memudahkan keluarga berobat sehingga sakit anak tidak berlarut-larut.",
}

PENJELASAN = {
    "air_layak": "Air minum dari sumber tidak layak mudah tercemar kuman sehingga anak lebih sering diare. Diare berulang mengurangi penyerapan zat gizi dan menghambat pertumbuhan.",
    "air_direbus": "Merebus air membunuh kuman penyebab diare. Air yang tidak direbus meningkatkan risiko infeksi saluran cerna pada anak.",
    "toilet_layak": "Toilet tanpa septic tank mencemari tanah dan air di sekitar rumah dengan tinja, sehingga anak mudah terkena diare dan cacingan yang mengganggu penyerapan gizi.",
    "sampah_layak": "Sampah yang dibuang sembarangan menjadi tempat berkembangnya lalat, tikus, dan kuman penyebab penyakit. Variabel ini juga mencerminkan perilaku hidup bersih rumah tangga.",
    "masalah_lingkungan_rumah": "Kotoran, tumpukan sampah, dan air tergenang di sekitar rumah menjadi sumber kuman dan sarang nyamuk. Makin banyak masalah, makin sering anak terpapar infeksi.",
    "ventilasi_cukup": "Ventilasi yang kurang membuat udara lembap dan asap dapur terperangkap di dalam rumah, sehingga anak lebih mudah terkena ISPA.",
    "dapur_satu_kamar_tidur": "Dapur yang menyatu dengan kamar tidur membuat anak menghirup asap memasak lebih lama, sehingga risiko ISPA meningkat.",
    "ada_perokok_RT": "Asap rokok meningkatkan risiko ISPA pada anak. Uang yang dipakai untuk rokok juga dapat mengurangi belanja pangan bergizi.",
    "masih_asi": "ASI tetap menjadi sumber gizi dan antibodi sampai usia 2 tahun bersama MPASI. Karena anak yang masih menyusu umumnya lebih muda, variabel ini dibaca bersama umur anak.",
    "asi_eksklusif_proxy": "ASI eksklusif selama 6 bulan memenuhi kebutuhan gizi awal bayi dan melindunginya dari infeksi.",
    "mpasi_dini": "MPASI yang diberikan sebelum 6 bulan meningkatkan risiko diare karena usus bayi belum siap, dan mengurangi asupan ASI.",
    "frek_makan_per_hari": "Frekuensi makan menentukan jumlah energi dan protein yang diterima anak. Kurang makan dalam waktu lama menyebabkan gagal tumbuh.",
    "jenis_makanan_7hr": "Makanan yang beragam mencerminkan kecukupan zat gizi mikro seperti zat besi, seng, dan vitamin A. Menu yang monoton berisiko kekurangan zat gizi untuk pertumbuhan.",
    "hari_protein_hewani_7hr": "Telur, ikan, dan daging mengandung protein bermutu tinggi, zat besi, dan seng yang penting untuk pertumbuhan tinggi badan.",
    "hari_susu_7hr": "Susu merupakan sumber protein dan kalsium yang mendukung pertumbuhan tulang.",
    "hari_makanan_olahan_7hr": "Mi instan, makanan cepat saji, minuman bersoda, gorengan, dan sambal umumnya rendah protein dan zat gizi mikro. Jika sering dimakan, makanan ini dapat menggantikan makanan bergizi sehingga anak kenyang tetapi kurang gizi.",
    "garam_beryodium": "Yodium dibutuhkan untuk hormon tiroid yang mengatur pertumbuhan. Kekurangan yodium dapat menghambat pertumbuhan dan perkembangan anak.",
    "hb_anak": "Hb rendah (anemia) menandakan kekurangan zat besi dan gizi. Anemia dan stunting sering muncul bersamaan karena penyebabnya mirip, yaitu asupan kurang dan infeksi.",
    "hb_ibu": "Ibu yang anemia berisiko melahirkan bayi kecil dengan cadangan zat besi rendah. Hb ibu juga mencerminkan kondisi gizi keluarga.",
    "frek_timbang_6bln": "Penimbangan rutin membuat gangguan pertumbuhan cepat terdeteksi dan segera ditangani. Variabel ini juga menunjukkan keaktifan keluarga memantau pertumbuhan anak.",
    "posyandu_4mgg": "Kunjungan ke Posyandu memberi akses ke pemantauan pertumbuhan, penyuluhan gizi, PMT, dan imunisasi.",
    "vit_a_6bln": "Vitamin A menjaga daya tahan tubuh dan mengurangi keparahan infeksi seperti diare dan campak. Variabel ini juga menjadi penanda keterjangkauan layanan Posyandu.",
    "punya_kartu_imun": "Kartu imunisasi atau buku KIA menunjukkan bahwa keluarga mencatat dan memantau imunisasi serta pertumbuhan anak.",
    "jumlah_jenis_imun": "Imunisasi yang lengkap mencegah penyakit infeksi (misalnya campak, TBC, dan diare rotavirus) yang dapat menghambat pertumbuhan.",
    "kartu_sehat": "Jaminan kesehatan (BPJS/KIS) memudahkan keluarga membawa anak berobat sehingga sakitnya tidak berlarut, serta memudahkan ibu memeriksakan kehamilan.",
}
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def label(v):
    return LABEL.get(v, v.replace("_", " "))


def jenis(v):
    if v in args.kontrol:
        return JENIS_KONTROL
    return JENIS_AKT if v in TINDAK_LANJUT else JENIS_LATAR


def tulis_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for nm, d in sheets.items():
            d.to_excel(xw, sheet_name=nm[:31], index=False)
            ws = xw.sheets[nm[:31]]
            for col in ws.columns:
                w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 60)
            ws.freeze_panes = "A2"


# ----------------------------------------------------------------------------
# HELPER DASHBOARD (identik dengan versi tahap 4; jelaskan_risiko = penjelasan per balita)
# ----------------------------------------------------------------------------
HELPER_PREDIKSI = '''# -*- coding: utf-8 -*-
"""Fungsi bantu dashboard: memuat model final dan menghitung probabilitas + kategori risiko stunting.
Pemakaian:
    import joblib, pandas as pd
    from prediksi_risiko import prediksi_risiko
    b = joblib.load("model_final.joblib")
    hasil = prediksi_risiko(b, pd.DataFrame([{...nilai predictor...}]))   # kolom yang tidak diisi = NaN (diimputasi)
Model & preprocessor TIDAK dilatih ulang (proposal 3.1.11.2)."""
import numpy as np
import pandas as pd


def prediksi_risiko(bundle, df_input):
    num, kat = bundle["fitur_numerik"], bundle["fitur_kategorik"]
    d = df_input.reindex(columns=num + kat).copy()
    for c in num + kat:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c, (lo, hi) in bundle["rentang"].items():
        if c in d.columns:
            d.loc[d[c].notna() & ~d[c].between(lo, hi), c] = np.nan
    Z = np.asarray(bundle["preprocessor"].transform(d), float)[:, bundle["indeks_terpilih"]]
    p = bundle["model"].predict_proba(Z)[:, 1]
    return pd.DataFrame({"probabilitas_stunting": p,
                         "kategori_risiko": np.where(p >= bundle["threshold"], "Risiko Lebih Tinggi", "Risiko Lebih Rendah")})
'''

HELPER_JELASKAN = '''# -*- coding: utf-8 -*-
"""Penjelasan per balita untuk dashboard (pendamping prediksi_risiko.py; letakkan di folder yang sama).
Metode oklusi: tiap predictor diganti nilai kosong (NaN) -> preprocessor mengimputasi dengan median/modus TRAIN,
lalu dihitung selisih probabilitas. kontribusi > 0 = nilai balita ini MENAIKKAN risiko dibanding balita tipikal.
Ini penjelasan asosiatif, bukan sebab-akibat.
Pemakaian:
    b = joblib.load("model_final.joblib")
    p, tabel = jelaskan_risiko(b, pd.DataFrame([{...}]), top=5)
"""
import numpy as np
import pandas as pd
from prediksi_risiko import prediksi_risiko


def jelaskan_risiko(bundle, df_input, top=5):
    kat = bundle["fitur_kategorik"]
    pakai = []
    for c in bundle["kolom_terpilih"]:
        v = next((k for k in kat if c.startswith(k + "_")), c)
        if v not in pakai:
            pakai.append(v)
    d0 = df_input.iloc[[0]].copy()
    p0 = float(prediksi_risiko(bundle, d0)["probabilitas_stunting"].iloc[0])
    baris = []
    for v in pakai:
        d1 = d0.copy()
        d1[v] = np.nan
        p1 = float(prediksi_risiko(bundle, d1)["probabilitas_stunting"].iloc[0])
        baris.append({"variabel": v, "nilai_input": d0[v].iloc[0] if v in d0.columns else np.nan, "kontribusi": p0 - p1})
    t = pd.DataFrame(baris).sort_values("kontribusi", key=abs, ascending=False).head(top).reset_index(drop=True)
    t["arah"] = np.where(t.kontribusi > 0, "menaikkan risiko", "menurunkan risiko")
    return p0, t
'''
(OUT / "prediksi_risiko.py").write_text(HELPER_PREDIKSI, encoding="utf-8")
(OUT / "jelaskan_risiko.py").write_text(HELPER_JELASKAN, encoding="utf-8")

# ============================================================================
# MUAT MODEL & DATA TRAIN
# ============================================================================
log("=== FEATURE IMPORTANCE IFLS5 (Proposal 3.1.10) ===")
bundle = joblib.load(MODEL / "model_final.joblib")
num, kat = bundle["fitur_numerik"], bundle["fitur_kategorik"]
kandidat = num + kat
tr = pd.read_csv(args.train).reset_index(drop=True)
assert "fold_cv" in tr.columns, "train.csv tidak punya kolom fold_cv; jalankan 3_Splitting.py"
y = tr[TARGET].values.astype(int)
fold_arr = tr.fold_cv.values
FOLDS = sorted(np.unique(fold_arr))
log(
    f"[data] train {tr.shape} | stunting {y.mean() * 100:.2f}% | fold CV {len(FOLDS)} | "
    f"model tahap 4: {bundle['skenario']} ({bundle['algoritma']}) | {len(kandidat)} kandidat variabel"
)
log("  Pemilihan fitur memakai data TRAIN saja (CV); data TEST tidak dipakai di skrip ini.")
tdk = [c for c in args.kontrol if c not in kandidat]
if tdk:
    log(f"  [peringatan] variabel kontrol tidak ada di kandidat fitur dan diabaikan: {', '.join(tdk)}")


def seragamkan(d):
    d = d[kandidat].copy()
    for c in kandidat:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for c, (lo, hi) in bundle["rentang"].items():
        if c in d.columns:
            d.loc[d[c].notna() & ~d[c].between(lo, hi), c] = np.nan
    return d


def variabel_asal(kol):
    for c in kat:
        if kol.startswith(c + "_"):
            return c
    kol = re.sub(r"^missingindicator_", "", kol)
    return re.sub(r"_sp_\d+$", "", kol)


X = seragamkan(tr)

# ============================================================================
# CV: importance bawaan + permutasi pada fold validasi
# ============================================================================
rng = np.random.default_rng(SEED)
oof = np.full(len(tr), np.nan)
bawaan_rows, perm_rows, fold_rows = [], [], []
log(f"\n[1] Permutasi per variabel ({len(kandidat)} variabel x {args.repeats} ulangan x {len(FOLDS)} fold) ...")
for k in FOLDS:
    i_tr, i_va = np.where(fold_arr != k)[0], np.where(fold_arr == k)[0]
    pre = bundle["preprocessor"]  # sudah di-fit di awal (preprocessing dulu)
    Zt = np.asarray(pre.transform(X.iloc[i_tr]), float)
    m = clone(bundle["model"]).fit(Zt, y[i_tr])  # semua kandidat fitur
    nama = [n.replace("num__", "").replace("kat__", "").replace("spl__", "") for n in pre.get_feature_names_out()]
    if hasattr(m, "feature_importances_"):
        imp = np.asarray(m.feature_importances_, float)
    elif hasattr(m, "coef_"):
        imp = np.abs(m.coef_[0])
    else:
        imp = np.ones(Zt.shape[1])  # model tanpa importance bawaan; acuan tetap permutasi
    b = pd.DataFrame({"variabel": [variabel_asal(c) for c in nama], "imp": imp}).groupby("variabel")["imp"].sum()
    b = b / b.sum() * 100
    for v, val in b.items():
        bawaan_rows.append({"fold": int(k), "variabel": v, "bawaan_pct": val})

    Xv, yv = X.iloc[i_va].copy(), y[i_va]
    p0 = m.predict_proba(np.asarray(pre.transform(Xv), float))[:, 1]
    oof[i_va] = p0
    auc0 = roc_auc_score(yv, p0)
    fold_rows.append({"fold": int(k), "n_training": len(i_tr), "n_validasi": len(i_va), "AUC_dasar_validasi": auc0})
    for v in kandidat:
        asli = Xv[v].values.copy()
        for r in range(args.repeats):
            Xv[v] = rng.permutation(asli)
            p = m.predict_proba(np.asarray(pre.transform(Xv), float))[:, 1]
            perm_rows.append({"fold": int(k), "variabel": v, "ulangan": r, "turun_AUC": auc0 - roc_auc_score(yv, p)})
        Xv[v] = asli
    log(f"    fold {int(k)}: n validasi {len(i_va)}, AUC dasar {auc0:.4f}")

fold_tab = pd.DataFrame(fold_rows)
perm = pd.DataFrame(perm_rows)
agg = perm.groupby("variabel")["turun_AUC"].agg(["mean", "std", "count"])
agg.columns = ["turun_AUC_rata2", "turun_AUC_SD", "n"]
agg["galat_baku"] = agg.turun_AUC_SD / np.sqrt(agg.n)
agg["nyata"] = np.where(agg.turun_AUC_rata2 - 2 * agg.galat_baku > 0, "ya", "tidak")
bw = pd.DataFrame(bawaan_rows).groupby("variabel")["bawaan_pct"].mean().rename("bawaan_pct")
tab = agg.join(bw).reset_index()
pos = tab.turun_AUC_rata2.clip(lower=0)
tab["kontribusi_pct_total"] = pos / pos.sum() * 100 if pos.sum() > 0 else 0.0
tab["jenis"] = tab.variabel.map(jenis)
tab["label"] = tab.variabel.map(label)
tab = tab.sort_values("turun_AUC_rata2", ascending=False).reset_index(drop=True)
tab.insert(0, "peringkat", range(1, len(tab) + 1))
rho = tab.bawaan_pct.corr(tab.turun_AUC_rata2, method="spearman")
log(
    f"\n[2] Peringkat teratas (permutasi): {', '.join(tab.variabel.head(5))} | "
    f"konsistensi bawaan vs permutasi (Spearman) {rho:.2f}"
    + ("  -> berbeda; permutasi dijadikan acuan (lebih andal pada fitur berkorelasi)" if rho < 0.6 else "")
)

# ============================================================================
# PEMILIHAN: kumulatif di dalam kelompok dapat-ditindaklanjuti
# ============================================================================
akt = tab[tab.jenis == JENIS_AKT].copy().sort_values("turun_AUC_rata2", ascending=False)
pa = akt.turun_AUC_rata2.clip(lower=0)
akt["porsi_pct"] = pa / pa.sum() * 100 if pa.sum() > 0 else 0.0
akt["kumulatif_pct"] = akt.porsi_pct.cumsum()
pilih, cum_prev = [], 0.0
for r in akt.itertuples():
    if cum_prev >= args.batas and len(pilih) >= args.min_var:
        break
    if args.tanpa_uji or r.nyata == "ya":
        pilih.append(r.variabel)
    cum_prev = r.kumulatif_pct
akt["dipilih"] = np.where(akt.variabel.isin(pilih), "ya", "tidak")
sens = []  # sensitivitas: berapa variabel terpilih bila batas kumulatif diubah
for b_ in (70, 80, 90):
    n_b, cp = 0, 0.0
    for r in akt.itertuples():
        if cp >= b_ and n_b >= args.min_var:
            break
        if args.tanpa_uji or r.nyata == "ya":
            n_b += 1
        cp = r.kumulatif_pct
    sens.append({"batas_kumulatif_pct": b_, "n_variabel_terpilih": n_b})
sens = pd.DataFrame(sens)
if len(pilih) < args.min_var:
    log(f"  [peringatan] hanya {len(pilih)} variabel dapat-ditindaklanjuti yang nyata (< min-var {args.min_var})")
kontrol = [c for c in args.kontrol if c in kandidat]
fitur_model = list(dict.fromkeys(kontrol + pilih))
cum_pilih = float(akt.loc[akt.variabel.isin(pilih), "porsi_pct"].sum())
log(
    f"\n[3] Variabel dapat-ditindaklanjuti: {len(akt)} kandidat -> {len(pilih)} dipilih "
    f"(kumulatif {cum_pilih:.1f}% dari kontribusi kelompok ini; batas {args.batas:.0f}%)"
)
log("    Sensitivitas batas: " + "; ".join(f"{r.batas_kumulatif_pct}% -> {r.n_variabel_terpilih} variabel" for r in sens.itertuples()))
log("    Dipilih: " + ", ".join(pilih))
log("    Kontrol: " + ", ".join(kontrol))
latar = tab[tab.jenis == JENIS_LATAR].head(5)
log(
    "    Faktor latar berkontribusi terbesar (TIDAK ikut model; masukkan ke --kontrol bila ingin dipertahankan): "
    + ", ".join(f"{r.variabel} ({r.kontribusi_pct_total:.1f}%)" for r in latar.itertuples())
)
log(f"    Fitur pemodelan ulang: {len(fitur_model)} variabel -> fitur_tahap3.txt")

# ============================================================================
# ARAH HUBUNGAN (dari probabilitas OOF; asosiatif)
# ============================================================================
arah_rows = []
for v in kandidat:
    s = X[v]
    ok = s.notna().values
    if v in kat:
        g = pd.DataFrame({"k": s[ok].values, "p": oof[ok]}).groupby("k")["p"].agg(["size", "mean"])
        arah_rows.append(
            {
                "variabel": v,
                "spearman_dengan_prob": np.nan,
                "selisih_prob_1_vs_0": np.nan,
                "arah": "nominal (lihat rata-rata per kategori)",
                "rata2_prob_per_kategori": "; ".join(f"{i:g}:{r['mean']:.2f}(n={int(r['size'])})" for i, r in g.iterrows() if r["size"] >= 20)[:300],
            }
        )
        continue
    sv = s[ok].values
    rho_v = pd.Series(sv).corr(pd.Series(oof[ok]), method="spearman")
    if set(np.unique(sv)) <= {0.0, 1.0}:
        d1 = float(oof[ok][sv == 1].mean() - oof[ok][sv == 0].mean())
        txt = "bernilai 1 -> risiko naik" if d1 > 0.01 else ("bernilai 1 -> risiko turun" if d1 < -0.01 else "hubungan lemah")
    else:
        d1 = np.nan
        txt = "nilai naik -> risiko naik" if rho_v > 0.05 else ("nilai naik -> risiko turun" if rho_v < -0.05 else "hubungan lemah/tidak monoton")
    arah_rows.append({"variabel": v, "spearman_dengan_prob": rho_v, "selisih_prob_1_vs_0": d1, "arah": txt, "rata2_prob_per_kategori": ""})
arah = pd.DataFrame(arah_rows)

# ============================================================================
# TABEL DASHBOARD
# ============================================================================
# kategori dan penjelasan (hanya label; perhitungan importance dan pemilihan tidak berubah)
akt["kategori"] = akt.variabel.map(KATEGORI).fillna("Lainnya")
akt["penjelasan"] = akt.variabel.map(PENJELASAN).fillna("")
akt["peringkat_kelompok"] = range(1, len(akt) + 1)
tanpa_kat = akt.loc[akt.kategori == "Lainnya", "variabel"].tolist()
if tanpa_kat:
    log("  [peringatan] belum punya kategori di KATEGORI: " + ", ".join(tanpa_kat))
ring_kat = (
    akt.groupby("kategori", sort=False)
    .agg(
        jumlah_variabel=("variabel", "size"),
        porsi_pct=("porsi_pct", "sum"),
        jumlah_nyata=("nyata", lambda s: int((s == "ya").sum())),
        jumlah_dipilih=("dipilih", lambda s: int((s == "ya").sum())),
        variabel=("variabel", ", ".join),
    )
    .sort_values("porsi_pct", ascending=False)
    .reset_index()
)
ring_kat.insert(0, "peringkat_kategori", range(1, len(ring_kat) + 1))
ring_kat["porsi_pct"] = ring_kat.porsi_pct.round(2)
ring_kat["penjelasan_kategori"] = ring_kat.kategori.map(PENJELASAN_KATEGORI).fillna("")
urut_kat = {k: i for i, k in enumerate(ring_kat.kategori)}
per_kat = (
    akt.merge(arah[["variabel", "arah"]], on="variabel", how="left")
    .assign(_u=lambda d: d.kategori.map(urut_kat))
    .sort_values(["_u", "turun_AUC_rata2"], ascending=[True, False])
    [["kategori", "peringkat_kelompok", "variabel", "label", "turun_AUC_rata2", "porsi_pct", "nyata", "dipilih", "arah", "penjelasan"]]
)
log("    Per kategori: " + "; ".join(f"{r.kategori} {r.porsi_pct:.1f}%" for r in ring_kat.itertuples()))

dash = akt[["variabel", "label", "kategori", "turun_AUC_rata2", "porsi_pct", "kumulatif_pct", "nyata", "dipilih"]].copy()
dash = dash.merge(arah[["variabel", "arah"]], on="variabel", how="left")
dash.insert(0, "peringkat", range(1, len(dash) + 1))
dash["porsi_pct"] = dash.porsi_pct.round(2)
dash["kumulatif_pct"] = dash.kumulatif_pct.round(2)
dash["turun_AUC_rata2"] = dash.turun_AUC_rata2.round(5)
dash["penjelasan_dashboard"] = dash.variabel.map(PENJELASAN).fillna("")
dash.to_csv(OUT / "importance_dashboard.csv", index=False, encoding="utf-8-sig")
(OUT / "fitur_tahap3.txt").write_text("\n".join(fitur_model) + "\n", encoding="utf-8")

# ============================================================================
# SIMPAN EXCEL
# ============================================================================
ringkas = pd.DataFrame(
    [
        ["Model acuan", f"{bundle['skenario']}: {bundle['algoritma']} (semua kandidat fitur)"],
        ["Data", "TRAIN saja (CV per fold_cv); test tidak dipakai"],
        ["Metode utama", f"Permutation importance (penurunan ROC-AUC), {args.repeats} ulangan x {len(FOLDS)} fold"],
        ["Kandidat variabel", len(kandidat)],
        ["Dapat ditindaklanjuti / dipilih", f"{len(akt)} / {len(pilih)}"],
        ["Batas kumulatif / minimal variabel", f"{args.batas:.0f}% / {args.min_var}"],
        ["Kumulatif variabel terpilih (%)", round(cum_pilih, 1)],
        ["Variabel kontrol", ", ".join(kontrol)],
        ["Fitur pemodelan ulang", ", ".join(fitur_model)],
        ["Konsistensi bawaan vs permutasi (Spearman)", round(float(rho), 2)],
        ["Catatan", "Importance = kontribusi relatif terhadap prediksi, BUKAN hubungan sebab-akibat (Subbab 2.6.6)"],
    ],
    columns=["indikator", "nilai"],
)
tulis_excel(
    OUT / "Feature_Importance.xlsx",
    {
        "00_Ringkasan": ringkas,
        "01_Semua_Variabel": tab.round(5),
        "02_Dapat_Ditindaklanjuti": akt.round(5),
        "02b_Per_Kategori": per_kat.round(5),
        "02c_Ringkasan_Kategori": ring_kat,
        "03_Faktor_Latar": tab[tab.jenis == JENIS_LATAR].round(5),
        "04_Fitur_Tahap3": pd.DataFrame({"no": range(1, len(fitur_model) + 1), "variabel": fitur_model, "label": [label(v) for v in fitur_model], "jenis": [jenis(v) for v in fitur_model]}),
        "05_Arah_Hubungan": arah.round(4),
        "06_Per_Fold": fold_tab.round(4),
        "07_Dashboard": dash,
        "08_Sensitivitas_Batas": sens,
    },
)

# ============================================================================
# GRAFIK
# ============================================================================
warna = {JENIS_AKT: "#F58518", JENIS_KONTROL: "#4C78A8", JENIS_LATAR: "#BAB0AC"}
top = tab.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.barh(top.variabel, top.turun_AUC_rata2, xerr=2 * top.galat_baku, color=[warna[j] for j in top.jenis])
ax.set(title="Permutation importance (penurunan AUC, CV pada train; +-2 galat baku)", xlabel="penurunan ROC-AUC")

ax.legend(handles=[Patch(color=c, label=j) for j, c in warna.items()], loc="lower right", fontsize=7)
fig.savefig(OUT / "fi_peringkat.png")
plt.close(fig)

if len(akt):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(akt))
    ax.bar(xs, akt.porsi_pct, color=np.where(akt.dipilih == "ya", "#F58518", "#E5E5E5"), edgecolor="gray")
    ax.set(xticks=xs, ylabel="porsi kontribusi (%)", title=f"Variabel dapat ditindaklanjuti: porsi dan kumulatif (batas {args.batas:.0f}%)")
    ax.set_xticklabels(akt.variabel, rotation=60, ha="right", fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(xs, akt.kumulatif_pct, "o-", color="#4C78A8", ms=3)
    ax2.axhline(args.batas, color="red", ls="--", lw=0.8)
    ax2.set(ylabel="kumulatif (%)", ylim=(0, 105))
    fig.savefig(OUT / "fi_kumulatif.png")
    plt.close(fig)

log(f"\nHasil disimpan di: {OUT}")
log("  Feature_Importance.xlsx, fitur_tahap3.txt, importance_dashboard.csv, fi_peringkat.png, fi_kumulatif.png,")
log("  prediksi_risiko.py + jelaskan_risiko.py (salin keduanya ke folder dashboard)")
log("LANGKAH BERIKUTNYA: python 2_PREPROCESSING.py --fitur-file <fitur_tahap3.txt> --out <folder> lalu "
    "python 4_PEMODELAN.py --protokol-paper --fitur-file <fitur_tahap3.txt> --pre <preprocessor> --out <folder> --kuota 0 --k <n>")
log("CATATAN: importance = kontribusi relatif terhadap prediksi model, BUKAN bukti sebab-akibat (proposal 2.6.6).")
(OUT / "laporan_fi.txt").write_text("\n".join(LOG), encoding="utf-8")
print("Selesai.")