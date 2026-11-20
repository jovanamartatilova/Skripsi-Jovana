"""
Pemetaan variabel paper rujukan -> variabel IFLS5 (sumber: kamus_variabel_IFLS5.xlsx)
-------------------------------------------------------------------------------------
Input : kamus_variabel_IFLS5.xlsx (hasil KamusVariabel.py), 1_EDA\EDA_IFLS5.xlsx (hasil 1_EDA.py)
Output: pemetaan_variabel_paper_ke_IFLS5.xlsx
          - Pemetaan           : konsep paper -> file & nama variabel IFLS5 (label & % terisi
                                 diambil dari kamus, bukan ditulis tangan)
          - Tidak_ada_langsung : konsep paper yang tidak punya variabel langsung di IFLS5
          - Fitur_Kandidat : setiap fitur kandidat -> asal file/variabel IFLS5, sumber acuan, status di model

Aturan: nama variabel di daftar PEMETAAN di bawah dipilih dengan melihat kamus.
Skrip ini MEMVERIFIKASI setiap (file, variabel) ada di kamus; kalau tidak ada, skrip berhenti
dengan pesan error (tidak ada nama yang dibiarkan tertebak).

Paper rujukan:
  Agushybana  = Agushybana et al. 2025, Critical Public Health (IFLS 2 & 5)
  Siramaneerat= Siramaneerat et al. 2024, BMC Public Health (IFLS5, urban-rural)
  Novalina    = Novalina et al. 2025, Bulletin EEI (benchmarking ML, IFLS5)
  Azriani     = Azriani et al. 2025, BMC Pediatrics (prediksi stunting, IFLS)
"""

import os
import sys

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# ====================== PENGATURAN (sudah terisi) ======================
FOLDER_OUT = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output"
KAMUS_XLSX = os.path.join(FOLDER_OUT, "kamus_variabel_IFLS5.xlsx")
EDA_XLSX = os.path.join(FOLDER_OUT, "1_EDA", "EDA_IFLS5.xlsx")
OUT_XLSX = os.path.join(FOLDER_OUT, "pemetaan_variabel_paper_ke_IFLS5.xlsx")
# =======================================================================
# Sumber acuan:
#   nama paper  = variabel dipakai di bagian metode paper tersebut
#   "pendahuluan <paper>" = hanya disebut sebagai faktor risiko di pendahuluan/pembahasan, bukan variabel paper
#   WHO         = kerangka faktor risiko stunting WHO di Subbab 2.3.1 (Kiik & Nuwa, 2024)
WHO = "Kerangka WHO (Kiik & Nuwa, 2024)"

# (kelompok, konsep, sumber acuan, file, variabel IFLS5, fitur di balita_ifls5.csv, catatan)
PEMETAAN = [
    ("Kunci", "ID rumah tangga", "semua", "b5_cov.dta", "hhid14", "-", "Kunci gabung RT; dipakai untuk split per keluarga (feature importance)"),
    ("Kunci", "ID individu", "semua", "b5_cov.dta", "pidlink", "-", "Kunci gabung antar file"),
    # ---------------- Anak ----------------
    ("Anak", "Umur anak (bulan)", "semua", "bk_ar1.dta", "ar08yr", "umur_bln", "Dihitung dari tanggal lahir (ar08yr/mth/day) dan tanggal wawancara"),
    ("Anak", "Umur anak (tahun, roster)", "semua", "bk_ar1.dta", "ar09", "-", "Tidak dipakai; umur dalam bulan lebih tepat"),
    ("Anak", "Tanggal lahir anak (Book 5)", "Azriani, Siramaneerat", "b5_cov.dta", "dob_yr", "-", "Tidak dipakai; tanggal lahir diambil dari bk_ar1"),
    ("Anak", "Jenis kelamin", "semua", "bk_ar1.dta", "ar07", "jk_laki", "1 = laki-laki, 0 = perempuan"),
    ("Anak", "Tinggi badan anak (cm) -> dasar HAZ", "semua", "bus_us.dta", "us04", "tb_anak", "Dasar label HAZ; us04x = apakah terukur"),
    ("Anak", "Berat badan anak (kg)", "Novalina", "bus_us.dta", "us06", "bb_anak, waz", "WAZ dihitung dari BB, umur, JK (WHO 2006)"),
    ("Anak", "Cara ukur (berdiri/berbaring)", "WHO 2006 (koreksi 0,7 cm)", "bus_us.dta", "us05", "-", "Untuk koreksi TB sebelum hitung HAZ"),
    ("Anak", "Kadar hemoglobin anak & ibu (g/dL)", WHO + "; pendahuluan Agushybana (anemia)", "bus_us.dta", "us13", "hb_anak, hb_ibu", "us13x = apakah terukur; anak < 1 tahun tidak diukur"),
    # ---------------- Kelahiran & kehamilan ----------------
    ("Lahir", "Berat lahir (kg)", "Azriani, Siramaneerat, Agushybana, Novalina", "b4_ch1.dta", "ch24", "bb_lahir_kg, bblr", "Hanya terisi untuk anak yang tercatat di riwayat kehamilan ibu"),
    ("Lahir", "Ditimbang setelah lahir?", "-", "b4_ch1.dta", "ch23", "-", "Membedakan 'tidak ditimbang' vs 'tidak tahu'"),
    ("Lahir", "Ukuran bayi relatif (persepsi ibu)", "Azriani (infant body size)", "b4_ch1.dta", "ch22", "ukuran_bayi", ""),
    ("Lahir", "Kelahiran kembar", "Azriani (category of babies)", "b4_ch1.dta", "ch06a", "anak_kembar", ""),
    ("Lahir", "Urutan & jarak kelahiran", "Azriani (parity); Siramaneerat, Agushybana (number of children)", "b4_ch1.dta", "ch05", "urutan_lahir, jarak_lahir_bln", "Dihitung dari urutan kehamilan (ch05) dan tanggal lahir (ch09yr/mth)"),
    ("Lahir", "Lama kehamilan", "Azriani (gestational age); Novalina (premature)", "b4_ch1.dta", "ch17", "lama_hamil_minggu, hamil_prematur_proxy", "Satuan lihat ch17x; tidak ada variabel 'prematur' langsung"),
    ("Lahir", "Tempat melahirkan", "Azriani, Siramaneerat, Agushybana", "b4_ch1.dta", "ch19", "tempat_lahir_grup", ""),
    ("Lahir", "Penolong persalinan", "Siramaneerat, Agushybana", "b4_ch1.dta", "ch20", "-", "Tidak dibentuk; tidak ada value label di kamus"),
    ("Lahir", "Pernah periksa kehamilan", "Siramaneerat, Agushybana, Azriani", "b4_ch1.dta", "ch14", "periksa_hamil", ""),
    ("Lahir", "Kunjungan ANC trimester 1", "Azriani (jumlah ANC)", "b4_ch1.dta", "ch16a", "anc_total", "anc_total = ch16a + ch16b + ch16c"),
    ("Lahir", "Kunjungan ANC trimester 2", "Azriani", "b4_ch1.dta", "ch16b", "anc_total", ""),
    ("Lahir", "Kunjungan ANC trimester 3", "Azriani", "b4_ch1.dta", "ch16c", "anc_total", ""),
    ("Lahir", "Dapat tablet besi", "Azriani (konsumsi Fe)", "b4_ch1.dta", "ch16f", "tablet_besi", ""),
    ("Lahir", "Jumlah tablet besi", "Azriani", "b4_ch1.dta", "ch16g", "jml_tablet_besi", ""),
    ("Lahir", "Usia ibu saat kehamilan berakhir", "Siramaneerat (maternal age)", "b4_ch1.dta", "ch10a", "-", "Terisi rendah (2%); umur ibu diambil dari roster (ar09)"),
    # ---------------- ASI & MPASI ----------------
    ("Menyusui", "Pernah disusui", "Agushybana, Siramaneerat, Novalina", "b4_ch1.dta", "ch24a", "asi_pernah, asi_eksklusif_proxy", ""),
    ("Menyusui", "Usia pertama kali diberi air", "Agushybana, Siramaneerat", "b4_ch1.dta", "ch24c", "usia_air_bln, air_dini, asi_eksklusif_proxy", ""),
    ("Menyusui", "Usia pertama kali diberi makanan lain (MPASI)", "Agushybana, Siramaneerat", "b4_ch1.dta", "ch24d", "usia_mpasi_bln, mpasi_dini, asi_eksklusif_proxy", ""),
    ("Menyusui", "Usia disapih", "Agushybana, Siramaneerat", "b4_ch1.dta", "ch24e", "usia_sapih_bln, masih_asi", "masih_asi = belum disapih saat wawancara"),
    ("Menyusui", "ASI eksklusif (modul menyusui ibu)", "Novalina", "b4_bf.dta", "bf09", "-", "Tidak dipakai; diganti asi_eksklusif_proxy dari b4_ch1"),
    # ---------------- Pola makan & infeksi anak ----------------
    ("Makan", "Frekuensi makan anak per hari", WHO, "b5_fma1.dta", "fma01", "frek_makan_per_hari, hanya_asi", ""),
    ("Makan", "Jenis makanan yang dimakan (7 hari)", WHO + " (keragaman makanan)", "b5_fma2.dta", "fma02", "jenis_makanan_7hr", "17 jenis makanan (fmatype A-Q)"),
    ("Makan", "Jumlah hari makan per jenis makanan (7 hari)", "Novalina (protein intake); " + WHO + " (protein hewani)", "b5_fma2.dta", "fma03",
     "hari_protein_hewani_7hr, hari_susu_7hr, hari_makanan_olahan_7hr, total_hari_makan_7hr, jenis_sayur_buah_7hr", "Protein hewani = telur + ikan + daging"),
    ("Infeksi", "Status kesehatan & hari sakit (4 minggu)", WHO + " (infeksi)", "b5_maa1.dta", "maa0b", "hari_sakit_4mgg, hari_terbaring_4mgg, status_sehat_umum", "maa0a = status sehat, maa0c = hari terbaring"),
    ("Infeksi", "Gejala sakit (4 minggu)", WHO + " (infeksi)", "b5_maa2.dta", "maa01", "jumlah_gejala_4mgg", "Jumlah gejala utama (kode maatype 2 huruf)"),
    # ---------------- Layanan kesehatan anak ----------------
    ("Layanan anak", "Kunjungan posyandu (4 minggu terakhir)", WHO + " (akses layanan kesehatan)", "b5_rja0.dta", "rja0a", "posyandu_4mgg", ""),
    ("Layanan anak", "Frekuensi ditimbang (6 bulan terakhir)", WHO + " (akses layanan kesehatan)", "b5_rja2.dta", "rja32", "frek_timbang_6bln", ""),
    ("Layanan anak", "Punya kartu imunisasi", "pendahuluan Agushybana (MCH book / health card)", "b5_rja2.dta", "rja26", "punya_kartu_imun", "Tidak ada variabel Buku KIA langsung"),
    ("Layanan anak", "Imunisasi (tanggal di kartu + ingatan ibu)", "pendahuluan Agushybana & Siramaneerat (imunisasi)", "b5_rja3.dta", "rja28ayr_b",
     "jumlah_jenis_imun, imun_bcg, imun_polio, imun_dpt, imun_hepb, imun_campak, imun_rota", "Huruf b-p = jenis vaksin; ingatan ibu rja30a-rja30h"),
    ("Layanan anak", "Vitamin A (6 bulan terakhir)", "pendahuluan Agushybana & Siramaneerat (vitamin A)", "b5_rja3.dta", "rja25a", "vit_a_6bln", ""),
    # ---------------- Orang tua ----------------
    ("Orang tua", "ID ayah (nomor urut di RT)", "Agushybana, Azriani, Siramaneerat", "bk_ar1.dta", "ar10", "ada_ayah_di_RT", "Untuk menyambung ke record ayah"),
    ("Orang tua", "ID ibu (nomor urut di RT)", "Agushybana, Azriani, Siramaneerat", "bk_ar1.dta", "ar11", "ada_ibu_di_RT", "Untuk menyambung ke record ibu"),
    ("Orang tua", "Pendidikan tertinggi (ayah/ibu)", "semua", "bk_ar1.dta", "ar16", "pend_ibu, pend_ayah", "Kategori dikelompokkan"),
    ("Orang tua", "Tinggi badan ayah/ibu (cm)", "Azriani, Agushybana", "bus_us.dta", "us04", "tinggi_ibu, tinggi_ayah", "Diambil dari baris ayah/ibu"),
    ("Orang tua", "Berat badan ayah/ibu (kg) -> IMT", "Agushybana, Siramaneerat", "bus_us.dta", "us06", "bb_ibu, imt_ibu, bb_ayah, imt_ayah", ""),
    ("Orang tua", "Umur ayah/ibu", "Siramaneerat, Azriani, Agushybana", "bk_ar1.dta", "ar09", "umur_ibu, umur_ayah", ""),
    ("Orang tua", "Status bekerja (ayah/ibu)", "Agushybana, Siramaneerat, Novalina, Azriani", "bk_ar1.dta", "ar15a", "kerja_ibu, kerja_ayah", "Bekerja tahun lalu"),
    ("Orang tua", "Ada anggota RT merokok", "Novalina, Azriani", "b3b_km.dta", "km01e", "ada_perokok_RT", "Diagregasi per hhid14"),
    # ---------------- Rumah tangga ----------------
    ("Rumah tangga", "Jumlah anggota rumah tangga", "Siramaneerat, Agushybana", "bk_ar1.dta", "ar01a", "jumlah_anggota_RT", "Anggota RT yang masih tinggal"),
    ("Rumah tangga", "Perkotaan/perdesaan", "Siramaneerat, Agushybana", "bk_sc1.dta", "sc05", "perkotaan", "1 = Urban, 2 = Rural"),
    ("Rumah tangga", "Provinsi", "-", "bk_sc1.dta", "sc01_14_14", "provinsi", "Kode BPS 2014"),
    ("Rumah tangga", "Sumber air minum utama", "Agushybana, Azriani, Novalina", "b2_kr.dta", "kr13", "air_minum_kode, air_layak", "Kategori layak didefinisikan sendiri"),
    ("Rumah tangga", "Air minum direbus", WHO + " (keamanan air)", "b2_kr.dta", "kr13a", "air_direbus", ""),
    ("Rumah tangga", "Fasilitas BAB/toilet", "Azriani, Novalina", "b2_kr.dta", "kr20", "toilet_kode, toilet_layak", "Kode 1 = toilet sendiri + septic tank"),
    ("Rumah tangga", "Pembuangan limbah cair", "Azriani", "b2_kr.dta", "kr21", "limbah_kode", ""),
    ("Rumah tangga", "Pembuangan sampah", "Azriani (waste management)", "b2_kr.dta", "kr22", "sampah_kode, sampah_layak", ""),
    ("Rumah tangga", "Punya kartu kesehatan/BPJS", "pendahuluan Agushybana (health card)", "b2_kr.dta", "kr26", "kartu_sehat", ""),
    ("Rumah tangga", "Bahan bakar memasak", "pendahuluan Azriani (biomass fuels)", "b2_kr.dta", "kr24", "bahan_bakar_padat", ""),
    ("Rumah tangga", "Garam beryodium", "pendahuluan Agushybana & Siramaneerat (iodized salt)", "b1_cov.dta", "us21", "garam_beryodium", ""),
    ("Rumah tangga", "Observasi: kotoran/sampah/air tergenang sekitar rumah", WHO + " (sanitasi & lingkungan rumah)", "bk_krk.dta", "krk02a", "masalah_lingkungan_rumah", "Jumlah dari krk02a, krk02b, krk02c"),
    ("Rumah tangga", "Observasi: ventilasi rumah", WHO + " (lingkungan rumah)", "bk_krk.dta", "krk02e", "ventilasi_cukup", ""),
    ("Rumah tangga", "Observasi: dapur menyatu kamar tidur", WHO + " (lingkungan rumah)", "bk_krk.dta", "krk02i", "dapur_satu_kamar_tidur", ""),
    ("Rumah tangga", "Observasi: jenis lantai", WHO + " (lingkungan rumah)", "bk_krk.dta", "krk08", "lantai_tanah", ""),
    ("Rumah tangga", "Kepemilikan aset rumah tangga", WHO + " (kondisi ekonomi); pendahuluan Agushybana (household wealth)", "b2_hr1.dta", "hr01", "indeks_kekayaan_pca", "PCA dari 13 jenis aset (hrtype)"),
    ("Rumah tangga", "Pengeluaran makanan", "Novalina (household food, PCE)", "b1_ks1.dta", "ks02", "log_pengeluaran_pangan_perkapita, log_pengeluaran_perkapita", "ks02 + ks03, diubah ke per bulan per kapita"),
    ("Rumah tangga", "Pengeluaran non-makanan", "Novalina (PCE)", "b1_ks2.dta", "ks06", "log_pengeluaran_perkapita", "Ditambah b1_ks3 (ks08, ks09a)"),
    ("Rumah tangga", "Pengeluaran non-makanan tahunan", "Novalina (PCE)", "b1_ks3.dta", "ks08", "log_pengeluaran_perkapita", ""),
]

TIDAK_ADA = [
    ("Status prematur (Novalina)", "Tidak ada variabel langsung; dipakai lama_hamil_minggu (b4_ch1.ch17)"),
    ("Buku KIA / Health Card (Agushybana)", "Tidak ada; dipakai punya_kartu_imun (b5_rja2.rja26) dan kartu_sehat (b2_kr.kr26)"),
    ("Pengeluaran protein (Novalina)", "Tidak ada per item; diganti hari_protein_hewani_7hr dari pola makan anak (b5_fma2)"),
    ("Imunisasi lengkap (Agushybana)", "Tidak ada variabel jadi; dirangkum sendiri menjadi jumlah_jenis_imun (b5_rja3)"),
    ("Wealth index", "Tidak ada variabel jadi; dibangun sendiri menjadi indeks_kekayaan_pca (PCA aset b2_hr1)"),
]


def status_fitur(fitur, status, alasan):
    if fitur in ("", "-"):
        return ""
    out = []
    for f in [x.strip() for x in fitur.split(",")]:
        s = status.get(f, "bukan kandidat")
        out.append(f"{f} = {s}" + (f" ({alasan[f]})" if s == "DIKELUARKAN" else ""))
    return "; ".join(out)


def main():
    if not os.path.exists(KAMUS_XLSX):
        sys.exit(f"Kamus tidak ditemukan: {KAMUS_XLSX}\nJalankan KamusVariabel.py dulu.")
    v = pd.read_excel(KAMUS_XLSX, sheet_name="Kamus_Variabel")
    vl = pd.read_excel(KAMUS_XLSX, sheet_name="Value_Labels")
    punya_vl = set(zip(vl["file"], vl["variabel"]))

    # status tiap fitur (DIPAKAI / DIKELUARKAN + alasan) dari hasil 1_EDA.py
    sel = pd.read_excel(EDA_XLSX, sheet_name="18_Seleksi_Variabel") if os.path.exists(EDA_XLSX) else None
    if sel is None:
        print(f"[peringatan] {EDA_XLSX} tidak ada; kolom status_di_model dikosongkan. Jalankan 1_EDA.py dulu.")
    status = {} if sel is None else dict(zip(sel["variabel"], sel["status"]))
    alasan = {} if sel is None else dict(zip(sel["variabel"], sel["alasan"]))

    rows, hilang = [], []
    for kel, konsep, paper, f, var, fitur, cat in PEMETAAN:
        r = v[(v["file"] == f) & (v["variabel"] == var)]
        if r.empty:
            hilang.append((f, var))
            continue
        r = r.iloc[0]
        rows.append(
            {
                "kelompok": kel,
                "konsep": konsep,
                "sumber_acuan": paper,
                "file": f,
                "variabel": var,
                "label_EN": r["label_EN"],
                "pct_terisi": r["pct_terisi"],
                "ada_value_label": "Ya" if (f, var) in punya_vl else "",
                "fitur_dataset": fitur,
                "status_di_model": status_fitur(fitur, status, alasan),
                "catatan": cat,
            }
        )
    if hilang:
        sys.exit(f"Variabel tidak ada di kamus (perbaiki PEMETAAN): {hilang}")

    d = pd.DataFrame(rows)
    t = pd.DataFrame(TIDAK_ADA, columns=["konsep", "keterangan"])
    lembar = {"Pemetaan": d, "Tidak_ada_langsung": t}

    # Sheet per fitur kandidat: asal file/variabel IFLS5 dan sumber acuannya
    if sel is not None:
        p = d.assign(fitur=d["fitur_dataset"].str.split(",")).explode("fitur")
        p["fitur"] = p["fitur"].str.strip()
        gab = lambda s: "; ".join(dict.fromkeys(s))
        g = p.groupby("fitur").agg(file=("file", gab), variabel_IFLS5=("variabel", gab), sumber_acuan=("sumber_acuan", gab))
        f = sel[["variabel", "status", "alasan"]].rename(columns={"variabel": "fitur"})
        f = f.merge(g, left_on="fitur", right_index=True, how="left").fillna({"file": "BELUM DIPETAKAN"})
        f = f.sort_values(["status", "fitur"], ascending=[False, True])
        lembar["Fitur_Kandidat"] = f
        n_pakai = int((f["status"] == "DIPAKAI").sum())
        belum = f.loc[f["file"] == "BELUM DIPETAKAN", "fitur"].tolist()
        print(f"Fitur kandidat: {len(f)} | dipakai model: {n_pakai} | dikeluarkan: {len(f) - n_pakai}")
        print(f"Belum dipetakan ke variabel IFLS5: {belum or '-'}")

    os.makedirs(FOLDER_OUT, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        for nm, df in lembar.items():
            df.to_excel(w, sheet_name=nm, index=False)
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for c in ws[1]:
                c.font = Font(name="Arial", bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F4E78")
            for col in ws.columns:
                L = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(L + 2, 10), 70)
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.font = Font(name="Arial", size=10)
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    print(f"Selesai: {len(d)} variabel terverifikasi di kamus -> {OUT_XLSX}")


if __name__ == "__main__":
    main()