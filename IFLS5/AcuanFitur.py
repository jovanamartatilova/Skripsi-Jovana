"""
BuildAcuanFitur_IFLS5.py  -  Excel "Acuan Fitur" (semua fitur kandidat vs literatur)
==========================================================================================================
Output: Acuan_Fitur_IFLS5_v2.xlsx
  Ringkasan          : jumlah fitur, rekap dukungan literatur (dihitung otomatis dari sheet Acuan_Fitur)
  Acuan_Fitur        : semua fitur kandidat x 4 paper (S/N/D), temuan paper (angka dari tabel paper), alasan masuk kandidat,
                       + statistik data (missing %, ukuran efek univariat, p BH) DIHITUNG dari balita_ifls5.csv + status di model (DIPAKAI/DIKELUARKAN + alasan) dari 1_EDA/EDA_IFLS5.xlsx
  Tabel33_vs_Final   : 34 variabel Tabel 3.3 proposal -> keputusan -> kolom dataset final
  Kandidat_Belum_Ada : konsep di paper yang belum ada di dataset
  Metode_Pemilihan   : dasar pemilihan fitur (literatur + IG/Gini di data training; p-value hanya deskriptif)
  Catatan_Paper      : ketidakkonsistenan di paper rujukan yang perlu diwaspadai saat mengutip
  Legenda

Kode dukungan paper (per paper):
  S = bermakna secara statistik di model multivariat ATAU terpilih oleh seleksi fitur paper tsb
  N = dipakai/diuji di paper tsb tetapi TIDAK bermakna / tidak terpilih
  D = hanya deskriptif (tabel rata-rata) atau hanya disebut di teks/pembahasan (tidak dianalisis)
  - = tidak ada di paper tsb

Semua angka paper dibaca langsung dari 4 paper yang diunggah (Agushybana 2025; Siramaneerat 2024;
Novalina 2025; Azriani 2025). Statistik data dihitung ulang dari CSV (bukan disalin), jadi bisa direproduksi.
Statistik univariat BERSIFAT DESKRIPTIF (seluruh data, mengabaikan klaster keluarga) dan BUKAN dasar seleksi fitur.

"""
import os
import sys

import numpy as np
import pandas as pd

# ============================= PENGATURAN (sudah terisi) =============================
CSV_DATASET = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output\BuildDataset\balita_ifls5.csv"
KAMUS_XLSX = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output\kamus_variabel_IFLS5.xlsx"
EDA_XLSX = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output\1_EDA\EDA_IFLS5.xlsx"
OUT_XLSX = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output\BuildDataset\Acuan_Fitur_IFLS5_v2.xlsx"
KOLOM_ID = ["pidlink", "hhid14", "pid14"]
KOLOM_TARGET = ["haz", "stunting"]
# =====================================================================================

PAPER = ["Agushybana (2025)", "Siramaneerat (2024)", "Novalina (2025)", "Azriani (2025)"]

# (fitur, kelompok, deskripsi, sumber IFLS5, status vs Tabel 3.3, A, S, N, Z, temuan paper, alasan kandidat, catatan desain)
FITUR = [
("umur_bln", "Pengukuran anak", "Umur anak (bulan)", "bk_ar1.ar08yr/mth/day, bus_time.ivwyr/mth/day",
 "Ada di Tabel 3.3 (us03; cara hitung berubah)", "S", "S", "S", "-",
 "Agushybana 2014: AOR 0,98 (p<0,001). Siramaneerat: OR 0,978 (CI 0,969-0,987). Novalina: p<0,05 dan terpilih (mutual information). Azriani memprediksi dari lahir sehingga tidak memakai umur.",
 "Dasar HAZ; kader mencatat tanggal lahir dan tanggal ukur.", "Komponen rumus HAZ (umur + TB + JK menentukan HAZ)."),
("jk_laki", "Pengukuran anak", "Jenis kelamin (1 = laki-laki)", "bk_ar1.ar07",
 "Ada di Tabel 3.3 (us01; sumber diganti roster)", "N", "N", "-", "N",
 "Tidak bermakna: Agushybana AOR 1,04 (p 0,690, 2014); Siramaneerat OR 1,015 (p 0,868); Azriani p 0,361 (IG 0,000).",
 "Variabel dasar anak; dipertahankan karena diisi kader dan menjadi komponen HAZ.", "Komponen rumus HAZ."),
("tb_anak", "Pengukuran anak", "Tinggi/panjang badan anak (cm)", "bus_us.us04 (+us04x=1)",
 "Baru (tidak ada di Tabel 3.3; us04)", "-", "-", "S", "-",
 "Novalina: tinggi badan p<0,05 dan terpilih. Namun label stunting Novalina tampaknya dihitung dari distribusi sampelnya sendiri dan fiturnya memuat TB sehingga akurasi tinggi (lihat Catatan_Paper).",
 "Kader mengukur TB saat kunjungan rumah (keputusan peneliti).", "RISIKO KEBOCORAN: TB adalah dasar HAZ; wajib ditulis di keterbatasan dan dibandingkan dengan skenario tanpa TB."),
("bb_anak", "Pengukuran anak", "Berat badan anak (kg)", "bus_us.us06 (+us06x=1)",
 "Ada di Tabel 3.3 (us06)", "-", "-", "S", "-",
 "Novalina: berat badan p<0,05 dan terpilih.",
 "Kader menimbang saat kunjungan rumah.", "Berkorelasi kuat dengan TB (cek di EDA)."),
("bb_lahir_kg", "Kelahiran", "Berat lahir (kg)", "b4_ch1.ch24 (+ch23=1)",
 "Baru (tidak ada di Tabel 3.3)", "S", "S", "S", "S",
 "Konsisten di keempat paper: Agushybana 2014 AOR 0,49 (2,5-3,99 kg) dan 0,20 (>=4 kg) vs <2,5 kg; Siramaneerat OR 0,495 dan 0,193; Azriani AOR 0,79 per kg (p<0,01); Novalina terpilih.",
 "Faktor paling konsisten di literatur.", "Missing karena hanya anak yang tercatat di riwayat kehamilan ibu (b4_ch1)."),
("bblr", "Kelahiran", "Berat lahir rendah (1 jika < 2,5 kg)", "dari bb_lahir_kg",
 "Baru (turunan bb_lahir_kg)", "S", "S", "-", "-",
 "Kategori <2,5 kg adalah pembanding di Agushybana dan Siramaneerat; kategori lebih berat bermakna lebih rendah (lihat bb_lahir_kg).",
 "Mengikuti kategori yang dipakai paper.", "Informasi tumpang tindih dengan bb_lahir_kg."),
("ukuran_bayi", "Kelahiran", "Ukuran bayi menurut persepsi ibu (1 sangat besar ... 5 sangat kecil)", "b4_ch1.ch22",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "S",
 "Azriani: bermakna (kategori 'large' AOR 0,24, p<0,05; IG 0,006).",
 "Proksi ukuran lahir bagi anak yang berat lahirnya tidak tercatat.", "Label kode 2 di kamus janggal ('0Bigger'); cek codebook. Label kategori di tabel Azriani juga tampak bergeser."),
("lama_hamil_minggu", "Kehamilan", "Lama kehamilan (minggu)", "b4_ch1.ch17 + ch17x",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "N",
 "Azriani: usia kehamilan tidak bermakna (p 0,985; IG 0,001). Novalina memakai status prematur, bukan lama kehamilan.",
 "Dasar proxy prematur.", "Satuan campur (minggu/bulan); bulan dibulatkan."),
("hamil_prematur_proxy", "Kehamilan", "Proxy prematur (1 jika < 37 minggu)", "dari lama_hamil_minggu",
 "Baru (turunan lama_hamil_minggu)", "-", "-", "S", "-",
 "Novalina: premature status terpilih (mutual information).",
 "IFLS5 tidak punya variabel prematur langsung, sehingga dibuat proxy.", "Proxy kasar; jangan disebut 'status prematur' tanpa kata proxy."),
("periksa_hamil", "Kehamilan", "Pernah periksa kehamilan (1 = ya)", "b4_ch1.ch14",
 "Baru (tidak ada di Tabel 3.3)", "N", "-", "-", "N",
 "Tidak bermakna: Agushybana 2014 AOR 1,55 (p 0,206); Azriani ANC status p 0,379 (IG 0,001).",
 "Pelayanan antenatal ada di Agushybana dan Azriani.", "Biasanya hampir semua 'ya' (cek kolom catatan data)."),
("anc_total", "Kehamilan", "Jumlah kunjungan ANC trimester 1-3", "b4_ch1.ch16a+ch16b+ch16c",
 "Baru (tidak ada di Tabel 3.3)", "-", "N", "-", "N",
 "Siramaneerat (>=4 kunjungan vs kurang): AOR 1,72, p 0,128 (tidak bermakna). Azriani jumlah ANC: p 0,709 (IG 0,005).",
 "Jumlah kunjungan dipakai di Siramaneerat dan Azriani.", "Cek nilai tidak wajar (> 20 kunjungan)."),
("tablet_besi", "Kehamilan", "Mendapat tablet besi (1 = ya)", "b4_ch1.ch16f",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "N",
 "Azriani: konsumsi zat besi (total) tidak bermakna (p 0,559; IG 0,001). Agushybana hanya membahas suplementasi besi di pembahasan.",
 "Gizi ibu hamil adalah jalur penting (Agushybana, pembahasan).", ""),
("jml_tablet_besi", "Kehamilan", "Jumlah tablet besi", "b4_ch1.ch16g",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "N",
 "Azriani: total iron consumption tidak bermakna (p 0,559).",
 "Mengikuti Azriani.", "Bernilai 0 jika tidak mendapat tablet; sisanya kosong bila jumlah tidak diketahui"),
("tempat_lahir_grup", "Kelahiran", "Tempat melahirkan (1 RS, 2 puskesmas/polindes, 3 praktik, 4 dukun, 5 rumah)", "b4_ch1.ch19",
 "Baru (tidak ada di Tabel 3.3)", "N", "S", "-", "S",
 "Siramaneerat: puskesmas/polindes vs RS OR 1,69 (p 0,001) pada seluruh data; di perkotaan puskesmas OR 2,33 dan rumah OR 1,75. Azriani: lahir di fasilitas kesehatan AOR 0,83 (p 0,010). Agushybana tidak bermakna.",
 "Proksi kualitas layanan persalinan.", "Pengelompokan adalah keputusan peneliti."),
("asi_pernah", "ASI dan MP-ASI", "Pernah disusui (1 = ya)", "b4_ch1.ch24a",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "-", "-",
 "Tidak bermakna di Agushybana (p 0,967, 2014) dan Siramaneerat (p 0,908). Novalina memakai ASI EKSKLUSIF (konsep berbeda) dan terpilih.",
 "Variabel menyusui ada di tiga paper.", "Biasanya hampir semua 'ya'; bukan ASI eksklusif."),
("usia_air_bln", "ASI dan MP-ASI", "Usia pertama diberi air (bulan)", "b4_ch1.ch24c, ch24c1, ch24cx",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "-", "-",
 "Tidak bermakna: Agushybana p 0,743 (2014); Siramaneerat p 0,544.",
 "Timing of introduction of water ada di dua paper.", "Missing struktural (anak yang belum diberi); cek nilai lebih besar dari umur anak."),
("usia_mpasi_bln", "ASI dan MP-ASI", "Usia pertama diberi makanan lain (bulan)", "b4_ch1.ch24d, ch24d1, ch24dx",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "-", "-",
 "Tidak bermakna: Agushybana p 0,158 (2014); Siramaneerat p 0,315.",
 "Timing of complementary feeding ada di dua paper.", "Missing struktural; cek nilai lebih besar dari umur anak."),
("usia_sapih_bln", "ASI dan MP-ASI", "Usia disapih (bulan)", "b4_ch1.ch24e, ch24ex",
 "Baru (tidak ada di Tabel 3.3)", "N", "S", "-", "-",
 "Siramaneerat: bermakna hanya di perkotaan (disapih <6 bulan OR 0,669); seluruh data p 0,149. Agushybana p 0,235.",
 "Age of weaning ada di dua paper.", "Missing besar (anak masih menyusu); cek nilai lebih besar dari umur anak."),
("masih_asi", "ASI dan MP-ASI", "Masih menyusu (1 = ya)", "b4_ch1.ch24ex",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di paper.",
 "Pelengkap usia_sapih_bln agar anak yang masih menyusu tidak hilang informasinya.", "Fitur buatan peneliti."),
("air_dini", "ASI dan MP-ASI", "Air diberikan < 6 bulan (1 = ya)", "dari usia_air_bln",
 "Baru (turunan usia_air_bln)", "N", "N", "-", "-",
 "Kategori <6 bulan seperti di kedua paper; tidak bermakna.",
 "Mengikuti kategori paper.", "Hanya dinilai pada anak >= 6 bulan."),
("mpasi_dini", "ASI dan MP-ASI", "MP-ASI diberikan < 6 bulan (1 = ya)", "dari usia_mpasi_bln",
 "Baru (turunan usia_mpasi_bln)", "N", "N", "-", "-",
 "Kategori <6 bulan seperti di kedua paper; tidak bermakna.",
 "Mengikuti kategori paper.", "Hanya dinilai pada anak >= 6 bulan"),
("posyandu_4mgg", "Layanan kesehatan anak", "Ke posyandu dalam 4 minggu terakhir", "b5_rja0.rja0a",
 "Pengganti rja0ca (lihat Tabel33_vs_Final)", "D", "-", "-", "-",
 "Tidak dianalisis di keempat paper. Agushybana mengutip Anwar et al. (2010) bahwa partisipasi posyandu memperbaiki status gizi.",
 "Relevan dengan konteks kader posyandu.", "Hanya dukungan tidak langsung."),
("vit_a_6bln", "Layanan kesehatan anak", "Mendapat vitamin A dalam 6 bulan terakhir", "b5_rja3.rja25a",
 "Ada di Tabel 3.3 (rja25a)", "D", "D", "-", "-",
 "Hanya disebut di pendahuluan Agushybana dan Siramaneerat (orang tua terdidik memberi vitamin A); tidak dianalisis.",
 "Pengasuhan kesehatan anak.", "Hanya dukungan tidak langsung."),
("punya_kartu_imun", "Layanan kesehatan anak", "Punya kartu imunisasi (1 = ya)", "b5_rja2.rja26",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Agushybana menyebut ketiadaan Buku KIA/Health Card di abstrak dan pembahasan, tetapi variabel itu TIDAK ada di tabel hasilnya.",
 "Proxy terdekat untuk Buku KIA di IFLS5.", "Bukan Buku KIA yang sebenarnya."),
("imun_bcg", "Layanan kesehatan anak", "Imunisasi BCG", "b5_rja3.rja28ayr_b / rja30a",
 "Diagregasi dari variabel Tabel 3.3 (rja28ady_b)", "D", "-", "-", "-",
 "Agushybana menyebut imunisasi tidak lengkap di pembahasan, tetapi tidak dianalisis di tabel.",
 "Kelanjutan variabel tanggal imunisasi di Tabel 3.3.", "Dirangkum dari tanggal kartu dan jawaban ibu."),
("imun_polio", "Layanan kesehatan anak", "Imunisasi polio (dosis 0-4)", "b5_rja3.rja28ayr_c,d,e,f,n / rja30b",
 "Diagregasi dari variabel Tabel 3.3 (rja28ady_c, d, n)", "D", "-", "-", "-",
 "Idem imun_bcg.", "Idem.", "Tabel 3.3 hanya memuat sebagian dosis; dataset memakai semua dosis."),
("imun_dpt", "Layanan kesehatan anak", "Imunisasi DPT (dosis 1-3)", "b5_rja3.rja28ayr_g,h,i / rja30c",
 "Diagregasi dari variabel Tabel 3.3 (rja28ady_h, i)", "D", "-", "-", "-",
 "Idem imun_bcg.", "Idem.", "Tabel 3.3 hanya memuat DPT 2 dan 3; dataset memakai DPT 1-3."),
("imun_hepb", "Layanan kesehatan anak", "Imunisasi Hepatitis B (dosis 1-3)", "b5_rja3.rja28ayr_k,l,m / rja30e",
 "Diagregasi dari variabel Tabel 3.3 (rja28ady_k, l, m)", "D", "-", "-", "-",
 "Idem imun_bcg.", "Idem.", ""),
("imun_campak", "Layanan kesehatan anak", "Imunisasi campak", "b5_rja3.rja28ayr_j / rja30d",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Idem imun_bcg.", "Idem.", "Bayi muda belum waktunya; status 'tidak' bergantung umur."),
("imun_rota", "Layanan kesehatan anak", "Imunisasi rotavirus (dosis 1-2)", "b5_rja3.rja28ayr_o,p / rja30g,h",
 "Diagregasi dari variabel Tabel 3.3 (rja28ady_o)", "D", "-", "-", "-",
 "Idem imun_bcg.", "Idem.", "Tabel 3.3 hanya memuat Rotavirus 1."),
("jumlah_jenis_imun", "Layanan kesehatan anak", "Jumlah jenis imunisasi diterima (0-6)", "dari imun_*",
 "Baru (turunan imun_*)", "D", "-", "-", "-",
 "Idem imun_bcg (imunisasi tidak lengkap disebut di pembahasan Agushybana).",
 "Ringkasan status imunisasi.", "Bergantung umur anak (bayi muda belum waktunya)."),
("tinggi_ibu", "Orang tua", "Tinggi badan ibu (cm)", "bk_ar1.ar11 -> bus_us.us04",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "-", "S",
 "Azriani: AOR 0,91 per cm (p<0,01) dan Information Gain tertinggi (0,045). Agushybana hanya mendeskripsikan di Tabel 2 (ibu anak stunting 149,8 vs 152,8 cm, 2014). Siramaneerat hanya menyebut tinggi orang tua di pendahuluan.",
 "Prediktor teratas di Azriani; faktor genetik/antargenerasi.", ""),
("tinggi_ayah", "Orang tua", "Tinggi badan ayah (cm)", "bk_ar1.ar10 -> bus_us.us04",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "-", "S",
 "Azriani: AOR 0,95 (p<0,01), IG 0,023 (peringkat 2). Agushybana hanya Tabel 2 (161,4 vs 164,3 cm, 2014).",
 "Prediktor kedua di Azriani.", "Missing lebih besar karena ayah tidak selalu tinggal serumah."),
("bb_ibu", "Orang tua", "Berat badan ibu (kg)", "bus_us.us06",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Agushybana hanya deskriptif (Tabel 2).", "Dasar IMT ibu.", "Redundan dengan imt_ibu."),
("bb_ayah", "Orang tua", "Berat badan ayah (kg)", "bus_us.us06",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Agushybana hanya deskriptif (Tabel 2).", "Dasar IMT ayah.", "Redundan dengan imt_ayah."),
("imt_ibu", "Orang tua", "IMT ibu (kg/m2)", "dari bb & tinggi",
 "Baru (turunan)", "S", "S", "-", "-",
 "Agushybana: status gizi ibu obesitas AOR 0,56 (p 0,008, 2014). Siramaneerat: tidak bermakna pada seluruh data (underweight OR 1,36, p 0,116), bermakna hanya di perkotaan (underweight OR 1,84).",
 "Status gizi ibu ada di dua paper (sebagai kategori).", "Paper memakai kategori status gizi, dataset memakai IMT kontinu."),
("imt_ayah", "Orang tua", "IMT ayah (kg/m2)", "dari bb & tinggi",
 "Baru (turunan)", "D", "-", "-", "-",
 "Agushybana hanya mendeskripsikan status gizi ayah di Tabel 1; tidak dimodelkan.", "Pasangan imt_ibu.", "Redundan dengan bb_ayah dan tinggi_ayah."),
("umur_ibu", "Orang tua", "Umur ibu (tahun)", "bk_ar1.ar09",
 "Baru (tidak ada di Tabel 3.3)", "N", "S", "-", "S",
 "Azriani: AOR 0,98 (p<0,01; IG 0,005). Siramaneerat: OR 0,967 hanya di perkotaan (seluruh data p 0,082). Agushybana p 0,097.",
 "Dipakai di tiga paper.", ""),
("umur_ayah", "Orang tua", "Umur ayah (tahun)", "bk_ar1.ar09",
 "Baru (tidak ada di Tabel 3.3)", "N", "S", "-", "N",
 "Siramaneerat: OR 0,979 hanya di perkotaan. Agushybana p 0,362; Azriani p 0,294.",
 "Dipakai di tiga paper.", "Berkorelasi dengan umur_ibu."),
("pend_ibu", "Orang tua", "Pendidikan ibu (0-4)", "bk_ar1.ar16",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "N", "S",
 "Azriani: pendidikan rendah AOR 1,49 (p 0,019), IG 0,019. Novalina: p<0,05 (tetapi tidak terpilih mutual information). Agushybana dan Siramaneerat tidak bermakna.",
 "Dipakai di keempat paper.", "Pengelompokan pendidikan adalah keputusan peneliti."),
("pend_ayah", "Orang tua", "Pendidikan ayah (0-4)", "bk_ar1.ar16",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "N", "S",
 "Novalina: p<0,05 tetapi tidak terpilih mutual information. Agushybana 1997: SD vs tidak sekolah AOR 2,64 (p 0,024), tidak bermakna pada 2014. Azriani: IG 0,020 (peringkat 3, masuk model prediksi) walau regresi tidak bermakna (p 0,080). Siramaneerat tidak bermakna.",
 "Dipakai di keempat paper.", "Pengelompokan pendidikan adalah keputusan peneliti."),
("kerja_ibu", "Orang tua", "Ibu bekerja (1 = ya)", "bk_ar1.ar15a",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "S", "N",
 "Novalina: terpilih (mutual information). Tiga paper lain tidak bermakna (Agushybana p 0,432; Siramaneerat p 0,251; Azriani p 0,910).",
 "Dipakai di keempat paper.", "ar15a = bekerja dalam setahun terakhir."),
("kerja_ayah", "Orang tua", "Ayah bekerja (1 = ya)", "bk_ar1.ar15a",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "N", "N",
 "Tidak bermakna di keempat paper (Agushybana p 0,318; Siramaneerat p 0,439; Azriani p 0,725).",
 "Dipakai di keempat paper.", "Biasanya hampir semua 'ya'."),
("ada_ibu_di_RT", "Teknis", "Ibu tercatat di roster rumah tangga (1 = ya)", "bk_ar1.ar11",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di paper.", "Penanda kelengkapan data ibu.", "Fitur teknis; pertimbangkan dikeluarkan."),
("ada_ayah_di_RT", "Teknis", "Ayah tercatat di roster rumah tangga (1 = ya)", "bk_ar1.ar10",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di paper.", "Penanda kelengkapan data ayah.", "Fitur teknis; dapat menjadi proxy ayah tidak serumah."),
("perkotaan", "Wilayah", "Tempat tinggal perkotaan (1 = ya)", "bk_sc1.sc05",
 "Baru (tidak ada di Tabel 3.3)", "S", "S", "-", "-",
 "Agushybana: perkotaan AOR 0,81 (p 0,029, 2014; 0,50 pada 1997). Siramaneerat: perdesaan OR 1,229 (p 0,033).",
 "Konsisten bermakna di dua paper.", ""),
("provinsi", "Wilayah", "Kode provinsi BPS 2014", "bk_sc1.sc01_14_14",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Bukan variabel di paper (paper hanya menyebut cakupan 13 provinsi).",
 "Kontrol geografis.", "Nominal; tidak ada di paper; tinjau apakah dipertahankan (dashboard kader tidak punya)."),
("air_minum_kode", "Lingkungan rumah tangga", "Sumber air minum (kode asli kr13)", "b2_kr.kr13",
 "Ada di Tabel 3.3 (kr13)", "-", "-", "N", "N",
 "Tidak bermakna: Azriani p 0,707 (IG 0,000); Novalina (water) tidak bermakna dan tidak terpilih.",
 "Ada di Tabel 3.3 dan dua paper.", ""),
("air_layak", "Lingkungan rumah tangga", "Air minum layak (1 jika kode 1, 2, 10)", "dari kr13",
 "Turunan dari variabel Tabel 3.3", "-", "-", "N", "N",
 "Idem air_minum_kode.", "Mengikuti Novalina (protected) dan Azriani (healthy source).", "Definisi layak = keputusan peneliti."),
("toilet_kode", "Lingkungan rumah tangga", "Tempat BAB (kode asli kr20)", "b2_kr.kr20",
 "Ada di Tabel 3.3 (kr20)", "-", "-", "N", "S",
 "Azriani: toilet sesuai standar AOR 0,86 (p 0,029; IG 0,015, peringkat 5). Novalina: tidak bermakna dan tidak terpilih.",
 "Ada di Tabel 3.3 dan dua paper.", ""),
("toilet_layak", "Lingkungan rumah tangga", "Toilet layak (1 jika kode 1)", "dari kr20",
 "Turunan dari variabel Tabel 3.3", "-", "-", "N", "S",
 "Idem toilet_kode.", "Mengikuti Novalina dan Azriani.", "Definisi layak = keputusan peneliti."),
("limbah_kode", "Lingkungan rumah tangga", "Pembuangan limbah cair (kode asli kr21)", "b2_kr.kr21",
 "Ada di Tabel 3.3 (kr21)", "-", "-", "-", "N",
 "Azriani: waste disposal tidak bermakna (p 0,193; IG 0,002).", "Ada di Tabel 3.3 dan Azriani.", ""),
("sampah_kode", "Lingkungan rumah tangga", "Pembuangan sampah (kode asli kr22)", "b2_kr.kr22",
 "Ada di Tabel 3.3 (kr22)", "-", "-", "-", "S",
 "Azriani: waste management sesuai standar AOR 0,79 (p<0,01; IG 0,014, peringkat 6).", "Ada di Tabel 3.3 dan Azriani.", ""),
("sampah_layak", "Lingkungan rumah tangga", "Pengelolaan sampah layak (1 jika kode 1)", "dari kr22",
 "Turunan dari variabel Tabel 3.3", "-", "-", "-", "S",
 "Idem sampah_kode.", "Mengikuti Azriani.", "Definisi layak = keputusan peneliti."),
("kartu_sehat", "Lingkungan rumah tangga", "RT punya kartu sehat/ASKESKIN/JAMKESMAS/BPJS (kr26)", "b2_kr.kr26",
 "Ada di Tabel 3.3 (kr26)", "-", "-", "-", "-",
 "Tidak dianalisis di paper mana pun. Ini asuransi kesehatan rumah tangga, BUKAN Buku KIA/KMS yang disebut Agushybana.",
  "Ada di Tabel 3.3 (proxy akses layanan).", "Level rumah tangga; jangan diklaim sebagai Buku KIA."),
("asi_eksklusif_proxy", "ASI dan MP-ASI", "ASI eksklusif, proxy (anak >= 6 bulan)", "dari asi_pernah, air_dini, mpasi_dini",
 "Baru (turunan)", "D", "-", "S", "-",
 "Novalina: ASI eksklusif terpilih (mutual information). Agushybana hanya membahasnya.",
 "Mendekati konsep ASI eksklusif.", "Proxy, bukan pengukuran langsung."),
("air_direbus", "Lingkungan rumah tangga", "Air minum direbus (1 = ya)", "b2_kr.kr13a",
 "Ada di Tabel 3.3 (kr13a)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Ada di Tabel 3.3; relevan dengan keamanan air.", "Tidak ditanyakan pada semua rumah tangga."),
("jumlah_anggota_RT", "Lingkungan rumah tangga", "Jumlah anggota rumah tangga", "bk_ar1",
 "Baru (tidak ada di Tabel 3.3)", "N", "N", "-", "-",
 "Tidak bermakna: Agushybana p 0,339 (2014); Siramaneerat p 0,599.", "Dipakai dua paper; pembagi per kapita.", "Dihitung dari roster dengan ar09 terisi."),
("ada_perokok_RT", "Lingkungan rumah tangga", "Ada perokok aktif di rumah", "b3b_km.km01d, km01e, km04",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "N", "N",
 "Novalina tidak bermakna dan tidak terpilih; Azriani p 0,374 pada model multivariat.", "Dipakai dua paper.", "Agregasi per rumah tangga."),
("log_pengeluaran_perkapita", "Sosial ekonomi", "Log pengeluaran per kapita per bulan", "b1_ks1, b1_ks2, b1_ks3",
 "Pengganti ks02a/ks07a", "-", "-", "S", "-",
 "Novalina: PCE terpilih (mutual information).", "Proxy kemampuan ekonomi.", "Periode pengeluaran perlu dicocokkan dengan kuesioner."),
("log_pengeluaran_pangan_perkapita", "Sosial ekonomi", "Log pengeluaran pangan per kapita per bulan", "b1_ks1.ks02, ks03",
 "Pengganti ks02a", "-", "-", "S", "-",
 "Novalina: household food terpilih (mutual information).", "Proxy akses pangan.", "Berkorelasi dengan pengeluaran total."),
# ---------------- Tambahan: fitur yang belum punya baris acuan ----------------
("waz", "Pengukuran anak", "Z-score berat badan menurut umur (BB/U, WHO 2006)", "dari bus_us.us06 + umur + JK",
 "Turunan dari us06 (Tabel 3.3)", "-", "-", "S", "-",
 "Novalina: berat badan (kg) p<0,05 dan terpilih (mutual information). Keempat paper tidak memakai WAZ; WAZ adalah BB yang dibakukan menurut umur dan JK.",
 "Kader menimbang BB setiap bulan di posyandu; WAZ dipakai agar BB tidak bergantung umur.", "Berkaitan erat dengan status gizi; ditulis di keterbatasan."),
("hb_anak", "Layanan kesehatan anak", "Kadar hemoglobin anak (g/dL)", "bus_us.us13 (+us13x=1)",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.",
 "Kerangka WHO Subbab 2.3.1 (zat gizi mikro dan infeksi); anemia dapat dirujuk ke Puskesmas.", "Tidak diukur pada anak < 1 tahun (missing struktural)."),
("hb_ibu", "Orang tua", "Kadar hemoglobin ibu (g/dL)", "bk_ar1.ar11 -> bus_us.us13",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Agushybana hanya di pembahasan: anemia ibu saat hamil berhubungan dengan stunting (mengutip penelitian lain); tidak dianalisis.",
 "Status gizi ibu (kerangka WHO Subbab 2.3.1).", "Hb saat survei, bukan saat hamil."),
("indeks_kekayaan_pca", "Sosial ekonomi", "Indeks kekayaan rumah tangga (PCA 13 jenis aset)", "b2_hr1.hr01 (hrtype)",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "D", "-",
 "Agushybana hanya menyebut status kekayaan rumah tangga di pendahuluan; Novalina hanya di tinjauan pustaka (wealth status). Di paper, ekonomi rumah tangga diwakili pengeluaran (PCE Novalina).",
 "Kondisi ekonomi (kerangka WHO, faktor kontekstual); metode PCA aset seperti DHS.", "Dibangun sendiri dari komponen 1 PCA."),
("hari_protein_hewani_7hr", "Pola makan anak", "Jumlah hari makan telur/ikan/daging (7 hari)", "b5_fma2.fma02, fma03 (fmatype B-D)",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "D", "-",
 "Novalina: protein intake terpilih, tetapi berupa PENGELUARAN protein rumah tangga, bukan konsumsi anak. Agushybana menyarankan pengeluaran pangan hewani di kesimpulan (tidak dianalisis).",
 "Asupan protein hewani (kerangka WHO, MP-ASI tidak adekuat); bisa diedukasi kader.", "Konsep berbeda dengan Novalina; tulis sebagai pendekatan, bukan variabel yang sama."),
("jenis_makanan_7hr", "Pola makan anak", "Jumlah jenis makanan yang dimakan (7 hari, dari 17 jenis)", "b5_fma2.fma02",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "D", "-",
 "Novalina hanya di tinjauan pustaka (keragaman makanan di bawah standar WHO, mengutip Paudel); tidak dianalisis.",
 "Keragaman makanan (kerangka WHO).", ""),
("hari_makanan_olahan_7hr", "Pola makan anak", "Jumlah hari makan mi instan/cepat saji/minuman bersoda/gorengan/camilan manis (7 hari)", "b5_fma2.fma03 (fmatype K-O)",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Kualitas pangan rendah (kerangka WHO); bisa diedukasi kader.", ""),
("hari_susu_7hr", "Pola makan anak", "Jumlah hari minum susu (7 hari)", "b5_fma2.fma03 (fmatype E)",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Asupan pangan hewani (kerangka WHO).", ""),
("frek_makan_per_hari", "Pola makan anak", "Frekuensi makan anak per hari", "b5_fma1.fma01",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Praktik pemberian makan (kerangka WHO).", "Kode kategori diubah menjadi kali per hari."),
("total_hari_makan_7hr", "Pola makan anak", "Total hari makan semua jenis makanan (7 hari)", "b5_fma2.fma03",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Ringkasan pola makan.", "Redundan dengan jenis_makanan_7hr."),
("jenis_sayur_buah_7hr", "Pola makan anak", "Jumlah jenis sayur dan buah (7 hari)", "b5_fma2.fma02 (fmatype F-J)",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "D", "-",
 "Agushybana menyarankan pangan non-serealia di kesimpulan; Novalina menyebut keragaman makanan di tinjauan pustaka. Tidak dianalisis.",
 "Keragaman makanan (kerangka WHO).", "Redundan dengan jenis_makanan_7hr."),
("hanya_asi", "ASI dan MP-ASI", "Masih hanya diberi ASI (belum makan lain)", "b5_fma1.fma01 (kode 96)",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada sebagai variabel; Novalina memakai ASI eksklusif (lihat asi_eksklusif_proxy).", "Pelengkap pola makan bayi.", "Hanya relevan pada bayi muda."),
("hari_sakit_4mgg", "Infeksi", "Jumlah hari sakit (4 minggu terakhir)", "b5_maa1.maa0b",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "D", "D",
 "Infeksi/diare hanya disebut di pendahuluan atau pembahasan keempat paper; tidak dianalisis.", "Infeksi (kerangka WHO).", ""),
("hari_terbaring_4mgg", "Infeksi", "Jumlah hari terbaring karena sakit (4 minggu terakhir)", "b5_maa1.maa0c",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "D", "D",
 "Idem hari_sakit_4mgg.", "Infeksi (kerangka WHO).", "Redundan dengan hari_sakit_4mgg."),
("jumlah_gejala_4mgg", "Infeksi", "Jumlah gejala sakit (4 minggu terakhir)", "b5_maa2.maa01",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "D", "D",
 "Idem hari_sakit_4mgg.", "Infeksi (kerangka WHO).", "Hanya gejala utama (kode maatype 2 huruf)."),
("status_sehat_umum", "Infeksi", "Status kesehatan anak menurut orang tua (1-4)", "b5_maa1.maa0a",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Gambaran kesehatan anak.", "Subjektif."),
("frek_timbang_6bln", "Layanan kesehatan anak", "Berapa kali ditimbang (6 bulan terakhir)", "b5_rja2.rja32",
 "Pengganti rja0ca (lihat Tabel33_vs_Final)", "D", "D", "-", "-",
 "Siramaneerat menyebut pemantauan pertumbuhan (growth monitoring) sebagai intervensi di abstrak; Agushybana mengutip manfaat partisipasi posyandu. Tidak dianalisis.",
 "Pemantauan pertumbuhan di posyandu, sesuai peran kader.", ""),
("masalah_lingkungan_rumah", "Lingkungan rumah tangga", "Jumlah masalah kebersihan sekitar rumah (kotoran, sampah, air tergenang; 0-3)", "bk_krk.krk02a, krk02b, krk02c",
 "Baru (tidak ada di Tabel 3.3)", "D", "-", "-", "-",
 "Agushybana hanya menyebut lingkungan rumah tidak sehat sebagai penyebab tidak langsung di pendahuluan; tidak dianalisis.",
 "Sanitasi dan lingkungan rumah (kerangka WHO).", "Observasi pewawancara."),
("garam_beryodium", "Lingkungan rumah tangga", "Garam rumah tangga beryodium (tes pewawancara)", "b1_cov.us21",
 "Baru (tidak ada di Tabel 3.3)", "D", "D", "-", "-",
 "Hanya disebut di pendahuluan Agushybana dan Siramaneerat (orang tua terdidik memberi imunisasi, vitamin A, dan garam beryodium); tidak dianalisis.",
 "Zat gizi mikro (kerangka WHO).", ""),
("ventilasi_cukup", "Lingkungan rumah tangga", "Ventilasi rumah cukup (1 = ya)", "bk_krk.krk02e",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "D", "-",
 "Novalina hanya di tinjauan pustaka (dapur tanpa ventilasi, mengutip Paudel); tidak dianalisis.", "Lingkungan rumah (kerangka WHO).", "Observasi pewawancara."),
("dapur_satu_kamar_tidur", "Lingkungan rumah tangga", "Dapur menyatu dengan kamar tidur (1 = ya)", "bk_krk.krk02i",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "D", "-",
 "Idem ventilasi_cukup.", "Lingkungan rumah (kerangka WHO).", "Observasi pewawancara."),
("bahan_bakar_padat", "Lingkungan rumah tangga", "Memasak dengan kayu/arang (1 = ya)", "b2_kr.kr24",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "D",
 "Azriani hanya di pendahuluan (paparan bahan bakar biomassa); tidak dianalisis.", "Lingkungan rumah (kerangka WHO).", ""),
("lantai_tanah", "Lingkungan rumah tangga", "Lantai rumah dari tanah (1 = ya)", "bk_krk.krk08",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "-",
 "Tidak ada di keempat paper.", "Kondisi rumah sebagai gambaran ekonomi dan kebersihan.", "Observasi pewawancara."),
("anak_kembar", "Kelahiran", "Lahir kembar (1 = ya)", "b4_ch1.ch06a",
 "Baru (tidak ada di Tabel 3.3)", "-", "-", "-", "N",
 "Azriani: category of babies (tunggal/kembar) tidak bermakna (p 0,069; IG 0,000).", "Dipakai Azriani.", "Jarang (sekitar 3%)."),
]

# (kode asli, file, deskripsi, keputusan, kolom di dataset final, alasan)
TABEL33 = [
("us01", "bus_cov.dta", "Jenis kelamin anak", "Diganti sumber", "jk_laki", "Dataset memakai roster bk_ar1.ar07 (sama dengan sumber tanggal lahir dan hubungan orang tua); kode diverifikasi silang dengan b4_ch1.ch08."),
("us03", "bus_cov.dta", "Usia anak (tahun)", "Diganti", "umur_bln", "Umur dalam tahun terlalu kasar untuk HAZ; dihitung dalam bulan."),
("us03", "bus_cov.dta", "Usia anak (bulan, aproksimasi)", "Dipertahankan (cara hitung berubah)", "umur_bln", "Bukan tahun x 12, melainkan selisih tanggal lahir dan tanggal wawancara; hari lahir tidak valid diisi 15."),
("us06", "bus_us.dta", "Berat badan (kg)", "Dipertahankan", "bb_anak", "Dipakai Novalina; kader menimbang saat kunjungan."),
("(turunan us06)", "", "Flag berat badan hilang", "Dibuang", "-", "Berat badan yang tidak terukur/tidak wajar sudah dijadikan NaN; flag tidak informatif."),
("kr13", "b2_kr.dta", "Sumber utama air minum", "Dipertahankan", "air_minum_kode, air_layak", "Ada di Novalina dan Azriani."),
("kr13a", "b2_kr.dta", "Air direbus sebelum diminum", "Dipertahankan", "air_direbus", "Melengkapi informasi keamanan air, walau tidak ada di keempat paper."),
("kr14", "b2_kr.dta", "Lokasi sumber air utama", "Dibuang", "-", "Tidak ada di keempat paper."),
("kr16", "b2_kr.dta", "Air untuk kebutuhan lain (mandi/cuci)", "Dibuang", "-", "Tidak ada di paper; redundan dengan kr13."),
("kr17", "b2_kr.dta", "Sumber air kebutuhan lain", "Dibuang", "-", "Tidak ada di paper; redundan dengan kr13."),
("kr18", "b2_kr.dta", "Lokasi sumber air kebutuhan lain", "Dibuang", "-", "Tidak ada di paper; redundan dengan kr13."),
("kr20", "b2_kr.dta", "Tempat BAB mayoritas ART", "Dipertahankan", "toilet_kode, toilet_layak", "Ada di Novalina dan Azriani (bermakna di Azriani)."),
("kr21", "b2_kr.dta", "Pembuangan air limbah", "Dipertahankan", "limbah_kode", "Ada di Azriani (waste disposal)."),
("kr22", "b2_kr.dta", "Cara membuang sampah", "Dipertahankan", "sampah_kode, sampah_layak", "Ada di Azriani (waste management, bermakna)."),
("kr23", "b2_kr.dta", "Penyimpanan makanan di lemari es", "Dibuang", "-", "Tidak ada di paper; hanya proxy aset."),
("kr26", "b2_kr.dta", "Kartu Sehat/ASKESKIN/JAMKESMAS/BPJS", "Dipertahankan", "kartu_sehat", "Proxy akses layanan; BUKAN Buku KIA (lihat Acuan_Fitur)."),
("kr27", "b2_kr.dta", "Partisipasi dana sehat", "Dibuang", "-", "Tidak ada di paper."),
("rja0ca", "b5_rja0.dta", "Pernah ditimbang di Posyandu", "Dibuang (struktural)", "-", "Pertanyaan lanjutan yang hanya ditanyakan kepada anak yang ke posyandu 4 minggu terakhir (rja0a); sisanya kosong. Informasi posyandu diwakili posyandu_4mgg (rja0a)."),
("rja0ce", "b5_rja0.dta", "Pernah diimunisasi", "Diganti", "imun_*, jumlah_jenis_imun", "Pertanyaan lanjutan (hanya anak yang ke posyandu); status imunisasi dirinci per jenis dari kartu dan jawaban ibu."),
("rja0cc", "b5_rja0.dta", "Pernah menerima kapsul vitamin A", "Diganti", "vit_a_6bln", "Pertanyaan lanjutan (hanya anak yang ke posyandu); diwakili rja25a."),
("rja25a", "b5_rja3.dta", "Vitamin A 6 bulan terakhir", "Dipertahankan", "vit_a_6bln", "Tercakup lebih luas daripada rja0cc."),
("rja30f", "b5_rja3.dta", "Pernah menerima vitamin A (jawaban ibu)", "Dibuang", "-", "Tumpang tindih dengan rja25a."),
("rja28ady_b", "b5_rja3.dta", "Tanggal imunisasi BCG", "Diagregasi", "imun_bcg", "Tanggal absolut tidak bermakna sebagai fitur; dirangkum menjadi status 0/1."),
("rja28ady_c", "b5_rja3.dta", "Tanggal imunisasi Polio 0", "Diagregasi", "imun_polio", "Idem; dataset memakai semua dosis polio (0-4)."),
("rja28ady_d", "b5_rja3.dta", "Tanggal imunisasi Polio 1", "Diagregasi", "imun_polio", "Idem."),
("rja28ady_h", "b5_rja3.dta", "Tanggal imunisasi DPT 2", "Diagregasi", "imun_dpt", "Idem; dataset memakai DPT 1-3."),
("rja28ady_i", "b5_rja3.dta", "Tanggal imunisasi DPT 3", "Diagregasi", "imun_dpt", "Idem."),
("rja28ady_k", "b5_rja3.dta", "Tanggal imunisasi Hep B 1", "Diagregasi", "imun_hepb", "Idem."),
("rja28ady_l", "b5_rja3.dta", "Tanggal imunisasi Hep B 2", "Diagregasi", "imun_hepb", "Idem."),
("rja28ady_m", "b5_rja3.dta", "Tanggal imunisasi Hep B 3", "Diagregasi", "imun_hepb", "Idem."),
("rja28ady_n", "b5_rja3.dta", "Tanggal imunisasi Polio 4", "Diagregasi", "imun_polio", "Idem."),
("rja28ady_o", "b5_rja3.dta", "Tanggal imunisasi Rotavirus 1", "Diagregasi", "imun_rota", "Idem; dataset memakai Rotavirus 1-2."),
("ks02a", "b1_ks0.dta", "Pengeluaran pangan 1 minggu terakhir (arti perlu dicek di kuesioner)", "Diganti", "log_pengeluaran_pangan_perkapita, log_pengeluaran_perkapita", "Dihitung dari rincian item b1_ks1 karena label ks02a di kamus tidak memastikan nilainya total pangan."),
("ks07a", "b1_ks0.dta", "Konsumsi non-pangan rumah tangga (arti perlu dicek di kuesioner)", "Diganti", "log_pengeluaran_perkapita", "Dihitung dari rincian item b1_ks2 dan b1_ks3 karena label ks07a di kamus terpotong."),
]

BELUM_ADA = [
("Penolong persalinan", "Agushybana, Siramaneerat", "b4_ch1.ch20: tipe string TANPA value label di kamus; cek codebook", "Sedang-sulit", "Tambahkan hanya setelah kode dipastikan dari codebook."),
("Paritas / jumlah anak", "Agushybana, Siramaneerat, Azriani (semua tidak bermakna)", "Hitung jumlah kehamilan per ibu dari b4_ch1", "Sedang", "Opsional."),
("Pengeluaran protein rumah tangga", "Novalina", "Harus dihitung dari b1_ks1 (per item) dan butuh codebook kode item", "Sulit", "Tidak dibuat; didekati dengan hari_protein_hewani_7hr (konsumsi anak). Tulis di keterbatasan."),
("ASI eksklusif (pengukuran langsung)", "Novalina; Agushybana (pembahasan)", "b4_bf.bf09 (cek definisi di codebook)", "Sedang", "Opsional; versi proxy sudah ada di asi_eksklusif_proxy."),
("Status gizi ibu/ayah (kategori)", "Agushybana, Siramaneerat", "Turunan IMT yang sudah ada", "Mudah", "Tidak perlu: diwakili imt_ibu dan imt_ayah."),
]

CATATAN_PAPER = [
("Agushybana 2025", "Abstrak dan Kesimpulan menyebut faktor bermakna: antenatal care tidak lengkap, sanitasi buruk, pemberian makanan pendamping, imunisasi tidak lengkap, dan ketiadaan Buku KIA/Health Card. Tabel 3 (hasil regresi) TIDAK mendukungnya: yang bermakna hanya umur anak (2014), berat lahir, status gizi ibu (obesitas, 2014), tempat tinggal; ANC (p 0,206) dan makanan pendamping (p 0,158) dan usia penyapihan (p 0,235) tidak bermakna, sedangkan sanitasi, imunisasi, dan Buku KIA tidak dianalisis.", "Jangan mengutip Agushybana sebagai bukti ANC, sanitasi, imunisasi, atau Buku KIA. Kutip dari Tabel 3, bukan abstrak."),
("Agushybana 2025", "Abstrak menyebut anak 24-60 bulan, tetapi teks hasil dan judul Tabel 1-3 menyebut 0-23 bulan. Jumlah sampel 2014: 2.571 (abstrak dan metode) vs n=2.517 di judul Tabel 1 (persentase 35,63% cocok dengan 2.571). Prevalensi di hasil (47,58% dan 35,63%) berbeda dengan pembahasan (29,7% dan 32,5%).", "Cantumkan bahwa rentang usia dan prevalensi di paper tidak konsisten bila dipakai sebagai rujukan."),
("Siramaneerat 2024", "Tabel 1: angka pekerjaan ayah (1.477 / 814 / 35,53 dan 93 / 44 / 32,12) identik dengan pekerjaan ibu; kemungkinan salah salin.", "Jangan memakai statistik pekerjaan ayah dari Tabel 1."),
("Siramaneerat 2024", "Abstrak dan teks hasil menyebut status gizi ibu berasosiasi pada analisis multivariat seluruh data, tetapi Tabel 3 menunjukkan status gizi ibu tidak bermakna (underweight OR 1,359; p 0,116); yang bermakna adalah umur anak, berat lahir, tempat lahir (puskesmas OR 1,69), dan tempat tinggal. Status gizi ibu bermakna hanya pada model perkotaan (OR 1,84).", "Sebut 'bermakna di perkotaan', bukan 'bermakna' saja."),
("Siramaneerat 2024", "Beberapa angka tidak konsisten: Tabel 3 ukuran rumah tangga AOR 0,940 dengan CI 0,976-1,183 (estimasi di luar CI); Tabel 4 pedesaan waktu MP-ASI OR 1,948 dengan CI 0,714-1,258.", "Hindari mengutip angka tersebut."),
("Novalina 2025", "Label stunting tampaknya dibuat dari z-score distribusi sampel sendiri, prevalensinya hanya 17,04% (Siramaneerat melaporkan 35,34%; sama-sama IFLS5, dengan rentang usia berbeda). Fitur memuat tinggi dan berat badan sehingga akurasi 96% tidak sebanding dengan hasil model tanpa TB/BB. Hasil juga memakai SMOTE/undersampling sehingga accuracy tidak mencerminkan penapisan.", "Kutip daftar faktor risikonya, jangan membandingkan angka performa."),
("Azriani 2025", "Label kategori ukuran bayi di Tabel 2 tampak bergeser (abstrak menyebut 'large infant size' AOR 0,24 sedangkan baris pertama tabel 'very large' 0,055). Prediktor ANC, pendidikan ayah, dan lainnya tidak bermakna di regresi tetapi sebagian masuk model prediksi berdasarkan Information Gain, yaitu prediksi dan eksplanasi sengaja dipisah. F1 84,5% adalah F1 tertimbang; F1 kelas stunting dari Tabel 5 adalah 0,776.", "Kutip dengan menyebut model prediksi (6 prediktor kNN) terpisah dari model eksplanatori (9 faktor)."),
]

METODE = [
("1. Kandidat dari literatur dan ketersediaan data", "Daftar kandidat prediktor disusun dari faktor risiko stunting di literatur (paper acuan, kerangka determinan) lalu dicocokkan dengan variabel yang tersedia di IFLS5.", "Azriani (2025): 23 prediktor dipilih dari scoping review faktor risiko stunting Asia Tenggara, kerangka malnutrisi UNICEF, dan ketersediaan data IFLS. Pola yang sama dipakai di sini.", "Inilah dasar 'mengapa variabel ini diambil'. Bukan p-value."),
("2. Pemeriksaan kualitas data", "Missing, outlier, logika, duplikat, korelasi, VIF (hasil EDA).", "Proposal Bab III 3.1.2", "Menentukan pembersihan dan penanganan missing, bukan membuang fitur secara otomatis."),
("3. Seleksi fitur dalam pemodelan", "Information Gain dan Gini Decrease dihitung HANYA pada data training (di dalam setiap fold), peringkat digabung.", "Azriani (2025) memakai Information Gain dan Gini Decrease; Novalina (2025) memakai mutual information.", "Seleksi berbasis data yang sah untuk model prediksi; terjadi SETELAH split."),
("p-value", "Dipakai untuk deskripsi/validasi hubungan, bukan untuk memilih fitur model.", "Novalina memakai analisis p-value bersama korelasi dan mutual information, sedangkan Agushybana memakai skrining univariat p<0,05 (Hosmer-Lemeshow) untuk regresi logistik; itu sesuai untuk model eksplanatori, kurang tepat untuk model prediksi.", "Alasan: bergantung ukuran sampel, mengabaikan interaksi, belum memperhitungkan klaster keluarga, dan bila dihitung pada seluruh data sebelum split menimbulkan kebocoran."),
("Variabel dipertahankan walau tidak bermakna", "Fitur yang diketahui dari literatur tetap menjadi kandidat.", "Agushybana mempertahankan variabel penting tanpa melihat signifikansi univariat.", "Contoh: jenis kelamin, pendidikan orang tua."),
("Arti kolom 'S/N/D' di Acuan_Fitur", "S = bermakna atau terpilih oleh paper; N = diuji tetapi tidak bermakna/tidak terpilih; D = hanya deskriptif atau disebut di teks.", "Dibaca langsung dari tabel hasil tiap paper.", "Hanya S yang boleh ditulis sebagai 'didukung literatur'; N dan D tidak boleh."),
]

LEGENDA = [
("S", "Bermakna secara statistik di model multivariat, atau terpilih oleh seleksi fitur paper tersebut"),
("N", "Dipakai/diuji di paper tersebut tetapi tidak bermakna atau tidak terpilih"),
("D", "Hanya deskriptif (tabel rata-rata) atau hanya disebut di teks/pembahasan, tidak dianalisis"),
("-", "Tidak ada di paper tersebut"),
("Catatan S Siramaneerat", "umur_ibu, umur_ayah, usia_sapih_bln, dan imt_ibu bermakna hanya pada model perkotaan, tidak pada model seluruh data."),
("Kategori bukti", "Kuat = S di >= 2 paper; Sedang = S di 1 paper; Diuji tidak bermakna = hanya N; Hanya disebut = hanya D; Tanpa paper = tidak ada"),
("Missing / ukuran efek / p (BH)", "Dihitung dari balita_ifls5.csv (deskriptif, seluruh data, abaikan klaster keluarga). AUC dipakai untuk fitur numerik (0,5 = tidak beda; < 0,5 berarti nilai lebih tinggi pada anak tidak stunting), Cramer's V untuk fitur kategorik/biner (<= 10 nilai). p dikoreksi Benjamini-Hochberg. BUKAN dasar seleksi fitur."),
]

NOMINAL = {"provinsi", "tempat_lahir_grup", "air_minum_kode", "toilet_kode", "limbah_kode", "sampah_kode"}


def kategori_bukti(a, s, n, z):
    kode = [a, s, n, z]
    ns, nn, nd = kode.count("S"), kode.count("N"), kode.count("D")
    if ns >= 2:
        return "Kuat (S di >= 2 paper)"
    if ns == 1:
        return "Sedang (S di 1 paper)"
    if nn >= 1:
        return "Diuji, tidak bermakna"
    if nd >= 1:
        return "Hanya disebut/deskriptif"
    return "Tanpa dukungan paper"


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    o = np.argsort(pv)
    r = pv[o] * pv.size / (np.arange(pv.size) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    res = np.empty_like(pv)
    res[o] = np.minimum(r, 1.0)
    out[ok] = res
    return out


def statistik_data(df):
    """missing %, ukuran efek univariat, p BH, catatan otomatis untuk tiap fitur."""
    from scipy.stats import chi2_contingency, mannwhitneyu
    y = df["stunting"].astype(int)
    fitur = [f[0] for f in FITUR]
    hasil, pvals = {}, []
    num_cols = [c for c in fitur if c in df.columns and c not in NOMINAL]
    rho = df[num_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman") if num_cols else pd.DataFrame()
    for f in fitur:
        if f not in df.columns:
            hasil[f] = dict(missing=np.nan, efek=np.nan, jenis="kolom tidak ada di CSV", p=np.nan, catatan="Kolom tidak ada di CSV")
            pvals.append(np.nan)
            continue
        x = pd.to_numeric(df[f], errors="coerce")
        m = x.notna()
        miss = round(100 * (1 - m.mean()), 1)
        xs, ys = x[m], y[m]
        efek, jenis, p = np.nan, "-", np.nan
        if len(xs) >= 30 and ys.nunique() == 2 and xs.nunique() >= 2:
            if xs.nunique() <= 10 or f in NOMINAL:
                tab = pd.crosstab(xs, ys)
                chi2, p, _, _ = chi2_contingency(tab, correction=False)
                k = min(tab.shape) - 1
                efek = float(np.sqrt(chi2 / (tab.values.sum() * k))) if k > 0 else np.nan
                jenis = "Cramer's V"
            else:
                n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
                u, p = mannwhitneyu(xs[ys == 1], xs[ys == 0], alternative="two-sided")
                efek = float(u / (n1 * n0))
                jenis = "AUC (0,5 = tidak beda)"
        cat = []
        if miss > 30:
            cat.append(f"Missing {miss}% (> 30%)")
        if xs.nunique() <= 10 and len(xs):
            top = xs.value_counts(normalize=True)
            if top.iloc[0] >= 0.95:
                cat.append(f"Hampir konstan ({top.iloc[0]*100:.1f}% bernilai {top.index[0]:g})")
        if f in rho.columns:
            pas = rho[f].drop(f).abs()
            pas = pas[pas >= 0.8].sort_values(ascending=False)
            if len(pas):
                cat.append("Berkorelasi tinggi dengan " + ", ".join(f"{k} (rho {rho.loc[f, k]:.2f})" for k in pas.index[:3]))
        if f == "anc_total":
            n_tw = int((x > 20).sum())
            if n_tw:
                cat.append(f"{n_tw} kasus > 20 kunjungan (tidak wajar)")
        if f in ("usia_air_bln", "usia_mpasi_bln", "usia_sapih_bln") and "umur_bln" in df.columns:
            n_tw = int((x > pd.to_numeric(df["umur_bln"], errors="coerce") + 1).sum())
            if n_tw:
                cat.append(f"{n_tw} nilai melebihi umur anak (tidak logis)")
        hasil[f] = dict(missing=miss, efek=efek, jenis=jenis, p=p, catatan="; ".join(cat))
        pvals.append(p)
    pbh = bh(np.array(pvals, dtype=float))
    for f, v in zip(fitur, pbh):
        hasil[f]["p_bh"] = v
    return hasil


def style(ws, lebar=None, wrap=True):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for c in ws[1]:
        c.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
        c.fill = PatternFill("solid", fgColor="1F4E78")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    for i, col in enumerate(ws.columns, 1):
        L = max(len(str(c.value or "")) for c in col)
        w = (lebar or {}).get(i, min(max(L + 2, 10), 60))
        ws.column_dimensions[get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.font = Font(name="Arial", size=10)
            c.alignment = Alignment(wrap_text=wrap, vertical="top")


def main():
    df = None
    if os.path.exists(CSV_DATASET):
        df = pd.read_csv(CSV_DATASET)
        fitur_csv = [c for c in df.columns if c not in KOLOM_ID + KOLOM_TARGET]
        fitur_py = [f[0] for f in FITUR]
        asing = [c for c in fitur_csv if c not in fitur_py]
        if asing:
            sys.exit(f"Kolom di CSV belum punya baris acuan di skrip ini: {asing}")
        print(f"CSV: {df.shape[0]} baris, {len(fitur_csv)} fitur (di tabel acuan: {len(fitur_py)})")
        st = statistik_data(df)
    else:
        print(f"[peringatan] CSV tidak ditemukan ({CSV_DATASET}); kolom statistik data dikosongkan.")
        st = {}

    rows = []
    for i, (f, kel, des, src, s33, a, s, n, z, temuan, alasan, desain) in enumerate(FITUR, 1):
        v = st.get(f, {})
        catatan = "; ".join(x for x in [desain, v.get("catatan", "")] if x)
        rows.append({
            "No": i, "Fitur (kolom dataset)": f, "Kelompok": kel, "Deskripsi": des, "Sumber IFLS5 (berkas.variabel)": src,
            "Status vs Tabel 3.3 proposal": s33,
            PAPER[0]: a, PAPER[1]: s, PAPER[2]: n, PAPER[3]: z,
            "Jumlah paper dengan S": [a, s, n, z].count("S"),
            "Kategori bukti literatur": kategori_bukti(a, s, n, z),
            "Temuan di paper (dari tabel hasil)": temuan, "Alasan masuk kandidat": alasan,
            "Catatan desain / kualitas data": catatan,
            "Missing (%)": v.get("missing", np.nan), "Ukuran efek univariat": v.get("efek", np.nan),
            "Jenis efek": v.get("jenis", ""), "p (BH)": v.get("p_bh", np.nan),
        })
    acuan = pd.DataFrame(rows)
    if os.path.exists(EDA_XLSX):
        sel = pd.read_excel(EDA_XLSX, sheet_name="18_Seleksi_Variabel").set_index("variabel")
        acuan["Status di model"] = acuan["Fitur (kolom dataset)"].map(sel["status"]).fillna("bukan kandidat")
        acuan["Alasan dipakai/dikeluarkan (EDA)"] = acuan["Fitur (kolom dataset)"].map(sel["alasan"])
    else:
        print(f"[peringatan] {EDA_XLSX} tidak ada; kolom status di model tidak dibuat. Jalankan 1_EDA.py dulu.")

    # ---- Tabel 3.3 vs final (persen terisi dari kamus) ----
    pct = {}
    if os.path.exists(KAMUS_XLSX):
        kv = pd.read_excel(KAMUS_XLSX, sheet_name="Kamus_Variabel")
        pct = {(r.file, r.variabel): r.pct_terisi for r in kv.itertuples()}
    t33 = pd.DataFrame([{
        "No (Tabel 3.3)": i, "Kode asli": k, "Deskripsi": d, "Keputusan": kp, "Kolom di dataset final": kol, "Alasan": al,
        "% terisi di IFLS5 (kamus)": pct.get((fl, k), np.nan)} for i, (k, fl, d, kp, kol, al) in enumerate(TABEL33, 1)])
    kp = t33["Keputusan"].str.split(" ").str[0].value_counts()

    # ---- Ringkasan (otomatis dari tabel) ----
    n_f = len(acuan)
    kat = acuan["Kategori bukti literatur"].value_counts()
    st33 = acuan["Status vs Tabel 3.3 proposal"].str.split(" ").str[0].value_counts()
    ring = [("Jumlah fitur kandidat di dataset final", n_f, "Kolom dataset dikurangi 3 ID dan 2 target (haz, stunting)")]
    if "Status di model" in acuan:
        pakai = acuan[acuan["Status di model"] == "DIPAKAI"]
        ring.append(("Fitur dipakai model / dikeluarkan", f"{len(pakai)} / {n_f - len(pakai)}", "Dari 1_EDA (sheet 18_Seleksi_Variabel)"))
        ring.append(("Fitur dipakai model dengan kode S di >= 1 paper", int((pakai["Jumlah paper dengan S"] > 0).sum()),
                     "Sisanya dari kerangka WHO (Subbab 2.3.1) atau hanya disebut di paper (D)"))
    for p in PAPER:
        c = acuan[p]
        ring.append((f"{p}: S / N / D", f"{(c=='S').sum()} / {(c=='N').sum()} / {(c=='D').sum()}",
                     "S = bermakna/terpilih; N = diuji tidak bermakna; D = hanya deskriptif/disebut"))
    for k in ["Kuat (S di >= 2 paper)", "Sedang (S di 1 paper)", "Diuji, tidak bermakna", "Hanya disebut/deskriptif", "Tanpa dukungan paper"]:
        ring.append((f"Fitur dengan bukti: {k}", int(kat.get(k, 0)), ""))
    ring += [("Variabel di Tabel 3.3 proposal", len(t33), "Baris Tabel 3.3 yang dibandingkan")]
    for k, v in kp.items():
        ring.append((f"... keputusan: {k}", int(v), ""))
    ring += [("Fitur Tabel 3.3 yang masih ada langsung / turunan / agregasi",
              int(sum(v for k, v in st33.items() if k != "Baru")), "Status 'Baru' = bukan dari Tabel 3.3: %d fitur" % int(st33.get("Baru", 0))),
             ("", "", ""),
             ("Jawaban singkat: dari mana variabel diambil?", "",
              "Kandidat diambil dari empat paper acuan yang memakai IFLS, kerangka faktor risiko WHO (Subbab 2.3.1), dan ketersediaan di IFLS5, meniru Azriani (2025). Seleksi akhir dilakukan oleh Information Gain + Gini Decrease pada data training. p-value hanya untuk deskripsi. 'Didukung literatur' hanya boleh ditulis untuk fitur dengan kode S; N dan D tidak."),
             ("Peringatan", "","TB + umur + JK menentukan HAZ. Model B (tanpa TB, dengan WAZ) adalah model final; Model A dan C (dengan TB) hanya pembanding batas atas.")]
    ringkasan = pd.DataFrame(ring, columns=["Indikator", "Nilai", "Keterangan"])

    sheets = {
        "Ringkasan": ringkasan, "Acuan_Fitur": acuan, "Tabel33_vs_Final": t33,
        "Kandidat_Belum_Ada": pd.DataFrame(BELUM_ADA, columns=["Konsep di paper", "Dipakai oleh", "Cara memperoleh dari IFLS5", "Tingkat kesulitan", "Rekomendasi"]),
        "Metode_Pemilihan": pd.DataFrame(METODE, columns=["Tahap", "Apa yang dilakukan", "Dasar / acuan", "Catatan"]),
        "Catatan_Paper": pd.DataFrame(CATATAN_PAPER, columns=["Paper", "Ketidakkonsistenan yang ditemukan", "Saran saat mengutip"]),
        "Legenda": pd.DataFrame(LEGENDA, columns=["Kode / istilah", "Arti"]),
    }
    os.makedirs(os.path.dirname(OUT_XLSX), exist_ok=True)
    from openpyxl.styles import PatternFill
    warna = {"S": "C6EFCE", "N": "FFEB9C", "D": "DDEBF7"}
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        for nm, d in sheets.items():
            d.to_excel(w, sheet_name=nm, index=False)
            lebar = {"Acuan_Fitur": {2: 22, 3: 20, 4: 34, 5: 30, 6: 32, 12: 22, 13: 70, 14: 40, 15: 50}}.get(nm)
            style(w.sheets[nm], lebar)
        ws = w.sheets["Acuan_Fitur"]
        for row in ws.iter_rows(min_row=2, min_col=7, max_col=10):
            for c in row:
                if c.value in warna:
                    c.fill = PatternFill("solid", fgColor=warna[c.value])
        for row in ws.iter_rows(min_row=2, min_col=17, max_col=19):
            for c in row:
                if isinstance(c.value, float):
                    c.number_format = "0.000"
    print(f"Selesai: {OUT_XLSX}")
    print(ringkasan.head(14).to_string(index=False))


if __name__ == "__main__":
    main()