"""
build_ifls5_dataset.py
======================
Membangun SATU dataset "siap model" (1 baris = 1 individu dewasa) dari data mentah
IFLS-5 (2014/2015, format Stata .dta).

Cara pakai
----------
    python build_ifls5_dataset.py                      # auto-detect folder IFLS5
    python build_ifls5_dataset.py --data_dir "Dataset/IFLS5"   # atau tentukan manual

Dependensi:  pip install pandas numpy xlsxwriter

Output
------
    ifls5_model_ready.xlsx   -> sheet: dataset | data_dictionary | missing_summary | readme
    ifls5_model_ready.csv    -> isi sheet 'dataset' (lebih enak dibaca pandas)

Unit analisis : individu (pidlink) yang menyelesaikan wawancara Buku 3A IFLS-5.
Penggabungan  : kunci individu = pidlink ; kunci rumah tangga = hhid14.
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# KONFIGURASI (opsional) - isi manual jika auto-detect gagal, contoh:
#   DATA_DIR = r"C:\\Users\\kamu\\Dataset\\IFLS5"
# Biarkan None agar script mencari sendiri folder yang berisi ptrack.dta
# --------------------------------------------------------------------------- #
DATA_DIR = None
OUT_DIR = None   # None -> folder "output" di sebelah file script ini


def find_data_dir(user_value=None):
    """Cari folder yang berisi file IFLS (.dta), dicek lewat keberadaan ptrack.dta."""
    here = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    cands = []
    if user_value:
        cands.append(user_value)
    for base in (cwd, here, os.path.dirname(here), os.path.dirname(cwd)):
        cands += [base,
                  os.path.join(base, "IFLS5"),
                  os.path.join(base, "Dataset", "IFLS5"),
                  os.path.join(base, "Dataset"),
                  os.path.join(base, "data", "IFLS5")]
    for c in cands:
        if os.path.isfile(os.path.join(c, "ptrack.dta")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Folder data IFLS5 (berisi ptrack.dta, b3a_cov.dta, dst.) tidak ditemukan.\n"
        "Solusi: jalankan dengan  --data_dir \"path/ke/IFLS5\"  atau isi DATA_DIR di bagian atas script.\n"
        f"Folder kerja saat ini: {cwd}")


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
BAD_TEXT = re.compile(r"(don'?t know|missing|refuse|tidak tahu|not applicable)", re.I)


def rd(data_dir, name, cols=None, labels=False):
    """Baca file .dta. labels=True -> kolom kategori jadi teks berlabel."""
    return pd.read_stata(
        os.path.join(data_dir, f"{name}.dta"),
        convert_categoricals=labels,
        columns=cols,
    )


def clean_label(s):
    """'1:Yes' -> 'Yes' ; '0Self owned => KR05a' -> 'Self owned'; DK/Missing -> NaN."""
    out = s.astype("object").copy()
    mask = out.map(lambda v: isinstance(v, str))
    cleaned = out[mask].map(
        lambda v: re.sub(r"\s*=>.*$", "", re.sub(r"^\s*\d+\s*:?\s*", "", v)).strip()
    )
    cleaned = cleaned.where(~cleaned.map(lambda v: bool(BAD_TEXT.search(v))), np.nan)
    out[mask] = cleaned
    return out


def yn(series):
    """Kode IFLS 1=Ya, 3=Tidak (selain itu NaN) -> 1/0."""
    return series.map({1: 1, 3: 0}).astype("float")


def num_clean(s, bad_ge=None, lo=None, hi=None):
    s = pd.to_numeric(s, errors="coerce")
    if bad_ge is not None:
        s = s.mask(s >= bad_ge)
    if lo is not None:
        s = s.mask(s < lo)
    if hi is not None:
        s = s.mask(s > hi)
    return s


PROVINCE = {
    11: "Aceh", 12: "Sumatera Utara", 13: "Sumatera Barat", 14: "Riau", 15: "Jambi",
    16: "Sumatera Selatan", 17: "Bengkulu", 18: "Lampung", 19: "Bangka Belitung",
    21: "Kepulauan Riau", 31: "DKI Jakarta", 32: "Jawa Barat", 33: "Jawa Tengah",
    34: "DI Yogyakarta", 35: "Jawa Timur", 36: "Banten", 51: "Bali",
    52: "Nusa Tenggara Barat", 53: "Nusa Tenggara Timur", 61: "Kalimantan Barat",
    62: "Kalimantan Tengah", 63: "Kalimantan Selatan", 64: "Kalimantan Timur",
    65: "Kalimantan Utara", 71: "Sulawesi Utara", 72: "Sulawesi Tengah",
    73: "Sulawesi Selatan", 74: "Sulawesi Tenggara", 75: "Gorontalo",
    76: "Sulawesi Barat", 81: "Maluku", 82: "Maluku Utara", 91: "Papua Barat", 94: "Papua",
}

# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def build(data_dir):
    log = lambda m: print(f"[IFLS5] {m}", flush=True)

    # ---------------- 1. Basis: individu dewasa dengan Buku 3A selesai ------- #
    log("Basis individu (b3a_cov) ...")
    cov = rd(data_dir, "b3a_cov", ["pidlink", "hhid14", "age", "sex", "marstat", "proxy", "result"], labels=True)
    cov_n = rd(data_dir, "b3a_cov", ["pidlink", "age", "proxy", "result"])
    base = pd.DataFrame({"pidlink": cov.pidlink, "hhid14": cov.hhid14})
    base["age"] = num_clean(cov_n.age, bad_ge=120, lo=0)
    base["sex"] = clean_label(cov.sex)
    base["marital_status"] = clean_label(cov.marstat)
    base["proxy_respondent"] = yn(cov_n.proxy)
    base["_result"] = cov_n.result.values
    base = base[base["_result"] == 1].drop(columns="_result")          # wawancara selesai

    # bobot & umur cadangan dari ptrack
    pt = rd(data_dir, "ptrack", ["pidlink", "age_14", "pwt14xa", "pwt14la"])
    pt = pt.drop_duplicates("pidlink")
    base = base.merge(pt, on="pidlink", how="left")
    base["age"] = base["age"].fillna(num_clean(base.pop("age_14"), bad_ge=120))
    base = base.rename(columns={"pwt14xa": "weight_cross_section", "pwt14la": "weight_longitudinal"})
    base = base[base.age >= 15].copy()
    log(f"  individu dewasa (>=15 th) dengan wawancara 3A selesai: {len(base):,}")

    # ---------------- 2. Wilayah ------------------------------------------- #
    log("Wilayah ...")
    sc = rd(data_dir, "bk_sc1", ["hhid14", "sc05", "sc01_14_14"]).drop_duplicates("hhid14")
    sc["province"] = sc.sc01_14_14.map(PROVINCE)
    sc["area_type"] = sc.sc05.map({1: "Urban", 2: "Rural"})
    base = base.merge(sc[["hhid14", "province", "area_type"]], on="hhid14", how="left")

    # ---------------- 3. Pendidikan & literasi ----------------------------- #
    log("Pendidikan ...")
    dl_l = rd(data_dir, "b3a_dl1", ["pidlink", "dl04", "dl06", "dl02", "dl03", "dl03b", "dl03d"], labels=True)
    dl_n = rd(data_dir, "b3a_dl1", ["pidlink", "dl07"])
    edu = pd.DataFrame({"pidlink": dl_l.pidlink})
    edu["ever_attended_school"] = yn(rd(data_dir, "b3a_dl1", ["dl04"]).dl04)
    edu["highest_education_level"] = clean_label(dl_l.dl06)
    edu["highest_grade_completed"] = num_clean(dl_n.dl07, bad_ge=90)
    edu["can_read_indonesian"] = yn(rd(data_dir, "b3a_dl1", ["dl02"]).dl02)
    edu["can_write_indonesian"] = yn(rd(data_dir, "b3a_dl1", ["dl03"]).dl03)
    edu["has_cellphone"] = yn(rd(data_dir, "b3a_dl1", ["dl03b"]).dl03b)
    edu["has_internet_access"] = yn(rd(data_dir, "b3a_dl1", ["dl03d"]).dl03d)
    base = base.merge(edu, on="pidlink", how="left")

    # ---------------- 4. Ketenagakerjaan ----------------------------------- #
    log("Ketenagakerjaan ...")
    tk1 = rd(data_dir, "b3a_tk1", ["pidlink", "tk01"], labels=True)
    tk1["main_activity_last_week"] = clean_label(tk1.pop("tk01"))
    base = base.merge(tk1, on="pidlink", how="left")

    tk2_l = rd(data_dir, "b3a_tk2", ["pidlink", "tk24a", "tk24a5"], labels=True)
    tk2_n = rd(data_dir, "b3a_tk2", ["pidlink", "tk25a1", "tk21a", "tk22a"])
    tk2 = pd.DataFrame({"pidlink": tk2_l.pidlink})
    tk2["job_type"] = clean_label(tk2_l.tk24a)
    tk2["has_work_contract"] = rd(data_dir, "b3a_tk2", ["tk24a5"]).tk24a5.map({1: 1, 2: 1, 3: 0}).astype("float")
    tk2["monthly_wage_idr"] = num_clean(tk2_n.tk25a1, bad_ge=99_999_990, lo=0)
    tk2["hours_worked_last_week"] = num_clean(tk2_n.tk21a, bad_ge=168, lo=0)
    tk2["is_employed"] = 1
    base = base.merge(tk2, on="pidlink", how="left")
    base["is_employed"] = base["is_employed"].fillna(0).astype(int)
    base["log_monthly_wage"] = np.log1p(base["monthly_wage_idr"])

    # ---------------- 5. Kesehatan subjektif, rokok, masa kecil ------------ #
    log("Kesehatan subjektif & perilaku ...")
    kk_l = rd(data_dir, "b3b_kk1", ["pidlink", "kk01", "kk02c"], labels=True)
    kk_n = rd(data_dir, "b3b_kk1", ["pidlink", "kk01", "kk02c", "kk02a"])
    kk = pd.DataFrame({"pidlink": kk_l.pidlink})
    kk["self_rated_health"] = clean_label(kk_l.kk01)
    kk["self_rated_health_score"] = kk_n.kk01.where(kk_n.kk01.isin([1, 2, 3, 4]))   # 1=sangat sehat ... 4=sangat tidak sehat
    kk["health_vs_last_year"] = clean_label(kk_l.kk02c)
    kk["days_missed_activity_4wk"] = num_clean(kk_n.kk02a, bad_ge=90, lo=0, hi=31)
    base = base.merge(kk, on="pidlink", how="left")

    km = rd(data_dir, "b3b_km", ["pidlink", "km01a", "km04", "km08"])
    smoke = np.select(
        [km.km01a == 3, (km.km01a == 1) & (km.km04 == 1), (km.km01a == 1) & (km.km04 == 3)],
        ["Never", "Current", "Former"], default=None)
    km2 = pd.DataFrame({"pidlink": km.pidlink, "smoking_status": smoke})
    km2["smoking_status"] = km2["smoking_status"].replace({"None": np.nan})
    km2["cigarettes_per_day"] = num_clean(km.km08, bad_ge=98, lo=0).where(km2.smoking_status == "Current")
    base = base.merge(km2, on="pidlink", how="left")

    eh = rd(data_dir, "b3b_eh", ["pidlink", "eh01", "eh08"], labels=True)
    eh2 = pd.DataFrame({"pidlink": eh.pidlink, "childhood_health": clean_label(eh.eh01)})
    eh2["childhood_hunger"] = yn(rd(data_dir, "b3b_eh", ["eh08"]).eh08)
    base = base.merge(eh2, on="pidlink", how="left")

    # ---------------- 6. Penyakit kronis (didiagnosis tenaga kesehatan) ----- #
    log("Penyakit kronis ...")
    cd3 = rd(data_dir, "b3b_cd3", ["pidlink", "cdtype", "cd05"])
    cd3["yes"] = (cd3.cd05 == 1).astype(float).where(cd3.cd05.isin([1, 3]))
    w = cd3.pivot_table(index="pidlink", columns="cdtype", values="yes", aggfunc="max")
    chronic = pd.DataFrame({
        "hypertension_diagnosed": w["A"],     # kode A = hipertensi (divalidasi: SBP terukur 153 vs 126 mmHg)
        "diabetes_diagnosed": w["B"],         # kode B = diabetes (divalidasi: 39% pengguna obat diabetes)
        "n_chronic_conditions": w.sum(axis=1, min_count=1),   # jumlah dari 16 kondisi kronis
    }).reset_index()
    base = base.merge(chronic, on="pidlink", how="left")

    # ---------------- 7. Depresi CES-D 10 ---------------------------------- #
    log("CES-D 10 ...")
    kp = rd(data_dir, "b3b_kp", ["pidlink", "kptype", "kp02"])
    kp["v"] = kp.kp02.map({1: 0, 2: 1, 3: 2, 4: 3})
    kp.loc[kp.kptype.isin(["E", "H"]), "v"] = 3 - kp.loc[kp.kptype.isin(["E", "H"]), "v"]  # item positif dibalik
    kpw = kp.pivot_table(index="pidlink", columns="kptype", values="v", aggfunc="first")
    cesd = pd.DataFrame({"pidlink": kpw.index, "cesd10_score": kpw.sum(axis=1, min_count=10).values})
    cesd["depressive_symptoms"] = (cesd.cesd10_score >= 10).astype(float).where(cesd.cesd10_score.notna())
    base = base.merge(cesd, on="pidlink", how="left")

    # ---------------- 8. Asuransi kesehatan & kesejahteraan subjektif ------ #
    log("Asuransi & kesejahteraan ...")
    ak = rd(data_dir, "b3b_ak1", ["pidlink", "ak01"])
    ak["y"] = ak.ak01.map({1: 1, 3: 0})
    ins = ak.groupby("pidlink")["y"].max().rename("has_health_insurance").reset_index()
    base = base.merge(ins, on="pidlink", how="left")

    sw_l = rd(data_dir, "b3a_sw", ["pidlink", "sw00", "sw06"], labels=True)
    sw_n = rd(data_dir, "b3a_sw", ["pidlink", "sw00", "sw01"])
    sw = pd.DataFrame({"pidlink": sw_l.pidlink})
    sw["life_satisfaction"] = clean_label(sw_l.sw00)
    sw["life_satisfaction_score"] = sw_n.sw00.where(sw_n.sw00.isin([1, 2, 3, 4, 5]))   # 1=sangat puas ... 5=sangat tidak puas
    sw["economic_step_now"] = sw_n.sw01.where(sw_n.sw01.isin([1, 2, 3, 4, 5, 6]))      # 1=termiskin ... 6=terkaya
    base = base.merge(sw, on="pidlink", how="left")

    # ---------------- 9. Pengukuran fisik (Buku US) ------------------------ #
    log("Pengukuran fisik ...")
    ucols = ["pidlink", "us06", "us04", "us06a", "us07a1", "us07a2", "us07b1", "us07b2", "us07c1", "us07c2",
             "us07ap", "us13", "us20a", "us20b", "us20c", "us20d", "us18ab", "us15a"]
    us = rd(data_dir, "bus_us", ucols).drop_duplicates("pidlink")
    ph = pd.DataFrame({"pidlink": us.pidlink})
    ph["height_cm"] = num_clean(us.us04, lo=100, hi=220)
    ph["weight_kg"] = num_clean(us.us06, lo=20, hi=250)
    ph["bmi"] = (ph.weight_kg / (ph.height_cm / 100) ** 2).where(lambda s: s.between(10, 60))
    ph["waist_cm"] = num_clean(us.us06a, lo=40, hi=200)

    def bp(sys_cols, dia_cols, lo, hi):
        m = us[sys_cols].where(lambda d: d.apply(lambda c: c.between(lo[0], hi[0])))
        return m
    sbp_all = us[["us07a1", "us07b1", "us07c1"]].where(lambda d: d.apply(lambda c: c.between(70, 270)))
    dbp_all = us[["us07a2", "us07b2", "us07c2"]].where(lambda d: d.apply(lambda c: c.between(40, 150)))
    # pola IFLS/standar: rata-rata pengukuran ke-2 & ke-3; jika tidak ada pakai yang tersedia
    sbp23 = sbp_all[["us07b1", "us07c1"]].mean(axis=1)
    dbp23 = dbp_all[["us07b2", "us07c2"]].mean(axis=1)
    ph["systolic_bp"] = sbp23.fillna(sbp_all.mean(axis=1)).values
    ph["diastolic_bp"] = dbp23.fillna(dbp_all.mean(axis=1)).values
    ph.loc[ph.systolic_bp <= ph.diastolic_bp, ["systolic_bp", "diastolic_bp"]] = np.nan
    ph["pulse_bpm"] = num_clean(us.us07ap, lo=30, hi=220)
    ph["hemoglobin_gdl"] = num_clean(us.us13, lo=3, hi=20)
    grip = us[["us20a", "us20b", "us20c", "us20d"]].where(lambda d: d.apply(lambda c: c.between(1, 90)))
    ph["grip_strength_max_kg"] = grip.max(axis=1)
    ph["takes_hypertension_medicine"] = yn(us.us18ab)
    ph["_pregnant"] = (us.us15a == 1)
    base = base.merge(ph, on="pidlink", how="left")

    # --- fitur turunan klinis ---
    adult18 = base.age >= 18
    bmi_cat = pd.cut(base.bmi, [0, 18.5, 23, 25, 30, 100], right=False,
                     labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II"])   # kriteria Asia-Pasifik
    base["bmi_category"] = np.where(adult18 & ~base["_pregnant"].fillna(False), bmi_cat.astype(object), np.nan)
    base["obese"] = np.where(base.bmi_category.isna(), np.nan, base.bmi_category.isin(["Obese I", "Obese II"]).astype(float))
    htn = ((base.systolic_bp >= 140) | (base.diastolic_bp >= 90) | (base.takes_hypertension_medicine == 1)).astype(float)
    base["hypertension_measured"] = htn.where(adult18 & base.systolic_bp.notna())
    low_hb = np.where(base.sex == "Female", base.hemoglobin_gdl < 12, base.hemoglobin_gdl < 13).astype(float)
    base["anemia"] = pd.Series(low_hb, index=base.index).where(
        base.hemoglobin_gdl.notna() & adult18 & ~base["_pregnant"].fillna(False) & base.sex.notna())
    base.drop(columns="_pregnant", inplace=True)

    # ---------------- 10. Rumah tangga: ukuran, perumahan, pengeluaran ------ #
    log("Rumah tangga ...")
    mem = rd(data_dir, "ptrack", ["hhid14", "member14"])
    hhsize = mem[mem.member14 == 1].groupby("hhid14").size().rename("household_size").reset_index()
    base = base.merge(hhsize, on="hhid14", how="left")

    kr_l = rd(data_dir, "b2_kr", ["hhid14", "kr03", "kr13", "kr20"], labels=True)
    kr = pd.DataFrame({"hhid14": kr_l.hhid14})
    kr["house_ownership"] = clean_label(kr_l.kr03)
    kr["has_electricity"] = yn(rd(data_dir, "b2_kr", ["kr11"]).kr11)
    kr["drinking_water_source"] = clean_label(kr_l.kr13)
    kr["toilet_type"] = clean_label(kr_l.kr20)
    base = base.merge(kr.drop_duplicates("hhid14"), on="hhid14", how="left")

    # Pengeluaran rumah tangga (per bulan, Rp) = pangan mingguan*52/12 + nonpangan bulanan + nonpangan tahunan/12
    ks1 = rd(data_dir, "b1_ks1", ["hhid14", "ks02", "ks03"])
    for c in ("ks02", "ks03"):
        ks1[c] = num_clean(ks1[c], bad_ge=99_999_990, lo=0)
    food = ks1.groupby("hhid14").agg(a=("ks02", "sum"), b=("ks03", "sum"))
    food = ((food.a + food.b) * 52 / 12).rename("food_month").reset_index()

    ks2 = rd(data_dir, "b1_ks2", ["hhid14", "ks06"]); ks2["ks06"] = num_clean(ks2.ks06, bad_ge=99_999_990, lo=0)
    nf_m = ks2.groupby("hhid14")["ks06"].sum().rename("nonfood_month").reset_index()

    ks3 = rd(data_dir, "b1_ks3", ["hhid14", "ks08", "ks09a"])
    ks3["ks08"] = num_clean(ks3.ks08, bad_ge=99_999_990, lo=0)
    ks3["ks09a"] = num_clean(ks3.ks09a, bad_ge=99_999_990, hi=1e9, lo=0)
    nf_y = ks3.groupby("hhid14").agg(a=("ks08", "sum"), b=("ks09a", "sum"))
    nf_y = ((nf_y.a + nf_y.b) / 12).rename("nonfood_annual_per_month").reset_index()

    ks0 = rd(data_dir, "b1_ks0", ["hhid14", "ks10aa", "ks11aa", "ks12aa"])
    for c in ("ks10aa", "ks11aa", "ks12aa"):
        ks0[c] = num_clean(ks0[c], bad_ge=99_999_990, lo=0)
    ks0["education_month"] = ks0[["ks10aa", "ks11aa", "ks12aa"]].sum(axis=1) / 12
    ex = food.merge(nf_m, on="hhid14", how="outer").merge(nf_y, on="hhid14", how="outer") \
             .merge(ks0[["hhid14", "education_month"]], on="hhid14", how="outer")
    ex["household_expenditure_month_idr"] = ex[["food_month", "nonfood_month", "nonfood_annual_per_month", "education_month"]].sum(axis=1, min_count=1)
    base = base.merge(ex[["hhid14", "household_expenditure_month_idr"]], on="hhid14", how="left")
    base["per_capita_expenditure_month_idr"] = base.household_expenditure_month_idr / base.household_size
    base["log_per_capita_expenditure"] = np.log(base.per_capita_expenditure_month_idr.where(lambda s: s > 0))
    base["expenditure_quintile"] = pd.qcut(base.per_capita_expenditure_month_idr.where(lambda s: s > 0), 5,
                                           labels=["Q1 (poorest)", "Q2", "Q3", "Q4", "Q5 (richest)"]).astype(object)

    # ---------------- 11. Rapikan ------------------------------------------ #
    base["age_group"] = pd.cut(base.age, [14, 24, 34, 44, 54, 64, 120],
                               labels=["15-24", "25-34", "35-44", "45-54", "55-64", "65+"]).astype(object)
    # buang kolom redundan
    base = base.drop(columns=["household_expenditure_month_idr"])
    # buang baris tanpa jenis kelamin (tidak bisa dipakai pemodelan) -> hanya 6 baris
    base = base[base.sex.notna()].reset_index(drop=True)
    return base


# --------------------------------------------------------------------------- #
# Kamus data
# --------------------------------------------------------------------------- #
DICT = [
    # (kolom, deskripsi, peran, sumber)
    ("pidlink", "ID unik individu (kunci gabung lintas gelombang IFLS)", "ID", "b3a_cov"),
    ("hhid14", "ID rumah tangga 2014", "ID", "b3a_cov"),
    ("age", "Umur (tahun)", "Fitur", "b3a_cov / ptrack"),
    ("age_group", "Kelompok umur", "Fitur", "turunan dari age"),
    ("sex", "Jenis kelamin", "Fitur", "b3a_cov"),
    ("marital_status", "Status perkawinan", "Fitur", "b3a_cov.marstat"),
    ("proxy_respondent", "1 = diwawancara lewat perwakilan (proxy)", "Metadata", "b3a_cov.proxy"),
    ("weight_cross_section", "Bobot survei cross-section 2014 (pakai untuk estimasi populasi)", "Bobot", "ptrack.pwt14xa"),
    ("weight_longitudinal", "Bobot longitudinal 2014", "Bobot", "ptrack.pwt14la"),
    ("province", "Provinsi tempat tinggal 2014 (13 provinsi sampel IFLS + sedikit migran ke provinsi lain)", "Fitur", "bk_sc1.sc01_14_14"),
    ("area_type", "Urban / Rural", "Fitur", "bk_sc1.sc05"),
    ("ever_attended_school", "1 = pernah/sedang bersekolah", "Fitur", "b3a_dl1.dl04"),
    ("highest_education_level", "Jenjang pendidikan tertinggi yang pernah diikuti", "Fitur", "b3a_dl1.dl06"),
    ("highest_grade_completed", "Kelas/tingkat tertinggi yang diselesaikan", "Fitur", "b3a_dl1.dl07"),
    ("can_read_indonesian", "1 = bisa membaca koran berbahasa Indonesia", "Fitur", "b3a_dl1.dl02"),
    ("can_write_indonesian", "1 = bisa menulis surat dalam bahasa Indonesia", "Fitur", "b3a_dl1.dl03"),
    ("has_cellphone", "1 = punya ponsel", "Fitur", "b3a_dl1.dl03b"),
    ("has_internet_access", "1 = punya akses internet", "Fitur", "b3a_dl1.dl03d"),
    ("main_activity_last_week", "Aktivitas utama seminggu terakhir", "Fitur", "b3a_tk1.tk01"),
    ("is_employed", "1 = punya pekerjaan utama (ada data pekerjaan di Buku 3A TK2)", "Fitur / Target", "b3a_tk2"),
    ("job_type", "Status pekerjaan utama (buruh, wirausaha, PNS, dst.)", "Fitur", "b3a_tk2.tk24a"),
    ("has_work_contract", "1 = bekerja dengan kontrak", "Fitur", "b3a_tk2.tk24a5"),
    ("monthly_wage_idr", "Upah/gaji sebulan terakhir pekerjaan utama (Rp). Hanya terisi untuk yang menjawab", "Target kandidat", "b3a_tk2.tk25a1"),
    ("log_monthly_wage", "log(1 + monthly_wage_idr)", "Target kandidat", "turunan"),
    ("hours_worked_last_week", "Total jam kerja minggu lalu (pekerjaan 1)", "Fitur", "b3a_tk2.tk21a"),
    ("self_rated_health", "Kesehatan umum (self-rated)", "Target kandidat", "b3b_kk1.kk01"),
    ("self_rated_health_score", "Skor self-rated health: 1=sangat sehat ... 4=sangat tidak sehat", "Target kandidat", "b3b_kk1.kk01"),
    ("health_vs_last_year", "Kesehatan dibanding 12 bulan lalu", "Fitur", "b3b_kk1.kk02c"),
    ("days_missed_activity_4wk", "Hari absen dari aktivitas utama karena sakit (4 minggu terakhir)", "Fitur", "b3b_kk1.kk02a"),
    ("smoking_status", "Never / Former / Current (semua jenis tembakau)", "Fitur", "b3b_km.km01a, km04"),
    ("cigarettes_per_day", "Batang rokok per hari (hanya perokok aktif)", "Fitur", "b3b_km.km08"),
    ("childhood_health", "Kesehatan semasa kecil (retrospektif)", "Fitur", "b3b_eh.eh01"),
    ("childhood_hunger", "1 = pernah kelaparan semasa kecil", "Fitur", "b3b_eh.eh08"),
    ("hypertension_diagnosed", "1 = pernah didiagnosis hipertensi oleh tenaga kesehatan", "Target kandidat", "b3b_cd3 (cdtype=A)"),
    ("diabetes_diagnosed", "1 = pernah didiagnosis diabetes", "Target kandidat", "b3b_cd3 (cdtype=B)"),
    ("n_chronic_conditions", "Jumlah kondisi kronis (dari 16 jenis) yang pernah didiagnosis", "Fitur / Target", "b3b_cd3"),
    ("cesd10_score", "Skor CES-D 10 (0-30; makin tinggi makin banyak gejala depresi)", "Target kandidat", "b3b_kp"),
    ("depressive_symptoms", "1 = CES-D 10 >= 10 (indikasi gejala depresi)", "Target kandidat", "turunan dari cesd10_score"),
    ("has_health_insurance", "1 = punya asuransi/jaminan kesehatan", "Fitur", "b3b_ak1.ak01"),
    ("life_satisfaction", "Kepuasan hidup secara keseluruhan", "Fitur / Target", "b3a_sw.sw00"),
    ("life_satisfaction_score", "1=sangat puas ... 5=sangat tidak puas", "Fitur / Target", "b3a_sw.sw00"),
    ("economic_step_now", "Posisi ekonomi sekarang (tangga 1=termiskin ... 6=terkaya)", "Fitur", "b3a_sw.sw01"),
    ("height_cm", "Tinggi badan terukur (cm)", "Fitur", "bus_us.us04"),
    ("weight_kg", "Berat badan terukur (kg)", "Fitur", "bus_us.us06"),
    ("bmi", "Indeks massa tubuh (kg/m2)", "Target kandidat", "turunan"),
    ("bmi_category", "Kategori BMI kriteria Asia-Pasifik (>=18 th, tidak hamil)", "Target kandidat", "turunan"),
    ("obese", "1 = BMI >= 25 (Obese I/II, Asia-Pasifik)", "Target kandidat", "turunan"),
    ("waist_cm", "Lingkar pinggang (cm)", "Fitur", "bus_us.us06a"),
    ("systolic_bp", "Tekanan darah sistolik (mmHg; rata-rata ukuran ke-2 & ke-3)", "Fitur / LEAKAGE jika target hipertensi", "bus_us.us07*1"),
    ("diastolic_bp", "Tekanan darah diastolik (mmHg)", "Fitur / LEAKAGE jika target hipertensi", "bus_us.us07*2"),
    ("pulse_bpm", "Denyut nadi (bpm)", "Fitur", "bus_us.us07ap"),
    ("hemoglobin_gdl", "Hemoglobin (g/dL)", "Fitur / LEAKAGE jika target anemia", "bus_us.us13"),
    ("grip_strength_max_kg", "Kekuatan genggaman maksimum (kg)", "Fitur", "bus_us.us20*"),
    ("takes_hypertension_medicine", "1 = sedang minum obat hipertensi", "Fitur / LEAKAGE jika target hipertensi", "bus_us.us18ab"),
    ("hypertension_measured", "1 = SBP>=140 atau DBP>=90 atau minum obat hipertensi (>=18 th)", "Target kandidat", "turunan"),
    ("anemia", "1 = Hb < 12 (perempuan) / < 13 (laki-laki), tidak hamil, >=18 th", "Target kandidat", "turunan"),
    ("household_size", "Jumlah anggota rumah tangga", "Fitur", "ptrack.member14"),
    ("house_ownership", "Status kepemilikan rumah", "Fitur", "b2_kr.kr03"),
    ("has_electricity", "1 = rumah tangga memakai listrik", "Fitur", "b2_kr.kr11"),
    ("drinking_water_source", "Sumber utama air minum", "Fitur", "b2_kr.kr13"),
    ("toilet_type", "Jenis fasilitas buang air besar", "Fitur", "b2_kr.kr20"),
    ("per_capita_expenditure_month_idr", "Perkiraan pengeluaran per kapita per bulan (Rp)", "Target kandidat", "b1_ks0/ks1/ks2/ks3 (turunan)"),
    ("log_per_capita_expenditure", "log pengeluaran per kapita", "Target kandidat", "turunan"),
    ("expenditure_quintile", "Kuintil pengeluaran per kapita (Q1=termiskin)", "Fitur / Target", "turunan"),
]


def make_dictionary(df):
    rows = []
    for col, desc, role, src in DICT:
        if col not in df:
            continue
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        row = {
            "variable": col, "description": desc, "role": role, "source": src,
            "type": "numeric" if is_num else "categorical",
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 1),
            "n_unique": int(s.nunique()),
        }
        if is_num:
            row["summary"] = (f"min={s.min():.4g} | median={s.median():.4g} | mean={s.mean():.4g} | max={s.max():.4g}")
        else:
            vc = s.value_counts().head(6)
            row["summary"] = " ; ".join(f"{k} ({v:,})" for k, v in vc.items())
        rows.append(row)
    return pd.DataFrame(rows)


README = [
    ["Dataset", "IFLS-5 (Indonesia Family Life Survey, 2014/15) - dataset analitik tingkat individu"],
    ["Unit analisis", "1 baris = 1 individu dewasa (>=15 tahun) dengan wawancara Buku 3A selesai"],
    ["Kunci", "pidlink (individu), hhid14 (rumah tangga)"],
    ["Cakupan variabel", "Demografi, wilayah, pendidikan, pekerjaan, kesehatan (subjektif & terukur), CES-D 10, penyakit kronis, "
                         "asuransi, kesejahteraan subjektif, perumahan, pengeluaran rumah tangga"],
    ["Peran kolom", "ID / Fitur / Target kandidat / Bobot / Metadata (lihat sheet data_dictionary). Pilih SATU target lalu "
                    "buang kolom turunannya & kolom bertanda LEAKAGE."],
    ["Nilai hilang", "Kosong = tidak ditanyakan (skip pattern), tidak terukur, atau jawaban 'tidak tahu/missing' (sudah dibersihkan ke kosong)."],
    ["Catatan CES-D", "Item positif (E, H) dibalik skor; skor hanya dihitung jika 10 item lengkap. Orientasi item E & H "
                      "(positif) divalidasi empiris: berkorelasi berlawanan arah dengan 8 item lain terhadap self-rated health & kepuasan hidup."],
    ["Catatan penyakit kronis", "Hanya kode A (hipertensi) dan B (diabetes) yang diberi nama karena sudah divalidasi dengan data terukur; "
                                "sisanya hanya masuk ke n_chronic_conditions."],
    ["Catatan pengeluaran", "Pengeluaran per kapita adalah perkiraan sederhana (pangan mingguan x 52/12 + nonpangan bulanan + "
                            "nonpangan tahunan/12 + biaya sekolah/12) / ukuran RT; bukan agregat resmi RAND."],
    ["Bobot", "Gunakan weight_cross_section bila ingin estimasi representatif populasi; tidak perlu untuk model prediktif biasa."],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="folder berisi file .dta IFLS5 (opsional, auto-detect)")
    ap.add_argument("--out_dir", default=None, help="folder output (default: ./output di sebelah script)")
    a = ap.parse_args()
    a.data_dir = find_data_dir(a.data_dir or DATA_DIR)
    a.out_dir = a.out_dir or OUT_DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[IFLS5] Membaca data dari : {a.data_dir}")
    print(f"[IFLS5] Output ke          : {a.out_dir}")

    df = build(a.data_dir)
    order = [c for c, *_ in DICT if c in df.columns]
    df = df[order]

    dd = make_dictionary(df)
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False).rename("pct_missing").reset_index()
    miss.columns = ["variable", "pct_missing"]

    csv_path = os.path.join(a.out_dir, "ifls5_model_ready.csv")
    xlsx_path = os.path.join(a.out_dir, "ifls5_model_ready.xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        wb = xw.book
        hdr = wb.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1, "valign": "top", "text_wrap": True})
        df.to_excel(xw, sheet_name="dataset", index=False)
        dd.to_excel(xw, sheet_name="data_dictionary", index=False)
        miss.to_excel(xw, sheet_name="missing_summary", index=False)
        pd.DataFrame(README, columns=["Item", "Keterangan"]).to_excel(xw, sheet_name="readme", index=False)

        for name, frame, widths in [
            ("dataset", df, None),
            ("data_dictionary", dd, {"variable": 30, "description": 70, "role": 28, "source": 28, "summary": 80}),
            ("missing_summary", miss, {"variable": 34}),
            ("readme", pd.DataFrame(README, columns=["Item", "Keterangan"]), {"Item": 24, "Keterangan": 140}),
        ]:
            ws = xw.sheets[name]
            for j, c in enumerate(frame.columns):
                ws.write(0, j, c, hdr)
                w = (widths or {}).get(c, min(max(len(c) + 2, 12), 28))
                ws.set_column(j, j, w)
            ws.freeze_panes(1, 1 if name == "dataset" else 0)
            ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)
        xw.sheets["dataset"].set_row(0, 32)

    print(f"\nSelesai. {len(df):,} baris x {df.shape[1]} kolom")
    print(" ->", xlsx_path)
    print(" ->", csv_path)


if __name__ == "__main__":
    main()