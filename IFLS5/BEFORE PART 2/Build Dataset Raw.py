#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ifls5_dataset.py
======================
Membangun SATU dataset "siap model" (1 baris = 1 individu dewasa) dari data mentah
IFLS-5 (2014/2015, format Stata .dta).

Cara pakai
----------
    python build_ifls5_dataset.py                      # mode RAW (apa adanya) - default
    python build_ifls5_dataset.py --mode full          # SEMUA variabel (CSV besar + kamus Excel)
    python build_ifls5_dataset.py --mode clean         # versi dibersihkan + kolom turunan
    python build_ifls5_dataset.py --data_dir "Dataset/IFLS5"   # atau tentukan manual

Dependensi:  pip install pandas numpy xlsxwriter

Output
------
    mode raw   (default): ifls5_raw.xlsx / ifls5_raw.csv  (kode asli, tanpa cleaning)
    mode clean          : ifls5_model_ready.xlsx / .csv
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


# --------------------------------------------------------------------------- #
# MODE RAW: gabungan apa adanya (tanpa cleaning, tanpa filter, tanpa kolom turunan)
# --------------------------------------------------------------------------- #
# (file, kolom-kolom asli, kolom_pembeda_untuk_pivot atau None, level kunci)
RAW_SPEC = [
    ("b3a_cov", ["hhid14", "age", "sex", "marstat", "proxy", "result", "resvis", "dob_yr"], None, "pid"),
    ("ptrack",  ["age_14", "member14", "pwt14xa", "pwt14la"], None, "pid"),
    ("b3a_dl1", ["dl04", "dl06", "dl07", "dl02", "dl03", "dl03b", "dl03d"], None, "pid"),
    ("b3a_tk1", ["tk01", "tk02", "tk03", "tk04"], None, "pid"),
    ("b3a_tk2", ["tk24a", "tk24a5", "tk25a1x", "tk25a1", "tk21a", "tk22a"], None, "pid"),
    ("b3a_sw",  ["sw00", "sw01", "sw06"], None, "pid"),
    ("b3b_kk1", ["kk01", "kk02a", "kk02c"], None, "pid"),
    ("b3b_km",  ["km01a", "km04", "km08"], None, "pid"),
    ("b3b_eh",  ["eh01", "eh08"], None, "pid"),
    ("b3b_cd3", ["cd05"], "cdtype", "pid"),       # -> cd05_A ... cd05_R  (A=hipertensi, B=diabetes)
    ("b3b_kp",  ["kp02"], "kptype", "pid"),        # -> kp02_A ... kp02_J  (item CES-D 10)
    ("b3b_ak1", ["ak01"], "aktype", "pid"),        # -> ak01_main, ak01_A ... (asuransi)
    ("bus_us",  ["us04", "us06", "us06a", "us07a1", "us07a2", "us07b1", "us07b2", "us07c1", "us07c2",
                 "us07ap", "us13", "us20a", "us20b", "us20c", "us20d", "us18ab", "us15a"], None, "pid"),
    ("bk_sc1",  ["sc01_14_14", "sc02_14_14", "sc03_14_14", "sc05"], None, "hh"),
    ("b2_kr",   ["kr03", "kr11", "kr13", "kr20"], None, "hh"),
    ("b1_ks0",  ["ks02a", "ks03a", "ks04b", "ks07a", "ks10aa", "ks11aa", "ks12aa"], None, "hh"),
]


def _label_map(data_dir, name, cols):
    """Peta kode -> teks label dari file .dta (dibangun dari pasangan nilai numerik & berlabel)."""
    num = rd(data_dir, name, cols, labels=False)
    lab = rd(data_dir, name, cols, labels=True)
    out = {}
    for c in cols:
        if c not in lab or not (str(lab[c].dtype) == "category" or lab[c].dtype == object):
            continue
        pair = pd.DataFrame({"code": num[c], "label": lab[c].astype(object)}).dropna(subset=["code"]).drop_duplicates()
        pair = pair[pair["label"].map(lambda v: isinstance(v, str))]
        if len(pair):
            out[c] = {k: v for k, v in zip(pair["code"], pair["label"])}
    return out


def build_raw(data_dir):
    log = lambda m: print(f"[IFLS5-RAW] {m}", flush=True)
    base = None
    var_info, value_labels = {}, []     # var -> (label, sumber)  |  baris value_labels

    for name, cols, pivot, level in RAW_SPEC:
        log(f"{name} ...")
        key = ["pidlink"] if level == "pid" else ["hhid14"]
        use = key + ([pivot] if pivot else []) + [c for c in cols if c not in key]
        full = rd(data_dir, name, None)            # untuk cek kolom yang benar-benar ada
        have = [c for c in use if c in full.columns]
        missing_cols = [c for c in use if c not in full.columns]
        if missing_cols:
            log(f"  (lewati kolom tak ada di {name}: {missing_cols})")
        df = full[have].copy()
        with pd.read_stata(os.path.join(data_dir, f"{name}.dta"), iterator=True) as r:
            vlab = r.variable_labels()
        lm = _label_map(data_dir, name, [c for c in have if c not in key + ([pivot] if pivot else [])])

        if pivot:
            df[pivot] = df[pivot].astype(str).str.strip().replace({"": "main", "nan": "main"})
            df = df.pivot_table(index=key, columns=pivot, values=[c for c in cols if c in df], aggfunc="first")
            df.columns = [f"{a}_{b}" for a, b in df.columns]
            df = df.reset_index()
            for c in cols:
                for newc in [x for x in df.columns if x.startswith(c + "_")]:
                    var_info[newc] = (f"{vlab.get(c, '')} [item {newc.split('_')[-1]}]", name)
                    for k, v in lm.get(c, {}).items():
                        value_labels.append((newc, k, v))
        else:
            if level == "hh":                      # tabel rumah tangga: satu baris per hhid14
                df = df.drop_duplicates("hhid14")
            elif df.duplicated("pidlink").any():
                df = df.drop_duplicates("pidlink")
            for c in have:
                if c in key:
                    continue
                var_info[c] = (vlab.get(c, ""), name)
                for k, v in lm.get(c, {}).items():
                    value_labels.append((c, k, v))

        if base is None:
            base = df
        else:
            on = key
            # hhid14 juga ada di b3a_cov -> gabung rumah tangga lewat hhid14
            base = base.merge(df, on=on, how="left")
        log(f"  -> {base.shape[0]:,} baris x {base.shape[1]} kolom")

    # urutan kolom: ID dulu
    first = ["pidlink", "hhid14"]
    base = base[first + [c for c in base.columns if c not in first]]
    base["pidlink"] = base["pidlink"].astype(str)
    base["hhid14"] = base["hhid14"].astype(str)

    # ---- kamus ----
    rows = []
    for c in base.columns:
        lbl, src = var_info.get(c, ("ID unik individu" if c == "pidlink" else "ID rumah tangga 2014", "b3a_cov"))
        s = base[c]
        isnum = pd.api.types.is_numeric_dtype(s)
        rows.append({
            "variable": c, "original_label": lbl, "source_file": src,
            "type": "numeric/kode" if isnum else "text",
            "n_missing(NaN)": int(s.isna().sum()), "pct_missing": round(100 * s.isna().mean(), 1),
            "n_unique": int(s.nunique()),
            "min": s.min() if isnum else "", "max": s.max() if isnum else "",
        })
    dd = pd.DataFrame(rows)
    vl = pd.DataFrame(value_labels, columns=["variable", "code", "label"]).drop_duplicates()
    vl = vl.sort_values(["variable", "code"]).reset_index(drop=True)
    return base, dd, vl


RAW_README = [
    ["Dataset", "IFLS-5 (2014/15) - gabungan APA ADANYA tingkat individu, untuk eksplorasi awal"],
    ["Unit", "1 baris = 1 pidlink di b3a_cov (SEMUA baris, tanpa filter umur/hasil wawancara/jenis kelamin)"],
    ["Nama kolom", "Nama variabel ASLI IFLS (mis. dl06, tk25a1, us07a1). Arti tiap kolom ada di sheet data_dictionary."],
    ["Nilai", "KODE ASLI, belum diterjemahkan. Arti kode ada di sheet value_labels (mis. sex: 1=Male, 3=Female; ya/tidak: 1=Yes, 3=No)."],
    ["Nilai kosong", "Kosong (NaN) = baris tidak punya data di modul itu atau pertanyaan dilewati (skip pattern)."],
    ["KODE KHUSUS", "TIDAK dibersihkan. Angka seperti 8/9, 98/99, 998/999, 9998, 99999998 dst. biasanya berarti 'tidak tahu / missing / menolak' "
                    "dan bukan nilai sebenarnya - tangani di tahap preprocessing."],
    ["Outlier", "Tidak ada filter nilai tidak masuk akal (tinggi, tensi, upah, dll.) dan tidak ada kolom turunan (BMI, skor CES-D, target)."],
    ["Kolom pivot", "cd05_A..R = diagnosis penyakit kronis per jenis (A=hipertensi, B=diabetes, lainnya belum diberi nama); "
                    "kp02_A..J = 10 item CES-D (item E & H berarti positif, perlu dibalik saat membuat skor); "
                    "ak01_* = asuransi/jaminan kesehatan per jenis."],
    ["Kolom rumah tangga", "Kolom dari bk_sc1, b2_kr, b1_ks0 adalah level rumah tangga (nilai sama untuk anggota satu hhid14). "
                           "Pengeluaran rinci (b1_ks1/ks2/ks3) belum diagregasi."],
    ["Bobot", "pwt14xa = bobot cross-section 2014; pwt14la = bobot longitudinal."],
]


def run_raw(a):
    df, dd, vl = build_raw(a.data_dir)
    csv_path = os.path.join(a.out_dir, "ifls5_raw.csv")
    xlsx_path = os.path.join(a.out_dir, "ifls5_raw.xlsx")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    rd_df = pd.DataFrame(RAW_README, columns=["Item", "Keterangan"])
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        hdr = xw.book.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1})
        sheets = [("dataset_raw", df, {}), ("data_dictionary", dd, {"variable": 18, "original_label": 80, "source_file": 14}),
                  ("value_labels", vl, {"variable": 18, "label": 60}), ("readme", rd_df, {"Item": 20, "Keterangan": 150})]
        for name, frame, widths in sheets:
            frame.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for j, c in enumerate(frame.columns):
                ws.write(0, j, c, hdr)
                ws.set_column(j, j, widths.get(c, min(max(len(str(c)) + 2, 10), 16)))
            ws.freeze_panes(1, 1 if name == "dataset_raw" else 0)
            ws.autofilter(0, 0, len(frame), len(frame.columns) - 1)
    print(f"\nSelesai (RAW). {len(df):,} baris x {df.shape[1]} kolom")
    print(" ->", xlsx_path)
    print(" ->", csv_path)


# --------------------------------------------------------------------------- #
# MODE FULL: SEMUA variabel dari semua file "lebar" (1 baris per orang / rumah tangga)
# --------------------------------------------------------------------------- #
ADMIN_COLS = {"hhid14", "hhid14_9", "pid14", "stemid", "pid", "version", "module"}   # kunci/metadata duplikat


def _shrink(df):
    """Hemat memori tanpa mengubah nilai: kolom bilangan bulat -> float32 bila aman (<= 2^24)."""
    for c in df.columns:
        s = df[c]
        if s.dtype == "float64":
            v = s.dropna()
            if len(v) and (v == v.round()).all() and v.abs().max() <= 16_777_216:
                df[c] = s.astype("float32")
    return df


def _load_all(data_dir, log):
    import glob
    person, hh, excluded, dict_rows, vlabels, file_rows = [], [], [], [], [], []
    files = sorted(glob.glob(os.path.join(data_dir, "*.dta")))
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)[:-4]
        with pd.read_stata(path, iterator=True, convert_categoricals=False) as r:
            df = r.read()
            vlab = r.variable_labels()
            vl = r.value_labels()
            lbl = list(r._lbllist) if hasattr(r, "_lbllist") else [""] * len(df.columns)
        n, k = df.shape
        key = "pidlink" if "pidlink" in df.columns else ("hhid14" if "hhid14" in df.columns else None)
        if key is None:
            excluded.append((name, n, k, "-", "", "tidak punya kunci pidlink/hhid14 (data komunitas/fasilitas)"))
            log(f"[{i}/{len(files)}] {name}: DIKECUALIKAN (tanpa kunci)"); continue
        kv = df[key].dropna()
        ratio = len(kv) / max(kv.nunique(), 1)
        if ratio > 1.05:
            excluded.append((name, n, k, key, round(ratio, 2),
                             f"format panjang: rata-rata {ratio:.1f} baris per {key}"))
            log(f"[{i}/{len(files)}] {name}: DIKECUALIKAN (panjang, {ratio:.1f} baris/{key})"); continue
        df = df[df[key].notna()].copy()
        df[key] = df[key].astype(str)
        n_dup = int(df[key].duplicated().sum())            # duplikat kunci kecil -> ambil baris pertama, dicatat
        if n_dup:
            df = df.drop_duplicates(key, keep="first")
        df = df.set_index(key)
        level = "individu" if key == "pidlink" else "rumah_tangga"
        # kamus (sebelum rename)
        for j, c in enumerate(df.columns):
            dict_rows.append({"original_name": c, "source_file": name, "level": level,
                              "label": vlab.get(c, ""), "dtype": str(df[c].dtype),
                              "n_rows_in_file": len(df), "n_non_null": int(df[c].notna().sum()),
                              "_lblset": lbl[list(vlab).index(c)] if c in vlab and list(vlab).index(c) < len(lbl) else ""})
        for lname, mapping in vl.items():
            for code, text in mapping.items():
                vlabels.append((name, lname, code, text))
        file_rows.append((name, level, len(df), len(df.columns), n_dup))
        (person if level == "individu" else hh).append((name, _shrink(df)))
        log(f"[{i}/{len(files)}] {name}: {level}, {len(df):,} baris x {len(df.columns)} kolom")
    return person, hh, excluded, dict_rows, vlabels, file_rows


def run_full(a):
    import gc
    log = lambda m: print(f"[IFLS5-FULL] {m}", flush=True)
    person, hh, excluded, dict_rows, vlabels, file_rows = _load_all(a.data_dir, log)
    person.sort(key=lambda t: t[0] != "ptrack")      # ptrack paling depan: pemilik resmi pidlink/hhid14

    # ---- penamaan kolom: nama asli; bila bentrok antar-file -> "<file>__<kolom>"; kolom admin duplikat dibuang ----
    first_seen = {}
    for name, df in person + hh:
        for c in df.columns:
            first_seen.setdefault(c, []).append(name)
    new_names, dropped_admin = {}, []
    for name, df in person + hh:
        mp = {}
        for c in df.columns:
            owners = first_seen[c]
            if c in ADMIN_COLS and owners[0] != name:
                mp[c] = None; dropped_admin.append((name, c)); continue
            mp[c] = c if len(owners) == 1 or (c in ADMIN_COLS) else f"{name}__{c}"
        new_names[name] = mp
    for lst in (person, hh):
        for idx, (name, df) in enumerate(lst):
            mp = new_names[name]
            df = df[[c for c in df.columns if mp[c] is not None]].rename(columns=mp)
            lst[idx] = (name, df)

    # ---- semesta baris = gabungan semua pidlink (ptrack dulu) ----
    order = [n for n, _ in person]
    ptrack_idx = dict(person)["ptrack"].index if "ptrack" in order else pd.Index([])
    universe = ptrack_idx
    for _, df in person:
        universe = universe.append(df.index.difference(universe))
    if a.sample:
        universe = universe[: a.sample]
    log(f"Semesta baris: {len(universe):,} individu (ptrack: {len(ptrack_idx):,})")
    hh_of = dict(person)["ptrack"]["hhid14"] if "ptrack" in order and "hhid14" in dict(person)["ptrack"].columns else None

    cols_person = [c for _, df in person for c in df.columns]
    cols_hh = [c for _, df in hh for c in df.columns]
    all_cols = ["pidlink"] + cols_person + cols_hh
    log(f"Total kolom data: {len(all_cols):,} (individu {len(cols_person):,} + rumah tangga {len(cols_hh):,})")

    csv_path = os.path.join(a.out_dir, "ifls5_full_wide.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    CH = 4000
    for start in range(0, len(universe), CH):
        ids = universe[start:start + CH]
        parts = [df.reindex(ids) for _, df in person]
        if hh and hh_of is not None:
            hid = hh_of.reindex(ids)
            for _, df in hh:
                part = df.reindex(hid.values)
                part.index = ids
                parts.append(part)
        chunk = pd.concat(parts, axis=1)
        chunk.index.name = "pidlink"
        chunk = chunk[[c for c in all_cols if c != "pidlink"]]
        chunk.to_csv(csv_path, mode="a", header=(start == 0), encoding="utf-8-sig", float_format="%.12g")
        log(f"  ditulis {min(start + CH, len(universe)):,}/{len(universe):,} baris")
        del parts, chunk; gc.collect()

    # ---- kamus & ringkasan ----
    dd = pd.DataFrame(dict_rows)
    dd["variable"] = dd.apply(lambda r: new_names[r.source_file].get(r.original_name), axis=1)
    dd = dd[dd.variable.notna()].copy()
    dd["pct_non_null_in_file"] = (100 * dd.n_non_null / dd.n_rows_in_file).round(1)
    dd = dd[["variable", "original_name", "source_file", "level", "label", "dtype",
             "n_rows_in_file", "n_non_null", "pct_non_null_in_file"]]
    vl = pd.DataFrame(vlabels, columns=["source_file", "label_set", "code", "label"]).drop_duplicates()
    ex = pd.DataFrame(excluded, columns=["source_file", "n_rows", "n_cols", "key", "rows_per_key", "reason"])
    fs = pd.DataFrame(file_rows, columns=["source_file", "level", "n_rows", "n_cols", "dup_key_rows_dropped"])
    adm = pd.DataFrame(dropped_admin, columns=["source_file", "dropped_duplicate_admin_column"])
    notes = pd.DataFrame([
        ["Dataset", "IFLS-5: gabungan SEMUA variabel dari file berformat lebar, apa adanya (tanpa cleaning/filter/turunan)"],
        ["Unit baris", "1 baris = 1 pidlink (gabungan semua pidlink dari semua file; ptrack dulu)"],
        ["Nilai", "Kode asli. Arti kode: sheet value_labels (label_set dihubungkan lewat file sumber)."],
        ["Nama kolom", "Nama asli IFLS. Jika nama sama muncul di >1 file -> '<file>__<nama>'."],
        ["Kolom rumah tangga", "Digabung lewat hhid14 dari ptrack; nilai sama untuk anggota satu rumah tangga."],
        ["Tidak ikut digabung", "File berformat panjang (banyak baris per orang/RT), file komunitas/fasilitas: lihat sheet excluded_files."],
        ["Kolom admin duplikat dibuang", "hhid14, hhid14_9, pid14, stemid, pid, version, module dari file ke-2 dan seterusnya (lihat sheet dropped_admin_cols)."],
    ], columns=["Item", "Keterangan"])
    xlsx_path = os.path.join(a.out_dir, "ifls5_full_dictionary.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
        hdr = xw.book.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white", "border": 1})
        for nm, fr, w in [("readme", notes, {"Item": 26, "Keterangan": 130}),
                          ("data_dictionary", dd, {"variable": 28, "original_name": 16, "source_file": 14, "label": 80}),
                          ("value_labels", vl, {"source_file": 14, "label_set": 18, "label": 60}),
                          ("excluded_files", ex, {"source_file": 14, "reason": 60}),
                          ("file_summary", fs, {"source_file": 14}),
                          ("dropped_admin_cols", adm, {"source_file": 14, "dropped_duplicate_admin_column": 30})]:
            fr.to_excel(xw, sheet_name=nm, index=False)
            ws = xw.sheets[nm]
            for j, c in enumerate(fr.columns):
                ws.write(0, j, c, hdr); ws.set_column(j, j, w.get(c, 14))
            ws.freeze_panes(1, 0); ws.autofilter(0, 0, len(fr), len(fr.columns) - 1)
    print(f"\nSelesai (FULL): {len(universe):,} baris x {len(all_cols):,} kolom")
    print(" ->", csv_path)
    print(" ->", xlsx_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="folder berisi file .dta IFLS5 (opsional, auto-detect)")
    ap.add_argument("--out_dir", default=None, help="folder output (default: ./output di sebelah script)")
    ap.add_argument("--mode", choices=["raw", "clean", "full"], default="raw",
                    help="raw = 111 variabel utama apa adanya (default) | full = SEMUA variabel file lebar | clean = dibersihkan + turunan")
    ap.add_argument("--sample", type=int, default=0, help="(opsional) hanya N individu pertama, untuk uji cepat mode full")
    a = ap.parse_args()
    a.data_dir = find_data_dir(a.data_dir or DATA_DIR)
    a.out_dir = a.out_dir or OUT_DIR or os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(a.out_dir, exist_ok=True)
    print(f"[IFLS5] Membaca data dari : {a.data_dir}")
    print(f"[IFLS5] Output ke          : {a.out_dir}")

    if a.mode == "raw":
        run_raw(a)
        return
    if a.mode == "full":
        run_full(a)
        return

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