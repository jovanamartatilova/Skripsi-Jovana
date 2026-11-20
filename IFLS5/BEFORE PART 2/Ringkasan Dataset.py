# -*- coding: utf-8 -*-
"""
A3_ringkasan_data.py - RINGKASAN DATA IFLS5: dari data mentah sampai kandidat predictor
=====================================================================================
Menampilkan "corong" (funnel):
  A. SUBJEK   : total PIDLINK -> punya data umur -> balita kasar (0-4 th) -> balita 0-59 bulan (usia eksak)
                -> HAZ valid (target) -> stunting / tidak -> train / test
  B. VARIABEL : kolom CSV gabungan -> berdata pada balita -> terisi >= 50% -> bukan administratif/leakage
                -> bukan (hampir) konstan -> kandidat otomatis (per tema) -> [pilihan finalmu, opsional]

Prasyarat : A2_daftar_variabel.py sudah dijalankan (cache). Bagian "usia eksak/HAZ" butuh A_EDA.py sukses.
Pakai     : python A3_ringkasan_data.py
            python A3_ringkasan_data.py --final kandidat_final.txt   (1 kode variabel per baris)
Output    : daftar_variabel/ringkasan_data.txt  +  daftar_kandidat_otomatis.txt
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

MIN_COV, MAX_DOM = 50.0, 95.0
T33 = {"us01", "us03", "us06", "kr13", "kr13a", "kr14", "kr16", "kr17", "kr18", "kr20", "kr21", "kr22", "kr23",
       "kr26", "kr27", "rja0ca", "rja0ce", "rja0cc", "rja25a", "rja30f", "ks02a", "ks07a"}


def code_of(c):
    return c.split("__", 1)[1] if "__" in c else c


def fm(n):
    return f"{int(n):,}".replace(",", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--final", default=None, help="txt berisi kode variabel final (1 per baris)")
    ap.add_argument("--min_cov", type=float, default=MIN_COV)
    ap.add_argument("--max_dom", type=float, default=MAX_DOM)
    ap.add_argument("--chunk", type=int, default=3000)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv = a.csv
    if csv is None:
        for b in (os.getcwd(), os.path.join(os.getcwd(), "output"), here, os.path.join(here, "output")):
            p = os.path.join(b, "ifls5_full_per_pidlink_named.csv")
            if os.path.isfile(p):
                csv = p
                break
    if not csv or not os.path.isfile(csv):
        raise FileNotFoundError("ifls5_full_per_pidlink_named.csv tidak ditemukan. Pakai --csv.")
    base = os.path.dirname(os.path.abspath(csv))
    vdir = os.path.join(base, "daftar_variabel")
    os.makedirs(vdir, exist_ok=True)
    L = []
    P = lambda s="": (print(s), L.append(s))

    # ------------------------------------------------------------------ #
    # A. SUBJEK
    # ------------------------------------------------------------------ #
    hdr = pd.read_csv(csv, nrows=0, encoding="utf-8-sig").columns.tolist()
    find = lambda k: next((c for c in hdr if code_of(c) == k), None)
    age_c, h_c, w_c = find("us03"), find("us04"), find("us06")
    use = [c for c in ("pidlink", "hhid14", age_c, h_c, w_c) if c]
    pids, hhs = set(), set()
    n_rows = n_age = n_k = n_k05 = n_k_h = n_k_w = n_k_hw = 0
    for ch in pd.read_csv(csv, usecols=use, chunksize=a.chunk, low_memory=False, encoding="utf-8-sig",
                          dtype={"pidlink": str, "hhid14": str}):
        n_rows += len(ch)
        pids.update(ch["pidlink"].dropna())
        if "hhid14" in ch:
            hhs.update(ch["hhid14"].dropna())
        ag = pd.to_numeric(ch[age_c], errors="coerce") if age_c else pd.Series(np.nan, index=ch.index)
        n_age += int(ag.notna().sum())
        n_k05 += int(ag.between(0, 5).sum())
        kb = ag.between(0, 4)
        n_k += int(kb.sum())
        if h_c and w_c:
            hv = pd.to_numeric(ch[h_c], errors="coerce").between(20, 200)
            wv = pd.to_numeric(ch[w_c], errors="coerce").between(0.5, 60)
            n_k_h += int((kb & hv).sum()); n_k_w += int((kb & wv).sum()); n_k_hw += int((kb & hv & wv).sum())

    P("=" * 78)
    P("A. CORONG SUBJEK")
    P("=" * 78)
    P(f"Total baris CSV gabungan                      : {fm(n_rows)}")
    P(f"Total PIDLINK unik (individu)                 : {fm(len(pids))}")
    P(f"Total rumah tangga unik (hhid14)              : {fm(len(hhs))}")
    P(f"  -> punya umur (us03, Buku US)               : {fm(n_age)}")
    P(f"  -> anak umur 0-5 tahun (kandidat)           : {fm(n_k05)}")
    P(f"  -> BALITA kasar, umur 0-4 tahun (us03)      : {fm(n_k)}")
    P(f"       + tinggi badan valid (20-200 cm)       : {fm(n_k_h)}")
    P(f"       + berat badan valid (0,5-60 kg)        : {fm(n_k_w)}")
    P(f"       + tinggi & berat valid                 : {fm(n_k_hw)}")

    sub_p = os.path.join(base, "eda_ifls5", "balita_ifls5_subset.csv")
    if os.path.isfile(sub_p):
        sh = pd.read_csv(sub_p, nrows=0, encoding="utf-8-sig").columns.tolist()
        need = [c for c in ("pidlink", "hhid14", "age_months_exact", "haz_valid", "stunting") if c in sh]
        sb = pd.read_csv(sub_p, usecols=need, dtype={"pidlink": str, "hhid14": str}, encoding="utf-8-sig")
        P(f"Subset hasil EDA (A_EDA.py)                   : {fm(len(sb))} baris")
        if "age_months_exact" in sb and sb["age_months_exact"].notna().any():
            am = sb["age_months_exact"]
            P(f"  -> usia eksak (bulan) terhitung             : {fm(am.notna().sum())}")
            P(f"  -> BALITA 0-59 BULAN (usia eksak)           : {fm(am.between(0, 60, inclusive='left').sum())}")
            if "haz_valid" in sb:
                hv = sb["haz_valid"].astype(str).str.lower().isin(["true", "1", "1.0"])
                P(f"  -> HAZ valid = TARGET VALID                 : {fm(hv.sum())}")
                if "stunting" in sb and hv.any():
                    st = sb.loc[hv, "stunting"]
                    P(f"       stunting (HAZ < -2)                    : {fm(st.sum())} ({100 * st.mean():.1f}%)")
                    P(f"       tidak stunting                         : {fm((st == 0).sum())} ({100 * (1 - st.mean()):.1f}%)")
                    g = sb.loc[hv].groupby(sb.loc[hv, "hhid14"].fillna("x" + sb.loc[hv, "pidlink"])).size()
                    P(f"       rumah tangga dgn >=2 balita            : {fm((g >= 2).sum())} ({fm(g[g >= 2].sum())} anak)")
        else:
            P("  !! usia eksak belum ada -> A_EDA.py belum diberi kolom tanggal (lihat cek_tanggal_ukur.py)")
    else:
        P("Subset hasil EDA belum ada -> jalankan A_EDA.py dulu untuk angka balita 0-59 bulan & target.")
    for nm in ("train_ifls5.csv", "test_ifls5.csv"):
        p = os.path.join(base, "splitting_ifls5", nm)
        if os.path.isfile(p):
            n = sum(1 for _ in open(p, encoding="utf-8-sig")) - 1
            P(f"Splitting {nm.split('_')[0]:5s}                              : {fm(n)} baris")

    # ------------------------------------------------------------------ #
    # B. VARIABEL
    # ------------------------------------------------------------------ #
    P("")
    P("=" * 78)
    P("B. CORONG VARIABEL")
    P("=" * 78)
    data_cols = [c for c in hdr if c not in ("pidlink", "hhid14")]
    P(f"Kolom CSV gabungan (di luar ID)               : {fm(len(data_cols))}  "
      f"({fm(len({code_of(c) for c in data_cols}))} kode unik)")
    mp = os.path.join(base, "ifls5_column_mapping.csv")
    if os.path.isfile(mp):
        m = pd.read_csv(mp, dtype=str, low_memory=False, encoding="utf-8-sig")
        extra = f", {m['source_file'].nunique()} file sumber" if "source_file" in m else ""
        P(f"Baris ifls5_column_mapping.csv                : {fm(len(m))}{extra}")
    cache = os.path.join(vdir, "_profil_cache.pkl")
    if not os.path.isfile(cache):
        P("!! Cache belum ada -> jalankan A2_daftar_variabel.py dulu, lalu ulangi skrip ini.")
    else:
        pf = pd.read_pickle(cache)
        if "pct_dominan" not in pf.columns:
            P("!! Cache versi lama -> jalankan: python A2_daftar_variabel.py --refresh")
            pf["pct_dominan"] = 0.0
        admin = pf["tema"].str.startswith("99")
        leak = pf["tema"].str.startswith("00")
        s1 = pf
        s2 = s1[s1["pct_terisi"] >= a.min_cov]
        s3 = s2[~(s2["tema"].str.startswith("99") | s2["tema"].str.startswith("00"))]
        s4 = s3[(s3["pct_dominan"] < a.max_dom) & (s3["n_unik_num"] >= 2)] if "n_unik_num" in s3 else s3
        step = lambda t, d: P(f"{t:46s}: {fm(len(d)):>7s}  ({fm(d['code'].nunique())} kode unik)")
        step("Kolom CSV berdata pada balita (>=1 terisi)", s1)
        step(f"  terisi >= {a.min_cov:.0f}% pada balita", s2)
        step("  bukan administratif & bukan LEAKAGE", s3)
        step(f"  bukan konstan/hampir konstan (dominan<{a.max_dom:.0f}%)", s4)
        P(f"    (dibuang administratif: {int(admin.sum())}, leakage: {int(leak.sum())} dari semua berdata)")
        P("")
        P("KANDIDAT OTOMATIS per tema (hasil 4 saringan di atas; BELUM seleksi teori/dashboard):")
        for t, g in s4.groupby("tema"):
            P(f"   {t[:70]:70s} {len(g):4d}")
        in33 = sorted(set(s4["code"]) & T33)
        P(f"\nVariabel Tabel 3.3 yang lolos saringan otomatis : {len(in33)} -> {', '.join(in33)}")
        out33 = sorted(T33 - set(s4["code"]))
        P(f"Variabel Tabel 3.3 yang TIDAK lolos             : {len(out33)} -> {', '.join(out33)}")
        with open(os.path.join(vdir, "daftar_kandidat_otomatis.txt"), "w", encoding="utf-8") as f:
            f.write(f"KANDIDAT OTOMATIS ({len(s4)} kolom). Saringan: terisi>={a.min_cov:.0f}%, bukan administratif/leakage, "
                    f"dominansi<{a.max_dom:.0f}%.\n")
            for t, g in s4.sort_values(["tema", "code"]).groupby("tema"):
                f.write(f"\n=== {t} ({len(g)}) ===\n")
                for _, r in g.iterrows():
                    f.write(f"{r['code']}: {r['label']}  | terisi {r['pct_terisi']:.0f}% | {r['tipe']} | {r['contoh']}\n")
        if a.final:
            fin = [x.strip() for x in open(a.final, encoding="utf-8") if x.strip() and not x.startswith("#")]
            hit = pf[pf["code"].isin(fin)]
            miss = sorted(set(fin) - set(pf["code"]))
            P(f"\nPILIHAN FINAL (--final)                         : {len(fin)} kode -> {len(hit)} kolom ditemukan")
            P(f"  terisi < {a.min_cov:.0f}%: {sorted(hit.loc[hit['pct_terisi'] < a.min_cov, 'code'])}")
            P(f"  tidak ditemukan / tanpa data balita: {miss}")

    P("\nCatatan: angka balita pada corong VARIABEL memakai umur tahun 0-4 (us03) sebagai pendekatan,")
    P("sedangkan 0-59 bulan eksak baru ada setelah A_EDA.py sukses dengan kolom tanggal.")
    with open(os.path.join(vdir, "ringkasan_data.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\nTersimpan:", os.path.join(vdir, "ringkasan_data.txt"))


if __name__ == "__main__":
    main()