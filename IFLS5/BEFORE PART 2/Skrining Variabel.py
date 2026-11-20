# -*- coding: utf-8 -*-
"""
A4_skrining_kandidat.py - SKRINING STATISTIK KANDIDAT PREDICTOR (hanya data TRAINING)
====================================================================================
Tujuan : dari ribuan variabel berdata pada balita, tunjukkan mana yang MEMBERI INFORMASI TAMBAHAN
         tentang stunting di luar umur dan jenis kelamin, lalu kelompokkan yang redundan.
         Ini alat PENEMUAN kandidat; keputusan akhir tetap lewat teori WHO + kelayakan dashboard.

Aturan anti-leakage : hanya membaca train_ifls5.csv (test tidak disentuh).

Langkah
  1. Saringan mekanis : terisi >= 50%, bukan administratif/leakage, bukan (hampir) konstan.
  2. Uji tiap variabel: Likelihood-Ratio test model logistik [umur, umur^2, kelompok umur, jenis kelamin] vs
     [umur, umur^2, jenis kelamin + variabel]  -> p-value, dikoreksi FDR (Benjamini-Hochberg).
     Ukuran efek: Cramer's V (kategorik) / Somers' D (numerik); plus Mutual Information.
  3. Redundansi : variabel signifikan yang saling berasosiasi kuat (Cramer's V >= 0.6) dikelompokkan;
     disarankan satu wakil per kelompok.
  4. Laporan per tema WHO (txt + csv). Tanda [T3.3] = sudah ada di Tabel 3.3; [BARU] = belum.

Pakai   : python A4_skrining_kandidat.py
          python A4_skrining_kandidat.py --semua_anak        # tanpa 1-anak-per-rumah-tangga
          python A4_skrining_kandidat.py --alpha 0.01 --topn 15
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score

warnings.filterwarnings("ignore")

import importlib.util


def _load_theme_module():
    """Muat modul daftar variabel (A2) dari folder yang sama, TAHAN terhadap penamaan ulang file
    (termasuk nama dengan spasi, yang tidak bisa di-import dengan 'import ...')."""
    d = os.path.dirname(os.path.abspath(__file__))
    for name in ("A2_daftar_variabel.py", "Daftar_Variabel.py", "Daftar Variabel.py", "daftar_variabel.py"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("modul_daftar_variabel", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    return None


_m = _load_theme_module()
if _m is None:
    # Sengaja BERHENTI (bukan fallback diam-diam): tanpa klasifikasi tema, variabel leakage
    # (tinggi badan) dan administratif tidak tersaring dan akan tampak "sangat signifikan".
    raise SystemExit("File daftar variabel (A2_daftar_variabel.py / Daftar Variabel.py) tidak ditemukan di folder "
                     "yang sama dengan skrip ini. Letakkan di folder yang sama lalu jalankan ulang.")
classify, THEMES, T33_CODES = _m.classify, _m.THEMES, _m.T33_CODES

TARGET = "stunting"
ID_COLS = ["pidlink", "hhid14"]
DERIVED = {"haz", "haz_flag_implausible", "haz_valid", "stunting", "is_balita", "age_days", "age_months_exact"}
AGE_COL, SEX_CODE, AGE_YEAR_CODE = "age_months_exact", "us01", "us03"
# Variabel administratif/teknis lapangan yang lolos klasifikasi tema A2 (kode supervisor, flag "able to answer", dst.)
ADMIN_EXTRA_TEXT = re.compile(r"supervisor|coordinator|capi check|able to answer|\bcapi\b|enumerator|field team", re.I)
ADMIN_EXTRA_CODE = re.compile(r"^(sc1[0-3]|ut01c)$|\d+[a-z]*x(_|$)", re.I)      # mis. sc13, ks06x_A4, us13x


def parse_name(c):
    return tuple(c.split("__", 1)) if "__" in c else ("", c)


def kind_of(s):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() >= 0.9 * s.notna().sum() and num.nunique() > 12:
        return "num", num
    if num.notna().sum() >= 0.9 * s.notna().sum():       # kode angka -> string rapi
        t = num.map(lambda v: (str(int(v)) if float(v).is_integer() else str(v)) if pd.notna(v) else np.nan)
        return "cat", t
    return "cat", s.astype("object").map(lambda v: str(v).strip() if pd.notna(v) else np.nan)


def merge_rare(s, n, max_levels=12):
    s = s.astype(object).where(s.notna(), "NA")
    vc = s.value_counts()
    keep = vc[vc >= max(15, 0.02 * n)].index[:max_levels]
    return s.where(s.isin(keep), "jarang")


def design(kind, s, n):
    """Matriks desain variabel (tanpa intercept). None bila tidak informatif."""
    if kind == "num":
        r = s.rank(pct=True)
        miss = r.isna()
        d = pd.DataFrame({"v": r.fillna(0.5)})
        if miss.mean() > 0.05:
            d["miss"] = miss.astype(float)
        return d if d["v"].nunique() > 1 else None
    m = merge_rare(s, n)
    if m.nunique() < 2:
        return None
    return pd.get_dummies(m, drop_first=True, dtype=float)


def loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=400)
    m.fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def cramers_v_codes(a, b):
    """Cramer's V pada baris yang terisi di KEDUANYA (kode -1 = kosong), supaya pola kosong tidak dihitung."""
    m = (a >= 0) & (b >= 0)
    if m.sum() < 100:
        return 0.0
    a = np.unique(a[m], return_inverse=True)[1]
    b = np.unique(b[m], return_inverse=True)[1]
    ka, kb = int(a.max()) + 1, int(b.max()) + 1
    ct = np.bincount(a * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    n = ct.sum()
    e = np.outer(ct.sum(1), ct.sum(0)) / n
    ok = e > 0
    chi2 = (((ct - e) ** 2)[ok] / e[ok]).sum()
    k = min(ka, kb) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def to_codes(kind, s, n):
    """Kode integer; -1 = kosong. Kategori jarang digabung."""
    if kind == "num":
        return pd.qcut(s, 5, duplicates="drop").cat.codes.values.astype(int)
    m = s.astype(object)
    vc = m.value_counts()
    keep = vc[vc >= max(15, 0.02 * n)].index[:12]
    m = m.where(m.isin(keep) | m.isna(), "jarang")
    return pd.factorize(m)[0].astype(int)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(r, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--min_cov", type=float, default=0.5)
    ap.add_argument("--max_dom", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=0.05, help="ambang FDR")
    ap.add_argument("--red_thr", type=float, default=0.6, help="ambang Cramer's V untuk redundansi")
    ap.add_argument("--topn", type=int, default=12, help="jumlah teratas per tema di laporan")
    ap.add_argument("--semua_anak", action="store_true", help="jangan ambil 1 anak per rumah tangga")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    log = lambda m: print(f"[SKRINING] {m}", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    path = a.train
    if path is None:
        for b in (os.getcwd(), os.path.join(os.getcwd(), "output"), here, os.path.join(here, "output")):
            p = os.path.join(b, "splitting_ifls5", "train_ifls5.csv")
            if os.path.isfile(p):
                path = p
                break
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("train_ifls5.csv tidak ditemukan. Jalankan B_splitting.py atau pakai --train.")
    out_dir = os.path.abspath(a.out_dir or os.path.join(os.path.dirname(path), "..", "seleksi_kandidat"))
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(path, dtype={"pidlink": str, "hhid14": str}, low_memory=False, encoding="utf-8-sig")
    y_all = df[TARGET].astype(int).values
    n_all = len(df)
    log(f"Training: {n_all:,} balita, prevalensi stunting {100 * y_all.mean():.1f}%")

    # ---- baseline: umur, umur^2, jenis kelamin ----
    sex_c = next((c for c in df.columns if parse_name(c)[1] == SEX_CODE), None)
    if AGE_COL not in df.columns or sex_c is None:
        raise ValueError(f"Butuh kolom {AGE_COL} dan jenis kelamin (kode {SEX_CODE}) di train.")
    age = pd.to_numeric(df[AGE_COL], errors="coerce")
    age = age.fillna(age.median()) / 60.0
    sex = (pd.to_numeric(df[sex_c], errors="coerce") == 3).astype(float)
    base = pd.DataFrame({"age": age, "age2": age ** 2, "sex": sex})
    age_m = age * 60.0
    for lo, hi in ((6, 12), (12, 24), (24, 36), (36, 48), (48, 61)):          # dummy kelompok umur (bulan)
        base[f"ag{lo}"] = ((age_m >= lo) & (age_m < hi)).astype(float)

    # ---- saringan mekanis ----
    skip_codes = {parse_name(c)[1] for c in df.columns if parse_name(c)[1] in (SEX_CODE, AGE_YEAR_CODE)}
    rows, drop_info = [], {"admin/leakage": 0, "cakupan": 0, "konstan": 0}
    cand = []
    for c in df.columns:
        label, code = parse_name(c)
        if c in ID_COLS or c in DERIVED or code in skip_codes:
            continue
        s = df[c]
        cov = s.notna().mean()
        label = label.replace("_", " ") if label else code
        tema = classify(code, (label + " " + code.replace("_", " ")).lower())
        if tema.startswith(("00", "99")) or ADMIN_EXTRA_TEXT.search(label) or ADMIN_EXTRA_CODE.search(code):
            drop_info["admin/leakage"] += 1
            continue
        if cov < a.min_cov:
            drop_info["cakupan"] += 1
            continue
        kind, sv = kind_of(s)
        vc = sv.value_counts(normalize=True)
        if len(vc) < 2 or vc.iloc[0] >= a.max_dom:
            drop_info["konstan"] += 1
            continue
        cand.append((c, code, label, tema, kind, sv, cov))
    log(f"Lolos saringan mekanis: {len(cand)} variabel | dibuang: {drop_info}")

    # ---- sampel analisis: 1 anak per rumah tangga (p-value valid) ----
    if a.semua_anak or "hhid14" not in df:
        idx = np.arange(n_all)
    else:
        g = df["hhid14"].fillna("x" + df["pidlink"])
        idx = df.assign(_g=g).groupby("_g").sample(1, random_state=a.seed).index.values
    y = y_all[idx]
    n = len(idx)
    log(f"Sampel uji: {n:,} baris ({'semua anak' if a.semua_anak else '1 anak per rumah tangga'})")
    Xb = base.iloc[idx].values
    ll0 = loglik(Xb, y)

    # ---- uji tiap variabel ----
    for i, (c, code, label, tema, kind, sv, cov) in enumerate(cand):
        sv_s = sv.iloc[idx]
        d = design(kind, sv_s, n)
        if d is None:
            continue
        try:
            ll1 = loglik(np.hstack([Xb, d.values]), y)
        except Exception:
            continue
        lr = max(2 * (ll1 - ll0), 0.0)
        p = float(stats.chi2.sf(lr, d.shape[1]))
        ok = sv_s.notna().values
        if kind == "num":
            nn = pd.to_numeric(sv_s, errors="coerce")
            ok = nn.notna().values
            auc = roc_auc_score(y[ok], nn.values[ok]) if len(np.unique(y[ok])) > 1 and ok.sum() > 30 else 0.5
            eff, prev = abs(2 * (auc - 0.5)), ""
            mi_s = pd.qcut(nn, 10, duplicates="drop").cat.codes.values
        else:
            m = merge_rare(sv_s, n)
            ct = pd.crosstab(m, y)
            eff = float(np.sqrt(stats.chi2_contingency(ct)[0] / (ct.values.sum() * (min(ct.shape) - 1)))) if min(ct.shape) > 1 else 0.0
            pr = pd.Series(y, index=m.index).groupby(m).agg(["mean", "size"])
            pr = pr[pr["size"] >= 30]
            prev = f"{100 * pr['mean'].min():.0f}%-{100 * pr['mean'].max():.0f}%" if len(pr) > 1 else ""
            mi_s = pd.factorize(m)[0]
        rows.append(dict(kolom=c, code=code, label=label, tema=tema, tipe=kind, pct_terisi=round(100 * cov, 1),
                         df=d.shape[1], LR_chi2=round(lr, 2), p=p, efek=round(eff, 3),
                         MI=round(float(mutual_info_score(mi_s, y)), 4), prevalensi_antar_kategori=prev,
                         di_T33=code in T33_CODES))
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(cand)} variabel diuji")
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("Tidak ada variabel yang bisa diuji.")
    res["p_FDR"] = bh(res["p"].values)
    res["signifikan"] = res["p_FDR"] < a.alpha
    res = res.sort_values(["tema", "p_FDR", "LR_chi2"], ascending=[True, True, False])
    res.to_csv(os.path.join(out_dir, "skrining_kandidat.csv"), index=False, encoding="utf-8-sig")
    sig = res[res["signifikan"]].sort_values("LR_chi2", ascending=False)
    log(f"Diuji: {len(res)} | signifikan (FDR<{a.alpha}): {len(sig)} | di antaranya sudah di Tabel 3.3: {int(sig['di_T33'].sum())}")

    # ---- redundansi antar variabel signifikan ----
    top = sig.head(150)
    codes_map = {}
    for r in top.itertuples():
        sv = next(x[5] for x in cand if x[0] == r.kolom).iloc[idx]
        kd = r.tipe
        try:
            codes_map[r.kolom] = to_codes(kd, sv, n)
        except Exception:
            pass
    names = list(codes_map)                      # sudah urut dari LR chi2 terbesar
    reps, members = [], {}
    for k in names:                              # pengelompokan berbasis WAKIL (tanpa efek rantai)
        for r in reps:
            if cramers_v_codes(codes_map[k], codes_map[r]) >= a.red_thr:
                members[r].append(k)
                break
        else:
            reps.append(k)
            members[k] = [k]
    groups = [g for g in members.values() if len(g) > 1]

    # ---- laporan ----
    lines = []
    W = lines.append
    W("SKRINING KANDIDAT PREDICTOR STUNTING (hanya data training)")
    W(f"Training {n_all:,} balita | sampel uji {n:,} ({'semua anak' if a.semua_anak else '1 anak/rumah tangga'}) | "
      f"prevalensi {100 * y_all.mean():.1f}%")
    W(f"Saringan mekanis -> {len(cand)} variabel | diuji {len(res)} | signifikan FDR<{a.alpha}: {len(sig)}")
    W("Uji = LR test: tambahan informasi variabel di luar umur (linear, kuadrat, kelompok umur) dan jenis kelamin. efek = Cramer's V / Somers' D.")
    W("PENTING: signifikan secara statistik BUKAN berarti layak. Variabel yang merupakan AKIBAT stunting, proxy umur,")
    W("atau tidak bisa diisi kader harus tetap dibuang. Gunakan laporan ini untuk menemukan kandidat, lalu tinjau teori.\n")
    for tema, g in res.groupby("tema"):
        gs = g[g["signifikan"]].sort_values("LR_chi2", ascending=False).head(a.topn)
        W("=" * 100)
        W(f"{tema}  (diuji {len(g)}, signifikan {int(g['signifikan'].sum())})")
        W("=" * 100)
        if gs.empty:
            W("  (tidak ada yang signifikan)\n")
            continue
        for r in gs.itertuples():
            tag = "[T3.3]" if r.di_T33 else "[BARU]"
            W(f"{r.code}: {r.label} {tag}")
            W(f"      terisi {r.pct_terisi:.0f}% | {r.tipe} | LR chi2={r.LR_chi2} (df={r.df}) | p_FDR={r.p_FDR:.2g} | "
              f"efek={r.efek} | MI={r.MI}" + (f" | prevalensi stunting antar kategori: {r.prevalensi_antar_kategori}" if r.prevalensi_antar_kategori else ""))
        W("")
    W("=" * 100)
    W(f"KELOMPOK REDUNDAN (Cramer's V >= {a.red_thr}) - pilih SATU wakil per kelompok")
    W("=" * 100)
    if not groups:
        W("  (tidak ada)")
    lk = res.set_index("kolom")
    for gi, g in enumerate(groups, 1):
        W(f"Kelompok {gi}:")
        best = max(g, key=lambda k: (lk.loc[k, "pct_terisi"] >= 80, lk.loc[k, "LR_chi2"]))
        for k in sorted(g, key=lambda k: -lk.loc[k, "LR_chi2"]):
            W(f"   {'*' if k == best else ' '} {lk.loc[k, 'code']}: {lk.loc[k, 'label']} | terisi {lk.loc[k, 'pct_terisi']:.0f}% | "
              f"LR chi2={lk.loc[k, 'LR_chi2']}")
        W("   (* = saran wakil: cakupan tinggi dan informasi terbesar; ganti bila ada yang lebih mudah diisi kader)")
    t33_miss = res[res["di_T33"] & ~res["signifikan"]]
    W("\nVariabel Tabel 3.3 yang diuji tetapi TIDAK signifikan: " +
      (", ".join(f"{r.code} (p_FDR={r.p_FDR:.2g})" for r in t33_miss.itertuples()) or "-"))
    with open(os.path.join(out_dir, "skrining_kandidat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:6]))
    print("\nFile:", os.path.join(out_dir, "skrining_kandidat.txt"), "\n      ", os.path.join(out_dir, "skrining_kandidat.csv"))


if __name__ == "__main__":
    main()