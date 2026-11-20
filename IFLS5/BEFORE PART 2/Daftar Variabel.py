# -*- coding: utf-8 -*-
"""
A2_daftar_variabel.py - DAFTAR SEMUA VARIABEL IFLS5 YANG PUNYA DATA PADA BALITA
================================================================================
Tujuan : membantu memilih variabel predictor stunting. Hasilnya TXT (bukan Excel):

    kode: label asli (Inggris)
        terisi 87% | kategorik | nilai: 1(3000) 3(100) 8(5) | file: b5_... | kamus: ...

Yang dilakukan
  1. Baca CSV besar per potongan (hemat memori), ambil balita (umur tahun us03 = 0..4).
  2. Per variabel: % terisi pada balita, tipe, jumlah nilai unik, contoh nilai (+jumlah), rentang.
  3. Gabungkan keterangan dari ifls5_full_dictionary.xlsx dan ifls5_column_mapping.csv
     (kolom kunci dideteksi otomatis; kalau gagal, ada catatan di terminal).
  4. Kelompokkan per TEMA (tebakan otomatis dari kata kunci, bukan penilaian kausal) mengikuti
     kerangka WHO di Subbab 2.3.1: ibu/rumah tangga, ASI, MPASI/gizi, infeksi, kontekstual.
  5. Tandai  [T3.3] = variabel yang sudah ada di Tabel 3.3 proposalmu, dan  [LEAKAGE] = berkaitan
     dengan tinggi badan/HAZ (tidak boleh jadi predictor).

Pakai (VS Code / terminal)
    python A2_daftar_variabel.py                       # cari CSV otomatis di folder output/
    python A2_daftar_variabel.py --csv "D:/.../output/ifls5_full_per_pidlink_named.csv"
    python A2_daftar_variabel.py --min_cov 0.3         # hanya yang terisi >= 30% pada balita
    python A2_daftar_variabel.py --cari "breast,asi,diarrh"   # cari kata kunci (pakai cache, cepat)

Output (folder daftar_variabel/ di sebelah CSV): daftar_variabel_balita.txt  +  _profil_cache.pkl
"""
import argparse
import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

AGE_CODE = "us03"                       # umur (tahun); balita kasar = 0..4
AGE_RANGE = (0, 4)
CAP_UNIQUE = 60                         # berhenti menghitung nilai unik setelah > 60 nilai berbeda
SPECIAL = {7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999, 99997, 99998, 99999,
           999997, 999998, 999999, 9999997, 9999998, 9999999}
T33_CODES = {"us01", "us03", "us06", "kr13", "kr13a", "kr14", "kr16", "kr17", "kr18", "kr20", "kr21", "kr22",
             "kr23", "kr26", "kr27", "rja0ca", "rja0ce", "rja0cc", "rja25a", "rja30f", "ks02a", "ks07a",
             "rja28ady_b", "rja28ady_c", "rja28ady_d", "rja28ady_h", "rja28ady_i", "rja28ady_k",
             "rja28ady_l", "rja28ady_m", "rja28ady_n", "rja28ady_o"}
LEAK_CODES = {"us04"}                   # tinggi/panjang badan (dipakai menghitung HAZ)

# --------------------------------------------------------------------------- #
# Tema (urutan = prioritas; yang cocok pertama dipakai)
# --------------------------------------------------------------------------- #
THEMES = [
    ("99 ADMINISTRATIF (tanggal, pewawancara, status) - biasanya diabaikan",
     r"date of birth|(day|month|year) of birth|birth (day|month|year)|birthplace|birth cert|date of (visit|interview)|"
     r"time interview|interviewer (id|check)|id (number )?of (field )?interviewer|code of .*interviewer|"
     r"language|result of|final result|status of interview|interview status|(^| )pidlink|hhid|household id"),
    ("02 KEHAMILAN & KELAHIRAN (berat/panjang lahir, prematur, persalinan, ANC)",
     r"birth|born|pregnan|gestation|deliver|premature|antenatal|prenatal|postnatal|cesarean|caesarean|twin|"
     r"miscarr|stillbirth|labor|trimester|weeks? of pregnan"),
    ("00 LEAKAGE - tinggi/panjang badan & z-score (JANGAN jadi predictor)",
     r"\bheight\b|\blength\b|\btinggi\b|stunt|\bhaz\b|\bwaz\b|\bwhz\b|z.?score"),
    ("03 ASI / MENYUSUI / MPASI",
     r"breast|nurs(e|ing)|colostr|exclusive|formula|bottle|wean|complementary|porridge|solid food|first food|"
     r"\basi\b|suckl|lactat"),
    ("04 LAYANAN KESEHATAN, IMUNISASI, VITAMIN A, ASURANSI",
     r"immuniz|vaccin|bcg|polio|dpt|hepatitis|rotavirus|measles|posyandu|weigh(ing|ed)|puskesmas|clinic|health card|"
     r"kartu sehat|askes|jamkes|bpjs|insurance|health fund|dana sehat|midwife|doctor|nurse|paramedic|"
     r"health (facility|center|post|worker)|checkup|check-up|vitamin a|vit a|kms\b"),
    ("05 INFEKSI & PENYAKIT ANAK (diare, ISPA, demam, cacingan)",
     r"diarrh|cough|fever|illness|sick|disease|worm|malaria|respir|infection|symptom|dysent|pneumon|tubercul|"
     r"health complaint|chronic|asthma|injur|hospitali|medicine|drug|oral rehydration|\bors\b|treat"),
    ("06 EKONOMI RUMAH TANGGA (pengeluaran, aset, pendapatan)",
     r"expenditure|income|asset|wealth|saving|loan|debt|wage|salary|earn|price|remittance|transfer|value of|"
     r"business|farm|land|credit|spending|poor|worth|rent"),
    ("07 GIZI / MAKANAN / SUPLEMEN",
     r"food|meal|\beat|eats|diet|egg|meat|fish|vegetable|fruit|appetite|hunger|nutrition|iron|supplement|calori|"
     r"milk|rice|protein|snack|drink"),
    ("08 AIR, SANITASI, PERUMAHAN",
     r"water|toilet|sanitation|sewage|garbage|trash|waste|latrine|dwelling|floor|roof|wall|ventilation|kitchen|"
     r"cooking|electric|house size|square meters|rooms?\b|stove|fuel|lighting|drain|stagnant|yard|animal|"
     r"boil|refrigerator"),
    ("09 ORANG TUA / PENGASUH / PENDIDIKAN",
     r"mother|father|parent|caregiver|care ?giver|spouse|wife|husband|education|school|literate|reading|writing|"
     r"schooling|grade|child care|take care|taken care|cared for|guardian"),
    ("10 DEMOGRAFI & WILAYAH",
     r"province|kabupaten|kecamatan|urban|rural|region|village|community|ethnic|religion|marital|household size|"
     r"number of (hh )?(members|persons|people)|relationship|head of|\bsex\b|gender|\bage\b|years old|"
     r"language spoken"),
]
THEME_OTHER = "11 LAIN-LAIN / BELUM TERKLASIFIKASI"
THEME_COMPILED = [(n, re.compile(p, re.I)) for n, p in THEMES]


def classify(code, text):
    if code in LEAK_CODES:
        return THEMES[2][0]
    for n, rx in THEME_COMPILED:
        if rx.search(text):
            return n
    return THEME_OTHER


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def parse_name(c):
    if "__" in c:
        a, b = c.split("__", 1)
        return a, b
    return "", c


def fmt_val(v):
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except Exception:
        return str(v)[:25]


def load_extra(path, full_lower, code_to_cols, notes, tag):
    """Gabungkan keterangan dari kamus/mapping. Kolom kunci dideteksi dari kecocokan dengan nama kolom CSV."""
    extra = {}
    if not path or not os.path.isfile(path):
        notes.append(f"[{tag}] file tidak ditemukan: {path}")
        return extra
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        else:
            sheets = {os.path.basename(path): pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8-sig")}
    except Exception as e:
        notes.append(f"[{tag}] gagal dibaca: {e}")
        return extra
    code_set = set(code_to_cols)
    full_set = set(full_lower)
    for sname, df in sheets.items():
        if df is None or df.empty:
            continue
        best = None
        for c in df.columns:
            vals = set(df[c].dropna().astype(str).str.strip().str.lower())
            if not vals:
                continue
            hf, hc = len(vals & full_set), len(vals & code_set)
            hit = max(hf, hc)
            if hit >= min(20, max(3, len(code_set) // 3)) and hit / len(vals) >= 0.3 and (best is None or hit > best[1]):
                best = (c, hit)
        if best is None:
            notes.append(f"[{tag}] sheet '{sname}': kolom kunci tidak terdeteksi (dilewati). Kolom: {list(df.columns)[:8]}")
            continue
        key = best[0]
        notes.append(f"[{tag}] sheet '{sname}': kunci = kolom '{key}' ({best[1]} cocok); kolom lain digabung: "
                     f"{[c for c in df.columns if c != key][:8]}")
        others = [c for c in df.columns if c != key]
        keys = df[key].astype(str).str.strip().str.lower()
        for k, g in df.groupby(keys):
            targets = full_lower[k] if isinstance(full_lower, dict) and k in full_lower else code_to_cols.get(k, [])
            if not targets:
                continue
            parts = []
            for _, r in g.head(12).iterrows():
                t = ", ".join(f"{c}={str(r[c]).strip()[:70]}" for c in others if pd.notna(r[c]) and str(r[c]).strip())
                if t:
                    parts.append(t)
            if not parts:
                continue
            txt = " || ".join(parts)[:350]
            for t in targets:
                extra.setdefault(t, []).append(txt)
    return extra


def line_for(r):
    flag = ""
    if r["tema"].startswith("00"):
        flag += " [LEAKAGE]"
    if r["code"] in T33_CODES:
        flag += " [T3.3]"
    out = f"{r['code']}: {r['label']}{flag}\n"
    out += f"      terisi {r['pct_terisi']:.0f}% (n={r['n_balita_terisi']}) | {r['tipe']} | nilai: {r['contoh']}"
    if r["file"]:
        out += f" | file: {r['file']}"
    if r["kamus"]:
        out += f"\n      kamus: {r['kamus']}"
    return out + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dict", default=None, help="ifls5_full_dictionary.xlsx (default: di folder CSV)")
    ap.add_argument("--map", default=None, help="ifls5_column_mapping.csv (default: di folder CSV)")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--min_cov", type=float, default=0.05, help="min. proporsi terisi pada balita (default 0.05)")
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--cari", default=None, help="kata kunci dipisah koma; cari di cache (cepat)")
    ap.add_argument("--refresh", action="store_true", help="abaikan cache, hitung ulang")
    a = ap.parse_args()
    log = lambda m: print(f"[VAR] {m}", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = a.csv
    if csv_path is None:
        for b in (os.getcwd(), os.path.join(os.getcwd(), "output"), here, os.path.join(here, "output")):
            p = os.path.join(b, "ifls5_full_per_pidlink_named.csv")
            if os.path.isfile(p):
                csv_path = p
                break
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError("ifls5_full_per_pidlink_named.csv tidak ditemukan. Pakai --csv.")
    base = os.path.dirname(os.path.abspath(csv_path))
    out_dir = a.out_dir or os.path.join(base, "daftar_variabel")
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "_profil_cache.pkl")

    # ------------------------------ mode cari ------------------------------ #
    if a.cari:
        if not os.path.isfile(cache):
            raise SystemExit("Cache belum ada. Jalankan dulu tanpa --cari.")
        prof = pd.read_pickle(cache)
        kws = [k.strip() for k in a.cari.split(",") if k.strip()]
        rx = re.compile("|".join((r"\b" if len(k) <= 3 else "") + re.escape(k) for k in kws), re.I)
        hay = (prof["code"] + " " + prof["label"] + " " + prof["kamus"]).fillna("")
        hit = prof[hay.map(lambda t: bool(rx.search(t.replace("_", " "))) or bool(rx.search(t)))]
        hit = hit[hit["pct_terisi"] >= 100 * a.min_cov].sort_values(["tema", "file", "code"])
        print(f"\n{len(hit)} variabel cocok dengan {kws} (terisi >= {100 * a.min_cov:.0f}% pada balita)\n")
        for tema, g in hit.groupby("tema"):
            print(f"=== {tema} ===")
            for _, r in g.head(80).iterrows():
                print(line_for(r))
        return

    # ------------------------------ profil ------------------------------ #
    if os.path.isfile(cache) and not a.refresh:
        log(f"Pakai cache: {cache} (pakai --refresh untuk hitung ulang)")
        prof = pd.read_pickle(cache)
        notes = ["(dari cache)"]
        n_bal = int(prof.attrs.get("n_balita", 0)) if hasattr(prof, "attrs") else 0
    else:
        header = pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()
        log(f"{len(header):,} kolom di CSV")
        age_col = next((c for c in header if parse_name(c)[1] == AGE_CODE), None)
        if age_col is None:
            raise ValueError(f"Kolom umur (kode {AGE_CODE}) tidak ditemukan.")
        data_cols = [c for c in header if c not in ("pidlink", "hhid14")]
        n_nn = Counter()
        counters, capped = {}, set()
        vmin, vmax, numok = {}, {}, Counter()
        n_bal = 0
        log("Membaca CSV per potongan (beberapa menit) ...")
        for i, ch in enumerate(pd.read_csv(csv_path, chunksize=a.chunk, low_memory=False, encoding="utf-8-sig",
                                           dtype={"pidlink": str, "hhid14": str})):
            age = pd.to_numeric(ch[age_col], errors="coerce")
            sub = ch[age.between(*AGE_RANGE)]
            if sub.empty:
                continue
            n_bal += len(sub)
            nn = sub[data_cols].notna().sum()
            for c in nn[nn > 0].index:
                s = sub[c].dropna()
                n_nn[c] += int(nn[c])
                num = pd.to_numeric(s, errors="coerce")
                isn = num.notna()
                numok[c] += int(isn.sum())
                if isn.any():
                    vmin[c] = min(vmin.get(c, np.inf), float(num.min()))
                    vmax[c] = max(vmax.get(c, -np.inf), float(num.max()))
                if c not in capped:
                    vc = (num[isn] if isn.mean() > 0.9 else s.astype(str)).value_counts()
                    cn = counters.setdefault(c, Counter())
                    for k, v in vc.items():
                        cn[k] += int(v)
                    if len(cn) > CAP_UNIQUE:
                        capped.add(c)
            if (i + 1) % 10 == 0:
                log(f"  potongan {i + 1} dibaca (balita terkumpul: {n_bal:,})")
        log(f"Balita (umur {AGE_RANGE[0]}-{AGE_RANGE[1]} th): {n_bal:,}")

        # keterangan tambahan dari kamus & mapping
        notes = []
        full_lower = {c.lower(): [c] for c in data_cols}
        code_to_cols = {}
        for c in data_cols:
            code_to_cols.setdefault(parse_name(c)[1].lower(), []).append(c)
        mapp = a.map or os.path.join(base, "ifls5_column_mapping.csv")
        dictp = a.dict or os.path.join(base, "ifls5_full_dictionary.xlsx")
        ex_map = load_extra(mapp, full_lower, code_to_cols, notes, "mapping")
        ex_dict = load_extra(dictp, full_lower, code_to_cols, notes, "kamus")

        rows = []
        for c in data_cols:
            if n_nn.get(c, 0) == 0:
                continue
            label, code = parse_name(c)
            label = label.replace("_", " ") if label else code
            cn = counters.get(c, Counter())
            many = c in capped
            is_num = numok[c] >= 0.9 * n_nn[c]
            if is_num and (many or len(cn) > 12):
                tipe = "numerik"
                contoh = f"rentang {fmt_val(vmin[c])} s/d {fmt_val(vmax[c])}"
            else:
                tipe = "kategorik/kode" if is_num else "teks"
                top = cn.most_common(7)
                contoh = " ".join(f"{fmt_val(k)}({v})" for k, v in top) + (" ..." if (many or len(cn) > 7) else "")
                sp = sorted({float(k) for k in cn if is_num and float(k) in SPECIAL})
                if sp:
                    contoh += f"  [kode khusus mungkin: {', '.join(fmt_val(x) for x in sp)}]"
            mt = ex_map.get(c, [""])[0] if c in ex_map else ""
            m_file = re.search(r"source_file=([^,|]+)", mt)
            kamus = " || ".join(dict.fromkeys(
                [t for t in ex_dict.get(c, [])] + [re.sub(r"source_file=[^,|]+,?\s*", "", t) for t in ex_map.get(c, [])
                                                    if re.sub(r"source_file=[^,|]+,?\s*", "", t).strip()]))[:350]
            text = (label + " " + code.replace("_", " ") + " " + kamus).lower()
            rows.append(dict(kolom=c, code=code, label=label, file=(m_file.group(1).strip() if m_file else ""),
                             n_balita_terisi=int(n_nn[c]), pct_terisi=100 * n_nn[c] / max(n_bal, 1),
                             tipe=tipe, n_unik=(">%d" % CAP_UNIQUE if many else len(cn)), contoh=contoh,
                             kamus=kamus, tema=classify(code, text)))
        prof = pd.DataFrame(rows)
        prof.attrs["n_balita"] = n_bal
        prof.to_pickle(cache)

    # ------------------------------ tulis TXT ------------------------------ #
    sel = prof[prof["pct_terisi"] >= 100 * a.min_cov].copy()
    sel = sel.sort_values(["tema", "file", "code"])
    path_txt = os.path.join(out_dir, "daftar_variabel_balita.txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("DAFTAR VARIABEL IFLS5 YANG PUNYA DATA PADA BALITA\n")
        f.write(f"Balita (umur {AGE_RANGE[0]}-{AGE_RANGE[1]} th, kasar): {n_bal:,} | variabel berdata: {len(prof):,} | "
                f"ditampilkan (terisi >= {100 * a.min_cov:.0f}%): {len(sel):,}\n")
        f.write("Format: kode: label asli | terisi | tipe | nilai(jumlah) | file | kamus\n")
        f.write("[T3.3] = sudah ada di Tabel 3.3 proposal | [LEAKAGE] = berkaitan tinggi badan/HAZ (jangan dipakai)\n")
        f.write("Tema = tebakan otomatis dari kata kunci pada label. Label IFLS dipotong ~60 karakter.\n")
        f.write("Variabel milik IBU/AYAH (pendidikan, tinggi badan ibu, dst.) ada pada baris orang tua, bukan baris anak:\n")
        f.write("untuk memakainya perlu digabung lewat ID ibu/ayah (cari variabel 'pidlink'/'ID of mother' pada tema 09).\n\n")
        f.write("RINGKASAN PER TEMA\n")
        for t, g in sel.groupby("tema"):
            f.write(f"  {t}: {len(g)} variabel\n")
        for tema, g in sel.groupby("tema"):
            f.write("\n" + "=" * 100 + f"\nTEMA {tema}  ({len(g)} variabel)\n" + "=" * 100 + "\n")
            for fl, gg in g.groupby("file", dropna=False):
                f.write(f"\n--- file: {fl or '(tidak diketahui)'} ---\n")
                for _, r in gg.iterrows():
                    f.write(line_for(r))
    print("\n==== RINGKASAN ====")
    print(f"Variabel berdata pada balita: {len(prof):,} | ditampilkan: {len(sel):,}")
    print(sel.groupby("tema").size().to_string())
    for n in notes:
        print(" *", n)
    print("\nFile daftar:", path_txt)
    print("Cari kata kunci:  python A2_daftar_variabel.py --cari \"breast,asi,diarrh\"")


if __name__ == "__main__":
    main()