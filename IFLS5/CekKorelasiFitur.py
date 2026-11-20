"""Cek multikolinearitas pada FITUR FINAL (isi fitur_tahap3.txt), memakai data TRAIN saja.
Aturan: |Spearman| >= 0,7 atau VIF >= 5 dianggap tinggi.
Jalankan: python CekKorelasiFitur.py --fitur-file <file .txt berisi satu nama fitur per baris>"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5")
ap = argparse.ArgumentParser()
ap.add_argument("--train", default=str(BASE / "output" / "3_Splitting" / "train.csv"))
ap.add_argument("--fitur-file", required=True, help="file .txt, satu nama fitur per baris (mis. fitur Model D)")
ap.add_argument("--batas-r", type=float, default=0.7)
ap.add_argument("--batas-vif", type=float, default=5.0)
args = ap.parse_args()

NOMINAL = {"provinsi", "tempat_lahir_grup", "limbah_kode"}
fitur = [l.strip() for l in Path(args.fitur_file).read_text(encoding="utf-8").splitlines() if l.strip()]
tr = pd.read_csv(args.train)
cols = [c for c in fitur if c in tr.columns and c not in NOMINAL]
X = tr[cols].apply(pd.to_numeric, errors="coerce")
print(f"Fitur dicek: {len(cols)} -> {', '.join(cols)}")

r = X.corr(method="spearman", min_periods=100)
pas = [(a, b, r.loc[a, b]) for i, a in enumerate(cols) for b in cols[i + 1:] if abs(r.loc[a, b]) >= 0.5]
pas = sorted(pas, key=lambda t: -abs(t[2]))
print(f"\nPasangan |Spearman| >= 0,5 (batas tinggi = {args.batas_r}):")
for a, b, v in pas:
    print(f"  {a:28s} {b:28s} {v:+.3f}  {'<-- TINGGI' if abs(v) >= args.batas_r else ''}")
if not pas:
    print("  tidak ada")

Xi = X.fillna(X.median())
Xi = Xi.loc[:, Xi.std() > 0]
vif = pd.Series(np.diag(np.linalg.pinv(np.corrcoef(Xi.values, rowvar=False))), index=Xi.columns).sort_values(ascending=False)
print(f"\nVIF (batas tinggi = {args.batas_vif}):")
for v, x in vif.items():
    print(f"  {v:28s} {x:6.2f}  {'<-- TINGGI' if x >= args.batas_vif else ''}")
n_r = sum(abs(v) >= args.batas_r for _, _, v in pas)
n_v = int((vif >= args.batas_vif).sum())
print(f"\nKesimpulan: {n_r} pasangan korelasi tinggi, {n_v} variabel VIF tinggi "
      f"-> {'tidak ada multikolinearitas berarti' if n_r == 0 and n_v == 0 else 'perlu ditangani'}")