# -*- coding: utf-8 -*-
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
