# -*- coding: utf-8 -*-
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
