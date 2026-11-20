"""
custom_transformer.py - Transformer buatan sendiri untuk pipeline IFLS5

KENAPA FILE TERPISAH:
joblib/pickle menyimpan NAMA modul + kelas. Kalau Winsorizer ditulis langsung di 4_PEMODELAN.py, namanya
tersimpan sebagai "__main__.Winsorizer", sehingga model_final.joblib TIDAK bisa dimuat oleh 5_EVALUASI.py,
6_FEATURE IMPORTANCE.py, maupun dashboard Streamlit (error: Can't get attribute 'Winsorizer' on <module '__main__'>).
Dengan file ini, nama yang tersimpan adalah "custom_transformer.Winsorizer".

LETAKKAN file ini di folder skrip (D:\\UA_Tugas Kuliah\\SKRIPSI\\ML\\IFLS5). Dashboard memuatnya lewat
sys.path (lihat perubahan di modelkader.py).

Winsorizer: membatasi (clip) nilai kolom terpilih ke persentil lo-hi yang DIHITUNG dari data yang di-fit
(train atau training fold), sehingga tidak ada informasi dari data test/validasi. NaN dibiarkan (diimputasi
oleh langkah berikutnya).
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class Winsorizer(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Clip kolom (indeks `idx`) ke [persentil lo, persentil hi] hasil fit."""

    def __init__(self, idx=None, lo=1, hi=99):
        self.idx = idx
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.n_features_in_ = X.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # kolom yang seluruhnya NaN
            self.lo_ = np.nanpercentile(X, self.lo, axis=0)
            self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, float)
        for j in self.idx or []:
            X[:, j] = np.clip(X[:, j], self.lo_[j], self.hi_[j])
        return X
