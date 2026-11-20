# -*- coding: utf-8 -*-
"""
Gambar_Alur_Dataset_IFLS5.py - Gambar alur pembentukan dataset balita dari file .dta IFLS5.

Jalankan:  python Gambar_Alur_Dataset_IFLS5.py
Output  :  gambar_alur_dataset_ifls5.png (300 dpi) di folder yang sama dengan file ini.
Gaya dibuat mirip diagram draw.io pada Gambar 3.1 (kotak abu kebiruan, panah lurus, panel berjudul).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

ISI, GARIS, TEKS = "#C3CDD9", "#3E4C5E", "#1F1F1F"   # warna kotak ala draw.io
ISI_AKHIR = "#8FA3BC"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# (judul kotak, keterangan di bawah judul)
LANGKAH = [
    ("Katalog File .dta", "152 file\n4.859 variabel"),
    ("Pemilihan File", "20 file terpilih\nkunci diseragamkan,\nduplikat dihapus"),
    ("Penggabungan Data", "bus_us + bk_ar1 + bus_time\n(hhid14 + pid14)\n48.139 individu"),
    ("Pemilihan Balita", "umur dihitung,\nusia 0–59 bulan\n5.387 balita"),
    ("Pembentukan Target", "HAZ dan WAZ dihitung,\nHAZ di luar ±6 dikeluarkan\n5.243 balita"),
    ("Penambahan Data", "orang tua, kehamilan, layanan,\nmakan, sakit, rumah tangga;\nrecode dan variabel turunan"),
    ("Dataset Balita", "5.243 baris\n85 variabel kandidat"),
]
BARIS = [[0, 1, 2, 3], [4, 5, 6]]       # baris 1: 4 kotak, baris 2: 3 kotak

W, H = 3.8, 1.75          # ukuran kotak
GAP = 0.75                # jarak antar kotak (panjang panah)
Y_BARIS = [0.0, -2.95]    # posisi tengah tiap baris

fig, ax = plt.subplots(figsize=(14, 6.4))
ax.set_axis_off()

pos = {}
for r, idxs in enumerate(BARIS):
    for c, i in enumerate(idxs):
        x = c * (W + GAP)
        y = Y_BARIS[r]
        pos[i] = (x, y)
        akhir = i == len(LANGKAH) - 1
        ax.add_patch(FancyBboxPatch((x, y - H / 2), W, H, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=ISI_AKHIR if akhir else ISI, edgecolor=GARIS, linewidth=1.1))
        judul, ket = LANGKAH[i]
        ax.text(x + W / 2, y + H / 2 - 0.3, judul, ha="center", va="center", fontsize=12,
                fontweight="bold", color=TEKS)
        ax.plot([x + 0.15, x + W - 0.15], [y + H / 2 - 0.55] * 2, color=GARIS, linewidth=0.7)
        ax.text(x + W / 2, y - 0.28, ket, ha="center", va="center", fontsize=11, color=TEKS,
                linespacing=1.35)


def panah(titik):
    xs, ys = zip(*titik)
    ax.plot(xs, ys, color=GARIS, linewidth=1.2, solid_capstyle="butt")
    (x0, y0), (x1, y1) = titik[-2], titik[-1]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=GARIS, lw=1.2, mutation_scale=14))


for idxs in BARIS:
    for a, b in zip(idxs, idxs[1:]):
        (xa, ya), (xb, _) = pos[a], pos[b]
        panah([(xa + W, ya), (xb, ya)])

# panah siku dari kotak terakhir baris 1 ke kotak pertama baris 2
(x3, y3), (x4, y4) = pos[BARIS[0][-1]], pos[BARIS[1][0]]
y_tengah = (y3 - H / 2 + y4 + H / 2) / 2
panah([(x3 + W / 2, y3 - H / 2), (x3 + W / 2, y_tengah), (x4 + W / 2, y_tengah), (x4 + W / 2, y4 + H / 2)])

# bingkai panel dengan tab judul, seperti di draw.io
kiri, kanan = -0.45, 3 * (W + GAP) + W + 0.45
atas, bawah = Y_BARIS[0] + H / 2 + 0.75, Y_BARIS[1] - H / 2 - 0.45
ax.add_patch(Rectangle((kiri, bawah), kanan - kiri, atas - bawah, fill=False, edgecolor=GARIS, linewidth=1.1))
tab_w, tab_h = 5.6, 0.5
ax.add_patch(Polygon([(kiri, atas), (kiri + tab_w, atas), (kiri + tab_w, atas - tab_h + 0.18),
                      (kiri + tab_w - 0.18, atas - tab_h), (kiri, atas - tab_h)],
                     closed=True, facecolor=ISI, edgecolor=GARIS, linewidth=1.1))
ax.text(kiri + 0.25, atas - tab_h / 2, "Pembentukan Dataset Balita IFLS5", ha="left", va="center",
        fontsize=12, color=TEKS)

ax.set_xlim(kiri - 0.1, kanan + 0.1)
ax.set_ylim(bawah - 0.1, atas + 0.1)
ax.set_aspect("equal")

out = Path(__file__).resolve().parent / "gambar_alur_dataset_ifls5.png"
fig.savefig(out)
print(f"Tersimpan: {out}")