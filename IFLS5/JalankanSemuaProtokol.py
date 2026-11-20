# Menjalankan semua protokol paper secara berurutan (tombol Run VS Code bisa dipakai)
import subprocess
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
DAFTAR = [
    ("ProtokolAzrianidanRahman.py", "azriani"),
    ("ProtokolAzrianidanRahman.py", "rahman"),
    ("ProtokolLainya.py", "ndagijimana"),
    ("ProtokolLainya.py", "shen"),
    ("ProtokolLainya.py", "mkungudza"),
]
for skrip, protokol in DAFTAR:
    print(f"\n{'#' * 80}\n# {skrip} --protokol {protokol}\n{'#' * 80}", flush=True)
    hasil = subprocess.run([sys.executable, str(FOLDER / skrip), "--protokol", protokol], cwd=FOLDER)
    if hasil.returncode != 0:
        sys.exit(f"Berhenti: {skrip} --protokol {protokol} gagal. Lihat pesan error di atas.")
print("\nSemua protokol selesai.")