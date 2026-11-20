"""
Kamus variabel IFLS5 (.dta) -> Excel
------------------------------------
Membaca SEMUA file .dta di folder IFLS5 (termasuk subfolder), lalu menyimpan:
  - Ringkasan_File   : daftar file, jumlah baris & variabel
  - Kamus_Variabel   : tiap variabel + label asli (EN) + terjemahan (ID) + tipe + % terisi
  - Value_Labels     : arti kode tiap variabel kategorik (mis. 1 = Ya, 3 = Tidak)

Nama variabel & label diambil LANGSUNG dari metadata file .dta (tidak ditebak).

Cara jalan:   python kamus_variabel_ifls5.py
Library     : pandas, openpyxl  (wajib)
              deep-translator   (opsional, untuk terjemah ke Indonesia; butuh internet)
              pip install pandas openpyxl deep-translator
"""

import os
import sys

import pandas as pd

# ====================== PENGATURAN (sudah terisi) ======================
FOLDER_DTA = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\Dataset\IFLS5"
FOLDER_OUT = r"D:\UA_Tugas Kuliah\SKRIPSI\ML\IFLS5\output"
NAMA_EXCEL = "kamus_variabel_IFLS5.xlsx"

HITUNG_PCT_TERISI = True
# =======================================================================


# ------------------------------- Excel --------------------------------
def tulis_excel(path_out, sheets):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(path_out, engine="openpyxl") as w:
        for nama, df in sheets.items():
            df.to_excel(w, sheet_name=nama, index=False)
            ws = w.sheets[nama]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(name="Arial", bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(vertical="center")
            for i, col in enumerate(df.columns, 1):
                panjang = (
                    df[col].astype(str).str.len().quantile(0.95) if len(df) else 10
                )
                ws.column_dimensions[get_column_letter(i)].width = int(
                    min(max(len(col) + 2, panjang + 2), 60)
                )
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.font = Font(name="Arial", size=10)


def cari_file_dta(folder):
    hasil = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".dta"):
                hasil.append(os.path.join(root, f))
    return sorted(hasil)


def baca_metadata(path):
    var_rows, val_rows = [], []
    nama_file = os.path.relpath(path, FOLDER_DTA)

    with pd.io.stata.StataReader(path) as rd:
        var_labels = rd.variable_labels()
        val_labels = rd.value_labels()
        lbllist = getattr(rd, "_lbllist", None)
        varlist = getattr(rd, "_varlist", None)
        peta_lbl = {}

        if lbllist is not None and varlist is not None:
            peta_lbl = {v: l for v, l in zip(varlist, lbllist) if l}

        n_baris = getattr(rd, "_nobs", None)

    df = (
        pd.read_stata(
            path, convert_categoricals=False, convert_dates=False, convert_missing=False
        )
        if HITUNG_PCT_TERISI
        else None
    )

    if df is None:
        with pd.io.stata.StataReader(path) as rd:
            kolom = list(rd.variable_labels().keys())
        tipe = {}
    else:
        kolom = list(df.columns)
        n_baris = len(df)
        tipe = {c: str(df[c].dtype) for c in kolom}

    for c in kolom:
        pct = None
        n_terisi = None

        if df is not None:
            n_terisi = int(df[c].notna().sum())
            pct = round(100 * n_terisi / n_baris, 2) if n_baris else None

        set_lbl = peta_lbl.get(c)

        var_rows.append(
            {
                "file": nama_file,
                "variabel": c,
                "label_EN": (var_labels.get(c) or "").strip(),
                "tipe": tipe.get(c, ""),
                "punya_value_label": "Ya" if set_lbl else "",
                "n_terisi": n_terisi,
                "pct_terisi": pct,
            }
        )

        if set_lbl and set_lbl in val_labels:
            for kode, teks in val_labels[set_lbl].items():
                val_rows.append(
                    {
                        "file": nama_file,
                        "variabel": c,
                        "kode": kode,
                        "arti_EN": str(teks).strip(),
                    }
                )

    return n_baris, var_rows, val_rows


def main():
    if not os.path.isdir(FOLDER_DTA):
        sys.exit(f"Folder tidak ditemukan: {FOLDER_DTA}")
    os.makedirs(FOLDER_OUT, exist_ok=True)

    files = cari_file_dta(FOLDER_DTA)
    print(f"Ditemukan {len(files)} file .dta di {FOLDER_DTA}")
    if not files:
        sys.exit("Tidak ada file .dta.")

    ringkas, semua_var, semua_val, gagal = [], [], [], []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.relpath(p, FOLDER_DTA)}")
        try:
            n, vr, vl = baca_metadata(p)
            ringkas.append(
                {
                    "file": os.path.relpath(p, FOLDER_DTA),
                    "n_baris": n,
                    "n_variabel": len(vr),
                    "ukuran_MB": round(os.path.getsize(p) / 1e6, 2),
                }
            )
            semua_var += vr
            semua_val += vl
        except Exception as e:
            gagal.append({"file": os.path.relpath(p, FOLDER_DTA), "error": str(e)})
            print(f"  [GAGAL] {e}")

    df_var = pd.DataFrame(semua_var)
    df_val = pd.DataFrame(semua_val)

    # variabel kunci (mis. id) muncul di banyak file
    df_var["muncul_di_n_file"] = df_var.groupby("variabel")["file"].transform("nunique")
    df_var.insert(0, "no", range(1, len(df_var) + 1))

    kolom = [
        "no",
        "file",
        "variabel",
        "label_EN",
        "tipe",
        "punya_value_label",
        "n_terisi",
        "pct_terisi",
        "muncul_di_n_file",
    ]

    df_var = df_var[kolom]

    sheets = {
        "Ringkasan_File": pd.DataFrame(ringkas),
        "Kamus_Variabel": df_var,
        "Value_Labels": df_val,
    }
    if gagal:
        sheets["File_Gagal"] = pd.DataFrame(gagal)

    out = os.path.join(FOLDER_OUT, NAMA_EXCEL)
    tulis_excel(out, sheets)
    print(f"\nSelesai. Total {len(df_var)} baris variabel dari {len(ringkas)} file.")
    print(f"Excel: {out}")


if __name__ == "__main__":
    main()
