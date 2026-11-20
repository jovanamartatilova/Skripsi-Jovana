from pathlib import Path
import base64
import html as html_lib
import re
import warnings

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import json

from modelkader import (INPUT, faktor_perlu_diperhatikan, isi_dari_baris, kosongkan_form, kunci_form,
                        muat_model, prediksi, render_form, tampilkan_penjelasan)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dashboard Monitor Gizi & Stunting",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR.parent

DATA_PATH = ML_DIR / "Output" / "data_stunting.xlsx"        # 1 baris per balita (pengukuran terakhir)
TREND_PATH = ML_DIR / "Output" / "data_status_gizi.xlsx"    # semua pengukuran, untuk grafik tren per bulan
# Model final: Model B, S4 Random Forest (45 fitur, tanpa TB anak)
MODEL_PATH = ML_DIR / "IFLS5" / "output" / "4_Modeling_B" / "model_final.joblib"
IMPORTANCE_PATH = ML_DIR / "IFLS5" / "output" / "5_FeatureImportance_B" / "importance_dashboard.csv"

# Logo Posyandu
LOGO_PATH = (
    BASE_DIR
    / "assets"
    / "logo_posyandu.png"
)

# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap'
    );

    html, body, * {
    font-family: 'Poppins', sans-serif !important;
}

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background-color: #f5f6f8;
    }

    .block-container {
    padding-top: 3.2rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

.dashboard-title {
    margin-top: 6px;
}

    [data-testid="stAppViewContainer"] {
        color: #202020;
    }

    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6 {
        color: #202020 !important;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] li,
    [data-testid="stAppViewContainer"] .stMarkdown,
    [data-testid="stAppViewContainer"] .stMarkdown * {
        color: #202020;
    }

    [data-testid="stAppViewContainer"] div[data-testid="stExpander"] {
        border: 1px solid #d8d8d8;
        border-radius: 10px;
        background-color: #ffffff;
        margin-bottom: 10px;
    }

    [data-testid="stAppViewContainer"] div[data-testid="stExpander"] summary {
        color: #202020 !important;
        background-color: #eef1f6;
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 14px;
    }

[data-testid="stAppViewContainer"] div[data-testid="stPlotlyChart"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 10px 22px rgba(0,0,0,0.18), 0 2px 6px rgba(0,0,0,0.10);
    transform: translateZ(0);
}
    [data-testid="stAppViewContainer"] div[data-testid="stExpander"] summary svg {
        fill: #202020 !important;
    }

    [data-testid="stAppViewContainer"] div[data-testid="stExpanderDetails"] {
        background-color: #ffffff;
        color: #202020 !important;
        padding: 12px 14px;
    }

    [data-testid="stAppViewContainer"] div[data-testid="stExpanderDetails"] * {
        color: #202020 !important;
    }

    [data-testid="stAppViewContainer"] .stCode,
    [data-testid="stAppViewContainer"] .stCode * {
        color: #202020 !important;
        background-color: #f5f5f5 !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] * {
        color: #202020 !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="input"] input,
    [data-testid="stAppViewContainer"] div[data-baseweb="base-input"] input {
        color: #202020 !important;
        background-color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="input"],
    [data-testid="stAppViewContainer"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
    }

    div[data-testid="stMetricValue"] {
        color: #183b91 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #303030 !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #303030 !important;
    }

    [data-testid="stCaptionContainer"],
    .stCaption {
        color: #666666 !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0b1220 0%,
            #05070d 55%,
            #000000 100%
        );
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.6rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-baseweb="select"],
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #cfd6e4 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="input"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] *,
    section[data-testid="stSidebar"] div[data-baseweb="base-input"] input {
        color: #202020 !important;
        background-color: transparent !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stSlider"] * {
        color: white !important;
    }

    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] * {
        color: #202020 !important;
    }

    div[data-baseweb="calendar"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="calendar"] * {
        color: #202020 !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }

    ul[data-baseweb="menu"] li,
    ul[data-baseweb="menu"] li * {
        color: #202020 !important;
    }

    div[data-baseweb="tag"] {
        background-color: #183b91 !important;
    }

    div[data-baseweb="tag"] * {
        color: #ffffff !important;
    }

    .sidebar-brand {
        text-align: center;
        margin-top: 4px;
        margin-bottom: 5px;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.2px;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #93a3c9 !important;
        opacity: 1 !important;
        margin-bottom: 4px;
    }

    .sidebar-line {
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin: 18px 0 22px 0;
    }

    .filter-title {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #7d90c2 !important;
        opacity: 1 !important;
        margin-bottom: 14px;
    }

    .sidebar-footer-note {
        font-size: 11px;
        color: #7c869e !important;
        text-align: center;
        margin-top: 60px;
        padding-top: 14px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    .sidebar-logo-wrapper {
        width: 66px;
        height: 66px;
        border-radius: 50%;
        overflow: hidden;
        margin: 4px auto 14px auto;
        border: 2px solid rgba(255, 255, 255, 0.85);
        background-color: white;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
    }

    .sidebar-logo-wrapper img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .sidebar-logo-fallback {
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(
            135deg,
            #17408f 0%,
            #55c6f2 100%
        );
    }

    .sidebar-logo-fallback span {
        font-size: 26px;
        font-weight: 700;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] {
        border: none;
        margin-bottom: 10px;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] details {
        border: none;
        background: transparent;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] summary {
        background: linear-gradient(
            90deg,
            #17408f 0%,
            #2f6fd6 55%,
            #55c6f2 100%
        );
        border-radius: 12px;
        padding: 11px 15px;
        font-weight: 600;
        font-size: 13.5px;
        color: white !important;
        box-shadow: 0 2px 8px rgba(23, 64, 143, 0.35);
        transition: box-shadow 0.15s ease;
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] summary:hover {
        box-shadow: 0 3px 12px rgba(23, 64, 143, 0.5);
    }

    section[data-testid="stSidebar"]
    div[data-testid="stExpander"] summary svg {
        fill: white;
    }

   section[data-testid="stSidebar"]
div[data-testid="stExpander"]
div[data-testid="stExpanderDetails"] {
    background-color: #ffffff;
    border: 1px solid #d8d8d8;
    border-radius: 0 0 12px 12px;
    padding: 14px 14px 14px 14px;
    margin-top: -4px;
    overflow: hidden;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"]
div[data-testid="stExpanderDetails"] * {
    color: #202020 !important;
}

    section[data-testid="stSidebar"]
    div[data-testid="stButton"] button {
        background: linear-gradient(
            90deg,
            #17408f 0%,
            #2f6fd6 55%,
            #55c6f2 100%
        );
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 0;
        box-shadow: 0 2px 8px rgba(23, 64, 143, 0.35);
    }

    [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(
            90deg,
            #17408f 0%,
            #2f6fd6 55%,
            #55c6f2 100%
        );
        border: none;
        border-radius: 10px;
        color: #ffffff !important;
        font-weight: 600;
        padding: 10px 22px;
        box-shadow: 0 2px 8px rgba(23, 64, 143, 0.30);
    }

    [data-testid="stAppViewContainer"] div[data-testid="stFormSubmitButton"] button * {
        color: #ffffff !important;
    }

    .dashboard-title {
        font-size: 29px;
        font-weight: 700;
        color: #202020;
        line-height: 1.15;
        margin-bottom: 3px;
    }

    .dashboard-subtitle {
        font-size: 15px;
        color: #686868;
        margin-bottom: 18px;
    }

.kpi-card {
    font-family: 'Poppins', sans-serif;
    background-color: white;
    border: 1.5px solid #183b91;
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 108px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
}

.kpi-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-top: 10px;
}

.kpi-icon {
    width: 46px;
    height: 46px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #183b91;
    line-height: 1;
}

.kpi-title {
    font-family: 'Poppins', sans-serif;
    font-size: 14px;
    font-weight: 700;
    color: #303030;
    text-transform: uppercase;
}


.kpi-subtitle {
    font-family: 'Poppins', sans-serif;
    font-size: 11px;
    color: #777777;
    margin-top: 4px;
}

    .section-card {
        background-color: white;
        border: 1.5px solid #183b91;
        border-radius: 16px;
        padding: 17px 18px;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #292929 !important;
    }

    .section-description {
        font-size: 13px;
        color: #666666 !important;
        margin-top: 2px;
    }

    .risk-high {
        display: inline-block;
        background-color: #d9534f;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: 700;
    }

    .risk-low {
        display: inline-block;
        background-color: #5cb85c;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: 700;
    }

    .risk-unknown {
        display: inline-block;
        background-color: #777777;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 12px;
        font-weight: 700;
    }

    .alert-box {
        padding: 12px 14px;
        border-radius: 7px;
        margin: 10px 0;
        border-left: 4px solid;
        font-size: 14px;
        line-height: 1.5;
    }

    .alert-info {
        background-color: #eef4ff;
        border-left-color: #183b91;
        color: #14213d !important;
    }

    .alert-success {
        background-color: #e6f4ea;
        border-left-color: #2e7d32;
        color: #1b3a1e !important;
    }

    .alert-warning {
        background-color: #fff6e0;
        border-left-color: #b8860b;
        color: #4d3a00 !important;
    }

    .alert-error {
        background-color: #fdecea;
        border-left-color: #c62828;
        color: #5c1414 !important;
    }

    .alert-box * {
        color: inherit !important;
    }

    .info-box {
        background-color: #eef4ff;
        border-left: 4px solid #183b91;
        padding: 12px 14px;
        border-radius: 7px;
        margin: 10px 0;
        color: #14213d !important;
    }

    .info-box * {
        color: #14213d !important;
    }

    .detail-card {
        background-color: white;
        border: 1px solid #d8d8d8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }

    .detail-label {
        font-size: 12px;
        color: #777777 !important;
    }

    .detail-value {
        font-size: 17px;
        font-weight: 600;
        color: #222222 !important;
    }

    .footer {
        text-align: center;
        color: #777777 !important;
        font-size: 12px;
        padding: 20px 0;
    }

    html, body {
        color-scheme: light !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] > div,
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[class*="ValueContainer"],
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] div[class*="control"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] *,
    [data-testid="stAppViewContainer"] div[data-baseweb="select"] input {
        color: #202020 !important;
    }

    [data-testid="stAppViewContainer"] div[data-baseweb="select"] svg {
        fill: #202020 !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div,
    ul[data-baseweb="menu"],
    div[role="listbox"] {
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] [role="option"],
    ul[data-baseweb="menu"] li,
    div[role="listbox"] [role="option"] {
        background-color: #ffffff !important;
        color: #202020 !important;
    }

    div[data-baseweb="popover"] li *,
    div[data-baseweb="popover"] [role="option"] *,
    div[data-baseweb="menu"] li *,
    div[data-baseweb="menu"] [role="option"] *,
    ul[data-baseweb="menu"] li *,
    div[role="listbox"] [role="option"] * {
        color: #202020 !important;
    }

    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [role="option"][aria-selected="true"],
    div[data-baseweb="menu"] li:hover,
    div[data-baseweb="menu"] [role="option"]:hover,
    ul[data-baseweb="menu"] li:hover,
    div[role="listbox"] [role="option"]:hover,
    div[role="listbox"] [role="option"][aria-selected="true"] {
        background-color: #eef4ff !important;
        color: #14213d !important;
    }
        /* Sidebar biru navy solid */
    section[data-testid="stSidebar"] {
        background: #1a2a7a !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        background: linear-gradient(90deg, #2f9be0 0%, #55c6f2 100%);
        padding: 8px 14px;
        font-size: 12.5px;
    }

    /* Header putih + garis biru di atas */
    header[data-testid="stHeader"] {
        background: #ffffff !important;
        border-top: 3px solid #2f6fd6;
    }
    header[data-testid="stHeader"] * { color: #202020 !important; }

    /* Card besar */
    [class*="st-key-card_"] {
        background: #ffffff;
        border: 1.5px solid #183b91;
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: 18px;
    }

    /* Hilangkan kartu putih + shadow di tiap chart */
    [data-testid="stAppViewContainer"] div[data-testid="stPlotlyChart"] {
        background: transparent;
        box-shadow: none;
        padding: 0;
    }

    .card-heading { font-size: 18px; font-weight: 700; color: #202020; }
    .card-caption { font-size: 12.5px; color: #666666; margin-bottom: 8px; }
    .chart-label  { font-size: 13px; font-weight: 600; text-align: center; color: #444; }
    .sub-heading  { font-size: 12.5px; font-weight: 700; letter-spacing: .6px;
                    text-transform: uppercase; color: #202020; margin-top: 10px; }

        section[data-testid="stSidebar"] div[data-baseweb="checkbox"] span:first-child {
        background-color: #ffffff !important;
        border: 1.5px solid #8a94b0 !important;
    }

        /* Header sidebar: logo kiri, teks kanan */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .sidebar-header .sidebar-logo-wrapper {
        margin: 0;
        flex-shrink: 0;
        width: 58px;
        height: 58px;
    }
    .sidebar-header .sidebar-brand {
        text-align: left;
        margin: 0;
    }
    .sidebar-header .sidebar-title {
        font-size: 24px;
        line-height: 1.1;
    }
    .sidebar-header .sidebar-subtitle {
        font-size: 14px;
        margin: 0;
    }

        /* Rapatkan sidebar ke atas (versi rapi) */
    [data-testid="stSidebarHeader"] {
        padding: 0.8rem 1rem 0 1rem !important;
        min-height: 0 !important;
        height: auto !important;
    }
    [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.7rem !important;
    }
    .sidebar-line {
        margin: 16px 0 18px 0 !important;
    }
    .filter-title {
        margin-top: 4px !important;
        margin-bottom: 12px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        margin-bottom: 6px !important;
    }
    .sidebar-footer-note {
        margin-top: 22px !important;
        padding-top: 14px !important;
    }

        /* Judul di bar putih atas (sticky) */
    header[data-testid="stHeader"] {
        display: flex !important;
        align-items: center !important;
    }
    header[data-testid="stHeader"]::before {
        content: "Dashboard Monitor Gizi & Stunting";
        font-size: 22px;
        font-weight: 700;
        color: #202020;
        margin-left: 3.5rem;
        white-space: nowrap;
    }
    @media (min-width: 769px) {
        .stApp:has(section[data-testid="stSidebar"][aria-expanded="true"])
        header[data-testid="stHeader"]::before {
            margin-left: 370px;
        }
    }
    .block-container {
        padding-top: 4.5rem !important;
    }
               /* Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 4.2rem 0.8rem 1.5rem 0.8rem !important;
        }
        header[data-testid="stHeader"]::before {
            font-size: 15px;
            margin-left: 3.2rem;
            content: "Monitor Gizi & Stunting";
        }
        .kpi-card {
            min-height: 90px;
            padding: 12px 14px;
        }
        .kpi-value { font-size: 24px; }
        .kpi-title { font-size: 12px; }
        .kpi-icon { width: 38px; height: 38px; }
        [class*="st-key-card_"] {
            padding: 12px;
        }
        .card-heading { font-size: 16px; }
        .chart-label { font-size: 12px; }
        [class*="st-key-card_input"] div[data-testid="stButton"] button { width: 100%; }
        .form-heading { font-size: 14px; }
        .info-box { font-size: 13px; padding: 10px 12px; }
        .tbl { font-size: 12px; }
        .tbl th, .tbl td { padding: 7px 8px; }
        .stat-num { font-size: 17px; }
                /* judul atas digeser, tombol buka filter (sidebar) dibuat jelas */
        header[data-testid="stHeader"]::before { margin-left: 4.9rem; font-size: 15px; }
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stExpandSidebarButton"] {
            background: #183b91 !important; border-radius: 8px !important;
            width: 36px !important; height: 36px !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stExpandSidebarButton"] svg {
            color: #ffffff !important; fill: #ffffff !important;
        }

        /* KPI 2 x 2 */
        .st-key-kpi_baris [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important; column-gap: 10px !important; row-gap: 12px !important;
        }
        /* tahan margin negatif bawaan st.markdown agar baris KPI tidak menempel */
        .st-key-kpi_baris [data-testid="stColumn"],
        .st-key-kpi_baris [data-testid="column"] { padding-bottom: 16px !important; }
        .st-key-kpi_baris .kpi-card { margin-bottom: 0 !important; }
        .st-key-kpi_baris [data-testid="stColumn"],
        .st-key-kpi_baris [data-testid="column"] {
            flex: 1 1 calc(50% - 5px) !important;
            width: calc(50% - 5px) !important;
            min-width: calc(50% - 5px) !important;
        }
        .kpi-card { min-height: 0; padding: 10px 12px; border-radius: 12px; }
        .kpi-title { font-size: 11px; }
        .kpi-header { gap: 8px; margin-top: 6px; }
        .kpi-icon { width: 32px; height: 32px; }
        .kpi-value { font-size: 22px; }
        .kpi-subtitle { font-size: 10px; }

        /* tombol halaman tabel tetap satu baris */
        .st-key-card_tabel [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 6px !important; }
        .st-key-card_tabel [data-testid="stColumn"],
        .st-key-card_tabel [data-testid="column"] { min-width: 0 !important; }
        .st-key-card_tabel div[data-testid="stButton"] button p { font-size: 12px !important; }
        .hal-info { font-size: 11.5px; padding-top: 10px; }
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg {
        fill: #202020 !important;
        color: #202020 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover {
        background-color: #d5dcf0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        color: #202020 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important;
    }

        /* ===== Filter sidebar: warna konsisten ===== */

    /* Checkbox: biru saat dicentang, putih saat kosong */
    section[data-testid="stSidebar"] label[data-baseweb="checkbox"] > span:first-child {
        background-color: #ffffff !important;
        border: 1.5px solid #8a94b0 !important;
        border-radius: 5px !important;
    }
    section[data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child {
        background-color: #2f6fd6 !important;
        border-color: #2f6fd6 !important;
    }

    /* Dropdown dan kotak angka: border dan sudut sama */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
        border-color: #2f6fd6 !important;
    }

    /* Tombol - dan + */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        background-color: #eef4ff !important;
        border: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg {
        fill: #183b91 !important;
        color: #183b91 !important;
    }

    /* Label kecil (Dari, Sampai, Dari bulan, dll.) */
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] label p {
        color: #303030 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }
        /* ===== Font seragam di semua filter sidebar ===== */

    /* Label (Dari bulan, Sampai, Dari, dll.) */
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] label p,
    section[data-testid="stSidebar"] div[data-testid="stExpanderDetails"] [data-testid="stWidgetLabel"] p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #202020 !important;
    }

    /* Teks checkbox (Laki-laki, Perempuan, Pilih semua, nama posyandu) */
    section[data-testid="stSidebar"] div[data-testid="stCheckbox"] p,
    section[data-testid="stSidebar"] div[data-testid="stCheckbox"] span {
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #202020 !important;
    }

    /* Isi dropdown dan kotak angka */
    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        font-size: 14px !important;
        font-weight: 400 !important;
        color: #202020 !important;
    }

    /* Caption ("Rentang usia (bulan)") */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #202020 !important;
    }

    /* Border kotak angka: ganti hitam jadi abu-biru seperti dropdown */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"],
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        outline: none !important;
    }

        /* ===== Kotak angka: satu kotak rapi ===== */

    /* Border hanya di pembungkus luar */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: none !important;
    }

    /* Hilangkan border dan latar di lapisan dalam */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background-color: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        background-color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Tombol - dan +: putih, dipisah garis tipis */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        background-color: #ffffff !important;
        border: none !important;
        border-left: 1px solid #e3e8f2 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover {
        background-color: #eef4ff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg {
        fill: #183b91 !important;
        color: #183b91 !important;
    }

    /* Caption "Rentang usia (bulan)" tidak pucat lagi */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
        opacity: 1 !important;
        color: #202020 !important;
    }
        /* ===== Kotak angka: SATU border saja ===== */

    /* 1. Hapus border, latar, sudut, bayangan di SEMUA lapisan dalam */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] *,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] *::before,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] *::after {
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }

    /* 2. Border + sudut hanya di pembungkus paling luar */
    section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"],
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] > div > div[data-baseweb="input"] {
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        overflow: hidden !important;
    }

    /* 3. Kalau pembungkus ganda ikut kena, cegah garis dobel */
    section[data-testid="stSidebar"] div[data-testid="stNumberInputContainer"] div[data-baseweb="input"] {
        border: none !important;
    }

    /* 4. Isi angka */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: #202020 !important;
        font-size: 14px !important;
        padding-left: 12px !important;
    }

    /* 5. Tombol - dan +: garis pemisah tipis saja */
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        background: #ffffff !important;
        border-left: 1px solid #e3e8f2 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button:hover {
        background: #eef4ff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg {
        fill: #183b91 !important;
        color: #183b91 !important;
    }

        /* ===== Form kader ===== */
    .form-heading {
        font-size: 15px; font-weight: 700; color: #183b91;
        margin: 6px 0 6px 0;
    }
    [class*="st-key-card_input"] div[data-testid="stButton"] button {
        background: #183b91; border: none; border-radius: 10px;
        padding: 10px 22px; font-weight: 600;
    }
    [class*="st-key-card_input"] div[data-testid="stButton"] button:hover {
        background: #2f6fd6;
    }
    [class*="st-key-card_input"] div[data-testid="stButton"] button * {
        color: #ffffff !important;
    }
    .result-card {
        background: #ffffff; border: 1px solid #d8d8d8;
        border-radius: 10px; padding: 14px; text-align: center;
    }
    .result-label { font-size: 13px; color: #666666 !important; }
    .result-value { font-size: 18px; font-weight: 700; color: #202020 !important; margin-top: 4px; }

    /* ===== Tabel intervensi ===== */
    .tbl-wrap {
        max-height: 460px; overflow: auto;
        border: 1px solid #e3e3e3; border-radius: 8px;
    }
    .tbl { width: 100%; border-collapse: collapse; font-size: 13px; }
    .tbl th {
        position: sticky; top: 0; background: #ececec; color: #303030;
        font-size: 11.5px; text-align: left; padding: 9px 10px;
        letter-spacing: .3px; white-space: nowrap;
    }
    .tbl td { padding: 9px 10px; border-top: 1px solid #ededed; color: #202020; }
    .tbl tr.row-high td { background: #fdecea; }
    .badge-high {
        display: inline-block; background: #c9524f; color: #ffffff;
        padding: 3px 9px; border-radius: 4px; font-size: 12px; font-weight: 600;
        white-space: nowrap;
    }
    .badge-low {
        display: inline-block; background: #f1f1f1; color: #303030;
        border: 1px solid #bbbbbb; padding: 3px 9px; border-radius: 4px;
        font-size: 12px; font-weight: 600; white-space: nowrap;
    }

        /* Number input area utama: sama dengan dropdown */
    [data-testid="stAppViewContainer"] div[data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {
        background: #ffffff !important;
        border: none !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stNumberInput"] button {
        background: #ffffff !important;
        border: none !important;
        border-left: 1px solid #e3e8f2 !important;
        border-radius: 0 !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stNumberInput"] button:hover {
        background: #eef4ff !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stNumberInput"] button svg {
        fill: #183b91 !important;
        color: #183b91 !important;
    }

        /* ===== Ruang bawah card ===== */
    [class*="st-key-card_"] {
        padding: 18px 20px 30px 20px !important;
        margin-bottom: 10px !important;
    }
    .block-container {
        padding-bottom: 4rem !important;
    }

    /* Kotak teks di area utama: sama dengan dropdown */
    [data-testid="stAppViewContainer"] div[data-testid="stTextInput"] div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #cfd6e4 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
        background: #ffffff !important;
        border: none !important;
    }
    [data-testid="stAppViewContainer"] div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: #202020 !important;
        font-size: 14px !important;
    }


    /* ===== Placeholder terbaca ===== */
    [data-testid="stAppViewContainer"] input::placeholder {
        color: #8a93a8 !important;
        opacity: 1 !important;
    }

    /* ===== Hilangkan tombol silang (x) di form kader ===== */
    [class*="st-key-card_input"] [title="Clear value"],
    [class*="st-key-card_input"] [aria-label="Clear value"] {
        display: none !important;
    }

    /* ===== Kartu hasil prediksi: tinggi sama ===== */
    .result-card {
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    /* ===== Diagram risiko kosong ===== */
    .empty-chart {
        height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
    }
    .empty-ring {
        width: 170px;
        height: 170px;
        border-radius: 50%;
        border: 26px solid #e8ebf2;
    }
    .empty-text {
        font-size: 13px;
        color: #777777;
    }

    /* ===== Tombol reset hasil prediksi ===== */
    [class*="st-key-card_risiko"] div[data-testid="stButton"] button {
        background: #ffffff;
        border: 1.5px solid #183b91;
        border-radius: 10px;
        padding: 8px 0;
    }
    [class*="st-key-card_risiko"] div[data-testid="stButton"] button * {
        color: #183b91 !important;
        font-weight: 600;
    }
    [class*="st-key-card_risiko"] div[data-testid="stButton"] button:hover {
        background: #eef4ff;
    }

    /* ===== Kolom RT/RW di tabel tidak patah baris ===== */
    .tbl td:last-child {
        white-space: nowrap;
    }

        /* ===== Tabel: isi selain nama ditengah ===== */
    .tbl th:not(:first-child),
    .tbl td:not(:first-child) {
        text-align: center;
        white-space: nowrap;
    }

    div[data-testid="stColumn"]:has(.st-key-card_risiko) div:has(.st-key-card_risiko),
    div[data-testid="column"]:has(.st-key-card_risiko) div:has(.st-key-card_risiko) {
        height: 100%;
    }

    /* ===== Diagram kosong lebih besar ===== */
    .empty-chart { height: 340px; }
    .empty-ring  { width: 200px; height: 200px; }

    /* ===== Kotak ringkasan ===== */
    .stat-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 14px;
    }
    .stat-box {
        background: #f5f7fb;
        border: 1px solid #e3e8f2;
        border-radius: 10px;
        padding: 10px 4px;
        text-align: center;
    }
    .stat-num { font-size: 20px; font-weight: 700; color: #183b91; line-height: 1.1; }
    .stat-num.high { color: #d9534f; }
    .stat-num.low  { color: #3d9a3d; }
    .stat-lbl { font-size: 11px; color: #666666; margin-top: 3px; }

        /* ===== Card risiko: diagram di tengah, ringkasan + tombol di dasar ===== */
    .st-key-card_risiko {
        display: flex !important;
        flex-direction: column !important;
    }
    .st-key-card_risiko [data-testid="stElementContainer"]:has(.empty-chart),
    .st-key-card_risiko [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {
        margin-top: auto !important;
        margin-bottom: auto !important;
    }
    .st-key-card_risiko [data-testid="stElementContainer"]:has(.stat-row) {
        margin-top: 0 !important;
    }
    .st-key-card_risiko [data-testid="stElementContainer"]:has(.empty-chart) .empty-chart {
        height: auto;
        gap: 18px;
        padding: 10px 0;
    }
        /* ===== Hilangkan border putih di header filter sidebar ===== */
    section[data-testid="stSidebar"] div[data-testid="stExpander"],
    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] details,
    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] div[data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
    [data-testid="stAppViewContainer"] section[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
        border: none !important;
        outline: none !important;
        background: linear-gradient(90deg, #2f9be0 0%, #55c6f2 100%) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:focus,
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

        /* ===== Card risiko: isi rapat di atas, tombol reset di dasar ===== */
    .st-key-card_risiko {
        display: flex !important;
        flex-direction: column !important;
    }
    .st-key-card_risiko [data-testid="stElementContainer"]:has(button) {
        margin-top: auto !important;
    }
    .empty-chart {
        height: auto !important;
        gap: 16px;
        padding: 18px 0 8px 0;
    }
    .empty-ring {
        width: 230px;
        height: 230px;
        border-width: 30px;
    }

    /* Panduan singkat */
    .howto {
        margin-top: 16px;
        background: #f5f7fb;
        border: 1px solid #e3e8f2;
        border-radius: 10px;
        padding: 12px 14px;
    }
    .howto-title {
        font-size: 12.5px;
        font-weight: 700;
        color: #183b91;
        margin-bottom: 8px;
    }
    .howto-step {
        display: flex;
        gap: 9px;
        align-items: flex-start;
        font-size: 12.5px;
        color: #303030;
        line-height: 1.4;
        margin-bottom: 7px;
    }
    .howto-step:last-child { margin-bottom: 0; }
    .howto-no {
        flex-shrink: 0;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #183b91;
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* ===== Tombol halaman dan hapus di tabel ===== */
    .hal-info { text-align: center; font-size: 13px; color: #555555; padding-top: 9px; }
    [class*="st-key-card_tabel"] div[data-testid="stButton"] button {
        background: #ffffff; border: 1.5px solid #183b91; border-radius: 10px;
    }
    [class*="st-key-card_tabel"] div[data-testid="stButton"] button * {
        color: #183b91 !important; font-weight: 600;
    }
    [class*="st-key-card_tabel"] div[data-testid="stButton"] button:hover { background: #eef4ff; }
    [class*="st-key-card_tabel"] div[data-testid="stButton"] button:disabled { opacity: 0.4; }

    /* ===== Filter sidebar: tanpa garis putih (taruh paling akhir) ===== */
    html body section[data-testid="stSidebar"] [data-testid="stExpander"],
    html body section[data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: 0 !important;
        border-color: transparent !important;
        outline: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    html body section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        border: 0 !important;
        outline: 0 !important;
        border-radius: 12px !important;
        background: linear-gradient(90deg, #2f9be0 0%, #55c6f2 100%) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
        padding: 10px 14px !important;
        transition: filter 0.15s ease;
    }
    html body section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary {
        border-radius: 12px 12px 0 0 !important;
    }
    html body section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        filter: brightness(1.06);
    }
    html body section[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        border: 0 !important;
        border-radius: 0 0 12px 12px !important;
        margin-top: 0 !important;
        background: #ffffff !important;
    }

        /* ===== Loading prediksi ===== */
    .loading-overlay {
        position: fixed; inset: 0; z-index: 999999;
        background: rgba(245, 246, 248, 0.75);
        display: flex; align-items: center; justify-content: center;
    }
    .loading-box {
        background: #ffffff; border: 1.5px solid #183b91; border-radius: 16px;
        padding: 22px 28px; display: flex; flex-direction: column; align-items: center; gap: 14px;
        box-shadow: 0 10px 30px rgba(24, 59, 145, 0.18);
        font-size: 14px; font-weight: 600; color: #183b91;
    }
    .loading-spinner {
        width: 46px; height: 46px; border-radius: 50%;
        border: 5px solid #e3e8f2; border-top-color: #183b91;
        animation: putar 0.8s linear infinite;
    }
    @keyframes putar { to { transform: rotate(360deg); } }

    /* ===== Loading umum: kotak tengah, sama dengan loading prediksi ===== */
    [data-testid="stStatusWidget"] { display: none !important; }

    .stApp[data-test-script-state="running"]:not(:has(.loading-overlay)):not(:has([class*="st-key-card_input"]:focus-within))::before {
        content: "";
        position: fixed; inset: 0; z-index: 999998;
        background: rgba(245, 246, 248, 0.75);
        opacity: 0;
        animation: muncul 0.15s ease 0.3s forwards;
    }
    .stApp[data-test-script-state="running"]:not(:has(.loading-overlay)):not(:has([class*="st-key-card_input"]:focus-within))::after {
        content: "Memuat data...";
        position: fixed; top: 50%; left: 50%; z-index: 999999;
        transform: translate(-50%, -50%);
        background: #ffffff url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='46' height='46' viewBox='0 0 46 46'><circle cx='23' cy='23' r='20' fill='none' stroke='%23e3e8f2' stroke-width='5'/><path d='M23 3 a20 20 0 0 1 20 20' fill='none' stroke='%23183b91' stroke-width='5'><animateTransform attributeName='transform' type='rotate' from='0 23 23' to='360 23 23' dur='0.8s' repeatCount='indefinite'/></path></svg>") no-repeat center 22px;
        border: 1.5px solid #183b91; border-radius: 16px;
        padding: 82px 40px 22px;
        box-shadow: 0 10px 30px rgba(24, 59, 145, 0.18);
        font-size: 14px; font-weight: 600; color: #183b91;
        opacity: 0;
        animation: muncul 0.15s ease 0.3s forwards;
    }
    .stApp[data-test-script-state="running"]:not(:has([class*="st-key-card_input"]:focus-within)) [data-stale="true"] {
        animation: muncul 0.15s ease 0.3s reverse forwards;
    }
    @keyframes muncul { from { opacity: 0; } to { opacity: 1; } }

        /* ===== Pesan angka di luar batas pada form kader ===== */
    .input-error {
        margin: -6px 0 10px 0;
        padding: 6px 12px;
        border-radius: 8px;
        background: #FDECEC;
        color: #C94A4A;
        font-size: 13px;
        font-weight: 500;
    }
    /* sembunyikan tulisan "Press Enter to apply" */
    div[data-testid="InputInstructions"] { display: none !important; }

    /* ===== Ikon ? di form kader dibuat lebih jelas ===== */
    [class*="st-key-card_input"] [data-testid="stTooltipIcon"] svg {
        color: #2B6FD6 !important;
        opacity: 1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5B. ALERT HELPER (PENGGANTI st.warning/st.error/st.info/st.success)
# ============================================================

def render_alert(message, kind="info"):
    css_class = {
        "info": "alert-info",
        "success": "alert-success",
        "warning": "alert-warning",
        "error": "alert-error",
    }.get(kind, "alert-info")

    st.markdown(
        f"""
        <div class="alert-box {css_class}">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 5C. CHART THEME HELPER
# ============================================================
def apply_light_theme(fig):

    if fig is None:
        return fig

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#202020"),
        title_font=dict(color="#202020"),
        uniformtext_minsize=9,
        uniformtext_mode="hide",
    )

    fig.update_xaxes(
        color="#202020",
        gridcolor="#e8e8e8",
        linecolor="#cccccc",
        title_font=dict(color="#202020"),
        tickfont=dict(color="#202020"),
    )

    fig.update_yaxes(
        color="#202020",
        gridcolor="#e8e8e8",
        linecolor="#cccccc",
        title_font=dict(color="#202020"),
        tickfont=dict(color="#202020"),
    )

    return fig

# ============================================================
# 6. LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(path):

    if not path.exists():
        return None, f"Dataset tidak ditemukan: {path}"

    try:
        data = pd.read_excel(path)

        if data.empty:
            return None, "Dataset berhasil dibaca tetapi kosong."

        return data, None

    except Exception as e:
        return None, f"Gagal membaca dataset: {e}"


# ============================================================
# 11. LOAD LOGO
# ============================================================

@st.cache_data(show_spinner=False)
def get_base64_image(path):

    if path.exists():

        with open(
            path,
            "rb",
        ) as f:

            return base64.b64encode(
                f.read()
            ).decode()

    return None


# ============================================================
# 12. LOAD ALL
# ============================================================

df, data_error = load_dataset(DATA_PATH)
gizi_df, _ = load_dataset(TREND_PATH)

model_bundle, model_error = muat_model(MODEL_PATH)

model_available = model_bundle is not None

# ============================================================
# 13. HELPER FUNCTIONS
# ============================================================

def find_column(
    data,
    exact_names=None,
    keywords=None,
):

    if data is None or data.empty:
        return None

    exact_names = exact_names or []
    keywords = keywords or []

    normalized = {
        str(c).strip().lower(): c
        for c in data.columns
    }

    for name in exact_names:

        key = str(name).strip().lower()

        if key in normalized:
            return normalized[key]

    for column in data.columns:

        col_lower = (
            str(column)
            .strip()
            .lower()
        )

        if any(
            keyword.lower()
            in col_lower
            for keyword in keywords
        ):

            return column

    return None


def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def normalize_bbu_category(value):

    text = normalize_text(value)

    if not text:
        return "Tidak diketahui"

    if "sangat" in text and "kurang" in text:
        return "Berat badan sangat kurang (severely underweight)"

    if "kurang" in text:
        return "Berat badan kurang (underweight)"

    if "lebih" in text or "risiko" in text:
        return "Risiko berat badan lebih"

    if "normal" in text:
        return "Berat badan normal"

    return str(value).strip()


def normalize_tbu_category(value):

    text = normalize_text(value)

    if not text:
        return "Tidak diketahui"

    if "sangat" in text and "pendek" in text:
        return "Sangat pendek (severely stunted)"

    if "pendek" in text:
        return "Pendek (stunted)"

    if "tinggi" in text:
        return "Tinggi"

    if "normal" in text:
        return "Normal"

    return str(value).strip()


def normalize_bbtb_category(value):

    text = normalize_text(value)

    if not text:
        return "Tidak diketahui"

    if "sangat" in text and "kurus" in text or "buruk" in text:
        return "Gizi buruk (severely wasted)"
    
    if "gizi" in text and "kurang" in text:
        return "Gizi kurang (wasted)"
    
    if "kurus" in text:
        return "Gizi kurang (wasted)"

    if "obes" in text:
        return "Obesitas (obese)"

    if "risiko" in text or "berisiko" in text:
        return "Berisiko gizi lebih (possible risk of overweight)"

    if "lebih" in text:
        return "Gizi lebih (overweight)"

    if "normal" in text or "baik" in text:
        return "Gizi baik (normal)"

    return str(value).strip()

def normalize_bblr_category(value):

    if pd.isna(value):
        return "Tidak diketahui"

    try:
        weight = float(value)
    except (TypeError, ValueError):
        return "Tidak diketahui"

    # Beberapa baris tercatat dalam gram, bukan kg
    # (mis. 2600 alih-alih 2.6). Berat lahir wajar bayi
    # berada di rentang 0.5-6 kg, sehingga nilai di atas
    # itu dianggap salah satuan dan dikonversi dari gram
    # ke kg terlebih dahulu.
    if weight > 25:
        weight = weight / 1000

    if weight < 2.5:
        return "BBLR (<2,5 kg)"

    return "Normal (≥2,5 kg)"


SHORT_LABELS = {
    "Berat badan sangat kurang (severely underweight)": "Sangat kurang",
    "Berat badan kurang (underweight)": "Kurang",
    "Risiko berat badan lebih": "Risiko lebih",
    "Berat badan normal": "Normal",
    "Sangat pendek (severely stunted)": "Sangat pendek",
    "Pendek (stunted)": "Pendek",
    "Gizi buruk (severely wasted)": "Gizi buruk",
    "Gizi kurang (wasted)": "Gizi kurang",
    "Gizi baik (normal)": "Gizi baik",
    "Berisiko gizi lebih (possible risk of overweight)": "Berisiko gizi lebih",
    "Gizi lebih (overweight)": "Gizi lebih",
    "Obesitas (obese)": "Obesitas",
    "Gizi Kurang": "Gizi kurang",
}

def bucket_age_puskesmas(value):

    try:
        months = float(value)
    except (TypeError, ValueError):
        return "Tidak diketahui"

    if pd.isna(months):
        return "Tidak diketahui"

    if months < 6:
        return "0-5"
    elif months < 12:
        return "6-11"
    elif months < 24:
        return "12-23"
    else:
        return "24-59"

def make_donut(data, column, title=None, normalize_func=None, height=400, urutan=None, warna=None):

    if column is None or data.empty:
        return None

    series = data[column].fillna("Tidak diketahui")

    if normalize_func is not None:
        series = series.apply(normalize_func)
    else:
        series = series.astype(str).str.strip().replace("", "Tidak diketahui")

    
    series = series.replace(SHORT_LABELS)


    chart_data = series.value_counts().reset_index()
    chart_data.columns = ["Kategori", "Jumlah"]

    # urutan kategori tetap (mis. kelompok usia), bukan dari jumlah terbanyak
    if urutan:
        posisi = {k: i for i, k in enumerate(urutan)}
        chart_data = chart_data.sort_values(
            "Kategori", key=lambda s: s.map(posisi).fillna(len(urutan)), kind="stable"
        )

    fig = px.pie(
        chart_data,
        names="Kategori",
        values="Jumlah",
        hole=0.55,
        color_discrete_sequence=warna or [
            "#3b5fe0", "#e0553b", "#2fae7a", "#8b5cf6",
            "#f2994a", "#3ecbd9", "#e0b53b", "#e03b8f",
        ],
    )

    # Area donut dikunci: ukuran sama di semua chart
    domain_y0 = 0.36
    center_y = (domain_y0 + 1.0) / 2

    fig.update_traces(
        domain=dict(x=[0.08, 0.92], y=[domain_y0, 1.0]),
        marker=dict(line=dict(color="#ffffff", width=2)),
        textfont=dict(size=11, color="#ffffff"),
        textposition="inside",
        sort=False,
    )

    fig.add_annotation(
        text=f"<b>{int(chart_data['Jumlah'].sum())}</b>",
        xref="paper", yref="paper",
        x=0.5, y=center_y, showarrow=False,
        font=dict(size=20, color="#183b91"),
    )

    fig.update_layout(
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v", yanchor="top", y=domain_y0 - 0.04,
            xanchor="left", x=0.0,
            font=dict(size=10, color="#202020"),
        ),
        margin=dict(l=5, r=5, t=5, b=5),
    )

    return apply_light_theme(fig)


# ============================================================
# 15. DATA CHECK
# ============================================================

if df is None:

    render_alert(data_error, "error")

    render_alert(
        "Periksa lokasi file data_stunting.xlsx.",
        "info",
    )

    st.stop()


# ============================================================
# 15B. BUAT KOLOM USIA NUMERIK (BULAN) OTOMATIS
# ============================================================
# Kolom "Usia Saat Ukur" di dataset berformat teks
# ("3 Tahun - 6 Bulan - 13 Hari"), sehingga to_numeric()
# selalu gagal dan chart usia tampil "Data usia kosong."
# Prioritas: pakai kolom numerik siap pakai (Age_Months_Pygrowup)
# jika ada; jika tidak, parsing teks "Usia Saat Ukur" jadi
# total bulan.


def parse_age_text_to_months(value):

    if pd.isna(value):
        return None

    text = str(value)

    years_match = re.search(r"(\d+)\s*Tahun", text)
    months_match = re.search(r"(\d+)\s*Bulan", text)
    days_match = re.search(r"(\d+)\s*Hari", text)

    if not (years_match or months_match or days_match):
        return None

    total_months = 0.0

    if years_match:
        total_months += int(years_match.group(1)) * 12

    if months_match:
        total_months += int(months_match.group(1))

    if days_match:
        total_months += int(days_match.group(1)) / 30

    return round(total_months, 1)


if (
    "Age_Months_Pygrowup" in df.columns
    and pd.to_numeric(df["Age_Months_Pygrowup"], errors="coerce").notna().any()
):

    df["Usia (Bulan)"] = pd.to_numeric(
        df["Age_Months_Pygrowup"],
        errors="coerce",
    )

elif "Usia Saat Ukur" in df.columns:

    df["Usia (Bulan)"] = (
        df["Usia Saat Ukur"]
        .apply(parse_age_text_to_months)
    )

# ============================================================
# 15C. SIMPAN HASIL PREDIKSI KE FILE (tidak hilang saat reload)
# ============================================================

HASIL_PATH = ML_DIR / "Output" / "hasil_prediksi.csv"


def kunci_balita(idx):
    """Penanda balita yang tetap sama walau file dibaca ulang: NIK, atau nama + tanggal lahir."""
    row = df.loc[idx]
    nik = str(row.get("NIK", "")).strip()
    if nik.endswith(".0"):
        nik = nik[:-2]
    if nik and nik.lower() not in ("nan", "none", "0", "-"):
        return nik
    return f"{str(row.get('Nama', '')).strip().upper()}|{row.get('Tanggal Lahir', '')}"

def isian_kader(nilai):
    """Isian form kader dalam bentuk teks seperti yang tampil di layar."""
    isian = {}
    for k in nilai:
        if k not in INPUT or nilai.get(k) is None:
            continue
        v = st.session_state.get(f"f_{k}")
        if v is None or v == []:
            continue
        if isinstance(v, list):
            v = ", ".join(v)
        elif isinstance(v, float):
            v = f"{v:g}".replace(".", ",")
        isian[INPUT[k][0]] = str(v)
    return isian


def baca_isian(teks):
    try:
        return json.loads(teks) if isinstance(teks, str) and teks else {}
    except ValueError:
        return {}

def muat_hasil():
    if not HASIL_PATH.exists():
        return {}
    try:
        h = pd.read_csv(HASIL_PATH, dtype={"kunci": str})
    except Exception:
        return {}
    ke_idx = {kunci_balita(i): i for i in df.index}
    hasil = {}
    for r in h.itertuples():
        idx = ke_idx.get(str(r.kunci))
        if idx is not None:
            hasil[idx] = {"prob": float(r.prob), "tinggi": bool(r.tinggi), "waktu": str(r.waktu),
                          "isian": baca_isian(getattr(r, "isian", ""))}
    return hasil


def simpan_hasil():
    hasil = st.session_state.get("pred_results", {})
    rows = [
        {
            "kunci": kunci_balita(idx),
            "nama": str(df.loc[idx].get("Nama", "")),
            "prob": v["prob"],
            "tinggi": v["tinggi"],
            "waktu": v.get("waktu", ""),
            "isian": json.dumps(v.get("isian", {}), ensure_ascii=False),
        }
        for idx, v in hasil.items()
        if idx in df.index
    ]
    try:
        pd.DataFrame(rows, columns=["kunci", "nama", "prob", "tinggi", "waktu", "isian"]).to_csv(
            HASIL_PATH, index=False, encoding="utf-8-sig"
        )
    except Exception as e:
        render_alert(f"Hasil prediksi gagal disimpan: {e}", "warning")


if "pred_results" not in st.session_state:
    st.session_state["pred_results"] = muat_hasil()

# ============================================================
# 16. IDENTIFY ORIGINAL COLUMNS
# ============================================================

date_candidates = [
    c
    for c in df.columns
    if any(
        keyword in str(c).lower()
        for keyword in [
            "tanggal",
            "tgl",
            "date",
            "periode",
            "pengukuran",
            "ukur",
        ]
    )
    and "lahir" not in str(c).lower()
]

if "Tanggal Pengukuran" in df.columns:
    date_col = "Tanggal Pengukuran"
elif date_candidates:
    date_col = date_candidates[0]
else:
    date_col = None
BULAN_MAP = {
    "januari": "January", "februari": "February", "maret": "March",
    "april": "April", "mei": "May", "juni": "June", "juli": "July",
    "agustus": "August", "september": "September",
    "oktober": "October", "november": "November", "desember": "December",
    "agu": "Aug", "okt": "Oct", "des": "Dec",
}


def parse_tanggal_indonesia(value):

    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value

    text = str(value).strip().lower()

    for indo, eng in BULAN_MAP.items():
        text = re.sub(rf"\b{indo}\b", eng, text)

    # Jika tahun tidak ditulis, anggap tahun 2026
    if not re.search(r"\d{4}", text):
        text = f"{text} 2026"

    return pd.to_datetime(text, dayfirst=True, errors="coerce")


if date_col is not None:

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):

        df[date_col] = df[date_col].apply(parse_tanggal_indonesia)


sex_candidates = [
    c
    for c in df.columns
    if str(c).strip().lower()
    in [
        "jk",
        "jenis kelamin",
        "jenis_kelamin",
        "jenis kelamin balita",
        "sex",
        "gender",
    ]
]

sex_col = (
    sex_candidates[0]
    if sex_candidates
    else None
)


if "Usia (Bulan)" in df.columns:

    age_col = "Usia (Bulan)"

else:

    age_candidates = [
        c
        for c in df.columns
        if any(
            keyword in str(c).lower()
            for keyword in [
                "usia",
                "umur",
                "age",
            ]
        )
    ]

    age_col = (
        age_candidates[0]
        if age_candidates
        else None
    )


posyandu_candidates = [
    c
    for c in df.columns
    if "posyandu"
    in str(c).lower()
]

posyandu_col = (
    posyandu_candidates[0]
    if posyandu_candidates
    else None
)
# ============================================================
# 16b. PANDUAN PENGGUNAAN (POP-UP)
# ============================================================
@st.dialog(" ", width="large")
def tampilkan_panduan():
    langkah = [
        ("Atur data yang ingin dilihat",
         "Pilih periode, jenis kelamin, kelompok usia, dan Posyandu melalui filter di sebelah kiri. "
         "Supaya grafik tidak terlalu padat, pilih satu Posyandu terlebih dahulu."),
        ("Lihat ringkasan dan grafik",
         "Jumlah balita pada kotak ringkasan dan grafik akan menyesuaikan filter yang dipilih. "
         "Setiap warna pada grafik menunjukkan kategori status gizi yang berbeda. "
         "Sentuh batang grafik untuk melihat jumlah balitanya."),
        ("Cari data balita",
         "Ketik nama balita di kolom pencarian pada Tabel Intervensi Pemantauan untuk melihat "
         "status gizi, status stunting, dan kategori risikonya."),
        ("Isi data untuk memprediksi risiko stunting",
         "Pada form <b>Input Data Balita oleh Kader</b>, pilih nama balita, lalu lengkapi semua pertanyaan "
         "yang ditandai bintang merah <b style='color:#E5484D'>*</b>. Setelah itu, tekan tombol "
         "<b>Prediksi Risiko Stunting</b>."),
        ("Perhatikan hasil prediksi",
         "Hasil ditampilkan berupa kategori risiko lebih tinggi atau lebih rendah, beserta faktor yang perlu "
         "diperhatikan. Hasil ini merupakan perkiraan untuk membantu kader menentukan balita yang perlu "
         "mendapat perhatian lebih lanjut, bukan diagnosis medis."),
    ]
    isi = "".join(
        f"""<div class="pd-step"><div class="pd-no">{i}</div>
        <div><div class="pd-judul">{j}</div><div class="pd-teks">{t}</div></div></div>"""
        for i, (j, t) in enumerate(langkah, start=1)
    )
    st.markdown(
        f"""
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {{
            background: #FFFFFF !important; border-radius: 22px !important;
        }}
        div[data-testid="stDialog"] button[aria-label="Close"] svg {{ color: #1B2A7B !important; }}
        .pd-title {{
            color: #1B2A7B; font-size: 26px; font-weight: 700; margin: -46px 0 18px 0;
        }}
        .pd-step {{
            display: flex; gap: 14px; align-items: flex-start;
            background: #EEF5FD; border-radius: 14px; padding: 12px 16px; margin-bottom: 10px;
        }}
        .pd-no {{
            flex: 0 0 32px; height: 32px; border-radius: 50%;
            background: linear-gradient(135deg, #2B6FD6, #4FC3F0);
            color: #FFFFFF; font-weight: 700; display: flex; align-items: center; justify-content: center;
        }}
        .pd-judul {{ color: #1B2A7B; font-weight: 700; font-size: 16px; }}
        .pd-teks {{ color: #3D4A6B; font-size: 14.5px; line-height: 1.5; }}
        .st-key-btn_tutup_panduan button {{
            background: linear-gradient(90deg, #2B6FD6, #4FC3F0) !important;
            border: none !important; border-radius: 14px !important;
        }}
        .st-key-btn_tutup_panduan button p {{ color: #FFFFFF !important; font-weight: 600 !important; }}
        </style>
        <div class="pd-title">Panduan Penggunaan</div>
        {isi}
        """,
        unsafe_allow_html=True,
    )
    if st.button("Tutup", use_container_width=True, key="btn_tutup_panduan"):
        st.rerun()
# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:
    logo_base64 = get_base64_image(LOGO_PATH)

    if logo_base64:
        logo_inner = f'<img src="data:image/png;base64,{logo_base64}">'
        logo_class = "sidebar-logo-wrapper"
    else:
        logo_inner = "<span>P</span>"
        logo_class = "sidebar-logo-wrapper sidebar-logo-fallback"

    st.markdown(
        f"""
        <div class="sidebar-header">
            <div class="{logo_class}">
                {logo_inner}
            </div>
            <div class="sidebar-brand">
                <div class="sidebar-title">Posyandu</div>
                <div class="sidebar-subtitle">Kel. Tambakrejo</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-line"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="filter-title">'
        'FILTER INTERAKTIF'
        '</div>',
        unsafe_allow_html=True,
    )
    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    selected_date_range = None

    with st.expander(
        "Periode Pengukuran"
    ):

        if date_col:

            month_periods = [
                pd.Period(f"2026-{m:02d}", freq="M")
                for m in range(1, 8)
            ]

            month_labels = [
                "Januari 2026", "Februari 2026", "Maret 2026",
                "April 2026", "Mei 2026", "Juni 2026", "Juli 2026",
            ]

            start_label = st.selectbox(
                "Dari bulan",
                month_labels,
                index=0,
                key="date_start",
            )

            end_label = st.selectbox(
                "Sampai bulan",
                month_labels,
                index=len(month_labels) - 1,
                key="date_end",
            )

            start_p = month_periods[month_labels.index(start_label)]
            end_p = month_periods[month_labels.index(end_label)]

            if start_p > end_p:
                start_p, end_p = end_p, start_p

            selected_date_range = (
                start_p.start_time.date(),
                end_p.end_time.date(),
            )

        else:

            st.caption(
                "Periode pengukuran belum teridentifikasi."
            )
    # --------------------------------------------------------
    # SEX FILTER
    # --------------------------------------------------------

    selected_sex = None

    with st.expander(
        "Jenis Kelamin"
    ):

        if sex_col:

            sex_values = sorted(
                df[sex_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            sex_labels = {
                "L": "Laki-laki",
                "P": "Perempuan",
                "l": "Laki-laki",
                "p": "Perempuan",
            }

            selected_sex = []

            for i, value in enumerate(sex_values):

                if st.checkbox(
                    sex_labels.get(value, value),
                    value=True,
                    key=f"sex_{i}",
                ):
                    selected_sex.append(value)

        else:

            st.caption(
                "Jenis kelamin belum teridentifikasi."
            )
                  # --------------------------------------------------------
    # AGE FILTER
    # --------------------------------------------------------

    selected_age = None

    with st.expander(
        "Kelompok Usia"
    ):

        if age_col:

            valid_age = pd.to_numeric(
                df[age_col],
                errors="coerce",
            ).dropna()

            if not valid_age.empty:

                age_min = 0
                age_max = 59

                st.caption("Rentang usia (bulan)")

                a1, a2 = st.columns(2)

                with a1:
                    age_from = st.number_input(
                        "Dari",
                        min_value=age_min,
                        max_value=age_max,
                        value=age_min,
                        step=1,
                        key="age_from",
                    )

                with a2:
                    age_to = st.number_input(
                        "Sampai",
                        min_value=age_min,
                        max_value=age_max,
                        value=age_max,
                        step=1,
                        key="age_to",
                    )

                if age_from > age_to:
                    age_from, age_to = age_to, age_from

                selected_age = (age_from, age_to)

            else:

                st.caption("Data usia kosong.")

        else:

            st.caption("Usia belum teridentifikasi.")
    # --------------------------------------------------------
    # POSYANDU FILTER
    # --------------------------------------------------------

    selected_posyandu = None

    with st.expander(
        "Posyandu"
    ):

        if posyandu_col:

            posyandu_values = (
                df[posyandu_col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            posyandu_values = sorted(
                posyandu_values
            )

            def toggle_all_posyandu(n):
                for j in range(n):
                    st.session_state[f"posyandu_{j}"] = (
                        st.session_state["posyandu_all"]
                    )

            st.checkbox(
                "Pilih semua",
                value=True,
                key="posyandu_all",
                on_change=toggle_all_posyandu,
                args=(len(posyandu_values),),
            )

            selected_posyandu = []

            with st.container(height=260):

                for i, value in enumerate(posyandu_values):

                    if st.checkbox(value, value=True, key=f"posyandu_{i}"):
                        selected_posyandu.append(value)

        else:

            st.caption(
                "Posyandu belum teridentifikasi."
            )

        # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    st.markdown("")

    def reset_filter():
        st.session_state["date_start"] = "Januari 2026"
        st.session_state["date_end"] = "Juli 2026"
        st.session_state["age_from"] = 0
        st.session_state["age_to"] = 59
        for key in list(st.session_state.keys()):
            if key.startswith(("sex_", "posyandu_")):
                st.session_state[key] = True

    st.button(
        "Reset Filter",
        use_container_width=True,
        on_click=reset_filter,
    )

    ilustrasi_b64 = get_base64_image(
        BASE_DIR / "assets" / "ilustrasi_sidebar.png"
    )

    if ilustrasi_b64:

        st.markdown(
            f"""
            <div style="margin-top: 28px; text-align: center;">
                <img
                    src="data:image/png;base64,{ilustrasi_b64}"
                    style="width: 66%; height: auto;"
                >
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.caption("Ilustrasi tidak ditemukan di folder assets.")

    st.markdown(
        """
        <div class="sidebar-footer-note">
            Data ditampilkan berdasarkan filter
            yang dipilih.
        </div>
        <style>
        .st-key-btn_panduan button {
            background: transparent !important;
            border: 1.5px solid rgba(255, 255, 255, 0.7) !important;
            border-radius: 14px !important;
            margin-top: 10px;
        }
        .st-key-btn_panduan button p {
            color: #FFFFFF !important;
            font-size: 15px !important;
        }
        .st-key-btn_panduan button:hover {
            background: rgba(255, 255, 255, 0.12) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Panduan Penggunaan", key="btn_panduan", use_container_width=True):
        tampilkan_panduan()


# ============================================================
# 18. APPLY FILTER
# ============================================================

filtered_df = df.copy()


if (
    date_col is not None
    and selected_date_range is not None
    and len(selected_date_range) == 2
):

    date_values = pd.to_datetime(
        filtered_df[date_col],
        errors="coerce",
    )

    start_date = pd.Timestamp(
        selected_date_range[0]
    )

    end_date = (
        pd.Timestamp(
            selected_date_range[1]
        )
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    filtered_df = filtered_df[
        date_values.between(
            start_date,
            end_date,
        )
    ]


if (
    sex_col is not None
    and selected_sex is not None
):

    if len(selected_sex) > 0:

        filtered_df = filtered_df[
            filtered_df[sex_col]
            .astype(str)
            .str.strip()
            .isin(selected_sex)
        ]

    else:

        filtered_df = filtered_df.iloc[0:0]


if (
    age_col is not None
    and selected_age is not None
    and tuple(selected_age) != (0, 59)
):

    age_values = pd.to_numeric(
        filtered_df[age_col],
        errors="coerce",
    )

    filtered_df = filtered_df[
               age_values.between(
            selected_age[0],
            selected_age[1] + 0.99,
        )
    ]


if (
    posyandu_col is not None
    and selected_posyandu is not None
):

    if len(selected_posyandu) > 0:

        filtered_df = filtered_df[
            filtered_df[posyandu_col]
            .astype(str)
            .str.strip()
            .isin(selected_posyandu)
        ]

    else:

        filtered_df = filtered_df.iloc[0:0]


# ============================================================
# 19. IDENTIFY DASHBOARD VARIABLES
# ============================================================

stunting_col = find_column(
    filtered_df,
    exact_names=[
        "stunting",
        "status stunting",
        "status_stunting",
    ],
    keywords=[
        "stunting",
        "tb/u",
        "tbu",
        "zs tb/u",
        "zs_tbu",
    ],
)


wasting_col = find_column(
    filtered_df,
    exact_names=[
        "wasting",
        "status wasting",
        "status_wasting",
    ],
    keywords=[
        "wasting",
        "bb/tb",
        "bbtb",
    ],
)


underweight_col = find_column(
    filtered_df,
    exact_names=[
        "underweight",
        "status underweight",
        "status_underweight",
    ],
    keywords=[
        "underweight",
        "bb/u",
        "bbu",
    ],
)

# pakai status hasil verifikasi pygrowup (sesuai Tabel 2.1) kalau ada
if "Status TB/U" in filtered_df.columns:
    stunting_col = "Status TB/U"
if "Status BB/U" in filtered_df.columns:
    underweight_col = "Status BB/U"
if "Status BB/TB" in filtered_df.columns:
    wasting_col = "Status BB/TB"


bblr_col = find_column(
    filtered_df,
    exact_names=[
        "bblr",
        "status bblr",
        "status_bblr",
        "bb lahir",
        "berat lahir",
        "berat badan lahir",
    ],
    keywords=[
        "bblr",
        "berat lahir",
        "bb lahir",
        "berat badan lahir",
    ],
)


mbg_col = find_column(
    filtered_df,
    exact_names=[
        "mbg",
        "status mbg",
        "keikutsertaan mbg",
    ],
    keywords=[
        "mbg",
    ],
)


rw_col = find_column(
    filtered_df,
    exact_names=[
        "rw",
        "wilayah",
        "no rw",
        "nomor rw",
    ],
    keywords=[
        "rw",
        "wilayah",
    ],
)


rt_col = find_column(
    filtered_df,
    exact_names=[
        "rt",
        "no rt",
        "nomor rt",
    ],
    keywords=[
        "rt",
    ],
)


# ============================================================
# 20. EMPTY FILTER RESULT CHECK
# ============================================================

if filtered_df.empty:

    render_alert(
        "Tidak terdapat data yang sesuai dengan filter.",
        "warning",
    )

    st.stop()


# ============================================================
# 21. KPI CALCULATION
# ============================================================

total_balita = len(
    filtered_df
)


stunting_count = (
    int(
        filtered_df[stunting_col]
        .apply(normalize_tbu_category)
        .isin(["Pendek (stunted)", "Sangat pendek (severely stunted)"])
        .sum()
    )
    if stunting_col is not None
    else 0
)

wasting_count = (
    int(
        filtered_df[wasting_col]
        .apply(normalize_bbtb_category)
        .isin(["Gizi kurang (wasted)", "Gizi buruk (severely wasted)"])
        .sum()
    )
    if wasting_col is not None
    else 0
)


underweight_count = (
    int(
        filtered_df[underweight_col]
        .apply(normalize_bbu_category)
        .isin(["Berat badan kurang (underweight)", "Berat badan sangat kurang (severely underweight)"])
        .sum()
    )
    if underweight_col is not None
    else 0
)


stunting_percentage = (
    stunting_count
    / total_balita
    * 100
    if total_balita > 0
    else 0
)


wasting_percentage = (
    wasting_count
    / total_balita
    * 100
    if total_balita > 0
    else 0
)


underweight_percentage = (
    underweight_count
    / total_balita
    * 100
    if total_balita > 0
    else 0
)

# ============================================================
# 24. TOP KPI
# ============================================================

kotak_kpi = st.container(key="kpi_baris")
kpi1, kpi2, kpi3, kpi4 = kotak_kpi.columns(4)

icon_balita_b64 = get_base64_image(BASE_DIR / "assets" / "kpi1.png")
icon_stunting_b64 = get_base64_image(BASE_DIR / "assets" / "kpi2.png")
icon_wasting_b64 = get_base64_image(BASE_DIR / "assets" / "kpi3.png")
icon_underweight_b64 = get_base64_image(BASE_DIR / "assets" / "kpi3.png")

with kpi1:

    st.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Jumlah Balita
        </div>
        <div class="kpi-header">
            <img class="kpi-icon" src="data:image/png;base64,{icon_balita_b64}">
            <div class="kpi-value">
                {total_balita:,}
            </div>
        </div>
        <div class="kpi-subtitle">
            Setelah filter
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with kpi2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">
                Balita Stunting
            </div>
            <div class="kpi-header">
                <img class="kpi-icon" src="data:image/png;base64,{icon_stunting_b64}">
                <div class="kpi-value">
                    {stunting_count:,}
                </div>
            </div>
            <div class="kpi-subtitle">
                {stunting_percentage:.1f}% dari data terfilter
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with kpi3:

 st.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Balita Wasting
        </div>
        <div class="kpi-header">
            <img class="kpi-icon" src="data:image/png;base64,{icon_wasting_b64}">
            <div class="kpi-value">
                {wasting_count:,}
            </div>
        </div>
        <div class="kpi-subtitle">
            {wasting_percentage:.1f}% dari data terfilter
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with kpi4:

   st.markdown(
    f"""
    <div class="kpi-card">
        <div class="kpi-title">
            Balita Underweight
        </div>
        <div class="kpi-header">
            <img class="kpi-icon" src="data:image/png;base64,{icon_underweight_b64}">
            <div class="kpi-value">
                {underweight_count:,}
            </div>
        </div>
        <div class="kpi-subtitle">
            {underweight_percentage:.1f}% dari data terfilter
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 25. STATUS GIZI (+ TREND) | RISIKO STUNTING
# ============================================================

left_col, right_col = st.columns([3, 1])
with left_col:

    with st.container(key="card_status"):

        st.markdown(
            '<div class="card-heading">Status Gizi</div>'
            '<div class="card-caption">Berdasarkan Indikator WHO (BB/U, TB/U, BB/TB)</div>',
            unsafe_allow_html=True,
        )

        d1, d2, d3 = st.columns(3)

        donut_specs = [
            (d1, "BB/U",  underweight_col, normalize_bbu_category),
            (d2, "TB/U",  stunting_col,    normalize_tbu_category),
            (d3, "BB/TB", wasting_col,     normalize_bbtb_category),
        ]

        for col, label, source_col, norm in donut_specs:
            with col:
                st.markdown(
                    f'<div class="chart-label">{label}</div>',
                    unsafe_allow_html=True,
                )
                fig = make_donut(filtered_df, source_col, normalize_func=norm)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                else:
                    render_alert(f"Variabel {label} belum ditemukan.", "info")

        st.markdown(
            '<div class="sub-heading">Perkembangan Status Gizi per Bulan (BB/U)</div>',
            unsafe_allow_html=True,
        )

        if date_col is not None and underweight_col is not None:

            # tren memakai SEMUA pengukuran (data_status_gizi.xlsx) dari balita yang lolos filter
            if gizi_df is not None and "NIK" in gizi_df.columns and "NIK" in filtered_df.columns:
                trend_df = gizi_df[gizi_df["NIK"].astype(str).isin(filtered_df["NIK"].astype(str))].copy()
            else:
                trend_df = filtered_df.copy()
            trend_col = "Status BB/U" if "Status BB/U" in trend_df.columns else underweight_col
            trend_df["_tanggal"] = pd.to_datetime(trend_df[date_col].apply(parse_tanggal_indonesia), errors="coerce")
            trend_df = trend_df.dropna(subset=["_tanggal"])
            if selected_date_range is not None:
                trend_df = trend_df[trend_df["_tanggal"].dt.date.between(*selected_date_range)]

            if not trend_df.empty:

                trend_df["_bulan"] = trend_df["_tanggal"].dt.to_period("M").astype(str)
                trend_df["_status"] = (
                    trend_df[trend_col]
                    .apply(normalize_bbu_category)
                    .replace(SHORT_LABELS)
                )

                trend = (
                    trend_df.groupby(["_bulan", "_status"], dropna=False)
                    .size().reset_index(name="Jumlah")
                )
                total_month = trend_df.groupby("_bulan").size().reset_index(name="Total")
                trend = trend.merge(total_month, on="_bulan", how="left")
                trend["Persentase"] = trend["Jumlah"] / trend["Total"] * 100
                trend = trend.sort_values("_bulan")

                BULAN_SHORT = [
                    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                    "Jul", "Agu", "Sep", "Okt", "Nov", "Des",
                ]

                trend["Bulan"] = trend["_bulan"].apply(
                    lambda s: f"{BULAN_SHORT[int(s[5:7]) - 1]} {s[:4]}"
                )

                bulan_order = trend.drop_duplicates("_bulan")["Bulan"].tolist()

                fig = px.bar(
                    trend,
                    x="Bulan",
                    y="Persentase",
                    color="_status",
                    barmode="group",
                    category_orders={
                        "Bulan": bulan_order,
                        "_status": [
                            "Normal", "Kurang", "Sangat kurang",
                            "Risiko lebih", "Tidak diketahui",
                        ],
                    },
                    color_discrete_map={
                        "Normal": "#183b91",
                        "Kurang": "#e0553b",
                        "Sangat kurang": "#d9a300",
                        "Risiko lebih": "#2fae7a",
                        "Tidak diketahui": "#b0b0b0",
                    },
                )

                fig.update_traces(
                    texttemplate="%{y:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{fullData.name}</b><br>%{x}<br>%{y:.1f}%<extra></extra>",
                )

                fig.update_layout(
                    height=340,
                    xaxis_title=None,
                    yaxis_title="Persentase (%)",
                    bargap=0.25,
                    bargroupgap=0.05,
                    margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(
                        title=None, orientation="h",
                        yanchor="bottom", y=1.02, xanchor="left", x=0,
                        font=dict(size=12, color="#202020"),
                    ),
                )
                fig.update_xaxes(type="category")
                fig.update_yaxes(range=[0, 115], ticksuffix="%")

                st.plotly_chart(apply_light_theme(fig), use_container_width=True, theme=None)

            else:
                render_alert("Data periode belum memiliki tanggal valid.", "info")
        else:
            render_alert(
                "Grafik belum dapat dibuat: kolom periode/status BB/U belum ditemukan.",
                "info",
            )

FORM_KEYS = ["pilih_balita", "cari_tabel", "hapus_pilih", "hal_tabel"] + kunci_form()


def reset_prediksi():

    st.session_state["pred_results"] = {}
    st.session_state.pop("last_result", None)
    for k in FORM_KEYS:
        st.session_state.pop(k, None)

    simpan_hasil()

with right_col:

    with st.container(key="card_risiko"):

        st.markdown(
            '<div class="card-heading">Risiko Stunting</div>'
            '<div class="card-caption">Hasil prediksi dari input kader</div>',
            unsafe_allow_html=True,
        )

        hasil_semua = {
            k: v
            for k, v in st.session_state.get("pred_results", {}).items()
            if k in filtered_df.index
        }

        n_total = len(hasil_semua)
        n_tinggi = sum(1 for v in hasil_semua.values() if v["tinggi"])
        n_rendah = n_total - n_tinggi

        if hasil_semua:

            labels, values, colors = [], [], []

            if n_tinggi:
                labels.append("Risiko Lebih Tinggi")
                values.append(n_tinggi)
                colors.append("#d9534f")

            if n_rendah:
                labels.append("Risiko Lebih Rendah")
                values.append(n_rendah)
                colors.append("#5cb85c")

            fig = go.Figure(
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    sort=False,
                    marker=dict(
                        colors=colors,
                        line=dict(color="#ffffff", width=2),
                    ),
                    textfont=dict(size=11, color="#ffffff"),
                )
            )

            fig.add_annotation(
                text=f"<b>{n_total}</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="#183b91"),
            )

            fig.update_layout(
                height=340,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="top", y=-0.02,
                    xanchor="center", x=0.5,
                    font=dict(size=11, color="#202020"),
                ),
                margin=dict(l=5, r=5, t=5, b=5),
            )

            st.plotly_chart(
                apply_light_theme(fig),
                use_container_width=True,
                theme=None,
            )

        else:

            st.markdown(
                '<div class="empty-chart">'
                '<div class="empty-ring"></div>'
                '<div class="empty-text">Belum ada hasil prediksi</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-num">{n_total}</div>
                    <div class="stat-lbl">Diprediksi</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num high">{n_tinggi}</div>
                    <div class="stat-lbl">Risiko tinggi</div>
                </div>
                <div class="stat-box">
                    <div class="stat-num low">{n_rendah}</div>
                    <div class="stat-lbl">Risiko rendah</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="padding-bottom: 26px;">
            <div class="howto">
                <div class="howto-title">Cara memakai</div>
                <div class="howto-step">
                    <span class="howto-no">1</span>
                    <span>Pilih nama balita di form Input Data Balita oleh Kader.</span>
                </div>
                <div class="howto-step">
                    <span class="howto-no">2</span>
                    <span>Isi faktor utama, lalu klik Prediksi Risiko Stunting.</span>
                </div>
                <div class="howto-step">
                    <span class="howto-no">3</span>
                    <span>Hasil muncul di diagram ini dan di Tabel Intervensi Pemantauan.</span>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "Reset hasil prediksi",
            key="btn_reset_pred",
            on_click=reset_prediksi,
            use_container_width=True,
        )
# ============================================================
# 28. KARAKTERISTIK BALITA
# ============================================================

with st.container(key="card_karakteristik"):

    st.markdown('<div class="card-heading">Karakteristik Balita</div>', unsafe_allow_html=True)

    char1, char2, char3, char4 = st.columns(4)

    with char1:
        st.markdown('<div class="chart-label">KELOMPOK USIA</div>', unsafe_allow_html=True)

        if age_col:
            fig = make_donut(
                filtered_df,
                age_col,
                normalize_func=lambda v: (
                    f"{bucket_age_puskesmas(v)} bln"
                    if bucket_age_puskesmas(v) != "Tidak diketahui"
                    else "Tidak diketahui"
                ),
                 urutan=["0-5 bln", "6-11 bln", "12-23 bln", "24-59 bln", "Tidak diketahui"],
                warna=["#e0b53b", "#e0553b", "#2fae7a", "#3b5fe0"],
            )
        else:
            fig = None

        if fig:
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            render_alert("Kolom usia belum ditemukan.", "info")

    for col, label, source_col, norm in [
        (char2, "JENIS KELAMIN",     sex_col,  None),
        (char3, "STATUS BBLR",       bblr_col, normalize_bblr_category),
        (char4, "KEIKUTSERTAAN MBG", mbg_col,  None),
    ]:
        with col:
            st.markdown(f'<div class="chart-label">{label}</div>', unsafe_allow_html=True)
            fig = make_donut(filtered_df, source_col, normalize_func=norm)
            if fig:
                st.plotly_chart(fig, use_container_width=True, theme=None)
            else:
                render_alert(f"Variabel {label.title()} belum ditemukan.", "info")
# ============================================================
# 30B. INPUT DATA BALITA OLEH KADER
# ============================================================

kader_name_col = find_column(
    filtered_df,
    exact_names=["nama balita", "nama", "name"],
    keywords=["nama"],
)

kader_weight_col = find_column(
    filtered_df,
    exact_names=["berat", "berat badan", "bb", "berat badan (kg)"],
)

def label_balita(idx):

    if idx is None:
        return "Ketik nama balita untuk mencari..."

    nama = (
        str(filtered_df.loc[idx, kader_name_col])
        if kader_name_col
        else f"Balita {idx}"
    )

    if posyandu_col:
        nama += f" - {filtered_df.loc[idx, posyandu_col]}"

    return nama

def isi_dari_data_balita():

    st.session_state.pop("last_result", None)

    idx = st.session_state.get("pilih_balita")

    if idx is None:
        isi_dari_baris({})
        return

    row = filtered_df.loc[idx]
    nilai = {}

    if sex_col is not None:
        nilai["jk_laki"] = 0 if str(row[sex_col]).strip().upper().startswith("P") else 1

    if age_col is not None:
        nilai["umur_bln"] = pd.to_numeric(row[age_col], errors="coerce")

    if kader_weight_col is not None:
        nilai["bb_anak"] = pd.to_numeric(row[kader_weight_col], errors="coerce")

    if bblr_col is not None:
        lahir = pd.to_numeric(row[bblr_col], errors="coerce")
        if pd.notna(lahir):
            nilai["bb_lahir_kg"] = lahir / 1000 if lahir > 25 else lahir

    isi_dari_baris(nilai)


if st.session_state.get("pilih_balita") not in ([None] + filtered_df.index.tolist()):
    st.session_state["pilih_balita"] = None


# fragment: mengisi form tidak memuat ulang seluruh dashboard
@st.fragment
def tampil_form_kader():

    with st.container(key="card_input"):

        st.markdown(
            '<div class="card-heading">Input Data Balita oleh Kader</div>',
            unsafe_allow_html=True,
        )

        if not model_available:

            render_alert(
                f"Model belum berhasil dimuat, input manual belum dapat diproses. ({model_error})",
                "warning",
            )

        else:

            pick_col, _ = st.columns([3, 2])

            if kader_name_col:
                sorted_idx = (
                    filtered_df[kader_name_col]
                    .astype(str)
                    .sort_values()
                    .index.tolist()
                )
            else:
                sorted_idx = filtered_df.index.tolist()

            with pick_col:
                st.selectbox(
                    "Cari data balita",
                    options=[None] + sorted_idx,
                    format_func=label_balita,
                    key="pilih_balita",
                    on_change=isi_dari_data_balita,
                    label_visibility="collapsed",
                )
            nilai_input, input_kosong = render_form(model_bundle, IMPORTANCE_PATH)

            predict_clicked = st.button(
                "Prediksi Risiko Stunting", key="btn_prediksi",
            )

            if predict_clicked:

                if input_kosong:

                    render_alert(
                        "Lengkapi dulu: " + ", ".join(input_kosong) + ".",
                        "warning",
                    )

                else:

                    prediksi_ok = False

                    # layar loading: tetap tampil sampai dashboard selesai dimuat ulang
                    loading = st.empty()
                    loading.markdown(
                        '<div class="loading-overlay"><div class="loading-box">'
                        '<div class="loading-spinner"></div>'
                        '<div>Memproses prediksi risiko stunting...</div>'
                        '</div></div>',
                        unsafe_allow_html=True,
                    )

                    try:

                        manual_probability = prediksi(model_bundle, nilai_input)

                        is_high = manual_probability >= float(model_bundle["threshold"])
                        selected_idx = st.session_state.get("pilih_balita")

                        st.session_state["last_result"] = {
                            "prob": manual_probability,
                            "tinggi": is_high,
                            "idx": selected_idx,
                            "faktor": faktor_perlu_diperhatikan(nilai_input),
                        }

                        if selected_idx is not None:
                            st.session_state.setdefault("pred_results", {})[selected_idx] = {
                                "prob": manual_probability,
                                "tinggi": is_high,
                                "waktu": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                                "isian": isian_kader(nilai_input),
                            }
                            simpan_hasil()

                        prediksi_ok = True

                    except Exception as e:

                        loading.empty()
                        render_alert("Prediksi gagal dijalankan.", "error")

                        with st.expander("Detail teknis"):
                            st.code(str(e))

                    if prediksi_ok:
                        st.rerun()
            hasil = st.session_state.get("last_result")

            if hasil:

                badge_html = (
                    '<span class="risk-high">Risiko Lebih Tinggi</span>'
                    if hasil["tinggi"]
                    else '<span class="risk-low">Risiko Lebih Rendah</span>'
                )

                r1, r2 = st.columns(2)

                with r1:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Probabilitas Stunting</div>'
                        f'<div class="result-value">{hasil["prob"]:.3f} ({hasil["prob"] * 100:.1f}%)</div></div>',
                        unsafe_allow_html=True,
                    )

                with r2:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">Hasil Prediksi</div>'
                        f'<div class="result-value">{badge_html}</div></div>',
                        unsafe_allow_html=True,
                    )

                if hasil.get("idx") is not None:
                    st.caption(
                        "Skor dan risiko balita ini juga muncul di diagram Risiko "
                        "Stunting dan Tabel Intervensi Pemantauan."
                    )

                faktor = hasil.get("faktor")
                if faktor:
                    isi = "".join(f"<li style='margin-bottom:4px'>{html_lib.escape(f)}</li>" for f in faktor)
                    st.markdown(
                        '<div class="result-card" style="text-align:left">'
                        '<div class="result-label">Faktor yang perlu diperhatikan pada balita ini</div>'
                        f'<ul style="margin:8px 0 6px 18px;padding:0;font-size:0.92rem">{isi}</ul>'
                        '<div style="font-size:0.8rem;color:#6b7280">Faktor di atas berasal dari variabel yang '
                        'dipakai model prediksi. Alasan setiap faktor berpengaruh terhadap stunting dapat dibaca '
                        'pada bagian "Kenapa faktor-faktor ini berpengaruh terhadap stunting?" di bawah.</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                elif faktor is not None:
                    st.caption(
                        "Dari isian kader, tidak ada faktor yang perlu diperhatikan secara khusus. "
                        "Tetap lanjutkan pemantauan rutin setiap bulan."
                    )

                st.caption(
                    "Hasil prediksi merupakan dukungan keputusan untuk prioritas "
                    "pemantauan, bukan diagnosis medis."
                )

                def prediksi_lain():
                    # kosongkan form untuk balita berikutnya; hasil yang tersimpan tidak dihapus
                    st.session_state["pilih_balita"] = None
                    st.session_state.pop("last_result", None)
                    kosongkan_form()

                st.button(
                    "Prediksi balita lain",
                    key="btn_lain",
                    on_click=prediksi_lain,
                    use_container_width=True,
                )

            # penjelasan untuk kader: kenapa tiap faktor berpengaruh terhadap stunting
            tampilkan_penjelasan(IMPORTANCE_PATH)


tampil_form_kader()
# ============================================================
# 31. TABEL INTERVENSI PEMANTAUAN
# ============================================================

pred_results = st.session_state.get("pred_results", {})


def fmt_num(v):

    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)

@st.dialog(" ", width="large")
def tampilkan_isian(idx):
    h = st.session_state.get("pred_results", {}).get(idx, {})
    tinggi = h.get("tinggi")
    kategori = "Risiko Lebih Tinggi" if tinggi else "Risiko Lebih Rendah"
    warna = "#C94A4A" if tinggi else "#3F7D46"
    isian = h.get("isian") or {}
    baris = "".join(
        f'<div class="is-row"><div class="is-lbl">{html_lib.escape(k)}</div>'
        f'<div class="is-val">{html_lib.escape(v)}</div></div>'
        for k, v in isian.items()
    ) or (
        '<div class="is-kosong">Isian form belum tersimpan karena balita ini diprediksi sebelum fitur ini ada. '
        'Lakukan prediksi ulang agar isiannya tersimpan.</div>'
    )
    st.markdown(
        f"""
        <style>
        div[data-testid="stDialog"] div[role="dialog"] {{ background: #FFFFFF !important; border-radius: 22px !important; }}
        div[data-testid="stDialog"] button[aria-label="Close"] svg {{ color: #1B2A7B !important; }}
        .is-title {{ color: #1B2A7B; font-size: 26px; font-weight: 700; margin: -46px 0 14px 0; }}
        .is-info {{ background: #EEF5FD; border-radius: 14px; padding: 12px 16px; margin-bottom: 14px;
                    color: #3D4A6B; font-size: 14.5px; line-height: 1.6; }}
        .is-info b {{ color: #1B2A7B; font-size: 16px; }}
        .is-list {{ max-height: 52vh; overflow-y: auto; border: 1px solid #E3E8F2; border-radius: 14px; }}
        .is-row {{ display: flex; gap: 16px; padding: 10px 16px; border-bottom: 1px solid #EEF1F6; }}
        .is-row:last-child {{ border-bottom: none; }}
        .is-lbl {{ flex: 1.4; color: #3D4A6B; font-size: 14px; }}
        .is-val {{ flex: 1; color: #1B2A7B; font-weight: 600; font-size: 14px; }}
        .is-kosong {{ padding: 16px; color: #3D4A6B; font-size: 14px; }}
        .st-key-btn_tutup_isian button {{ background: linear-gradient(90deg, #2B6FD6, #4FC3F0) !important;
                                         border: none !important; border-radius: 14px !important; }}
        .st-key-btn_tutup_isian button p {{ color: #FFFFFF !important; font-weight: 600 !important; }}
        </style>
        <div class="is-title">Isian Form Kader</div>
        <div class="is-info"><b>{html_lib.escape(label_balita(idx))}</b><br>
        Diprediksi {h.get("waktu", "-")} · Skor {h.get("prob", 0):.2f} ·
        <span style="color:{warna}; font-weight:700;">{kategori}</span></div>
        <div class="is-list">{baris}</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Tutup", use_container_width=True, key="btn_tutup_isian"):
        st.rerun()

# fragment: mencari nama di tabel tidak memuat ulang seluruh dashboard
@st.fragment
def tampil_tabel_intervensi():

    with st.container(key="card_tabel"):

        st.markdown(
            '<div class="card-heading">Tabel Intervensi Pemantauan</div>',
            unsafe_allow_html=True,
        )

        tbl_names = (
            sorted(
                filtered_df[kader_name_col]
                .dropna().astype(str).str.strip().unique().tolist(),
                key=str.upper,
            )
            if kader_name_col
            else []
        )
        def ke_halaman_1():
            st.session_state["hal_tabel"] = 1

        f1, f2, f3 = st.columns([2, 1.2, 1.2])
        with f1:
            search_text = st.selectbox(
                "Cari nama balita",
                options=tbl_names,
                index=None,
                placeholder="Ketik nama balita, lalu klik nama yang muncul...",
                key="cari_tabel",
                on_change=ke_halaman_1,
            )
        with f2:
            tampilkan = st.selectbox(
                "Tampilkan",
                ["Semua balita", "Sudah diprediksi", "Risiko lebih tinggi", "Belum diprediksi",
                 "Sudah stunting", "Belum stunting"],
                key="filter_tabel",
                on_change=ke_halaman_1,
            )
        with f3:
            urutan = st.selectbox(
                "Urutkan",
                ["Nama (A-Z)", "Skor prioritas tertinggi"],
                key="urut_tabel",
                on_change=ke_halaman_1,
            )

        if stunting_col or wasting_col or underweight_col:

            table_df = filtered_df.copy()

            table_df["_flag_stunting"] = (
                table_df[stunting_col]
                .apply(normalize_tbu_category)
                .isin(["Pendek (stunted)", "Sangat pendek (severely stunted)"])
                if stunting_col
                else False
            )

            table_df["_flag_wasting"] = (
                table_df[wasting_col]
                .apply(normalize_bbtb_category)
                .isin(["Gizi kurang (wasted)", "Gizi buruk (severely wasted)"])
                if wasting_col
                else False
            )

            table_df["_flag_underweight"] = (
                table_df[underweight_col]
                .apply(normalize_bbu_category)
                .isin([
                    "Berat badan kurang (underweight)",
                    "Berat badan sangat kurang (severely underweight)",
                ])
                if underweight_col
                else False
            )

            table_df["_urut"] = (
                table_df["_flag_stunting"].astype(int)
                + table_df["_flag_wasting"].astype(int)
                + table_df["_flag_underweight"].astype(int)
            )

            tbl_name_col = kader_name_col

            if tbl_name_col:
                table_df = table_df.sort_values(
                    tbl_name_col,
                    key=lambda s: s.astype(str).str.upper(),
                )

            if search_text and tbl_name_col:
                table_df = table_df[
                    table_df[tbl_name_col].astype(str).str.strip() == search_text
                ]

            # filter dan urutan tabel
            hasil_pred = st.session_state.get("pred_results", {})
            sudah = table_df.index.isin(list(hasil_pred))
            if tampilkan == "Sudah diprediksi":
                table_df = table_df[sudah]
            elif tampilkan == "Belum diprediksi":
                table_df = table_df[~sudah]
            elif tampilkan == "Risiko lebih tinggi":
                tinggi = pd.Series([bool(hasil_pred.get(i, {}).get("tinggi")) for i in table_df.index],
                                   index=table_df.index, dtype=bool)
                table_df = table_df[tinggi]
            elif tampilkan == "Sudah stunting":
                table_df = table_df[table_df["_flag_stunting"].astype(bool)]
            elif tampilkan == "Belum stunting":
                table_df = table_df[~table_df["_flag_stunting"].astype(bool)]

            if urutan == "Skor prioritas tertinggi":
                skor = [hasil_pred.get(i, {}).get("prob", -1.0) for i in table_df.index]
                table_df = table_df.assign(_skor=skor).sort_values("_skor", ascending=False, kind="stable")

            # 10 balita per halaman
            PER_HAL = 10
            n_hal = max(1, -(-len(table_df) // PER_HAL))
            hal = min(max(int(st.session_state.get("hal_tabel", 1)), 1), n_hal)
            st.session_state["hal_tabel"] = hal
            jumlah_balita = len(table_df)
            table_df = table_df.iloc[(hal - 1) * PER_HAL: hal * PER_HAL]

            if table_df.empty:

                render_alert("Tidak ada balita yang cocok dengan pencarian.", "info")

            else:

                headers = [
                    "NAMA BALITA", "USIA (BULAN)", "STATUS GIZI", "STUNTING",
                    "BBLR", "SKOR PRIORITAS", "RISIKO STUNTING", "WILAYAH",
                ]

                rows_html = ""

                for idx, r in table_df.iterrows():

                    nama = str(r[tbl_name_col]) if tbl_name_col else f"Balita {idx}"

                    usia = "-"
                    if age_col:
                        m = pd.to_numeric(r[age_col], errors="coerce")
                        if pd.notna(m):
                            usia = str(int(round(m)))

                    status_gizi = "-"
                    if underweight_col:
                        kategori = normalize_bbu_category(r[underweight_col])
                        status_gizi = SHORT_LABELS.get(kategori, kategori)

                    stunting_txt = (
                        ("Ya" if r["_flag_stunting"] else "Tidak")
                        if stunting_col
                        else "-"
                    )

                    bblr_txt = "-"
                    if bblr_col:
                        cat = normalize_bblr_category(r[bblr_col])
                        if cat.startswith("BBLR"):
                            bblr_txt = "Ya"
                        elif cat != "Tidak diketahui":
                            bblr_txt = "Tidak"

                    hasil = pred_results.get(idx)

                    if hasil:
                        skor_txt = f"{hasil['prob']:.2f}"
                        risiko_html = (
                            '<span class="badge-high">Risiko Lebih Tinggi</span>'
                            if hasil["tinggi"]
                            else '<span class="badge-low">Risiko Lebih Rendah</span>'
                        )
                        row_class = ' class="row-high"' if hasil["tinggi"] else ""
                    else:
                        skor_txt = ""
                        risiko_html = ""
                        row_class = ""

                    wilayah = (
                        f"{fmt_num(r[rt_col])}/{fmt_num(r[rw_col])}"
                        if rt_col and rw_col
                        else "-"
                    )

                    cells = [
                        html_lib.escape(nama),
                        usia,
                        html_lib.escape(status_gizi),
                        stunting_txt,
                        bblr_txt,
                        skor_txt,
                        risiko_html,
                        html_lib.escape(wilayah),
                    ]

                    rows_html += (
                        f"<tr{row_class}>"
                        + "".join(f"<td>{c}</td>" for c in cells)
                        + "</tr>"
                    )

                table_html = (
                    '<div style="padding-bottom: 28px;">'
                    '<div class="tbl-wrap"><table class="tbl"><thead><tr>'
                    + "".join(f"<th>{h}</th>" for h in headers)
                    + "</tr></thead><tbody>"
                    + rows_html
                    + "</tbody></table></div></div>"
                )

                st.markdown(table_html, unsafe_allow_html=True)

                def geser_halaman(d):
                    st.session_state["hal_tabel"] = st.session_state.get("hal_tabel", 1) + d

                p1, p2, p3 = st.columns([1, 2, 1])
                with p1:
                    st.button("‹ Sebelumnya", key="hal_prev", on_click=geser_halaman, args=(-1,),
                              disabled=hal <= 1, use_container_width=True)
                with p2:
                    st.markdown(
                        f'<div class="hal-info">Halaman {hal} dari {n_hal} · {jumlah_balita} balita</div>',
                        unsafe_allow_html=True,
                    )
                with p3:
                    st.button("Berikutnya ›", key="hal_next", on_click=geser_halaman, args=(1,),
                              disabled=hal >= n_hal, use_container_width=True)

            hasil_ada = st.session_state.get("pred_results", {})
            if hasil_ada:
                with st.expander("Lihat isian form kader"):
                    l1, l2 = st.columns([3, 1])
                    with l1:
                        pilih_lihat = st.selectbox(
                            "Pilih balita", options=[i for i in hasil_ada if i in filtered_df.index],
                            format_func=label_balita,
                            index=None, placeholder="Pilih nama balita...", key="lihat_pilih",
                            label_visibility="collapsed",
                        )
                    with l2:
                        if st.button("Lihat", key="btn_lihat_isian", disabled=pilih_lihat is None,
                                     use_container_width=True):
                            tampilkan_isian(pilih_lihat)

            if hasil_ada:

                def hapus_satu():
                    idx_h = st.session_state.get("hapus_pilih")
                    st.session_state.get("pred_results", {}).pop(idx_h, None)
                    if (st.session_state.get("last_result") or {}).get("idx") == idx_h:
                        st.session_state.pop("last_result", None)
                    st.session_state["hapus_pilih"] = None
                    simpan_hasil()

                with st.expander(f"Hapus hasil prediksi balita tertentu ({len(hasil_ada)} sudah diprediksi)"):
                    h1, h2 = st.columns([3, 1])
                    with h1:
                        st.selectbox(
                            "Pilih balita", options=[i for i in hasil_ada if i in filtered_df.index],
                            format_func=label_balita,
                            index=None, placeholder="Pilih nama balita...", key="hapus_pilih",
                            label_visibility="collapsed",
                        )
                    with h2:
                        if st.button("Hapus", key="btn_hapus", on_click=hapus_satu,
                                     disabled=st.session_state.get("hapus_pilih") is None,
                                     use_container_width=True):
                            st.rerun()

        else:

            render_alert(
                "Tabel belum dapat dibuat karena kolom status gizi "
                "(Stunting/Wasting/Underweight) belum ditemukan pada dataset.",
                "info",
            )


tampil_tabel_intervensi()
# ============================================================
# 33. PERSEBARAN WILAYAH
# ============================================================

def make_sorted_bar(data, column, label, height=320):

    if column is None or data.empty:
        return None

    wilayah = data[column].fillna("Tidak diketahui").astype(str).str.strip()

    # stunting = TB/U pendek atau sangat pendek (Tabel 2.1)
    if stunting_col is not None:
        is_stunting = (
            data[stunting_col]
            .apply(normalize_tbu_category)
            .isin(["Pendek (stunted)", "Sangat pendek (severely stunted)"])
        )
    else:
        is_stunting = pd.Series(False, index=data.index)

    counts = (
        pd.DataFrame({label: wilayah, "stunting": is_stunting})
        .groupby(label)["stunting"]
        .agg(Jumlah="size", Stunting="sum")
        .reset_index()
    )
    counts["Tidak stunting"] = counts["Jumlah"] - counts["Stunting"]

    counts["_n"] = pd.to_numeric(counts[label], errors="coerce")
    counts = counts.sort_values(["_n", label], na_position="last")
    urutan = counts[label].tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=urutan, y=counts["Tidak stunting"], name="Tidak stunting",
        marker_color="#183b91",
        hovertemplate=f"{label} %{{x}}<br>Tidak stunting: %{{y}}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=urutan, y=counts["Stunting"], name="Stunting",
        marker_color="#d9534f",
        text=[str(int(v)) if v > 0 else "" for v in counts["Stunting"]],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#ffffff", size=11),
        hovertemplate=f"{label} %{{x}}<br>Stunting: %{{y}}<extra></extra>",
    ))

    # total balita di atas batang
    fig.add_trace(go.Scatter(
        x=urutan, y=counts["Jumlah"], mode="text",
        text=counts["Jumlah"].astype(int).astype(str),
        textposition="top center", showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        barmode="stack",
        height=height,
        xaxis_title=label,
        yaxis_title="Jumlah Balita",
        bargap=0.3,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11, color="#202020"),
        ),
    )
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=urutan)
    fig.update_yaxes(range=[0, counts["Jumlah"].max() * 1.2])

    return apply_light_theme(fig)


with st.container(key="card_wilayah"):

    st.markdown(
        '<div class="card-heading">Persebaran Wilayah</div>',
        unsafe_allow_html=True,
    )

    area1, area2 = st.columns(2)

    with area1:
        st.markdown(
            '<div class="chart-label">JUMLAH BALITA DAN STUNTING PER RW</div>',
            unsafe_allow_html=True,
        )
        fig = make_sorted_bar(filtered_df, rw_col, "RW")
        if fig:
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            render_alert("Kolom RW/wilayah belum ditemukan.", "info")

    with area2:
        st.markdown(
            '<div class="chart-label">JUMLAH BALITA DAN STUNTING PER RT</div>',
            unsafe_allow_html=True,
        )
        fig = make_sorted_bar(filtered_df, rt_col, "RT")
        if fig:
            st.plotly_chart(fig, use_container_width=True, theme=None)
        else:
            render_alert("Kolom RT belum ditemukan.", "info")


# ============================================================
# 37. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        &copy; 2026 Posyandu Kelurahan Tambakrejo
        <br>
        Dashboard Monitor Gizi &amp; Stunting
    </div>
    """,
    unsafe_allow_html=True,
)