"""
Pentecost University — AI Academic Performance Tracker
app.py  v6.0  |  Non-Technical Edition
Authors : Steven Asante-Poku Jnr & Frank Amoah  |  2025
Supervisor : Mr Harry Attieku-Boateng

Design principles:
  - Zero manual data entry for predictions (CSV-driven)
  - Upload CSV → predictions run automatically
  - Search within uploaded data by name or ID
  - Plain English language throughout
  - Role-based views with faculty-gated access
  - One-page scrollable layout — no complex tabs
"""

import streamlit as st
import pickle, json, os, io, datetime
import numpy  as np
import pandas as pd
import plotly.graph_objects as go

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "PU Academic Tracker",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
FACULTIES = ["FESAC", "FBA", "FEHAS", "PSTM"]
SEMESTERS = ["2019_S1","2019_S2","2020_S1","2020_S2",
             "2021_S1","2021_S2","2022_S1","2022_S2"]

RISK_LABEL = {0:"Low Risk",    1:"Medium Risk",  2:"High Risk"}
RISK_ICON  = {0:"🟢",           1:"🟡",            2:"🔴"}
RISK_COLOR = {0:"#22c55e",     1:"#f59e0b",      2:"#ef4444"}
RISK_BG    = {0:"#102a1c",     1:"#2e2208",      2:"#2d0e0e"}
RISK_BORDER= {0:"#22c55e",     1:"#f59e0b",      2:"#ef4444"}

RISK_MEANING = {
    0: "This student is performing well and is not currently at risk of academic difficulty.",
    1: "This student shows some signs of academic difficulty. A check-in is recommended.",
    2: "This student is at significant risk of failing next semester. Immediate action is needed.",
}

FACULTY_FULL = {
    "FESAC": "Faculty of Engineering & Applied Sciences",
    "FBA"  : "Faculty of Business Administration",
    "FEHAS": "Faculty of Education, Humanities & Applied Sciences",
    "PSTM" : "Pentecost School of Theology & Ministry",
}
FAC_COLOR = {
    "FESAC":"#60a5fa","FBA":"#f59e0b","FEHAS":"#22c55e","PSTM":"#a78bfa"
}

FEATURE_COLS = [
    "avg_attendance","avg_total_mark","avg_ca_score","avg_exam_score",
    "total_credits","num_courses","gender_enc","semester_index",
    "prev_gpa","gpa_trend","consec_fails","trend_x_fail",
    "fac_FESAC","fac_FBA","fac_FEHAS","fac_PSTM",
]

GRAD_CLASSES = [
    (3.60, 4.00, "First Class",        "#FFD700", "🥇"),
    (3.00, 3.59, "Second Class Upper", "#C0C0C0", "🥈"),
    (2.00, 2.99, "Second Class Lower", "#CD7F32", "🥉"),
    (1.50, 1.99, "Third Class",        "#f59e0b", "📜"),
    (1.00, 1.49, "Pass",               "#94a3b8", "📋"),
    (0.00, 0.99, "Fail",               "#ef4444", "❌"),
]


# ══════════════════════════════════════════════════════════════════════════════
# ROLES
# ══════════════════════════════════════════════════════════════════════════════
ROLES = {
    "Academic Advisor — FESAC" : {"faculty":"FESAC","icon":"👨‍🏫","pwd_key":"ADVISOR_FESAC_PASSWORD","default":"advisor_fesac_2025"},
    "Academic Advisor — FBA"   : {"faculty":"FBA",  "icon":"👨‍🏫","pwd_key":"ADVISOR_FBA_PASSWORD",  "default":"advisor_fba_2025"},
    "Academic Advisor — FEHAS" : {"faculty":"FEHAS","icon":"👨‍🏫","pwd_key":"ADVISOR_FEHAS_PASSWORD","default":"advisor_fehas_2025"},
    "Academic Advisor — PSTM"  : {"faculty":"PSTM", "icon":"👨‍🏫","pwd_key":"ADVISOR_PSTM_PASSWORD", "default":"advisor_pstm_2025"},
    "Head of Department — FESAC":{"faculty":"FESAC","icon":"🏛️", "pwd_key":"HOD_FESAC_PASSWORD",    "default":"hod_fesac_2025"},
    "Head of Department — FBA"  :{"faculty":"FBA",  "icon":"🏛️", "pwd_key":"HOD_FBA_PASSWORD",      "default":"hod_fba_2025"},
    "Head of Department — FEHAS":{"faculty":"FEHAS","icon":"🏛️", "pwd_key":"HOD_FEHAS_PASSWORD",    "default":"hod_fehas_2025"},
    "Head of Department — PSTM" :{"faculty":"PSTM", "icon":"🏛️", "pwd_key":"HOD_PSTM_PASSWORD",     "default":"hod_pstm_2025"},
    "Dean of Students"          :{"faculty":None,   "icon":"🎓", "pwd_key":"DEAN_PASSWORD",          "default":"dean_2025"},
}


# ══════════════════════════════════════════════════════════════════════════════
# CSS — clean, simple, friendly
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
    --accent1:#4f7cff; --accent2:#8b5cf6;
    --grad-accent:linear-gradient(135deg,#4f7cff 0%,#8b5cf6 100%);
    --grad-green:linear-gradient(135deg,#22c55e 0%,#16a34a 100%);
    --grad-amber:linear-gradient(135deg,#f59e0b 0%,#d97706 100%);
    --grad-red:linear-gradient(135deg,#ef4444 0%,#dc2626 100%);
    --surface:#131a26; --surface2:#1a2332; --border:#232b3d;
    --txt:#f1f5f9; --txt2:#94a3b8; --txt3:#64748b;
}

html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background:#080b12;}
.stApp{background:radial-gradient(ellipse 1200px 600px at 50% -10%,
        rgba(79,124,255,.10), transparent 60%), #080b12;}
.block-container{padding:1.3rem 1.8rem 3rem !important;}
* { -webkit-tap-highlight-color: transparent; }

@keyframes fadeUp{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.block-container > div { animation: fadeUp .35s ease-out; }

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0d1320 0%,#0a0e17 100%) !important;
    border-right:1px solid var(--border) !important;}
[data-testid="stSidebar"] *{color:var(--txt2) !important;}
[data-testid="stSidebar"] hr{border-color:var(--border) !important;}

/* ── Top banner ── */
.top-banner{background:linear-gradient(120deg,#1b2a52 0%,#3349a8 55%,#5b3fae 100%);
    border-radius:18px;padding:1.4rem 1.8rem;
    display:flex;align-items:center;gap:1.2rem;
    margin-bottom:1.5rem;position:relative;overflow:hidden;
    box-shadow:0 12px 32px -12px rgba(79,124,255,.45);}
.top-banner::after{content:'';position:absolute;inset:0;
    background:radial-gradient(circle at 85% 20%, rgba(255,255,255,.16), transparent 45%);}
.top-banner-icon{width:52px;height:52px;border-radius:14px;
    background:rgba(255,255,255,.14);backdrop-filter:blur(6px);
    display:flex;align-items:center;justify-content:center;
    font-size:1.7rem;flex-shrink:0;border:1px solid rgba(255,255,255,.2);
    position:relative;z-index:1;}
.top-banner h1{font-size:1.35rem;font-weight:800;
    color:white !important;margin:0;letter-spacing:-.01em;
    position:relative;z-index:1;}
.top-banner .sub{color:#cdd9ff;font-size:.8rem;margin-top:.25rem;
    position:relative;z-index:1;display:flex;gap:.4rem;flex-wrap:wrap;
    align-items:center;}
.top-banner .sub .pill{background:rgba(255,255,255,.14);
    border-radius:20px;padding:.12rem .65rem;font-size:.74rem;
    border:1px solid rgba(255,255,255,.18);}

/* ── Step cards ── */
.step-card{background:linear-gradient(160deg,var(--surface) 0%,#0f1626 100%);
    border:1px solid var(--border);border-radius:16px;
    padding:1.6rem 1.3rem;text-align:center;
    transition:transform .2s, box-shadow .2s, border-color .2s;}
.step-card:hover{transform:translateY(-4px);
    border-color:#4f7cff66;
    box-shadow:0 16px 32px -16px rgba(79,124,255,.4);}
.step-icon{width:52px;height:52px;border-radius:50%;
    background:var(--grad-accent);
    color:white;font-size:1.4rem;font-weight:800;
    display:flex;align-items:center;justify-content:center;
    margin:0 auto .9rem;box-shadow:0 8px 20px -6px rgba(79,124,255,.55);}
.step-title{color:var(--txt);font-weight:700;font-size:.98rem;margin-bottom:.4rem;
    letter-spacing:-.01em;}
.step-desc{color:var(--txt2);font-size:.83rem;line-height:1.55;}

/* ── Summary cards ── */
.sum-card{background:linear-gradient(160deg,var(--surface) 0%,#0f1626 100%);
    border:1px solid var(--border);
    border-radius:16px;padding:1.3rem 1.4rem;text-align:left;
    position:relative;overflow:hidden;
    transition:transform .2s, box-shadow .2s;}
.sum-card:hover{transform:translateY(-3px);
    box-shadow:0 16px 36px -18px var(--cglow);}
.sum-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:var(--c);}
.sum-icon{font-size:1.3rem;margin-bottom:.5rem;opacity:.9;}
.sum-val{font-size:2.3rem;font-weight:800;color:var(--c);line-height:1;
    letter-spacing:-.02em;}
.sum-lbl{font-size:.78rem;color:var(--txt2);text-transform:uppercase;
    letter-spacing:.08em;margin-top:.4rem;font-weight:600;}
.sum-sub{font-size:.78rem;color:var(--txt3);margin-top:.25rem;}

/* ── Search box ── */
.stTextInput input{background:var(--surface) !important;
    color:var(--txt) !important;border:1px solid var(--border) !important;
    border-radius:12px !important;font-size:.95rem !important;
    padding:.65rem 1.1rem !important;transition:border-color .15s;}
.stTextInput input:focus{border-color:#4f7cff !important;
    box-shadow:0 0 0 3px rgba(79,124,255,.15) !important;}
.stTextInput label{display:none !important;}

/* ── Detail panel — stat tile grid ── */
.stat-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:.6rem;margin:.6rem 0;}
@media (max-width:640px){.stat-grid{grid-template-columns:1fr}}
.stat-tile{background:linear-gradient(160deg,#0e1421 0%,#0a0e17 100%);
    border:1px solid var(--border);border-radius:12px;
    padding:.7rem .9rem;}
.stat-tile .stat-lbl{font-size:.72rem;color:var(--txt2);
    text-transform:uppercase;letter-spacing:.06em;
    display:flex;align-items:center;gap:.35rem;margin-bottom:.25rem;}
.stat-tile .stat-val{font-size:1.05rem;font-weight:700;color:var(--txt);}

/* ── Rec item ── */
.rec-item{background:linear-gradient(160deg,var(--surface) 0%,#0f1626 100%);
    border:1px solid var(--border);
    border-left:3px solid var(--rc);
    border-radius:0 12px 12px 0;padding:.75rem 1rem;margin:.4rem 0;
    font-size:.86rem;color:#cbd5e1;line-height:1.55;
    display:flex;gap:.7rem;align-items:flex-start;}
.rec-icon{width:26px;height:26px;border-radius:8px;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    font-size:.95rem;background:var(--rbg2);margin-top:.05rem;}

/* ── Risk meaning box ── */
.risk-meaning{border-radius:14px;padding:1rem 1.2rem;margin:.6rem 0;
    font-size:.88rem;color:#e2e8f0;line-height:1.65;
    border:1px solid var(--rc);
    background:linear-gradient(135deg,var(--rbg) 0%,#0a0e17 100%);
    display:flex;gap:.8rem;align-items:flex-start;}
.risk-meaning .ricon{font-size:1.8rem;flex-shrink:0;}

/* ── Upload area ── */
[data-testid="stFileUploaderDropzone"]{
    background:linear-gradient(160deg,var(--surface) 0%,#0f1626 100%) !important;
    border-radius:14px !important;
    border:2px dashed var(--border) !important;
    transition:border-color .2s;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#4f7cff88 !important;}

/* ── Buttons ── */
.stButton>button{background:var(--grad-accent) !important;color:white !important;
    border:none !important;border-radius:10px !important;
    font-weight:700 !important;padding:.5rem 1.3rem !important;
    box-shadow:0 6px 18px -8px rgba(79,124,255,.55) !important;
    transition:all .18s !important;letter-spacing:-.005em;}
.stButton>button:hover{transform:translateY(-2px) !important;
    box-shadow:0 10px 24px -8px rgba(79,124,255,.7) !important;
    filter:brightness(1.08);}
.stButton>button:active{transform:translateY(0) !important;}

/* ── Download button (secondary style) ── */
.stDownloadButton>button{
    background:var(--surface2) !important;color:var(--txt) !important;
    border:1px solid var(--border) !important;border-radius:10px !important;
    font-weight:600 !important;padding:.5rem 1.3rem !important;
    transition:all .18s !important;}
.stDownloadButton>button:hover{border-color:#4f7cff99 !important;
    transform:translateY(-2px) !important;
    box-shadow:0 8px 20px -10px rgba(79,124,255,.5) !important;}

/* ── Graduation card ── */
.grad-card{border-radius:16px;padding:1.2rem 1.4rem;margin:.5rem 0;
    border:1px solid var(--rc);
    background:linear-gradient(135deg,var(--rbg) 0%,#0a0e17 70%);
    position:relative;overflow:hidden;}
.grad-card::after{content:'';position:absolute;top:-30%;right:-15%;
    width:140px;height:140px;border-radius:50%;
    background:radial-gradient(circle, var(--rc)33, transparent 70%);}
.grad-emoji{font-size:2.6rem;line-height:1;filter:drop-shadow(0 4px 10px rgba(0,0,0,.4));}
.grad-upgrade{background:rgba(255,255,255,.04);border-radius:10px;
    padding:.6rem .85rem;margin-top:.6rem;font-size:.81rem;
    color:var(--txt2);border:1px solid rgba(255,255,255,.06);}

/* ── Progress bar ── */
.prog-bg{background:#1c2436;border-radius:8px;height:9px;overflow:hidden;margin:.35rem 0;
    box-shadow:inset 0 1px 2px rgba(0,0,0,.3);}
.prog-fill{height:100%;border-radius:8px;
    box-shadow:0 0 8px -1px var(--glow);
    transition:width .5s ease;}

/* ── Expander ── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary{
    background:linear-gradient(160deg,var(--surface) 0%,#0f1626 100%) !important;
    border:1px solid var(--border) !important;border-radius:12px !important;
    color:#cbd5e1 !important;font-size:.92rem !important;font-weight:500 !important;
    transition:border-color .15s, transform .15s !important;}
[data-testid="stExpander"]:hover summary{border-color:#4f7cff55 !important;}
.streamlit-expanderContent, [data-testid="stExpander"] [data-testid="stExpanderDetails"]{
    background:#0d1320 !important;
    border:1px solid var(--border) !important;
    border-top:none !important;
    border-radius:0 0 12px 12px !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#080b12;}
::-webkit-scrollbar-thumb{background:#2a3344;border-radius:3px;}
::-webkit-scrollbar-thumb:hover{background:#4f7cff;}

/* ── Selectbox / multiselect ── */
.stSelectbox>div>div{background:var(--surface) !important;
    border-color:var(--border) !important;border-radius:10px !important;
    color:#cbd5e1 !important;}

/* ── Headings spacing ── */
h1,h2,h3{letter-spacing:-.01em;}

/* ── Alert/info boxes from st.info / st.success / st.warning / st.error ── */
[data-testid="stAlert"]{border-radius:12px !important;
    border:1px solid var(--border) !important;
    background:var(--surface) !important;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _secret(key, fallback=""):
    try:
        v = st.secrets.get(key, "")
        if v: return v
    except Exception:
        pass
    return os.environ.get(key, fallback)


def _rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert #RRGGBB to an rgba(...) string for glow/tint effects."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_grad_class(cgpa: float):
    for low, high, label, color, emoji in GRAD_CLASSES:
        if low <= round(cgpa, 3) <= high:
            return label, color, emoji
    return "Fail", "#ef4444", "❌"


def project_graduation(cum_gpa, gpa_trend, level_num, comp_cr, prog_cr):
    remaining = 2 if level_num == 300 else 1
    rem_cr    = max(0, prog_cr - comp_cr)
    proj_sem  = min(4.0, max(0.0, cum_gpa + gpa_trend * 0.5 * remaining))
    if prog_cr > 0:
        proj = (comp_cr * cum_gpa + rem_cr * proj_sem) / prog_cr
    else:
        proj = cum_gpa
    proj  = round(min(4.0, max(0.0, proj)), 3)
    label, color, emoji = get_grad_class(proj)
    next_cls = []
    for low, high, cls, clr, emj in GRAD_CLASSES:
        if low > proj and rem_cr > 0:
            needed = ((low * prog_cr) - (comp_cr * cum_gpa)) / rem_cr
            needed = round(needed, 2)
            if 0.0 <= needed <= 4.0:
                next_cls.append({"class":cls,"color":clr,"emoji":emj,
                                  "needed":needed,"target":low})
    return {"proj_cgpa":proj,"label":label,"color":color,"emoji":emoji,
            "remaining":remaining,"next":next_cls[:2]}


def project_trajectory(current_gpa: float, gpa_trend: float, n_steps: int = 4) -> list:
    """
    Project a student's GPA forward n_steps semesters assuming the current
    GPA trend continues at half its observed rate each semester (the same
    dampening assumption used in project_graduation). Each point is tagged
    with its risk zone (0=Low, 1=Medium, 2=High) using the fixed Q33/Q66
    thresholds.
    """
    delta  = gpa_trend * 0.5
    points = []
    gpa    = current_gpa
    for step in range(n_steps + 1):
        if step > 0:
            gpa = min(4.0, max(0.0, gpa + delta))
        zone = 2 if gpa < Q33 else (1 if gpa < Q66 else 0)
        points.append({"step": step, "gpa": round(gpa, 3), "zone": zone})
    return points


def trajectory_insight(points: list, name: str = "This student") -> tuple:
    """
    Returns (icon, message) describing the trajectory in plain English —
    whether the student is projected to cross into a different risk zone.
    """
    zone_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    current_zone = points[0]["zone"]

    for p in points[1:]:
        if p["zone"] != current_zone:
            if p["zone"] < current_zone:
                return ("📈", f"{name} is projected to improve from "
                        f"{zone_label[current_zone]} into "
                        f"{zone_label[p['zone']]} within {p['step']} "
                        f"semester(s) if the current trend continues.")
            else:
                return ("📉", f"{name} is projected to decline from "
                        f"{zone_label[current_zone]} into "
                        f"{zone_label[p['zone']]} within {p['step']} "
                        f"semester(s) if nothing changes. Early "
                        f"intervention now could change this trajectory.")

    return ("➡️", f"{name} is projected to remain in the "
            f"{zone_label[current_zone]} zone over the next "
            f"{points[-1]['step']} semesters if current patterns continue.")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artefacts():
    try:
        with open("best_model.pkl",    "rb") as f: mdl = pickle.load(f)
        with open("scaler.pkl",        "rb") as f: scl = pickle.load(f)
        with open("feature_cols.json", "r")  as f: fc  = json.load(f)
        with open("thresholds.json",   "r")  as f: thr = json.load(f)
        return mdl, scl, fc, thr, True
    except FileNotFoundError:
        return None, None, None, None, False

model, scaler, _fcols, thresholds, artefacts_ok = load_artefacts()
if _fcols: FEATURE_COLS = _fcols

Q33  = thresholds.get("Q33",        2.0)   if thresholds else 2.0
Q66  = thresholds.get("Q66",        3.0)   if thresholds else 3.0
MF1  = thresholds.get("macro_f1", 0.6383) if thresholds else 0.6383

# ── Model transparency constants (from training notebook results) ─────────
ECE_SCORE = 0.0957   # Expected Calibration Error
BV_GAP    = 0.329    # Bias-Variance gap (SMOTETomek artefact, documented in Ch4)

# Top SHAP features — (feature_key, plain-English name, relative importance 0-1)
SHAP_IMPORTANCE = [
    ("avg_total_mark", "Overall Average Mark",            0.382),
    ("avg_exam_score", "Exam Performance",                0.334),
    ("gpa_trend",      "GPA Trend (improving/declining)", 0.291),
    ("avg_ca_score",   "Continuous Assessment Score",     0.238),
    ("consec_fails",   "Consecutive Low-GPA Semesters",   0.191),
    ("trend_x_fail",   "Declining + Failing Pattern",     0.153),
]

# Intersectional fairness audit — (group label, colour, macro F1)
FAIRNESS_AUDIT = [
    ("Female", "#a78bfa", 0.660),
    ("Male",   "#60a5fa", 0.613),
    ("FESAC",  "#60a5fa", 0.869),
    ("FBA",    "#f59e0b", 0.801),
    ("FEHAS",  "#22c55e", 0.582),
    ("PSTM",   "#a78bfa", 0.666),
]
FAIRNESS_THRESHOLD = 0.45


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_batch_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as the training notebook Cell 6,
    then run the LightGBM model on every student-semester row.
    """
    df = df_raw.copy().sort_values(["student_id", "semester"])
    df["prev_gpa"]     = df.groupby("student_id")["semester_gpa"].shift(1)
    df["gpa_trend"]    = df["semester_gpa"] - df["prev_gpa"]
    df["is_fail"]      = (df["semester_gpa"] < 1.5).astype(int)
    df["consec_fails"] = df.groupby("student_id")["is_fail"].transform(
        lambda x: x.rolling(window=2, min_periods=1).sum())
    df["trend_x_fail"] = df["gpa_trend"] * df["consec_fails"]
    for fac in FACULTIES:
        df[f"fac_{fac}"] = (df["faculty"] == fac).astype(int)
    df["gender_enc"]     = (df["gender"].str.strip().str.title()
                            .map({"Female":1,"Male":0}).fillna(0).astype(int))
    sem_map              = {s:i for i,s in enumerate(SEMESTERS)}
    df["semester_index"] = df["semester"].map(sem_map).fillna(0).astype(int)
    df = df.dropna(subset=["prev_gpa"]).reset_index(drop=True)

    X     = df[FEATURE_COLS].fillna(0).values
    X_sc  = scaler.transform(X)
    probs = model.predict_proba(X_sc)
    preds = probs.argmax(axis=1)

    df["risk_class"] = preds
    df["risk_label"] = [RISK_LABEL[p] for p in preds]
    df["prob_low"]   = probs[:,0].round(3)
    df["prob_med"]   = probs[:,1].round(3)
    df["prob_high"]  = probs[:,2].round(3)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def get_recommendations(row: dict) -> list:
    recs = []
    att   = row.get("avg_attendance",  0)
    exam  = row.get("avg_exam_score",  0)
    ca    = row.get("avg_ca_score",    0)
    trend = row.get("gpa_trend",       0)
    cf    = row.get("consec_fails",    0)
    pgpa  = row.get("prev_gpa",        0)

    if att < 3.0:
        recs.append(("🔴", f"Attendance is very low ({att:.1f}/5). "
            "Contact the student immediately to understand the reason "
            "and create an attendance plan."))
    elif att < 3.5:
        recs.append(("🟡", f"Attendance ({att:.1f}/5) is below the recommended level. "
            "Remind the student that attendance directly affects their grades."))

    if exam/60 < 0.4:
        recs.append(("🔴", f"Exam performance is critically low ({exam:.0f}/60). "
            "Refer the student to the Academic Support Centre for exam preparation help."))
    elif exam/60 < 0.5:
        recs.append(("🟡", f"Exam scores ({exam:.0f}/60) are below average. "
            "Encourage the student to join study groups and seek help from lecturers."))

    if ca/40 < 0.5:
        recs.append(("🟡", f"Continuous Assessment score ({ca:.0f}/40) is below average. "
            "Check that all assignments have been submitted on time."))

    if trend < -0.2:
        recs.append(("🔴", f"The student's GPA has dropped significantly. "
            "Schedule an urgent meeting to understand what is affecting their performance."))
    elif trend < 0:
        recs.append(("🟡", "The student's GPA is slowly declining. "
            "A check-in meeting is recommended before the situation worsens."))

    if cf >= 2:
        recs.append(("🔴", f"This student has had a GPA below 1.5 for {int(cf)} "
            "semesters in a row. Consider referring to the counselling service "
            "and reviewing their course selection."))

    if not recs:
        recs.append(("🟢", "This student is performing well. "
            "Continue standard monitoring through the regular review process."))

    return recs[:5]


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(row: dict, recs: list) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units     import cm
        from reportlab.lib           import colors
        from reportlab.platypus      import (SimpleDocTemplate, Paragraph,
                                              Spacer, Table, TableStyle,
                                              HRFlowable)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        ss    = getSampleStyleSheet()
        BLUE  = colors.HexColor("#4f7cff")
        GREY  = colors.HexColor("#666")
        LIGHT = colors.HexColor("#f0f4f8")
        story = []

        def p(txt, size=10, bold=False, color=colors.HexColor("#333"),
              after=8, align=0):
            return Paragraph(txt, ParagraphStyle("p", parent=ss["Normal"],
                fontSize=size, fontName="Helvetica-Bold" if bold else "Helvetica",
                textColor=color, spaceAfter=after, alignment=align))

        def section(txt):
            return Paragraph(txt, ParagraphStyle("h", parent=ss["Heading2"],
                fontSize=13, textColor=BLUE, fontName="Helvetica-Bold",
                spaceBefore=14, spaceAfter=6))

        # Header
        story += [
            p("PENTECOST UNIVERSITY", 18, True, BLUE, after=4, align=1),
            p("AI Academic Performance Tracker — Student Risk Report", 10,
              color=GREY, after=16, align=1),
            HRFlowable(width="100%", thickness=2, color=BLUE),
            Spacer(1,10),
        ]

        risk_class = row.get("risk_class", 0)
        rc_colors  = {0:colors.HexColor("#27ae60"),
                      1:colors.HexColor("#f39c12"),
                      2:colors.HexColor("#e74c3c")}
        rc_color   = rc_colors.get(int(risk_class), GREY)

        # Student info table
        story.append(section("Student Information"))
        info = Table([
            ["Student ID",  str(row.get("student_id","N/A")),
             "Name",        str(row.get("name","N/A"))],
            ["Faculty",     FACULTY_FULL.get(str(row.get("faculty","")),
                            str(row.get("faculty","N/A"))),
             "Semester",    str(row.get("semester","N/A"))],
            ["Gender",      str(row.get("gender","N/A")),
             "Date",        datetime.date.today().strftime("%d %B %Y")],
        ], colWidths=[3.5*cm,5.5*cm,3.5*cm,5.5*cm])
        info.setStyle(TableStyle([
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,0),(0,-1),BLUE),
            ("TEXTCOLOR",(2,0),(2,-1),BLUE),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("PADDING",(0,0),(-1,-1),7),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,colors.white]),
        ]))
        story += [info, Spacer(1,12)]

        # Risk result
        story.append(section("Risk Assessment Result"))
        res = Table([
            ["Risk Level", row.get("risk_label","N/A"),
             "Confidence", f"{max(row.get('prob_low',0),row.get('prob_med',0),row.get('prob_high',0)):.0%}"],
            ["Low Risk",   f"{row.get('prob_low',0):.0%}",
             "Medium Risk",f"{row.get('prob_med',0):.0%}"],
            ["High Risk",  f"{row.get('prob_high',0):.0%}",
             "Previous GPA",f"{row.get('prev_gpa',0):.2f}"],
        ], colWidths=[3.5*cm,5.5*cm,3.5*cm,5.5*cm])
        res.setStyle(TableStyle([
            ("BACKGROUND",(1,0),(1,0),rc_color),
            ("TEXTCOLOR",(1,0),(1,0),colors.white),
            ("FONTNAME",(1,0),(1,0),"Helvetica-Bold"),
            ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",(2,0),(2,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",(0,0),(0,-1),BLUE),
            ("TEXTCOLOR",(2,0),(2,-1),BLUE),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("PADDING",(0,0),(-1,-1),7),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,colors.white]),
        ]))
        story += [res, Spacer(1,12)]

        # Recommendations
        story.append(section("Recommended Actions"))
        for i,(icon,text) in enumerate(recs,1):
            story.append(p(f"{i}. {text}", after=6))

        # Footer
        story += [
            Spacer(1,20),
            HRFlowable(width="100%",thickness=1,color=colors.HexColor("#dee2e6")),
            Spacer(1,6),
            p(f"Pentecost University | AI Academic Performance Tracker | "
              f"Ghana DPA 2012 (Act 843) | "
              f"{datetime.datetime.now().strftime('%d %B %Y %H:%M')}",
              size=7.5, color=GREY, align=1),
        ]
        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════
def donut_chart(n_lr, n_mr, n_hr):
    total = max(n_lr+n_mr+n_hr, 1)
    fig = go.Figure(go.Pie(
        labels=["Low Risk","Medium Risk","High Risk"],
        values=[n_lr, n_mr, n_hr], hole=0.65,
        marker=dict(colors=["#22c55e","#f59e0b","#ef4444"],
                    line=dict(color="#0a0e17",width=3)),
        textinfo="label+percent",
        textfont=dict(size=11,color="#f1f5f9"),
        pull=[0.02,0.02,0.05],
        hovertemplate="<b>%{label}</b><br>%{value:,} students<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='color:#94a3b8;font-size:11px'>Students</span>",
        x=0.5,y=0.5,showarrow=False,
        font=dict(size=20,color="#f1f5f9"),align="center")
    fig.update_layout(
        paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h",yanchor="bottom",y=-0.2,
                    xanchor="center",x=0.5,
                    font=dict(size=10,color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    return fig


def faculty_bar(df):
    fac_data = {}
    for fac in FACULTIES:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        counts = sub["risk_class"].value_counts() if len(sub) else {}
        fac_data[fac] = [counts.get(k,0) for k in [0,1,2]]

    fig = go.Figure()
    for k,lbl,c in [(0,"Low Risk","#22c55e"),(1,"Medium Risk","#f59e0b"),(2,"High Risk","#ef4444")]:
        vals = [fac_data[f][k] for f in FACULTIES]
        fig.add_trace(go.Bar(
            name=lbl, x=FACULTIES, y=vals, marker_color=c,
            marker_line_width=0,
            text=[str(v) if v>0 else "" for v in vals],
            textposition="inside",
            textfont=dict(size=10,color="white"),
        ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0a0e17",plot_bgcolor="#131a26",
        height=280,margin=dict(l=40,r=10,t=10,b=40),
        font=dict(family="Inter",color="#cbd5e1"),
        xaxis=dict(gridcolor="#232b3d",tickcolor="#94a3b8"),
        yaxis=dict(gridcolor="#232b3d",tickcolor="#94a3b8"),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,
                    xanchor="right",x=1,font=dict(size=10)),
        showlegend=True,
    )
    return fig


def chart_faculty_gpa(df):
    """Average GPA per faculty — horizontal bar with risk thresholds."""
    faculties, avgs, clrs = [], [], []
    for fac in FACULTIES:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        if "semester_gpa" in sub.columns and len(sub):
            faculties.append(fac)
            avgs.append(round(sub["semester_gpa"].mean(), 3))
            clrs.append(FAC_COLOR.get(fac, "#60a5fa"))
    if not faculties:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=240,
            annotations=[dict(text="No GPA data available", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#5b6b8c", size=13))])
        return fig
    fig = go.Figure(go.Bar(
        x=avgs, y=faculties, orientation="h",
        marker_color=clrs, marker_line_width=0,
        text=[f"{v:.2f}" for v in avgs], textposition="outside",
        textfont=dict(size=11, color="#f1f5f9"),
        hovertemplate="<b>%{y}</b><br>Avg GPA: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=2.0, line_dash="dash", line_color="#ef4444",
                  line_width=1.5, opacity=0.7,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#ef4444",
                  annotation_position="top right")
    fig.add_vline(x=3.0, line_dash="dash", line_color="#f59e0b",
                  line_width=1.5, opacity=0.7,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=260, margin=dict(l=10, r=60, t=20, b=30),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(title="Average GPA", range=[0, 4.2],
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        showlegend=False,
    )
    return fig


def chart_faculty_risk_pct(df):
    """Risk percentage split per faculty — 100% stacked bar."""
    faculties, hr_pcts, mr_pcts, lr_pcts = [], [], [], []
    for fac in FACULTIES:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        n = len(sub)
        faculties.append(fac)
        hr_pcts.append(round((sub["risk_class"]==2).sum()/n*100, 1))
        mr_pcts.append(round((sub["risk_class"]==1).sum()/n*100, 1))
        lr_pcts.append(round((sub["risk_class"]==0).sum()/n*100, 1))
    if not faculties:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=260)
        return fig
    fig = go.Figure()
    for vals, lbl, c in [
        (hr_pcts, "High Risk",   "#ef4444"),
        (mr_pcts, "Medium Risk", "#f59e0b"),
        (lr_pcts, "Low Risk",    "#22c55e"),
    ]:
        fig.add_trace(go.Bar(
            name=lbl, x=faculties, y=vals, marker_color=c,
            marker_line_width=0,
            text=[f"{v:.0f}%" for v in vals], textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=280, margin=dict(l=40, r=10, t=10, b=40),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="Percentage of Students (%)",
                   gridcolor="#232b3d", tickcolor="#94a3b8", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


def chart_gender_gpa(df):
    """Average GPA by gender per faculty — grouped bars."""
    if "gender" not in df.columns or "semester_gpa" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=300,
            annotations=[dict(text="No gender data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#5b6b8c", size=13))])
        return fig
    genders = [g for g in ["Female","Male"]
               if g in df["gender"].str.strip().str.title().unique()]
    g_colors = {"Female":"#a78bfa", "Male":"#60a5fa"}
    fig = go.Figure()
    for g in genders:
        avgs = []
        for fac in FACULTIES:
            sub = df.copy()
            if "faculty" in sub.columns:
                sub = sub[sub["faculty"]==fac]
            sub = sub[sub["gender"].str.strip().str.title()==g]
            avgs.append(round(sub["semester_gpa"].mean(), 2) if len(sub) else 0)
        fig.add_trace(go.Bar(
            name=g, x=FACULTIES, y=avgs,
            marker_color=g_colors.get(g, "#60a5fa"),
            marker_line_width=0, opacity=0.88,
            text=[f"{v:.2f}" if v > 0 else "" for v in avgs],
            textposition="outside",
            textfont=dict(size=9, color="#f1f5f9"),
            hovertemplate=f"<b>%{{x}}</b><br>{g}: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ef4444",
                  line_width=1.2, opacity=0.6,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#ef4444")
    fig.add_hline(y=3.0, line_dash="dash", line_color="#f59e0b",
                  line_width=1.2, opacity=0.6,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=320, margin=dict(l=50, r=20, t=20, b=40),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="Average GPA", gridcolor="#232b3d",
                   tickcolor="#94a3b8", range=[0, 4.5]),
        legend=dict(font=dict(size=10, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_gender_risk_split(df):
    """Risk split by gender — side-by-side donut charts."""
    if "gender" not in df.columns or "risk_class" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=280)
        return fig
    genders = [g for g in ["Female","Male"]
               if g in df["gender"].str.strip().str.title().unique()]
    clrs = ["#22c55e","#f59e0b","#ef4444"]
    fig  = go.Figure()
    for i, g in enumerate(genders):
        sub  = df[df["gender"].str.strip().str.title()==g]
        vals = [(sub["risk_class"]==k).sum() for k in [0,1,2]]
        x0   = i * 0.52
        fig.add_trace(go.Pie(
            labels=["Low Risk","Medium Risk","High Risk"],
            values=vals, hole=0.55, name=g,
            domain={"x":[x0, x0+0.45], "y":[0,1]},
            marker=dict(colors=clrs, line=dict(color="#0a0e17", width=2)),
            textinfo="percent", textfont=dict(size=10, color="#f1f5f9"),
            hovertemplate=f"<b>{g}</b><br>%{{label}}: %{{value:,}}<br>%{{percent}}<extra></extra>",
            title=dict(text=g, font=dict(size=13, color="#f1f5f9")),
        ))
    fig.update_layout(
        paper_bgcolor="#0a0e17", height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5,
                    font=dict(size=10, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    return fig


def chart_gpa_trend(df):
    """Semester-on-semester average GPA trend lines per faculty."""
    if "semester" not in df.columns or "semester_gpa" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=340,
            annotations=[dict(
                text="Upload multi-semester data to see GPA trends",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#5b6b8c", size=13))])
        return fig
    sem_ord = {s:i for i,s in enumerate(SEMESTERS)}
    fig = go.Figure()
    drawn = 0
    for fac in FACULTIES:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        trend = (sub.groupby("semester")["semester_gpa"].mean()
                 .reset_index()
                 .assign(order=lambda x: x["semester"].map(sem_ord))
                 .sort_values("order"))
        if len(trend) < 2:
            continue
        c = FAC_COLOR.get(fac, "#60a5fa")
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["semester_gpa"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color="#0a0e17",width=1.5)),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.05)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>Avg GPA: %{{y:.2f}}<extra></extra>",
        ))
        drawn += 1
    if drawn == 0:
        fig.add_annotation(text="Need at least 2 semesters per faculty for trends",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#5b6b8c", size=13))
    fig.add_hrect(y0=0,   y1=2.0, fillcolor="rgba(248,81,73,0.05)",  line_width=0)
    fig.add_hrect(y0=2.0, y1=3.0, fillcolor="rgba(240,136,62,0.03)", line_width=0)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#ef4444",
                  line_width=1, opacity=0.5,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#ef4444")
    fig.add_hline(y=3.0, line_dash="dash", line_color="#f59e0b",
                  line_width=1, opacity=0.5,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f59e0b",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=340, margin=dict(l=50, r=20, t=20, b=50),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(title="Semester", tickangle=-30,
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="Average GPA", range=[0, 4.2],
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        legend=dict(font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def chart_risk_trend(df):
    """High Risk count per semester per faculty — trend lines."""
    if "semester" not in df.columns or "risk_class" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0a0e17", height=340,
            annotations=[dict(
                text="Upload multi-semester data to see risk trends",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#5b6b8c", size=13))])
        return fig
    sem_ord = {s:i for i,s in enumerate(SEMESTERS)}
    fig = go.Figure()
    drawn = 0
    for fac in FACULTIES:
        sub = df[(df["faculty"]==fac)&(df["risk_class"]==2)]               if "faculty" in df.columns else pd.DataFrame()
        if len(sub) == 0:
            continue
        trend = (sub.groupby("semester").size()
                 .reset_index(name="count")
                 .assign(order=lambda x: x["semester"].map(sem_ord))
                 .sort_values("order"))
        if len(trend) < 2:
            continue
        c = FAC_COLOR.get(fac, "#60a5fa")
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["count"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color="#0a0e17",width=1.5)),
            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.06)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>High Risk: %{{y}}<extra></extra>",
        ))
        drawn += 1
    if drawn == 0:
        fig.add_annotation(text="Need at least 2 semesters per faculty for trends",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#5b6b8c", size=13))
    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=340, margin=dict(l=50, r=20, t=20, b=50),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(title="Semester", tickangle=-30,
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="High Risk Students",
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        legend=dict(font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def graduation_summary(df):
    """Graduation projection table for Level 300/400 students."""
    if "level" not in df.columns:
        return pd.DataFrame()
    final = df[df["level"].isin(["Level 300","Level 400"])].copy()
    if len(final) == 0:
        return pd.DataFrame()
    rows = []
    for _, row in final.iterrows():
        lv   = int(str(row.get("level","300")).split()[-1])
        cum  = row.get("cumulative_gpa",  row.get("prev_gpa", 1.8))
        comp = row.get("completed_credits", 90)
        prog = row.get("programme_credits", 120)
        gtr  = row.get("gpa_trend", 0)
        grad = project_graduation(cum, gtr, lv, comp, prog)
        rows.append({
            "Student ID"      : row.get("student_id",""),
            "Name"            : row.get("name",""),
            "Faculty"         : row.get("faculty",""),
            "Level"           : row.get("level",""),
            "Current CGPA"    : round(cum, 2),
            "Projected CGPA"  : grad["proj_cgpa"],
            "Classification"  : grad["emoji"] + "  " + grad["label"],
            "Risk Level"      : row.get("risk_label",""),
        })
    return pd.DataFrame(rows)


def chart_trajectory(points: list) -> go.Figure:
    """Line chart showing a student's projected GPA over future semesters,
    with risk zone bands and colour-coded markers."""
    x_labels = ["Now"] + [f"+{p['step']} Sem" for p in points[1:]]
    y_vals   = [p["gpa"] for p in points]
    zone_clr = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}
    marker_colors = [zone_clr[p["zone"]] for p in points]

    fig = go.Figure()

    # Risk zone background bands
    fig.add_hrect(y0=0,   y1=Q33, fillcolor="rgba(239,68,68,0.07)",  line_width=0)
    fig.add_hrect(y0=Q33, y1=Q66, fillcolor="rgba(245,158,11,0.05)", line_width=0)
    fig.add_hrect(y0=Q66, y1=4.0, fillcolor="rgba(34,197,94,0.05)",  line_width=0)

    fig.add_trace(go.Scatter(
        x=x_labels, y=y_vals,
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2.5, dash="dot"),
        marker=dict(size=13, color=marker_colors,
                    line=dict(color="#0a0e17", width=2)),
        hovertemplate="<b>%{x}</b><br>Projected GPA: %{y:.2f}<extra></extra>",
        showlegend=False,
    ))

    fig.add_hline(y=Q33, line_dash="dash", line_color="#ef4444",
                  line_width=1, opacity=0.6,
                  annotation_text=f"High Risk ({Q33})",
                  annotation_font_color="#ef4444")
    fig.add_hline(y=Q66, line_dash="dash", line_color="#f59e0b",
                  line_width=1, opacity=0.6,
                  annotation_text=f"Medium Risk ({Q66})",
                  annotation_font_color="#f59e0b",
                  annotation_position="bottom right")

    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=240, margin=dict(l=40, r=20, t=15, b=30),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="Projected GPA", range=[0, 4.2],
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        showlegend=False,
    )
    return fig


def chart_shap_importance() -> go.Figure:
    """Horizontal bar chart — what matters most to the AI (top SHAP features)."""
    names = [f[1] for f in SHAP_IMPORTANCE][::-1]
    vals  = [f[2] for f in SHAP_IMPORTANCE][::-1]
    max_v = max(vals)
    pct   = [round(v / max_v * 100) for v in vals]

    fig = go.Figure(go.Bar(
        x=pct, y=names, orientation="h",
        marker_color="#4f7cff", marker_line_width=0,
        text=[f"{p}%" for p in pct], textposition="outside",
        textfont=dict(size=11, color="#f1f5f9"),
        hovertemplate="<b>%{y}</b><br>Relative importance: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=260, margin=dict(l=10, r=50, t=10, b=30),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(title="Relative influence on prediction", range=[0, 115],
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        showlegend=False,
    )
    return fig


def chart_fairness_audit() -> go.Figure:
    """Bar chart — AI performance (F1) across gender and faculty groups,
    all passing the fairness threshold."""
    labels = [f[0] for f in FAIRNESS_AUDIT]
    colors = [f[1] for f in FAIRNESS_AUDIT]
    vals   = [f[2] for f in FAIRNESS_AUDIT]

    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker_color=colors, marker_line_width=0,
        text=[f"{v:.2f} ✅" for v in vals], textposition="outside",
        textfont=dict(size=11, color="#f1f5f9"),
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=FAIRNESS_THRESHOLD, line_dash="dash",
                  line_color="#94a3b8", line_width=1.2, opacity=0.7,
                  annotation_text=f"Minimum acceptable ({FAIRNESS_THRESHOLD})",
                  annotation_font_color="#94a3b8")
    fig.update_layout(
        paper_bgcolor="#0a0e17", plot_bgcolor="#131a26",
        height=260, margin=dict(l=40, r=20, t=15, b=40),
        font=dict(family="Inter", color="#cbd5e1"),
        xaxis=dict(gridcolor="#232b3d", tickcolor="#94a3b8"),
        yaxis=dict(title="AI Accuracy Score (F1)", range=[0, 1.05],
                   gridcolor="#232b3d", tickcolor="#94a3b8"),
        showlegend=False,
    )
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1.5rem">
          <div style="font-size:3.5rem">🎓</div>
          <h1 style="font-size:1.7rem;font-weight:700;color:#f1f5f9;
                     letter-spacing:-.02em;margin:.4rem 0">
              Pentecost University</h1>
          <p style="color:#60a5fa;font-size:.82rem;font-weight:600;
                    letter-spacing:.1em;text-transform:uppercase;margin:0">
              Academic Performance Tracker</p>
          <p style="color:#94a3b8;font-size:.84rem;margin-top:.5rem">
              Sign in to view and manage student academic risk predictions</p>
          <div style="height:2px;background:linear-gradient(90deg,
                      transparent,#60a5fa,transparent);
                      margin:1rem auto;width:60%"></div>
        </div>""", unsafe_allow_html=True)

        if not artefacts_ok:
            st.warning("**Setup required:** The prediction model files are not found. "
                       "Please upload `best_model.pkl`, `scaler.pkl`, "
                       "`feature_cols.json`, and `thresholds.json` to the repository.")

        st.markdown("""
        <div style="background:#131a26;border:1px solid #2a3344;
                    border-radius:14px;padding:1.6rem 1.8rem">""",
                    unsafe_allow_html=True)

        with st.form("login"):
            st.markdown('<p style="color:#94a3b8;font-size:.78rem;'
                        'text-transform:uppercase;letter-spacing:.1em;'
                        'margin-bottom:.5rem">Who are you?</p>',
                        unsafe_allow_html=True)
            role = st.selectbox("Role", list(ROLES.keys()),
                                label_visibility="collapsed")
            st.markdown('<p style="color:#94a3b8;font-size:.78rem;'
                        'text-transform:uppercase;letter-spacing:.1em;'
                        'margin:.6rem 0 .3rem">Password</p>',
                        unsafe_allow_html=True)
            pwd  = st.text_input("Password", type="password",
                                 placeholder="Enter your password",
                                 label_visibility="collapsed")
            ok   = st.form_submit_button("Sign In →",
                                         use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if ok:
            cfg     = ROLES[role]
            correct = _secret(cfg["pwd_key"], cfg["default"])
            if pwd == correct:
                st.session_state.update({
                    "auth": True, "role": role, "df": None,
                })
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

        st.markdown("""
        <p style="text-align:center;color:#64748b;
                  font-size:.71rem;margin-top:1.2rem">
          2025 Pentecost University &nbsp;·&nbsp;
          Ghana Data Protection Act 2012 (Act 843) Compliant
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(role: str, df):
    cfg     = ROLES[role]
    faculty = cfg["faculty"]

    with st.sidebar:
        fac_line = ("All Faculties" if not faculty
                    else f"{faculty} — {FACULTY_FULL.get(faculty,'')[:22]}...")
        st.markdown(f"""
        <div style="text-align:center;padding:.6rem 0 1rem">
          <div style="width:56px;height:56px;border-radius:16px;margin:0 auto .6rem;
                      background:var(--grad-accent);display:flex;
                      align-items:center;justify-content:center;font-size:1.7rem;
                      box-shadow:0 8px 22px -8px rgba(79,124,255,.55)">
            {cfg['icon']}
          </div>
          <div style="font-weight:700;font-size:.95rem;color:#f1f5f9">
              {role.split(' — ')[0]}</div>
          <div style="font-size:.72rem;color:#60a5fa;margin-top:.2rem;
                      background:rgba(96,165,250,.12);display:inline-block;
                      padding:.15rem .6rem;border-radius:20px">
              {fac_line}</div>
        </div>
        <hr style="border-color:#232b3d;margin:.3rem 0 .8rem">
        """, unsafe_allow_html=True)

        # Dataset status
        if df is not None:
            n    = len(df)
            n_hr = (df["risk_class"]==2).sum()
            n_mr = (df["risk_class"]==1).sum()
            n_lr = (df["risk_class"]==0).sum()
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,#102a1c 0%,#0a0e17 100%);
                        border:1px solid #22c55e44;
                        border-radius:12px;padding:.8rem .9rem;font-size:.82rem;
                        margin-bottom:.6rem">
              <div style="color:#22c55e;font-weight:700;display:flex;
                          align-items:center;gap:.4rem">
                <span style="font-size:1rem">✅</span> Dataset loaded
              </div>
              <div style="color:#94a3b8;margin-top:.2rem">{n:,} students analysed</div>
            </div>
            <div style="font-size:.8rem;margin:.3rem 0;line-height:1.9">
              🔴 <b style="color:#ef4444">{n_hr}</b> need immediate attention<br>
              🟡 <b style="color:#f59e0b">{n_mr}</b> need monitoring<br>
              🟢 <b style="color:#22c55e">{n_lr}</b> on track
            </div>""", unsafe_allow_html=True)
            st.markdown('<hr style="border-color:#232b3d">',
                        unsafe_allow_html=True)

        # Quick guide
        with st.expander("How to use this system", key="sidebar_help_expander"):
            st.markdown("""
            **Step 1 — Upload your data**
            Upload a CSV file containing your students' semester records.

            **Step 2 — View predictions**
            The system automatically analyses every student and shows
            who needs attention.

            **Step 3 — Take action**
            Click any student to see their details and recommended actions.
            Download individual PDF reports or all results as a spreadsheet.
            """)

        st.markdown('<hr style="border-color:#232b3d">', unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            for k in ["auth","role","df"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("""
        <div style="font-size:.64rem;color:#64748b;
                    text-align:center;margin-top:.5rem">
          Ghana DPA 2012 Compliant
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT DETAIL CARD
# ══════════════════════════════════════════════════════════════════════════════
def student_detail(row: dict, uid: str = ""):
    """Expandable detail panel for one student — redesigned, richer layout."""
    risk_class = int(row.get("risk_class", 0))
    rc         = RISK_COLOR[risk_class]
    rbg        = RISK_BG[risk_class]
    recs       = get_recommendations(row)

    name = str(row.get("name","Unknown Student"))
    parts = [p for p in name.split() if p]
    initials = ("".join(p[0].upper() for p in parts[:2])) or "?"
    sid  = str(row.get("student_id",""))
    fac  = str(row.get("faculty",""))
    gend = str(row.get("gender",""))
    sem  = str(row.get("semester",""))
    gpa  = row.get("semester_gpa", 0)

    # ── Profile header ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.1rem;
                padding-bottom:1rem;border-bottom:1px solid #232b3d">
      <div style="width:50px;height:50px;border-radius:13px;flex-shrink:0;
                  background:var(--grad-accent);display:flex;
                  align-items:center;justify-content:center;
                  font-size:1.1rem;font-weight:800;color:white;
                  box-shadow:0 6px 16px -6px rgba(79,124,255,.5)">
        {initials}
      </div>
      <div style="flex:1;min-width:0">
        <div style="font-weight:700;color:#f1f5f9;font-size:1.05rem">{name}</div>
        <div style="color:#94a3b8;font-size:.8rem;margin-top:.15rem">
          🆔 {sid} &nbsp;·&nbsp; 🏛️ {fac} &nbsp;·&nbsp; {gend} &nbsp;·&nbsp; {sem}
        </div>
      </div>
      <div style="text-align:right;flex-shrink:0">
        <div style="font-size:1.6rem;font-weight:800;color:{rc};line-height:1">{gpa:.2f}</div>
        <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:.06em;margin-top:.1rem">GPA</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Performance scores ────────────────────────────────────────────────
    st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                'margin-bottom:.5rem">📊 Academic Performance</p>',
                unsafe_allow_html=True)

    def score_bar(label, value, maximum, color):
        pct = min(value/maximum*100, 100) if maximum > 0 else 0
        ok  = "✅" if pct >= 50 else "⚠️" if pct >= 40 else "🚨"
        st.markdown(f"""
        <div style="margin:.35rem 0">
          <div style="display:flex;justify-content:space-between;
                      font-size:.8rem;color:#94a3b8;margin-bottom:.2rem">
            <span>{ok} {label}</span>
            <span style="color:{color};font-weight:700">
                {value:.1f} / {maximum}</span>
          </div>
          <div class="prog-bg">
            <div class="prog-fill"
                 style="width:{pct:.0f}%;
                        background:linear-gradient(90deg,{color}aa,{color});
                        --glow:{_rgba(color,.6)}"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        score_bar("Total Mark",  row.get("avg_total_mark",0), 100, "#60a5fa")
        score_bar("CA Score",    row.get("avg_ca_score",  0),  40, "#22c55e")
    with sc2:
        score_bar("Exam Score",  row.get("avg_exam_score",0),  60, "#f59e0b")
        score_bar("Attendance",  row.get("avg_attendance",0),   5, "#a78bfa")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key facts — stat tile grid ──────────────────────────────────────────
    st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                'margin-bottom:.5rem">🔍 Key Facts</p>',
                unsafe_allow_html=True)

    gpa_trend = row.get("gpa_trend", 0)
    if gpa_trend > 0.05:
        trend_txt, trend_icon = f"+{gpa_trend:.2f}", "📈"
    elif gpa_trend < -0.05:
        trend_txt, trend_icon = f"{gpa_trend:.2f}", "📉"
    else:
        trend_txt, trend_icon = "Stable", "➡️"

    conf = max(row.get("prob_low",0), row.get("prob_med",0), row.get("prob_high",0))

    tiles = [
        ("📊", "Previous GPA",          f"{row.get('prev_gpa',0):.2f} / 4.00", "#f1f5f9"),
        (trend_icon, "GPA Direction",   trend_txt, "#f1f5f9"),
        ("⚠️", "Sems Below 1.5 GPA",    f"{int(row.get('consec_fails',0))}", "#f1f5f9"),
        ("📚", "Credits This Semester", f"{int(row.get('total_credits',0))}", "#f1f5f9"),
        ("🎯", "Courses Enrolled",      f"{int(row.get('num_courses',0))}", "#f1f5f9"),
        ("🤖", "Prediction Confidence", f"{conf:.0%}", rc),
    ]
    tiles_html = ""
    for icon, lbl, val, color in tiles:
        tiles_html += (
            '<div class="stat-tile">'
            '<div class="stat-lbl">' + icon + ' ' + lbl + '</div>'
            '<div class="stat-val" style="color:' + color + '">' + val + '</div>'
            '</div>'
        )
    st.markdown(f'<div class="stat-grid">{tiles_html}</div>', unsafe_allow_html=True)

    # ── What this means ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="risk-meaning" style="--rc:{rc};--rbg:{rbg}">
      <div class="ricon">{RISK_ICON[risk_class]}</div>
      <div>
        <b style="color:{rc}">What this means</b><br>
        {RISK_MEANING[risk_class]}
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Recommended actions ───────────────────────────────────────────────
    st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                'margin:1rem 0 .5rem">💡 Recommended Actions</p>',
                unsafe_allow_html=True)
    for icon, text in recs:
        color = ("#ef4444" if icon=="🔴" else
                 "#f59e0b" if icon=="🟡" else "#22c55e")
        st.markdown(f"""
        <div class="rec-item" style="--rc:{color};--rbg2:{_rgba(color,.15)}">
          <div class="rec-icon">{icon}</div>
          <div>{text}</div>
        </div>""", unsafe_allow_html=True)

    # ── Graduation projection (Level 300/400) ─────────────────────────────
    level = row.get("level","")
    if level in ["Level 300","Level 400"]:
        lv_num   = int(str(level).split()[1])
        cum_gpa  = row.get("cumulative_gpa", row.get("prev_gpa",1.8))
        comp_cr  = row.get("completed_credits", 90)
        prog_cr  = row.get("programme_credits", 120)
        g_trend  = row.get("gpa_trend",0)
        grad     = project_graduation(cum_gpa, g_trend, lv_num, comp_cr, prog_cr)

        st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                    'margin:1rem 0 .5rem">🎓 Graduation Classification Projection</p>',
                    unsafe_allow_html=True)

        upgrade_html = ""
        for t in grad["next"]:
            upgrade_html += (
                '<div class="grad-upgrade">'
                'To reach <b style="color:' + t["color"] + '">'
                + t["class"] +
                '</b>, needs avg GPA of <b style="color:' + t["color"] + '">'
                + str(round(t["needed"], 2)) +
                '</b> in remaining semester(s)</div>'
            )

        grad_html = (
            '<div class="grad-card" style="--rc:' + grad["color"]
            + ';--rbg:' + _rgba(grad["color"], .18) + '">'
            '<div style="display:flex;align-items:center;gap:1.1rem;'
            'position:relative;z-index:1">'
            '<div class="grad-emoji">' + grad["emoji"] + '</div>'
            '<div>'
            '<div style="font-weight:800;color:#f1f5f9;font-size:1.1rem">'
            + grad["label"] + '</div>'
            '<div style="color:#94a3b8;font-size:.84rem;margin-top:.15rem">'
            'Projected Final CGPA: '
            '<b style="color:' + grad["color"] + ';font-size:1rem">'
            + str(grad["proj_cgpa"]) + '</b>'
            ' &nbsp;·&nbsp; ' + str(grad["remaining"]) +
            ' semester(s) remaining</div>'
            '</div></div>'
            + upgrade_html +
            '</div>'
        )
        st.markdown(grad_html, unsafe_allow_html=True)

    # ── Multi-semester GPA trajectory ─────────────────────────────────────
    st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                'margin:1rem 0 .4rem">📈 Risk Trajectory — Next 4 Semesters</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:.81rem;margin-bottom:.5rem">'
                'Based on the current GPA trend, the system projects where this '
                'student\'s academic performance is heading. This is a forecast, '
                'not a certainty — early intervention can change the trajectory.</p>',
                unsafe_allow_html=True)

    traj = project_trajectory(
        current_gpa = row.get("semester_gpa", row.get("prev_gpa", 1.8)),
        gpa_trend   = row.get("gpa_trend", 0),
        n_steps     = 4,
    )
    traj_icon, traj_msg = trajectory_insight(traj, name.split()[0])

    traj_color = "#22c55e" if traj_icon == "📈" else \
                 "#ef4444" if traj_icon == "📉" else "#60a5fa"

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{_rgba(traj_color,.12)} 0%,
                #0a0e17 100%);border:1px solid {_rgba(traj_color,.35)};
                border-radius:12px;padding:.8rem 1rem;margin-bottom:.6rem;
                font-size:.86rem;color:#e2e8f0;line-height:1.6;
                display:flex;gap:.8rem;align-items:flex-start">
      <span style="font-size:1.4rem;flex-shrink:0">{traj_icon}</span>
      <span>{traj_msg}</span>
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_trajectory(traj),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"traj_{uid}")


    # ── What-If Simulator ─────────────────────────────────────────────────
    sim_toggle_key = f"sim_open_{uid}"
    if sim_toggle_key not in st.session_state:
        st.session_state[sim_toggle_key] = False

    st.markdown("<br>", unsafe_allow_html=True)

    sim_label = ("🔬 Close Simulator" if st.session_state[sim_toggle_key]
                 else "🔬 What-If Simulator — Adjust & Re-Predict")
    if st.button(sim_label, key=f"sim_btn_{uid}", use_container_width=True):
        st.session_state[sim_toggle_key] = not st.session_state[sim_toggle_key]

    if st.session_state[sim_toggle_key]:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(79,124,255,.10) 0%,
                    #0a0e17 100%);border:1px solid rgba(79,124,255,.3);
                    border-radius:14px;padding:1rem 1.2rem;margin:.5rem 0 .8rem">
          <p style="color:#bcd4ff;font-size:.84rem;margin:0;line-height:1.6">
            <b style="color:#f1f5f9">\U0001f52c Interactive Simulator</b><br>
            Adjust the sliders to instantly re-predict this student's risk.
            Use this to identify which improvements make the biggest difference.
          </p>
        </div>""", unsafe_allow_html=True)

        s1, s2 = st.columns(2)
        with s1:
            sim_att  = st.slider("Attendance (0-5)",    0.0, 5.0,
                                 float(round(row.get("avg_attendance", 3.0), 1)),
                                 0.1, key=f"sim_att_{uid}")
            sim_mark = st.slider("Avg Total Mark (0-100)", 0.0, 100.0,
                                 float(round(row.get("avg_total_mark", 55.0), 0)),
                                 0.5, key=f"sim_mark_{uid}")
            sim_ca   = st.slider("CA Score (0-40)", 0.0, 40.0,
                                 float(round(row.get("avg_ca_score", 20.0), 0)),
                                 0.5, key=f"sim_ca_{uid}")
        with s2:
            sim_exam  = st.slider("Exam Score (0-60)", 0.0, 60.0,
                                  float(round(row.get("avg_exam_score", 30.0), 0)),
                                  0.5, key=f"sim_exam_{uid}")
            sim_trend = st.slider("GPA Trend (-2 to +2)", -2.0, 2.0,
                                  float(round(row.get("gpa_trend", 0.0), 2)),
                                  0.05, key=f"sim_trend_{uid}")
            sim_cf    = st.slider("Consecutive Fail Sems (0-8)", 0, 8,
                                  int(row.get("consec_fails", 0)),
                                  key=f"sim_cf_{uid}")

        sim_feats = dict(row)
        sim_feats["avg_attendance"]  = sim_att
        sim_feats["avg_total_mark"]  = sim_mark
        sim_feats["avg_ca_score"]    = sim_ca
        sim_feats["avg_exam_score"]  = sim_exam
        sim_feats["gpa_trend"]       = sim_trend
        sim_feats["consec_fails"]    = float(sim_cf)
        sim_feats["trend_x_fail"]    = sim_trend * float(sim_cf)

        try:
            sim_vec    = np.array([float(sim_feats.get(c, 0.0))
                                   for c in FEATURE_COLS]).reshape(1,-1)
            sim_vec_sc = scaler.transform(sim_vec)
            sim_probs  = model.predict_proba(sim_vec_sc)[0]
            sim_pred   = int(np.argmax(sim_probs))

            orig_pred  = risk_class
            orig_probs = np.array([row.get("prob_low",0),
                                   row.get("prob_med",0),
                                   row.get("prob_high",0)])
            orig_rc    = RISK_COLOR[orig_pred]
            sim_rc     = RISK_COLOR[sim_pred]
            delta      = sim_probs[2] - orig_probs[2]

            st.markdown("<br>", unsafe_allow_html=True)
            comp_l, comp_sep, comp_r = st.columns([1, 0.15, 1])

            with comp_l:
                st.markdown(
                    '<div style="background:' + _rgba(orig_rc,.12) +
                    ';border:1px solid ' + _rgba(orig_rc,.4) +
                    ';border-radius:12px;padding:1rem;text-align:center">' +
                    '<div style="font-size:.75rem;color:#94a3b8;' +
                    'text-transform:uppercase;letter-spacing:.08em;' +
                    'margin-bottom:.4rem">Current Prediction</div>' +
                    '<div style="font-size:2rem">' + RISK_ICON[orig_pred] + '</div>' +
                    '<div style="font-weight:800;color:' + orig_rc +
                    ';font-size:1rem;margin:.25rem 0">' + RISK_LABEL[orig_pred] + '</div>' +
                    '<div style="font-size:.82rem;color:#94a3b8">High Risk: <b style="color:' +
                    orig_rc + '">' + f"{orig_probs[2]:.0%}" + '</b></div></div>',
                    unsafe_allow_html=True)

            with comp_sep:
                st.markdown(
                    '<div style="display:flex;height:100%;align-items:center;' +
                    'justify-content:center;font-size:1.5rem;color:#4f7cff;' +
                    'padding-top:1.8rem">\u2192</div>',
                    unsafe_allow_html=True)

            with comp_r:
                st.markdown(
                    '<div style="background:' + _rgba(sim_rc,.12) +
                    ';border:1px solid ' + _rgba(sim_rc,.4) +
                    ';border-radius:12px;padding:1rem;text-align:center">' +
                    '<div style="font-size:.75rem;color:#94a3b8;' +
                    'text-transform:uppercase;letter-spacing:.08em;' +
                    'margin-bottom:.4rem">Simulated Prediction</div>' +
                    '<div style="font-size:2rem">' + RISK_ICON[sim_pred] + '</div>' +
                    '<div style="font-weight:800;color:' + sim_rc +
                    ';font-size:1rem;margin:.25rem 0">' + RISK_LABEL[sim_pred] + '</div>' +
                    '<div style="font-size:.82rem;color:#94a3b8">High Risk: <b style="color:' +
                    sim_rc + '">' + f"{sim_probs[2]:.0%}" + '</b></div></div>',
                    unsafe_allow_html=True)

            if sim_pred < orig_pred:
                icon_c, clr = "\u2705", "#22c55e"
                msg = (f"These changes improve the prediction from <b>{RISK_LABEL[orig_pred]}</b>"
                       f" to <b style='color:{sim_rc}'>{RISK_LABEL[sim_pred]}</b>. "
                       f"High Risk probability drops by <b style='color:#22c55e'>{abs(delta):.0%}</b>.")
            elif sim_pred > orig_pred:
                icon_c, clr = "\u26a0\ufe0f", "#f59e0b"
                msg = (f"These values would worsen risk to <b style='color:{sim_rc}'>"
                       f"{RISK_LABEL[sim_pred]}</b>. "
                       f"High Risk probability rises by <b style='color:#ef4444'>{abs(delta):.0%}</b>.")
            else:
                icon_c, clr = "\u2139\ufe0f", "#60a5fa"
                msg = (f"Risk class stays at <b>{RISK_LABEL[sim_pred]}</b>. "
                       f"High Risk probability changes by <b>{delta:+.0%}</b>.")

            st.markdown(
                '<div style="background:' + _rgba(clr,.10) +
                ';border:1px solid ' + _rgba(clr,.35) +
                ';border-radius:10px;padding:.8rem 1rem;margin:.8rem 0;' +
                'font-size:.86rem;color:#e2e8f0;line-height:1.6;' +
                'display:flex;gap:.7rem">' +
                '<span style="font-size:1.2rem;flex-shrink:0">' + icon_c + '</span>' +
                '<span>' + msg + '</span></div>',
                unsafe_allow_html=True)

        except Exception as sim_err:
            st.info(f"Simulator requires model artefacts. ({sim_err})")

    # ── PDF download — lazy generation with unique key ───────────────────
    st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                'margin:1rem 0 .5rem">📄 Export Report</p>',
                unsafe_allow_html=True)
    pdf_key    = f"gen_pdf_{uid}"
    dl_key     = f"dl_pdf_{uid}"
    cache_key  = f"pdf_cache_{uid}"

    if st.button("📄 Generate PDF Report", key=pdf_key,
                 use_container_width=True):
        with st.spinner("Preparing report..."):
            pdf_bytes = generate_pdf(row, recs)
        if pdf_bytes:
            st.session_state[cache_key] = pdf_bytes
        else:
            st.info("Install `reportlab` to enable PDF export.")

    if st.session_state.get(cache_key):
        st.download_button(
            "⬇️ Download PDF Report",
            data=st.session_state[cache_key],
            file_name=f"report_{row.get('student_id','student')}.pdf",
            mime="application/pdf",
            key=dl_key,
            use_container_width=True,
        )



# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def main_dashboard(role: str):
    cfg     = ROLES[role]
    faculty = cfg["faculty"]

    render_sidebar(role, st.session_state.get("df"))

    # ── Top banner ─────────────────────────────────────────────────────────
    fac_display = FACULTY_FULL.get(faculty, "All Faculties") if faculty else "All Faculties"
    st.markdown(f"""
    <div class="top-banner">
      <div class="top-banner-icon">🎓</div>
      <div>
        <h1>Academic Performance Tracker</h1>
        <div class="sub">
          <span class="pill">{cfg['icon']} {role}</span>
          <span class="pill">🏛️ {fac_display}</span>
          <span class="pill">📅 {datetime.date.today().strftime("%d %B %Y")}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not artefacts_ok:
        st.error("The prediction model is not set up yet. "
                 "Please contact your system administrator.")
        return

    df = st.session_state.get("df")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL TRANSPARENCY DASHBOARD — always visible, collapsed by default
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("🔬 About the AI Model — How It Makes Predictions",
                     expanded=False, key="model_transparency_expander"):

        st.markdown('<p style="color:#94a3b8;font-size:.85rem;'
                    'margin-bottom:1rem">'
                    'This section explains what the AI model is, how it was '
                    'built, how accurate it is, and whether it treats all '
                    'students fairly. This transparency is essential for '
                    'responsible use of AI in academic decision-making.</p>',
                    unsafe_allow_html=True)

        # ── Model performance KPIs ─────────────────────────────────────────
        st.markdown('<p style="color:#f1f5f9;font-weight:700;font-size:.92rem;'
                    'margin-bottom:.6rem">⚡ Model Performance</p>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col, icon, val, lbl, sub, c, glow in [
            (m1, "🎯", f"{MF1:.4f}", "Accuracy (F1)",
             "Across all 3 risk classes",
             "#4f7cff", "rgba(79,124,255,.3)"),
            (m2, "📐", "0.0957", "Calibration (ECE)",
             "Lower = more trustworthy probabilities",
             "#22c55e", "rgba(34,197,94,.3)"),
            (m3, "🔗", "+0.329", "Bias-Variance Gap",
             "Expected training artefact — not real overfitting",
             "#f59e0b", "rgba(245,158,11,.3)"),
            (m4, "✅", "All Pass", "Fairness Audit",
             "All 8 gender × faculty subgroups pass",
             "#22c55e", "rgba(34,197,94,.3)"),
        ]:
            with col:
                st.markdown(f"""
                <div class="sum-card" style="--c:{c};--cglow:{glow}">
                  <div class="sum-icon">{icon}</div>
                  <div class="sum-val" style="font-size:1.5rem">{val}</div>
                  <div class="sum-lbl">{lbl}</div>
                  <div class="sum-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP + Fairness side by side ──────────────────────────────────
        sh1, sh2 = st.columns(2)
        with sh1:
            st.markdown('<p style="color:#f1f5f9;font-weight:700;'
                        'font-size:.92rem;margin-bottom:.3rem">'
                        '🔍 What the AI Looks At Most</p>',
                        unsafe_allow_html=True)
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;'
                        'margin-bottom:.5rem">'
                        'These are the six factors that influence the '
                        'prediction most — ranked by how much each one '
                        'affects the AI\'s decision. The longer the bar, '
                        'the more that factor matters.</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_shap_importance(),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="shap_importance_top")

        with sh2:
            st.markdown('<p style="color:#f1f5f9;font-weight:700;'
                        'font-size:.92rem;margin-bottom:.3rem">'
                        '⚖️ Is the AI Fair to All Students?</p>',
                        unsafe_allow_html=True)
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;'
                        'margin-bottom:.5rem">'
                        'The model was tested separately on male vs female '
                        'students, and on each faculty. Every group scored '
                        'above the minimum fairness threshold of '
                        f'{FAIRNESS_THRESHOLD:.2f}. No demographic group '
                        'is systematically disadvantaged.</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_fairness_audit(),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="fairness_audit_top")

        # ── How to interpret the probability ──────────────────────────────
        st.markdown('<p style="color:#f1f5f9;font-weight:700;'
                    'font-size:.92rem;margin:.8rem 0 .4rem">'
                    '🎲 How to Read the Confidence Percentage</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, pct, title, body, color in [
            (c1, "≥ 85%", "Very Confident",
             "The AI has very strong evidence this student is at risk. "
             "Prioritise this case immediately.",
             "#ef4444"),
            (c2, "60–84%", "Moderately Confident",
             "Risk signals present but not overwhelming. "
             "Treat as a flag for a check-in meeting — not a certainty.",
             "#f59e0b"),
            (c3, "< 60%", "Low Confidence",
             "The AI is uncertain. "
             "Use your own professional judgment alongside this signal.",
             "#60a5fa"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#131a26;border:1px solid #232b3d;
                            border-radius:12px;padding:.9rem 1rem;
                            border-top:3px solid {color}">
                  <div style="font-size:1.3rem;font-weight:800;
                              color:{color}">{pct}</div>
                  <div style="font-weight:700;color:#f1f5f9;
                              font-size:.88rem;margin:.25rem 0">{title}</div>
                  <div style="color:#94a3b8;font-size:.81rem;
                              line-height:1.5">{body}</div>
                </div>""", unsafe_allow_html=True)

        # ── Important limitation note ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a1a2e 0%,#0a0e17 100%);
                    border:1px solid #2a3344;border-radius:12px;
                    padding:.9rem 1.1rem;font-size:.84rem;color:#94a3b8;
                    line-height:1.65">
          ⚠️ <b style="color:#f1f5f9">Important:</b>
          This AI prediction is a <b>supporting tool</b>, not a verdict.
          It was trained on historical data (2019–2022) and identifies
          patterns that are <em>associated</em> with academic difficulty —
          it cannot account for personal circumstances, mental health,
          family situations, or sudden life events.
          Always apply professional academic judgment before taking any action.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # STATE 1 — No data uploaded yet
    # ══════════════════════════════════════════════════════════════════════
    if df is None:

        # How it works — 3 steps
        st.markdown('<p style="color:#f1f5f9;font-size:1rem;font-weight:600;'
                    'margin-bottom:.8rem">Get started in 3 simple steps</p>',
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        for col, icon, title, desc in [
            (s1, "📤", "Upload your student data",
             "Upload a CSV file with your students' records for the current semester."),
            (s2, "⚡", "View predictions instantly",
             "The system automatically identifies which students need attention."),
            (s3, "✅", "Take action",
             "See recommended actions for each student and download reports."),
        ]:
            with col:
                st.markdown(f"""
                <div class="step-card">
                  <div class="step-icon">{icon}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Upload section
        st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                    'margin-bottom:.3rem">Upload your student records</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.84rem;margin-bottom:.8rem">'
                    'Your file should include one row per student per semester. '
                    'Download the template below if you are unsure of the format.</p>',
                    unsafe_allow_html=True)

        # Template download
        sample = pd.DataFrame({
            "student_id"     : [100001]*2 + [100002]*2,
            "name"           : ["Alice Mensah"]*2 + ["Kofi Asante"]*2,
            "faculty"        : ["FESAC","FESAC","FBA","FBA"],
            "gender"         : ["Female","Female","Male","Male"],
            "semester"       : ["2021_S2","2022_S1","2021_S2","2022_S1"],
            "semester_gpa"   : [2.8, 2.5, 1.3, 0.9],
            "avg_attendance" : [4.0, 3.5, 2.0, 1.5],
            "avg_total_mark" : [65.0, 60.0, 42.0, 35.0],
            "avg_ca_score"   : [28.0, 25.0, 17.0, 14.0],
            "avg_exam_score" : [37.0, 35.0, 25.0, 21.0],
            "total_credits"  : [18, 18, 21, 21],
            "num_courses"    : [6, 6, 7, 7],
        })
        buf = io.StringIO()
        sample.to_csv(buf, index=False)

        dl1, dl2 = st.columns([1,3])
        with dl1:
            st.download_button("📥 Download template",
                               buf.getvalue(), "student_template.csv",
                               "text/csv", use_container_width=True)

        # ── Faculty scope reminder ─────────────────────────────────────────
        if faculty:
            fac_color = FAC_COLOR.get(faculty, "#60a5fa")
            st.markdown(f"""
            <div style="background:{_rgba(fac_color,.1)};
                        border:1px solid {_rgba(fac_color,.4)};
                        border-left:4px solid {fac_color};
                        border-radius:0 12px 12px 0;
                        padding:.7rem 1rem;margin-bottom:.8rem;
                        font-size:.86rem;color:#f1f5f9">
              <b style="color:{fac_color}">🏛️ Faculty Filter Active</b><br>
              <span style="color:#94a3b8">
                You are signed in as <b style="color:#f1f5f9">{role}</b>.
                When you upload a CSV, the system will automatically
                extract only <b style="color:{fac_color}">{faculty}</b>
                ({FACULTY_FULL.get(faculty, "")}) students for prediction.
                Students from other faculties in the file will be ignored.
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(96,165,250,.08);
                        border:1px solid rgba(96,165,250,.3);
                        border-left:4px solid #60a5fa;
                        border-radius:0 12px 12px 0;
                        padding:.7rem 1rem;margin-bottom:.8rem;
                        font-size:.86rem;color:#f1f5f9">
              <b style="color:#60a5fa">🎓 All-Faculty Access</b><br>
              <span style="color:#94a3b8">
                You are signed in as <b style="color:#f1f5f9">Dean of Students</b>.
                All students in the uploaded CSV will be included
                across all four faculties.
              </span>
            </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                df_raw = pd.read_csv(uploaded)
                total_rows = len(df_raw)

                # ── Case-insensitive faculty column detection ──────────────
                col_map = {c.lower().strip(): c for c in df_raw.columns}
                fac_col = col_map.get("faculty", None)

                if faculty:
                    # Non-Dean roles: filter to their assigned faculty only
                    if fac_col is None:
                        st.error(
                            f"Your CSV does not have a **faculty** column. "
                            f"Please add a column named `faculty` with values "
                            f"like FESAC, FBA, FEHAS, or PSTM so the system "
                            f"can extract your students correctly.")
                        st.stop()

                    # Normalise faculty column for comparison
                    df_raw[fac_col] = df_raw[fac_col].astype(str).str.strip().str.upper()
                    faculty_upper   = faculty.upper()
                    df_filtered     = df_raw[df_raw[fac_col] == faculty_upper].copy()

                    # Also restore original case for display
                    df_filtered[fac_col] = faculty

                    matched   = len(df_filtered)
                    excluded  = total_rows - matched

                    if matched == 0:
                        st.error(
                            f"No **{faculty}** students were found in this file. "
                            f"The file contains {total_rows:,} rows but none have "
                            f"`faculty = {faculty}`. "
                            f"Please check the file and try again.")
                        st.stop()

                    # Show extraction summary
                    fac_color = FAC_COLOR.get(faculty, "#60a5fa")
                    st.markdown(f"""
                    <div style="background:{_rgba(fac_color,.1)};
                                border:1px solid {_rgba(fac_color,.35)};
                                border-radius:12px;padding:.8rem 1.1rem;
                                margin:.5rem 0;font-size:.86rem">
                      <div style="font-weight:700;color:{fac_color};
                                  margin-bottom:.3rem">
                        ✅ Faculty Extraction Complete
                      </div>
                      <div style="color:#94a3b8;line-height:1.7">
                        📂 File uploaded:
                        <b style="color:#f1f5f9">{total_rows:,}</b> total rows<br>
                        🏛️ {faculty} students found:
                        <b style="color:{fac_color}">{matched:,}</b> rows<br>
                        🚫 Other faculties excluded:
                        <b style="color:#64748b">{excluded:,}</b> rows
                      </div>
                    </div>""", unsafe_allow_html=True)

                    df_raw = df_filtered

                else:
                    # Dean: all faculties — but show a breakdown
                    if fac_col:
                        df_raw[fac_col] = df_raw[fac_col].astype(str).str.strip()
                        fac_counts = df_raw[fac_col].value_counts().to_dict()
                        count_lines = "  ".join(
                            f"<b style='color:{FAC_COLOR.get(k,'#60a5fa')}'>"
                            f"{k}</b>: {v:,}"
                            for k, v in sorted(fac_counts.items()))
                        st.markdown(f"""
                        <div style="background:rgba(96,165,250,.08);
                                    border:1px solid rgba(96,165,250,.3);
                                    border-radius:12px;padding:.8rem 1.1rem;
                                    margin:.5rem 0;font-size:.86rem">
                          <div style="font-weight:700;color:#60a5fa;
                                      margin-bottom:.3rem">
                            ✅ All Faculties Loaded
                          </div>
                          <div style="color:#94a3b8;line-height:1.7">
                            📂 Total rows: <b style="color:#f1f5f9">{total_rows:,}</b><br>
                            🏛️ Faculty breakdown: {count_lines}
                          </div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.info(f"File loaded: {total_rows:,} rows. "
                                "No faculty column detected — "
                                "all rows will be included.")

                with st.spinner(f"Analysing {len(df_raw):,} records... "
                                "This usually takes a few seconds."):
                    df_result = run_batch_pipeline(df_raw)

                if len(df_result) == 0:
                    st.warning(
                        "No results could be generated. "
                        "Make sure each student has at least 2 rows "
                        "in the file so the system can calculate "
                        "their GPA trend.")
                else:
                    st.session_state["df"] = df_result
                    st.rerun()

            except Exception as e:
                st.error(f"There was a problem reading your file: {e}")

        # Required columns note
        with st.expander("What columns does my CSV file need?", key="csv_columns_help_expander"):
            st.markdown("""
            Your file must include these columns:

            | Column | What it means |
            |---|---|
            | `student_id` | The student's ID number |
            | `name` | The student's full name |
            | `faculty` | FESAC, FBA, FEHAS, or PSTM |
            | `gender` | Male or Female |
            | `semester` | e.g. 2022_S1 |
            | `semester_gpa` | GPA this semester (0-4) |
            | `avg_attendance` | Attendance score (0-5) |
            | `avg_total_mark` | Average total mark (0-100) |
            | `avg_ca_score` | Average CA score (0-40) |
            | `avg_exam_score` | Average exam score (0-60) |
            | `total_credits` | Total credits this semester |
            | `num_courses` | Number of courses enrolled |

            **Important:** Each student must appear at least twice
            (two semesters) so the system can detect trends.
            """)
        return

    # ══════════════════════════════════════════════════════════════════════
    # STATE 2 — Data loaded, show dashboard
    # ══════════════════════════════════════════════════════════════════════
    n    = len(df)
    n_hr = (df["risk_class"]==2).sum()
    n_mr = (df["risk_class"]==1).sum()
    n_lr = (df["risk_class"]==0).sum()

    # ── Summary cards ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, icon, val, lbl, sub, css_color, glow in [
        (k1, "👥", n,    "Students Analysed", "Full cohort",         "#60a5fa", "rgba(96,165,250,.35)"),
        (k2, "🔴", n_hr, "High Risk",         "Need immediate attention", "#ef4444", "rgba(239,68,68,.35)"),
        (k3, "🟡", n_mr, "Medium Risk",       "Need monitoring",          "#f59e0b", "rgba(245,158,11,.35)"),
        (k4, "🟢", n_lr, "Low Risk",          "On track",                 "#22c55e", "rgba(34,197,94,.35)"),
    ]:
        with col:
            st.markdown(f"""
            <div class="sum-card" style="--c:{css_color};--cglow:{glow}">
              <div class="sum-icon">{icon}</div>
              <div class="sum-val">{val:,}</div>
              <div class="sum-lbl">{lbl}</div>
              <div class="sum-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([1, 1.8])
    with ch1:
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.3rem">'
                    'Risk breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(n_lr, n_mr, n_hr),
                use_container_width=True,
                config={"displayModeBar": False},
                key="donut_risk_breakdown")
    with ch2:
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.3rem">'
                    'Risk by faculty</p>', unsafe_allow_html=True)
        st.plotly_chart(faculty_bar(df),
                use_container_width=True,
                config={"displayModeBar": False},
                key="faculty_bar_main")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS — Faculty comparison, gender analysis, trends, graduation
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("📊 View Detailed Analytics (Faculty Comparison, Gender, Trends, Graduation)",
                     expanded=False, key="analytics_expander"):

        # ── Component 1: Faculty Performance Comparison ────────────────────
        st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">🏛️ Faculty Performance Comparison</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.6rem">'
                    'Compare average GPA and risk levels across all four faculties '
                    'to identify which departments need the most support.</p>',
                    unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'Average GPA by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_faculty_gpa(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="faculty_gpa")
        with fc2:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'Risk Level Breakdown by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_faculty_risk_pct(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="faculty_risk_pct")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 2: Gender Performance Analysis ────────────────────────
        st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">👥 Male vs Female Performance</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.6rem">'
                    'Compare academic performance and risk distribution '
                    'between male and female students across faculties.</p>',
                    unsafe_allow_html=True)

        gc1, gc2 = st.columns([1.6, 1])
        with gc1:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'Average GPA by Gender per Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gender_gpa(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="gender_gpa")
        with gc2:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'Risk Split by Gender</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gender_risk_split(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="gender_risk_split")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 3: Semester Trend Lines ───────────────────────────────
        st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">📈 Performance Trends Over Time</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.6rem">'
                    'See how each faculty\'s GPA and risk levels have changed '
                    'across semesters. Requires data with at least 2 semesters '
                    'per faculty.</p>', unsafe_allow_html=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'GPA Trend by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gpa_trend(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="gpa_trend")
        with tc2:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.2rem">'
                        'High Risk Count Trend by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_risk_trend(df),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="risk_trend")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 4: Graduation Projection Summary ──────────────────────
        grad_df = graduation_summary(df)
        if len(grad_df) > 0:
            st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                        'margin:.5rem 0">🎓 Graduation Classification Summary '
                        '(Level 300 &amp; 400)</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.6rem">'
                        'Projected final classification for final-year students '
                        'based on current CGPA and performance trend.</p>',
                        unsafe_allow_html=True)

            # Summary counts by classification
            class_counts = grad_df["Classification"].value_counts()
            cls_cols = st.columns(min(len(class_counts), 6))
            for col, (cls, count) in zip(cls_cols, class_counts.items()):
                with col:
                    st.markdown(f"""
                    <div style="background:#131a26;border:1px solid #2a3344;
                                border-radius:10px;padding:.7rem;text-align:center">
                      <div style="font-size:1.4rem">{cls.split()[0]}</div>
                      <div style="font-size:1.3rem;font-weight:700;color:#f1f5f9">
                        {count}</div>
                      <div style="font-size:.7rem;color:#94a3b8">
                        {' '.join(cls.split()[1:])}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(grad_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background:#131a26;border:1px dashed #2a3344;
                        border-radius:10px;padding:1rem;text-align:center;
                        color:#94a3b8;font-size:.85rem">
              🎓 No graduation projections available.<br>
              <span style="font-size:.78rem">
                Add a <code>level</code> column (Level 300 / Level 400) and
                <code>cumulative_gpa</code>, <code>completed_credits</code>,
                <code>programme_credits</code> columns to your CSV to enable
                graduation classification predictions.
              </span>
            </div>""", unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 5: Model Transparency ────────────────────────────────
        st.markdown(
            '<div style="height:1px;background:linear-gradient(90deg,' +
            'transparent,#2a3344,transparent);margin:.5rem 0 1.2rem"></div>',
            unsafe_allow_html=True)
        st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;' +
                    'margin:.5rem 0">🤖 How the AI Makes Decisions</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.8rem">' +
                    'This section shows what the AI model learned from training data — ' +
                    'which features drive predictions most, and whether the model ' +
                    'treats all student groups fairly.</p>',
                    unsafe_allow_html=True)

        tr1, tr2 = st.columns(2)

        with tr1:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.3rem">' +
                        'What the model weighs most heavily (SHAP importance)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_shap_importance(),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="shap_importance_bottom")
            st.markdown(
                '<div style="background:#131a26;border:1px solid #2a3344;' +
                'border-left:3px solid #60a5fa;border-radius:0 8px 8px 0;' +
                'padding:.65rem .9rem;font-size:.81rem;color:#94a3b8;line-height:1.6">' +
                '<b style="color:#f1f5f9">What this means:</b> The bars show how much ' +
                'each factor influences the model prediction. A student overall ' +
                'marks and exam performance have the biggest impact. ' +
                'Longer bars = stronger influence on the risk classification.</div>',
                unsafe_allow_html=True)

        with tr2:
            st.markdown('<p style="color:#94a3b8;font-size:.8rem;margin-bottom:.3rem">' +
                        'Fairness audit — model accuracy across student groups</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(chart_fairness_audit(),
                            use_container_width=True,
                            config={"displayModeBar": False},
                            key="fairness_audit_bottom")
            st.markdown(
                '<div style="background:#131a26;border:1px solid #2a3344;' +
                'border-left:3px solid #22c55e;border-radius:0 8px 8px 0;' +
                'padding:.65rem .9rem;font-size:.81rem;color:#94a3b8;line-height:1.6">' +
                '<b style="color:#f1f5f9">What this means:</b> Every student group ' +
                '(by gender and faculty) has a fairness score above the minimum ' +
                'threshold of 0.45. The model does not systematically ' +
                'disadvantage any group.</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Model performance KPIs
        mk1, mk2, mk3, mk4 = st.columns(4)
        for mcol, mval, mlbl, msub, mclr in [
            (mk1, f"{MF1:.4f}", "Model Accuracy (F1)", "Test set performance", "#60a5fa"),
            (mk2, f"{ECE_SCORE:.4f}", "Calibration (ECE)", "Lower = more reliable", "#22c55e"),
            (mk3, "All pass ✅", "Fairness Audit", "8 groups checked", "#22c55e"),
            (mk4, "16", "Features Used", "No student PII exposed", "#a78bfa"),
        ]:
            with mcol:
                st.markdown(
                    f'<div style="background:#131a26;border:1px solid #2a3344;' +
                    f'border-top:3px solid {mclr};border-radius:12px;' +
                    f'padding:.9rem 1rem;text-align:center">' +
                    f'<div style="font-size:1.4rem;font-weight:800;' +
                    f'color:{mclr}">{mval}</div>' +
                    f'<div style="font-size:.73rem;color:#f1f5f9;font-weight:600;' +
                    f'text-transform:uppercase;letter-spacing:.06em;' +
                    f'margin:.25rem 0">{mlbl}</div>' +
                    f'<div style="font-size:.72rem;color:#64748b">{msub}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action row: search + filter + download ─────────────────────────────
    st.markdown('<p style="color:#f1f5f9;font-size:.95rem;font-weight:600;'
                'margin-bottom:.5rem">Find a student</p>',
                unsafe_allow_html=True)

    a1, a2, a3 = st.columns([3, 1.5, 1.5])
    with a1:
        search = st.text_input("search", placeholder="Search by name or student ID...",
                               label_visibility="collapsed")
    with a2:
        risk_filter = st.selectbox("Filter by risk",
                                   ["All Students","High Risk","Medium Risk","Low Risk"],
                                   label_visibility="collapsed")
    with a3:
        if st.button("Clear & Upload New File", use_container_width=True):
            st.session_state["df"] = None
            st.rerun()

    # ── Download all results ────────────────────────────────────────────────
    show_cols = [c for c in ["student_id","name","faculty","gender",
                              "semester","semester_gpa","risk_label",
                              "prob_low","prob_med","prob_high"]
                 if c in df.columns]
    dl_df = df[show_cols].sort_values(
        "risk_label",
        key=lambda s: s.map({"High Risk":0,"Medium Risk":1,"Low Risk":2}))
    st.download_button(
        "📥 Download all predictions (spreadsheet)",
        data=dl_df.to_csv(index=False),
        file_name=f"predictions_{datetime.date.today()}.csv",
        mime="text/csv",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filter logic ───────────────────────────────────────────────────────
    display_df = df.copy()
    if search.strip():
        q = search.strip().lower()
        mask = pd.Series([False]*len(display_df), index=display_df.index)
        if "student_id" in display_df.columns:
            mask |= display_df["student_id"].astype(str).str.lower().str.contains(q)
        if "name" in display_df.columns:
            mask |= display_df["name"].astype(str).str.lower().str.contains(q)
        display_df = display_df[mask]

    if risk_filter != "All Students":
        display_df = display_df[display_df["risk_label"]==risk_filter]

    # Sort: High Risk first
    display_df = display_df.sort_values(
        "risk_class", ascending=False).reset_index(drop=True)

    # ── Student count ──────────────────────────────────────────────────────
    total_shown = len(display_df)
    st.markdown(f'<p style="color:#94a3b8;font-size:.82rem;margin-bottom:.6rem">'
                f'Showing {total_shown:,} student{"s" if total_shown!=1 else ""}'
                f'{" — " + risk_filter if risk_filter!="All Students" else ""}'
                f'</p>', unsafe_allow_html=True)

    if total_shown == 0:
        st.info("No students match your search. Try a different name or ID.")
        return

    # ── Student cards ──────────────────────────────────────────────────────
    for idx, row in display_df.iterrows():
        risk_class = int(row.get("risk_class", 0))
        rc         = RISK_COLOR[risk_class]
        rbg        = RISK_BG[risk_class]

        sid  = str(row.get("student_id",""))
        name = str(row.get("name","Unknown Student"))
        fac  = str(row.get("faculty",""))
        sem  = str(row.get("semester",""))
        gpa  = row.get("semester_gpa", 0)
        gend = str(row.get("gender",""))

        label     = RISK_LABEL[risk_class]
        prob_high = row.get("prob_high", 0)

        # Card header (always visible)
        expander_label = (
            f"{RISK_ICON[risk_class]}  {name}  ({sid})"
            f"  ·  {fac}  ·  GPA: {gpa:.2f}  ·  {label}"
        )

        uid = str(row.get("student_id","")) + "_" + str(row.get("semester","")) + "_" + str(idx)
        with st.expander(expander_label, expanded=(risk_class == 2), key=f"exp_{uid}"):
            student_detail(row.to_dict(), uid=uid)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if "auth" not in st.session_state:
        st.session_state["auth"] = False
    if "df" not in st.session_state:
        st.session_state["df"] = None

    if not st.session_state["auth"]:
        login_page()
    else:
        main_dashboard(st.session_state["role"])


if __name__ == "__main__":
    main()
