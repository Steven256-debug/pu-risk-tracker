"""
Pentecost University - AI-Powered Academic Performance Tracker
app.py  v5.0
Authors: Steven Asante-Poku Jnr & Frank Amoah | 2025
Supervisor: Mr Harry Attieku-Boateng

Role Architecture:
  Academic Advisor (per faculty) - predict individual students in their faculty
  HOD FESAC / FBA / FEHAS / PSTM  - faculty predictions + faculty analytics
  Dean of Students                 - all faculties, all predictions, all analytics
"""

import streamlit as st
import pickle, json, os, io, datetime
import numpy  as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title = "PU Academic Performance Tracker",
    page_icon  = "🎓",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS  — must match notebook Cell 6 exactly
# ══════════════════════════════════════════════════════════════════════════
FACULTIES = ["FESAC", "FBA", "FEHAS", "PSTM"]
SEMESTERS = ["2019_S1","2019_S2","2020_S1","2020_S2",
             "2021_S1","2021_S2","2022_S1","2022_S2"]

RISK_MAP   = {0:"Low Risk",   1:"Medium Risk",  2:"High Risk"}
RISK_EMOJI = {0:"🟢",          1:"🟡",            2:"🔴"}
RISK_FG    = {0:"#3fb950",    1:"#f0883e",      2:"#f85149"}
RISK_BG    = {0:"#0d2b1a",    1:"#2b1f0a",      2:"#2b0d0d"}
RISK_CSS   = {0:"risk-low",   1:"risk-medium",  2:"risk-high"}

FACULTY_FULL = {
    "FESAC": "Faculty of Engineering & Applied Sciences",
    "FBA"  : "Faculty of Business Administration",
    "FEHAS": "Faculty of Education, Humanities & Applied Sciences",
    "PSTM" : "Pentecost School of Theology & Ministry",
}
FAC_COLOR = {
    "FESAC": "#58a6ff",
    "FBA"  : "#f0883e",
    "FEHAS": "#3fb950",
    "PSTM" : "#bc8cff",
}

FEATURE_COLS = [
    "avg_attendance","avg_total_mark","avg_ca_score","avg_exam_score",
    "total_credits","num_courses","gender_enc","semester_index",
    "prev_gpa","gpa_trend","consec_fails","trend_x_fail",
    "fac_FESAC","fac_FBA","fac_FEHAS","fac_PSTM",
]

# ══════════════════════════════════════════════════════════════════════════
# ROLE DEFINITIONS
# Each role maps to:
#   faculty  : which faculty data they can see (None = all)
#   tabs     : which tabs they have access to
#   label    : display name
# ══════════════════════════════════════════════════════════════════════════
ROLES = {
    "Academic Advisor (FESAC)": {
        "faculty": "FESAC", "tabs": ["predict"],
        "icon": "👨‍🏫", "pwd_key": "ADVISOR_FESAC_PASSWORD",
        "default_pwd": "advisor_fesac_2025",
    },
    "Academic Advisor (FBA)": {
        "faculty": "FBA",   "tabs": ["predict"],
        "icon": "👨‍🏫", "pwd_key": "ADVISOR_FBA_PASSWORD",
        "default_pwd": "advisor_fba_2025",
    },
    "Academic Advisor (FEHAS)": {
        "faculty": "FEHAS", "tabs": ["predict"],
        "icon": "👨‍🏫", "pwd_key": "ADVISOR_FEHAS_PASSWORD",
        "default_pwd": "advisor_fehas_2025",
    },
    "Academic Advisor (PSTM)": {
        "faculty": "PSTM",  "tabs": ["predict"],
        "icon": "👨‍🏫", "pwd_key": "ADVISOR_PSTM_PASSWORD",
        "default_pwd": "advisor_pstm_2025",
    },
    "HOD — FESAC": {
        "faculty": "FESAC", "tabs": ["predict","analytics"],
        "icon": "🏛️", "pwd_key": "HOD_FESAC_PASSWORD",
        "default_pwd": "hod_fesac_2025",
    },
    "HOD — FBA": {
        "faculty": "FBA",   "tabs": ["predict","analytics"],
        "icon": "🏛️", "pwd_key": "HOD_FBA_PASSWORD",
        "default_pwd": "hod_fba_2025",
    },
    "HOD — FEHAS": {
        "faculty": "FEHAS", "tabs": ["predict","analytics"],
        "icon": "🏛️", "pwd_key": "HOD_FEHAS_PASSWORD",
        "default_pwd": "hod_fehas_2025",
    },
    "HOD — PSTM": {
        "faculty": "PSTM",  "tabs": ["predict","analytics"],
        "icon": "🏛️", "pwd_key": "HOD_PSTM_PASSWORD",
        "default_pwd": "hod_pstm_2025",
    },
    "Dean of Students": {
        "faculty": None,    "tabs": ["predict","analytics","batch"],
        "icon": "🎓", "pwd_key": "DEAN_PASSWORD",
        "default_pwd": "dean_2025",
    },
}


# ══════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background:#0d1117;}
.stApp{background:#0d1117;}
.block-container{padding:1.5rem 2rem !important;}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0d1117 0%,#161b22 100%) !important;
    border-right:1px solid #30363d !important;}
[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
[data-testid="stSidebar"] hr{border-color:#30363d !important;}

/* Header */
.pu-header{background:linear-gradient(135deg,#0d1117,#161b22,#1c2128);
    border:1px solid #30363d;border-left:4px solid #58a6ff;
    padding:1.2rem 1.8rem;border-radius:12px;margin-bottom:1.5rem;
    display:flex;align-items:center;gap:1rem;}
.pu-header h1{font-size:1.35rem;font-weight:700;color:#e6edf3 !important;
    margin:0;letter-spacing:-.02em;}
.pu-header .sub{color:#58a6ff;font-size:.77rem;font-weight:500;
    letter-spacing:.08em;text-transform:uppercase;margin-top:.15rem;}

/* Section title */
.sec-title{font-size:.95rem;font-weight:600;color:#e6edf3;
    border-left:3px solid #58a6ff;padding-left:.8rem;
    margin:1.6rem 0 .6rem;}

/* ML Insight box — replaces generic chart titles */
.ml-insight{background:#161b22;border:1px solid #30363d;
    border-left:3px solid #58a6ff;border-radius:0 8px 8px 0;
    padding:.7rem 1rem;margin:.4rem 0 1rem;font-size:.83rem;
    color:#8b949e;line-height:1.5;}
.ml-insight b{color:#c9d1d9;}

/* Risk badges */
.risk-badge{display:inline-flex;align-items:center;gap:.4rem;
    padding:.3rem 1rem;border-radius:20px;font-weight:600;
    font-size:.85rem;border:1px solid;}
.risk-low{background:#0d2b1a;color:#3fb950;border-color:#3fb950;}
.risk-medium{background:#2b1f0a;color:#f0883e;border-color:#f0883e;}
.risk-high{background:#2b0d0d;color:#f85149;border-color:#f85149;}

/* Alert boxes */
.alert{border-radius:8px;padding:.7rem 1rem;margin:.4rem 0;
    font-size:.84rem;border-left:4px solid;}
.alert-red{background:#2b0d0d;border-color:#f85149;color:#ff9a9a;}
.alert-amber{background:#2b1f0a;border-color:#f0883e;color:#ffc97a;}
.alert-green{background:#0d2b1a;border-color:#3fb950;color:#7ee787;}
.alert-blue{background:#0d1f2b;border-color:#58a6ff;color:#79c0ff;}

/* Recommendation card */
.rec-card{background:#161b22;border:1px solid #30363d;
    border-radius:10px;padding:.9rem 1.1rem;margin:.45rem 0;}

/* KPI card */
.kpi-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:1.1rem 1.3rem;position:relative;overflow:hidden;
    transition:transform .15s;}
.kpi-card:hover{transform:translateY(-2px);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;
    height:3px;background:var(--accent,#58a6ff);border-radius:12px 12px 0 0;}
.kpi-card.green{--accent:#3fb950;}.kpi-card.amber{--accent:#f0883e;}
.kpi-card.red{--accent:#f85149;}.kpi-card.blue{--accent:#58a6ff;}
.kpi-val{font-size:1.9rem;font-weight:700;color:#e6edf3;line-height:1;}
.kpi-lbl{font-size:.71rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:.07em;margin-top:.25rem;}
.kpi-sub{font-size:.77rem;color:#484f58;margin-top:.3rem;}

/* Score bar */
.score-row{margin:.45rem 0;}
.score-label{font-size:.77rem;color:#8b949e;margin-bottom:.15rem;
    display:flex;justify-content:space-between;}
.score-bar-bg{background:#21262d;border-radius:6px;height:7px;overflow:hidden;}
.score-bar-fill{height:100%;border-radius:6px;}

/* Faculty tag */
.fac-tag{display:inline-flex;align-items:center;gap:.35rem;
    padding:.2rem .7rem;border-radius:12px;font-size:.78rem;
    font-weight:600;border:1px solid;}

/* Buttons */
.stButton>button{background:#238636 !important;color:white !important;
    border:1px solid #2ea043 !important;border-radius:8px !important;
    font-weight:600 !important;transition:all .15s !important;}
.stButton>button:hover{background:#2ea043 !important;
    transform:translateY(-1px) !important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#161b22 !important;
    border-bottom:1px solid #30363d !important;
    border-radius:8px 8px 0 0;padding:0 .5rem;}
.stTabs [data-baseweb="tab"]{background:transparent !important;
    color:#8b949e !important;font-weight:500 !important;
    padding:.7rem 1.2rem !important;
    border-bottom:2px solid transparent !important;}
.stTabs [aria-selected="true"]{color:#e6edf3 !important;
    font-weight:600 !important;border-bottom-color:#58a6ff !important;}

/* Inputs */
.stSelectbox>div>div,.stNumberInput input,.stTextInput input{
    background:#161b22 !important;color:#c9d1d9 !important;
    border-color:#30363d !important;border-radius:8px !important;}
.stSelectbox label,.stNumberInput label,.stTextInput label,
.stSlider label{color:#8b949e !important;font-size:.82rem !important;}

/* Expander */
.streamlit-expanderHeader{background:#161b22 !important;
    border-color:#30363d !important;color:#c9d1d9 !important;
    border-radius:8px !important;}

::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0d1117;}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SECRETS
# ══════════════════════════════════════════════════════════════════════════
def _secret(key, fallback=""):
    try:
        v = st.secrets.get(key, "")
        if v: return v
    except Exception:
        pass
    return os.environ.get(key, fallback)


# ══════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════
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
if _fcols:
    FEATURE_COLS = _fcols

Q33         = thresholds.get("Q33",        2.0)        if thresholds else 2.0
Q66         = thresholds.get("Q66",        3.0)        if thresholds else 3.0
MODEL_NAME  = thresholds.get("best_model", "LightGBM") if thresholds else "LightGBM"
BASELINE_F1 = thresholds.get("macro_f1",   0.6383)     if thresholds else 0.6383
TOP_SHAP    = thresholds.get("top_shap_features", [
    "avg_total_mark","avg_exam_score","gpa_trend","avg_ca_score",
    "consec_fails","trend_x_fail","fac_FEHAS","prev_gpa",
    "fac_FESAC","gender_enc",
]) if thresholds else []


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════════════════
def predict_one(feat_dict: dict):
    """Scale and predict. Returns (class_int, probability_array)."""
    row    = np.array([float(feat_dict.get(c, 0.0)) for c in FEATURE_COLS]).reshape(1, -1)
    row_sc = scaler.transform(row)
    probs  = model.predict_proba(row_sc)[0]
    return int(np.argmax(probs)), probs


def build_features(avg_attendance, avg_total_mark, avg_ca_score, avg_exam_score,
                   total_credits, num_courses, gender, semester_index,
                   prev_gpa, gpa_trend, consec_fails, faculty) -> dict:
    """Mirror Cell 6 feature engineering exactly."""
    return {
        "avg_attendance" : avg_attendance,
        "avg_total_mark" : avg_total_mark,
        "avg_ca_score"   : avg_ca_score,
        "avg_exam_score" : avg_exam_score,
        "total_credits"  : total_credits,
        "num_courses"    : num_courses,
        "gender_enc"     : int(gender == "Female"),
        "semester_index" : semester_index,
        "prev_gpa"       : prev_gpa,
        "gpa_trend"      : gpa_trend,
        "consec_fails"   : consec_fails,
        "trend_x_fail"   : gpa_trend * consec_fails,
        "fac_FESAC"      : int(faculty == "FESAC"),
        "fac_FBA"        : int(faculty == "FBA"),
        "fac_FEHAS"      : int(faculty == "FEHAS"),
        "fac_PSTM"       : int(faculty == "PSTM"),
    }


def apply_pipeline_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Replicate notebook Cell 6 feature engineering for batch input."""
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
    return df.dropna(subset=["prev_gpa"]).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════
def generate_recommendations(feats: dict, pred: int) -> list:
    """
    Rule-based academic recommendations derived from feature values.
    Each rule maps a measurable threshold to a concrete advisory action.
    """
    recs = []
    att   = feats.get("avg_attendance",  0)
    exam  = feats.get("avg_exam_score",  0)
    ca    = feats.get("avg_ca_score",    0)
    pgpa  = feats.get("prev_gpa",        0)
    trend = feats.get("gpa_trend",       0)
    cf    = feats.get("consec_fails",    0)
    tc    = feats.get("total_credits",  18)

    # Attendance
    if att < 3.0:
        recs.append(("🔴", "Critical: Attendance",
            f"Attendance score {att:.1f}/5 is below the minimum threshold of 3.0. "
            "The student risks course disqualification. Arrange an urgent welfare "
            "check and create a formal attendance improvement plan this week."))
    elif att < 3.5:
        recs.append(("🟡", "Attendance Below Recommended",
            f"Attendance {att:.1f}/5 falls short of the recommended 3.5. "
            "Remind the student that consistent presence directly improves "
            "assessment performance and GPA."))

    # Exam performance
    if exam / 60 < 0.4:
        recs.append(("🔴", "Exam Performance — Urgent",
            f"Exam score {exam:.1f}/60 ({exam/60*100:.0f}%) is below 40%. "
            "Refer the student immediately to the Academic Support Centre for "
            "exam technique coaching and past paper practice."))
    elif exam / 60 < 0.5:
        recs.append(("🟡", "Exam Performance — Below Average",
            f"Exam score {exam:.1f}/60 ({exam/60*100:.0f}%) is below 50%. "
            "Encourage the student to join faculty-run study groups and seek "
            "clarification on weak topics from lecturers."))

    # CA
    if ca / 40 < 0.5:
        recs.append(("🟡", "Continuous Assessment — Below Average",
            f"CA score {ca:.1f}/40 ({ca/40*100:.0f}%) is below 50%. "
            "Confirm that all assignments have been submitted. "
            "Review grading with the relevant lecturer."))

    # GPA trend
    if trend < -0.3:
        recs.append(("🔴", "Significant GPA Decline",
            f"GPA has dropped {abs(trend):.2f} points compared to the previous semester. "
            "Investigate root cause — this may indicate personal difficulties, "
            "course overload, or mental health concerns."))
    elif trend < 0:
        recs.append(("🟡", "GPA Declining",
            f"GPA trend of {trend:+.2f} indicates a gradual decline. "
            "Schedule a check-in meeting and monitor closely next semester."))

    # Consecutive failures
    if cf >= 2:
        recs.append(("🔴", "Consecutive Academic Failures",
            f"{int(cf)} consecutive semesters with GPA below 1.5. "
            "The student may be approaching academic probation. "
            "Refer to counselling services and consider a formal academic "
            "recovery agreement with the faculty."))
    elif cf == 1:
        recs.append(("🟡", "Recent Semester Below Threshold",
            "GPA was below 1.5 last semester. Review current course selections "
            "and provide targeted academic support before the situation escalates."))

    # Course load
    if tc > 22:
        recs.append(("🟡", "Course Load — Potentially Excessive",
            f"Student is enrolled in {tc} credits this semester, above the "
            "recommended maximum of 21. Consider advising a reduction next "
            "semester to prevent burnout affecting GPA."))

    # Previous GPA position
    if pgpa < Q33:
        recs.append(("🔴", "Previous GPA in High-Risk Zone",
            f"Previous semester GPA of {pgpa:.2f} is below the High Risk "
            f"threshold of {Q33:.1f}. A structured academic recovery plan "
            "with clearly defined milestones is required."))

    # All clear
    if pred == 0 and not recs:
        recs.append(("🟢", "No Immediate Concerns",
            "No risk signals detected in the current semester data. "
            "The student appears on track. Continue standard monitoring "
            "through semester-end reviews."))

    return recs[:6]


# ══════════════════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════════════════
def get_alerts(feats: dict, pred: int, probs: np.ndarray) -> list:
    """Generate real-time threshold-based alerts for the prediction result."""
    alerts = []

    if probs[2] >= 0.75:
        alerts.append(("red", "CRITICAL — High Risk Probability",
            f"The model assigns a {probs[2]:.0%} probability of High Risk. "
            "This student requires immediate advisory intervention."))
    elif probs[2] >= 0.50:
        alerts.append(("amber", "WARNING — Elevated Risk",
            f"High Risk probability is {probs[2]:.0%}. "
            "Schedule a follow-up meeting within the week."))

    if feats.get("avg_attendance", 5) < 3.0:
        alerts.append(("red", "Attendance Below Minimum",
            f"Score {feats['avg_attendance']:.1f}/5 is below the 3.0 threshold."))

    if feats.get("avg_exam_score", 60) / 60 < 0.40:
        alerts.append(("red", "Exam Failure Risk",
            "Exam score below 40% of maximum — high probability of failing examined courses."))

    if feats.get("consec_fails", 0) >= 2:
        alerts.append(("red", "Consecutive Semester Failures",
            f"{int(feats['consec_fails'])} consecutive semesters with GPA below 1.5."))

    if feats.get("gpa_trend", 0) < -0.3:
        alerts.append(("amber", "Sharp GPA Decline",
            f"GPA dropped {abs(feats['gpa_trend']):.2f} points vs last semester."))

    if pred == 0 and not alerts:
        alerts.append(("green", "Student is On Track",
            "No threshold breaches detected. Model predicts Low Risk for next semester."))

    return alerts


# ══════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════
def generate_pdf(sid, sname, faculty, semester, pred, probs, feats, recs):
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
        BLUE  = colors.HexColor("#003087")
        GREY  = colors.HexColor("#666666")
        LIGHT = colors.HexColor("#f8f9fa")
        GRID  = colors.HexColor("#dee2e6")
        story = []

        def st_par(txt, size=10, bold=False, color=colors.HexColor("#333333"),
                   space_before=0, space_after=8):
            return Paragraph(txt, ParagraphStyle("p", parent=ss["Normal"],
                fontSize=size, fontName="Helvetica-Bold" if bold else "Helvetica",
                textColor=color, spaceBefore=space_before, spaceAfter=space_after))

        def section_h(txt):
            return Paragraph(txt, ParagraphStyle("h", parent=ss["Heading2"],
                fontSize=13, textColor=BLUE,
                fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))

        def info_table(rows):
            t = Table(rows, colWidths=[3.5*cm,5.5*cm,3.5*cm,5.5*cm])
            t.setStyle(TableStyle([
                ("FONTNAME", (0,0),(0,-1), "Helvetica-Bold"),
                ("FONTNAME", (2,0),(2,-1), "Helvetica-Bold"),
                ("TEXTCOLOR",(0,0),(0,-1), BLUE),
                ("TEXTCOLOR",(2,0),(2,-1), BLUE),
                ("FONTSIZE", (0,0),(-1,-1), 9),
                ("PADDING",  (0,0),(-1,-1), 7),
                ("GRID",     (0,0),(-1,-1), 0.5, GRID),
                ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,colors.white]),
            ]))
            return t

        # Cover
        story += [
            st_par("PENTECOST UNIVERSITY", 18, True, BLUE, space_after=4),
            st_par("AI Academic Performance Tracker — Risk Assessment Report",
                   10, False, GREY, space_after=16),
            HRFlowable(width="100%", thickness=2, color=BLUE),
            Spacer(1, 12),
        ]

        # Student info
        story.append(section_h("Student Information"))
        story.append(info_table([
            ["Student ID", sid or "N/A", "Name", sname or "N/A"],
            ["Faculty",    FACULTY_FULL.get(faculty, faculty),
             "Semester",   semester],
            ["Report Date",
             datetime.date.today().strftime("%d %B %Y"),
             "ML Model",   MODEL_NAME],
        ]))
        story.append(Spacer(1, 14))

        # Prediction
        story.append(section_h("Prediction Result"))
        rc = colors.HexColor(
            "#27ae60" if pred==0 else "#f39c12" if pred==1 else "#e74c3c")
        res = [
            ["Risk Level",   RISK_MAP[pred], "Confidence", f"{probs[pred]:.1%}"],
            ["Low Risk P",   f"{probs[0]:.1%}",
             "Medium Risk P",f"{probs[1]:.1%}"],
            ["High Risk P",  f"{probs[2]:.1%}", "Model F1", f"{BASELINE_F1:.4f}"],
        ]
        rt = Table(res, colWidths=[3.5*cm,5.5*cm,3.5*cm,5.5*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (1,0),(1,0), rc),
            ("TEXTCOLOR",  (1,0),(1,0), colors.white),
            ("FONTNAME",   (1,0),(1,0), "Helvetica-Bold"),
            ("FONTNAME",   (0,0),(0,-1),"Helvetica-Bold"),
            ("FONTNAME",   (2,0),(2,-1),"Helvetica-Bold"),
            ("TEXTCOLOR",  (0,0),(0,-1), BLUE),
            ("TEXTCOLOR",  (2,0),(2,-1), BLUE),
            ("FONTSIZE",   (0,0),(-1,-1), 9),
            ("PADDING",    (0,0),(-1,-1), 7),
            ("GRID",       (0,0),(-1,-1), 0.5, GRID),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[LIGHT,colors.white]),
        ]))
        story += [rt, Spacer(1, 14)]

        # Metrics
        story.append(section_h("Performance Metrics"))
        mets_data = [
            ["Metric","Value","Status"],
            ["Avg Total Mark",
             f"{feats.get('avg_total_mark',0):.1f} / 100",
             "Pass" if feats.get("avg_total_mark",0)>=50 else "Fail"],
            ["Avg CA Score",
             f"{feats.get('avg_ca_score',0):.1f} / 40",
             "Pass" if feats.get("avg_ca_score",0)>=20 else "Fail"],
            ["Avg Exam Score",
             f"{feats.get('avg_exam_score',0):.1f} / 60",
             "Pass" if feats.get("avg_exam_score",0)>=30 else "Fail"],
            ["Attendance Score",
             f"{feats.get('avg_attendance',0):.1f} / 5",
             "Pass" if feats.get("avg_attendance",0)>=3.0 else "Fail"],
            ["Previous GPA",
             f"{feats.get('prev_gpa',0):.2f} / 4.0",
             "Good" if feats.get("prev_gpa",0)>=Q66 else
             "Warning" if feats.get("prev_gpa",0)>=Q33 else "Critical"],
            ["GPA Trend",
             f"{feats.get('gpa_trend',0):+.2f}",
             "Improving" if feats.get("gpa_trend",0)>0 else "Declining"],
        ]
        mt = Table(mets_data, colWidths=[5*cm,5*cm,8*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), BLUE),
            ("TEXTCOLOR", (0,0),(-1,0), colors.white),
            ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",  (0,0),(-1,-1), 9),
            ("PADDING",   (0,0),(-1,-1), 7),
            ("GRID",      (0,0),(-1,-1), 0.5, GRID),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[LIGHT,colors.white]),
        ]))
        story += [mt, Spacer(1, 14)]

        # Recommendations
        story.append(section_h("Academic Recommendations"))
        for i,(icon,title,text) in enumerate(recs, 1):
            story.append(st_par(f"<b>{i}. {title}</b>",
                                color=colors.HexColor("#003087")))
            story.append(st_par(text))
            story.append(Spacer(1, 5))

        # Footer
        story += [
            Spacer(1, 20),
            HRFlowable(width="100%", thickness=1,
                       color=colors.HexColor("#dee2e6")),
            Spacer(1, 6),
            st_par(
                f"Pentecost University | AI Academic Performance Tracker | "
                f"Ghana Data Protection Act 2012 (Act 843) | "
                f"Generated: {datetime.datetime.now().strftime('%d %B %Y %H:%M')}",
                size=7.5, color=GREY, space_after=0),
        ]

        doc.build(story)
        return buf.getvalue()
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════
# PLOTLY CHARTS  — dark theme, each with an ML interpretation
# ══════════════════════════════════════════════════════════════════════════
BG   = "#0d1117"
PLOT = "#161b22"
GRID = "#21262d"
TICK = "#8b949e"
TEXT = "#c9d1d9"
cfg  = {"displayModeBar": False}


def _base(title="", h=360):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=PLOT,
        font=dict(family="Inter", color=TEXT),
        title=dict(text=title, font=dict(size=13, color="#e6edf3")),
        height=h, margin=dict(l=50,r=30,t=50,b=50),
        xaxis=dict(gridcolor=GRID, linecolor="#30363d",
                   tickcolor=TICK, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, linecolor="#30363d",
                   tickcolor=TICK, zerolinecolor=GRID),
        legend=dict(bgcolor=PLOT, bordercolor="#30363d",
                    borderwidth=1, font=dict(color=TEXT)),
    )


def chart_gauge(prob_high: float) -> go.Figure:
    color = ("#3fb950" if prob_high < 0.33 else
             "#f0883e" if prob_high < 0.66 else "#f85149")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_high * 100, 1),
        number=dict(suffix="%", font=dict(size=44, color=color, family="Inter")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor=TICK,
                      tickfont=dict(size=10,color=TICK), dtick=25),
            bar=dict(color=color, thickness=0.28),
            bgcolor=PLOT, bordercolor="#30363d", borderwidth=1,
            steps=[dict(range=[0,33],  color="#0d2b1a"),
                   dict(range=[33,66], color="#2b1f0a"),
                   dict(range=[66,100],color="#2b0d0d")],
        ),
        title=dict(text="HIGH RISK PROBABILITY",
                   font=dict(size=11,color=TICK)),
    ))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      height=240, margin=dict(l=25,r=25,t=25,b=5),
                      font=dict(family="Inter",color=TEXT))
    return fig


def chart_prob_bar(probs: np.ndarray) -> go.Figure:
    labels = ["Low Risk","Medium Risk","High Risk"]
    colors = ["#3fb950","#f0883e","#f85149"]
    fig = go.Figure()
    for lbl,p,c in zip(labels, probs, colors):
        fig.add_trace(go.Bar(
            name=lbl, x=[p*100], y=[""], orientation="h",
            marker_color=c, marker_opacity=0.9,
            text=[f"{p:.1%}"], textposition="inside",
            textfont=dict(color="white",size=12,family="Inter"),
            hovertemplate=f"<b>{lbl}</b>: {p:.1%}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", paper_bgcolor=BG, plot_bgcolor=PLOT,
        height=70, margin=dict(l=0,r=0,t=0,b=0),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="left", x=0,
                    font=dict(size=10,color=TICK),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, showticklabels=False, range=[0,100]),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    return fig


def chart_feature_contributions(feats: dict) -> go.Figure:
    """
    Approximates SHAP-style feature contributions by measuring
    how far each key feature deviates from the safe-zone midpoint.
    This is an interpretability chart, not a true SHAP computation.
    """
    items = [
        ("Total Mark",   feats.get("avg_total_mark", 0)/100),
        ("Exam Score",   feats.get("avg_exam_score", 0)/60),
        ("CA Score",     feats.get("avg_ca_score",   0)/40),
        ("Attendance",   feats.get("avg_attendance", 0)/5),
        ("Previous GPA", feats.get("prev_gpa",       0)/4),
        ("GPA Trend",    (feats.get("gpa_trend",     0)+2)/4),
    ]
    names     = [i[0] for i in items]
    contribs  = [round(max(0, 0.65 - i[1]) * 100, 1) for i in items]
    bar_colors= ["#3fb950" if c<10 else "#f0883e" if c<25 else "#f85149"
                 for c in contribs]
    fig = go.Figure(go.Bar(
        x=names, y=contribs, marker_color=bar_colors, marker_line_width=0,
        text=[str(c) for c in contribs], textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate="<b>%{x}</b><br>Risk Contribution: %{y:.1f}<extra></extra>",
    ))
    layout = _base("Feature Risk Contributions", h=300)
    layout.update(showlegend=False,
                  yaxis_title="Risk Score (higher = more concern)",
                  xaxis=dict(tickangle=-15))
    fig.update_layout(**layout)
    return fig


def chart_radar(feats: dict, fac_avgs: dict) -> go.Figure:
    cats  = ["Attendance","Total Mark","CA Score","Exam Score","Prev GPA"]
    keys  = ["avg_attendance","avg_total_mark","avg_ca_score",
             "avg_exam_score","prev_gpa"]
    maxes = [5, 100, 40, 60, 4]
    sv    = [feats.get(k,0)/m*100 for k,m in zip(keys,maxes)]
    fv    = [fac_avgs.get(k,60)/m*100 for k,m in zip(keys,maxes)]
    cats2 = cats+[cats[0]]; sv2=sv+[sv[0]]; fv2=fv+[fv[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=fv2, theta=cats2, fill="toself",
        name="Faculty Average",
        fillcolor="rgba(88,166,255,0.1)",
        line=dict(color="#58a6ff",width=2)))
    fig.add_trace(go.Scatterpolar(r=sv2, theta=cats2, fill="toself",
        name="This Student",
        fillcolor="rgba(248,81,73,0.15)",
        line=dict(color="#f85149",width=2.5)))
    fig.update_layout(
        polar=dict(
            bgcolor=PLOT,
            radialaxis=dict(visible=True, range=[0,100], gridcolor=GRID,
                            tickfont=dict(size=9,color=TICK)),
            angularaxis=dict(gridcolor=GRID, tickfont=dict(size=10,color=TEXT)),
        ),
        paper_bgcolor=BG, height=320,
        margin=dict(l=55,r=55,t=45,b=45),
        legend=dict(bgcolor=PLOT, bordercolor="#30363d", borderwidth=1,
                    font=dict(color=TEXT, size=10)),
        title=dict(text="Student vs Faculty Average",
                   font=dict(size=13,color="#e6edf3")),
    )
    return fig


def chart_donut(n_lr, n_mr, n_hr) -> go.Figure:
    total = max(n_lr+n_mr+n_hr, 1)
    fig = go.Figure(go.Pie(
        labels=["Low Risk","Medium Risk","High Risk"],
        values=[n_lr, n_mr, n_hr], hole=0.60,
        marker=dict(colors=["#3fb950","#f0883e","#f85149"],
                    line=dict(color=BG, width=3)),
        textinfo="label+percent",
        textfont=dict(size=11, color="#e6edf3"),
        pull=[0.02, 0.02, 0.06],
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='color:{TICK};font-size:11px'>Students</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#e6edf3"), align="center")
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=340,
        margin=dict(l=20,r=20,t=50,b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                    xanchor="center", x=0.5,
                    font=dict(size=10, color=TICK),
                    bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Risk Level Distribution",
                   font=dict(size=13, color="#e6edf3")),
    )
    return fig


def chart_gpa_trend(df: pd.DataFrame, fac_filter=None) -> go.Figure:
    if "semester" not in df.columns or "semester_gpa" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_base("GPA Trend by Semester"))
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5,
                           showarrow=False,
                           font=dict(color="#4a6a8a", size=13))
        return fig

    faculties  = [fac_filter] if fac_filter else FACULTIES
    sem_ord    = {s:i for i,s in enumerate(SEMESTERS)}
    fig = go.Figure()

    for fac in faculties:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        if len(sub)==0: continue
        trend = (sub.groupby("semester")["semester_gpa"].mean()
                 .reset_index()
                 .assign(order=lambda x: x["semester"].map(sem_ord))
                 .sort_values("order"))
        c = FAC_COLOR.get(fac, "#58a6ff")
        r,g,b = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["semester_gpa"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color=BG, width=1.5)),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.04)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>Avg GPA: %{{y:.2f}}<extra></extra>",
        ))

    fig.add_hrect(y0=0,   y1=Q33, fillcolor="rgba(248,81,73,0.06)",  line_width=0)
    fig.add_hrect(y0=Q33, y1=Q66, fillcolor="rgba(240,136,62,0.04)", line_width=0)
    fig.add_hline(y=Q33, line_dash="dash", line_color="#f85149",
                  line_width=1, opacity=0.6,
                  annotation_text=f"High Risk threshold ({Q33})",
                  annotation_font_color="#f85149")
    fig.add_hline(y=Q66, line_dash="dash", line_color="#f0883e",
                  line_width=1, opacity=0.6,
                  annotation_text=f"Medium Risk threshold ({Q66})",
                  annotation_font_color="#f0883e",
                  annotation_position="bottom right")

    layout = _base("Average GPA per Semester", h=360)
    layout.update(
        xaxis=dict(title="Semester", tickangle=-30, gridcolor=GRID),
        yaxis=dict(title="Average GPA", range=[0, 4.2]),
        hovermode="x unified")
    fig.update_layout(**layout)
    return fig


def chart_risk_trend(df: pd.DataFrame, fac_filter=None) -> go.Figure:
    if "semester" not in df.columns or "risk_label" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_base("High-Risk Count per Semester"))
        fig.add_annotation(text="No data loaded", x=0.5, y=0.5,
                           showarrow=False,
                           font=dict(color="#4a6a8a", size=13))
        return fig

    faculties = [fac_filter] if fac_filter else FACULTIES
    sem_ord   = {s:i for i,s in enumerate(SEMESTERS)}
    fig = go.Figure()

    for fac in faculties:
        sub = df[(df["faculty"]==fac) & (df["risk_label"]==2)] \
              if "faculty" in df.columns else pd.DataFrame()
        if len(sub)==0: continue
        trend = (sub.groupby("semester").size().reset_index(name="count")
                 .assign(order=lambda x: x["semester"].map(sem_ord))
                 .sort_values("order"))
        c = FAC_COLOR.get(fac, "#58a6ff")
        r,g,b = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["count"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color=BG, width=1.5)),
            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.06)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>High Risk: %{{y}}<extra></extra>",
        ))

    layout = _base("High-Risk Students per Semester", h=360)
    layout.update(
        xaxis=dict(title="Semester", tickangle=-30, gridcolor=GRID),
        yaxis=dict(title="High Risk Count"),
        hovermode="x unified")
    fig.update_layout(**layout)
    return fig


def chart_stacked(df: pd.DataFrame, fac_filter=None) -> go.Figure:
    faculties = [fac_filter] if fac_filter else FACULTIES
    fac_data  = {}
    for fac in faculties:
        sub    = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        counts = sub["risk_label"].value_counts() if len(sub) else {}
        fac_data[fac] = [counts.get(k,0) for k in [0,1,2]]

    fig = go.Figure()
    for k,lbl,color in [(0,"Low Risk","#3fb950"),
                        (1,"Medium Risk","#f0883e"),
                        (2,"High Risk","#f85149")]:
        vals = [fac_data[f][k] for f in faculties]
        fig.add_trace(go.Bar(
            name=lbl, x=faculties, y=vals,
            marker_color=color, marker_line_width=0,
            text=[str(v) if v>0 else "" for v in vals],
            textposition="inside",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"<b>%{{x}}</b><br>{lbl}: %{{y}}<extra></extra>",
        ))
    layout = _base("Risk Distribution by Faculty", h=340)
    layout.update(
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    fig.update_layout(**layout)
    return fig


def chart_heatmap(df: pd.DataFrame, fac_filter=None) -> go.Figure:
    faculties = [fac_filter] if fac_filter else FACULTIES
    if "semester" not in df.columns or "faculty" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_base("High Risk % — Faculty x Semester", h=280))
        return fig

    sems = [s for s in SEMESTERS if s in df["semester"].unique()]
    if not sems:
        fig = go.Figure()
        fig.update_layout(**_base("", h=280))
        return fig

    data = []
    for fac in faculties:
        row = []
        for sem in sems:
            sub = df[(df["faculty"]==fac) & (df["semester"]==sem)]
            row.append((sub["risk_label"]==2).sum()/len(sub)*100
                       if len(sub)>0 else 0)
        data.append(row)

    fig = go.Figure(go.Heatmap(
        z=data, x=sems, y=faculties,
        colorscale=[[0,"#0d2b1a"],[0.33,"#2b2b0a"],
                    [0.66,"#2b1a0a"],[1,"#8b0000"]],
        text=[[f"{v:.0f}%" for v in row] for row in data],
        texttemplate="%{text}",
        textfont=dict(size=11, family="Inter"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>High Risk: %{z:.1f}%<extra></extra>",
        colorbar=dict(
            title=dict(text="High Risk %", font=dict(color=TICK,size=11)),
            tickfont=dict(color=TICK), bgcolor=PLOT, bordercolor="#30363d"),
        xgap=2, ygap=2,
    ))
    layout = _base("High Risk % by Faculty and Semester", h=int(120+80*len(faculties)))
    layout.update(xaxis=dict(side="bottom", tickangle=-30))
    fig.update_layout(**layout)
    return fig


# ══════════════════════════════════════════════════════════════════════════
# UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════
def sec(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)


def ml_insight(text):
    """Displays an ML interpretation box below every chart."""
    st.markdown(f'<div class="ml-insight">{text}</div>', unsafe_allow_html=True)


def kpi(val, label, css="", sub=None):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-card {css}">
      <div class="kpi-val">{val}</div>
      <div class="kpi-lbl">{label}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def alert_box(level, title, message):
    st.markdown(f"""
    <div class="alert alert-{level}">
      <strong>{title}</strong><br>
      <span style="opacity:.88">{message}</span>
    </div>""", unsafe_allow_html=True)


def score_bar(label, value, maximum, color="#58a6ff"):
    pct = min(value/maximum*100, 100)
    st.markdown(f"""
    <div class="score-row">
      <div class="score-label">
        <span>{label}</span>
        <span style="color:{color};font-weight:600">{value:.1f} / {maximum}</span>
      </div>
      <div class="score-bar-bg">
        <div class="score-bar-fill"
             style="width:{pct:.0f}%;background:{color}"></div>
      </div>
    </div>""", unsafe_allow_html=True)


def faculty_tag(fac):
    c = FAC_COLOR.get(fac, "#58a6ff")
    st.markdown(
        f'<span class="fac-tag" style="background:{c}22;color:{c};'
        f'border-color:{c}55">{fac} — {FACULTY_FULL.get(fac,"")}</span>',
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2.5rem 0 1.5rem">
          <div style="font-size:3.5rem;margin-bottom:.5rem">🎓</div>
          <h1 style="font-size:1.7rem;font-weight:700;color:#e6edf3;
                     margin:.3rem 0;letter-spacing:-.03em">
              Pentecost University</h1>
          <p style="color:#58a6ff;font-size:.8rem;font-weight:600;
                    letter-spacing:.12em;text-transform:uppercase;margin:0">
              AI Academic Performance Tracker</p>
          <div style="height:2px;background:linear-gradient(90deg,
                      transparent,#58a6ff,transparent);
                      margin:1rem auto;width:60%"></div>
        </div>""", unsafe_allow_html=True)

        if not artefacts_ok:
            st.error("Model artefacts not found. Upload `best_model.pkl`, "
                     "`scaler.pkl`, `feature_cols.json`, `thresholds.json` "
                     "to your GitHub repository.")

        st.markdown('<div style="background:#161b22;border:1px solid #30363d;'
                    'border-radius:12px;padding:1.5rem 1.8rem">',
                    unsafe_allow_html=True)

        with st.form("lf"):
            st.markdown('<p style="color:#8b949e;font-size:.78rem;'
                        'text-transform:uppercase;letter-spacing:.1em;'
                        'margin-bottom:.4rem">Select your role and sign in</p>',
                        unsafe_allow_html=True)
            role = st.selectbox("Role", list(ROLES.keys()),
                                label_visibility="collapsed")
            pwd  = st.text_input("Password", type="password",
                                 placeholder="Password",
                                 label_visibility="collapsed")
            ok   = st.form_submit_button("Sign In",
                                         use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if ok:
            role_cfg = ROLES[role]
            correct  = _secret(role_cfg["pwd_key"], role_cfg["default_pwd"])
            if pwd == correct:
                st.session_state.update({
                    "auth": True, "role": role,
                    "batch_df": None, "last_pred": None,
                })
                st.rerun()
            else:
                st.error("Incorrect password.")

        st.markdown("""
        <p style="text-align:center;color:#484f58;
                  font-size:.7rem;margin-top:1.2rem">
          2025 Pentecost University — Ghana DPA 2012 (Act 843) Compliant
        </p>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
def render_sidebar(role: str):
    rcfg    = ROLES[role]
    faculty = rcfg["faculty"]

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:1rem 0 .7rem;text-align:center">
          <div style="font-size:2rem">🎓</div>
          <div style="font-weight:700;font-size:.95rem;color:#e6edf3;margin:.2rem 0">
              Pentecost University</div>
          <div style="font-size:.7rem;color:#58a6ff;text-transform:uppercase;
                      letter-spacing:.1em">Academic Tracker</div>
        </div>
        <hr style="border-color:#30363d;margin:.4rem 0">
        """, unsafe_allow_html=True)

        icon = rcfg["icon"]
        st.markdown(f"**{icon} {role}**")

        if faculty:
            fc = FAC_COLOR.get(faculty,"#58a6ff")
            st.markdown(f"""
            <div style="background:{fc}18;border:1px solid {fc}44;
                        border-radius:7px;padding:.4rem .7rem;
                        margin:.4rem 0;font-size:.8rem;color:{fc}">
              Viewing: <b>{faculty}</b><br>
              <span style="opacity:.7;font-size:.73rem">
                  {FACULTY_FULL.get(faculty,"")}
              </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#58a6ff18;border:1px solid #58a6ff44;
                        border-radius:7px;padding:.4rem .7rem;margin:.4rem 0;
                        font-size:.8rem;color:#58a6ff">
              All Faculties Access
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#30363d">', unsafe_allow_html=True)
        st.markdown('<p style="color:#8b949e;font-size:.73rem;'
                    'text-transform:uppercase;letter-spacing:.08em">'
                    'Risk Thresholds (Fixed)</p>', unsafe_allow_html=True)
        for emoji, label, rng, color in [
            ("🔴","High Risk",   f"GPA < {Q33:.1f}",        "#f85149"),
            ("🟡","Medium Risk", f"GPA {Q33:.1f}–{Q66:.1f}", "#f0883e"),
            ("🟢","Low Risk",    f"GPA >= {Q66:.1f}",        "#3fb950"),
        ]:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;
                        border-left:3px solid {color};border-radius:0 6px 6px 0;
                        padding:.35rem .65rem;margin:.2rem 0;font-size:.79rem">
              {emoji} <b style="color:#e6edf3">{label}</b>
              <span style="color:#484f58;font-size:.71rem;float:right">{rng}</span>
            </div>""", unsafe_allow_html=True)

        batch = st.session_state.get("batch_df")
        if batch is not None:
            st.markdown('<hr style="border-color:#30363d">',
                        unsafe_allow_html=True)
            visible = batch[batch["faculty"]==faculty] \
                      if faculty else batch
            st.markdown(f"""
            <div style="background:#0d2b1a;border:1px solid #3fb950;
                        border-radius:8px;padding:.5rem .8rem;font-size:.8rem">
              <span style="color:#3fb950">Dataset loaded</span><br>
              <span style="color:#8b949e">{len(visible):,} records visible</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#30363d">',
                    unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            for k in ["auth","role","batch_df","last_pred"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("""
        <div style="font-size:.64rem;color:#484f58;
                    text-align:center;margin-top:.7rem">
          Ghana DPA 2012 Compliant
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB — INDIVIDUAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════
def tab_predict(role: str):
    rcfg    = ROLES[role]
    faculty = rcfg["faculty"]   # None means Dean (all faculties)

    sec("Individual Student Risk Prediction")
    st.markdown(
        '<p style="color:#8b949e;font-size:.85rem;margin-bottom:.8rem">'
        'Enter a student\'s current semester data. The LightGBM model will '
        'output a risk classification and probability for each risk class.</p>',
        unsafe_allow_html=True)

    if faculty:
        faculty_tag(faculty)

    with st.form("pf"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<p style="color:#58a6ff;font-size:.78rem;font-weight:600;'
                        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">'
                        'Student Identity</p>', unsafe_allow_html=True)
            sid   = st.text_input("Student ID", placeholder="e.g. PUIT/23120001")
            sname = st.text_input("Student Name (optional)")

            if faculty:
                fac = faculty
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #30363d;
                            border-radius:8px;padding:.5rem .8rem;
                            font-size:.83rem;color:{FAC_COLOR.get(fac,'#58a6ff')};
                            margin-top:.2rem">
                  Faculty locked: <b>{fac}</b>
                </div>""", unsafe_allow_html=True)
            else:
                fac = st.selectbox("Faculty", FACULTIES,
                    format_func=lambda x: f"{x} — {FACULTY_FULL[x][:28]}...")

            gen   = st.selectbox("Gender", ["Male","Female"])
            sem_i = st.selectbox("Current Semester", range(len(SEMESTERS)),
                                 format_func=lambda i: SEMESTERS[i], index=7)

        with c2:
            st.markdown('<p style="color:#58a6ff;font-size:.78rem;font-weight:600;'
                        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">'
                        'Current Semester Scores</p>', unsafe_allow_html=True)
            atm  = st.slider("Avg Total Mark (0–100)",  0.0, 100.0, 55.0, 0.5)
            aca  = st.slider("Avg CA Score   (0–40)",   0.0,  40.0, 22.0, 0.5)
            aex  = st.slider("Avg Exam Score (0–60)",   0.0,  60.0, 33.0, 0.5)
            aatt = st.slider("Attendance Score (0–5)",  0.0,   5.0,  3.5, 0.1)

        with c3:
            st.markdown('<p style="color:#58a6ff;font-size:.78rem;font-weight:600;'
                        'text-transform:uppercase;letter-spacing:.08em;margin-bottom:.3rem">'
                        'Enrolment & History</p>', unsafe_allow_html=True)
            tc   = st.number_input("Total Credits",                    1, 30, 18)
            nc   = st.number_input("Courses Enrolled",                 1, 12,  6)
            pgpa = st.number_input("Previous Semester GPA (0–4)",  0.0, 4.0, 1.8, 0.01)
            gtr  = st.number_input("GPA Trend (current − previous)", -4.0, 4.0, 0.0, 0.01,
                                   help="Positive = improving, Negative = declining")
            cf   = st.number_input("Consecutive Sems GPA < 1.5",    0, 8, 0)

        go = st.form_submit_button("Run Prediction", use_container_width=True)

    if go:
        if not artefacts_ok:
            st.error("Model artefacts not found."); return
        feats       = build_features(aatt, atm, aca, aex, tc, nc,
                                     gen, sem_i, pgpa, gtr, cf, fac)
        pred, probs = predict_one(feats)
        st.session_state["last_pred"] = dict(
            sid=sid, sname=sname, fac=fac,
            semester=SEMESTERS[sem_i], gen=gen,
            pred=pred, probs=probs.tolist(), feats=feats,
        )

    if not st.session_state.get("last_pred"):
        return

    lp    = st.session_state["last_pred"]
    pred  = lp["pred"]
    probs = np.array(lp["probs"])
    feats = lp["feats"]
    fac   = lp["fac"]
    recs  = generate_recommendations(feats, pred)
    alerts= get_alerts(feats, pred, probs)

    # ── Prediction result banner ───────────────────────────────────────────
    rc = ["risk-low","risk-medium","risk-high"][pred]
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;
                border-radius:12px;padding:1.1rem 1.5rem;margin:.8rem 0;
                border-left:4px solid {RISK_FG[pred]}">
      <div style="display:flex;align-items:center;gap:1rem">
        <div style="font-size:2.6rem">{RISK_EMOJI[pred]}</div>
        <div style="flex:1">
          <div style="font-size:1.2rem;font-weight:700;color:#e6edf3">
            {lp['sname'] or 'Student'}{f" ({lp['sid']})" if lp['sid'] else ""}
          </div>
          <div style="color:#8b949e;font-size:.8rem;margin-top:.15rem">
            {FACULTY_FULL.get(fac,fac)} &nbsp;·&nbsp;
            {lp['gen']} &nbsp;·&nbsp; {lp['semester']}
          </div>
        </div>
        <span class="risk-badge {rc}">
          {RISK_EMOJI[pred]} {RISK_MAP[pred]}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Gauge + Radar ──────────────────────────────────────────────────────
    v1, v2 = st.columns([1, 1.35])
    with v1:
        st.plotly_chart(chart_gauge(probs[2]),
                        use_container_width=True, config=cfg)
        st.plotly_chart(chart_prob_bar(probs),
                        use_container_width=True, config=cfg)
        ml_insight(
            "The <b>gauge</b> shows the model's probability that this student "
            "will have GPA below 2.0 next semester. "
            "The <b>bar</b> shows the full probability split across all three risk classes. "
            "These are outputs from the LightGBM softmax function — "
            "not a rule, but a learned statistical estimate.")

    with v2:
        bdf = st.session_state.get("batch_df")
        if bdf is not None and "faculty" in bdf.columns:
            fd = bdf[bdf["faculty"]==fac]
            fac_avgs = {k: fd[k].mean() if k in fd.columns else 60
                        for k in ["avg_attendance","avg_total_mark",
                                  "avg_ca_score","avg_exam_score","prev_gpa"]}
        else:
            fac_avgs = {"avg_attendance":3.8,"avg_total_mark":62,
                        "avg_ca_score":25,"avg_exam_score":37,"prev_gpa":2.0}
        st.plotly_chart(chart_radar(feats, fac_avgs),
                        use_container_width=True, config=cfg)
        ml_insight(
            "The <b>radar chart</b> compares this student's normalised scores "
            "(as a percentage of the maximum possible) against the faculty average. "
            "Axes pointing inward on the red polygon indicate where the student "
            "falls below the faculty average — these are the most likely "
            "contributors to the predicted risk classification.")

    # ── Performance breakdown ──────────────────────────────────────────────
    sec("Performance Breakdown")
    b1,b2,b3,b4 = st.columns(4)
    for col,lbl,key,mx,clr in [
        (b1,"Total Mark","avg_total_mark",100,"#58a6ff"),
        (b2,"CA Score",  "avg_ca_score",   40,"#3fb950"),
        (b3,"Exam Score","avg_exam_score",  60,"#f0883e"),
        (b4,"Attendance","avg_attendance",   5,"#bc8cff"),
    ]:
        with col:
            score_bar(lbl, feats.get(key,0), mx, clr)

    # ── Feature contribution chart ─────────────────────────────────────────
    sec("Feature Risk Contributions")
    st.plotly_chart(chart_feature_contributions(feats),
                    use_container_width=True, config=cfg)
    ml_insight(
        "This chart approximates each feature's contribution to the risk prediction. "
        "It measures how far each feature deviates from the expected safe-zone range "
        "(calibrated at 0.65 of the normalised maximum). "
        "<b>Red bars</b> indicate features with the highest contribution to the "
        "High Risk classification. This is an interpretability aid — "
        "the model's actual decision is based on 16 features including "
        "interaction terms (trend_x_fail) and faculty encodings.")

    # ── Alerts ─────────────────────────────────────────────────────────────
    sec("Real-Time Threshold Alerts")
    for level, title, msg in alerts:
        alert_box(level, title, msg)
    ml_insight(
        "Alerts are generated by <b>rule-based thresholds</b> applied directly "
        "to the input feature values — they are separate from the ML model. "
        "They serve as a human-readable summary of the most critical signals "
        "in the data, independently of what the model predicted.")

    # ── Recommendations ────────────────────────────────────────────────────
    sec("Academic Recommendations")
    for i, (icon, title, text) in enumerate(recs, 1):
        st.markdown(f"""
        <div class="rec-card">
          <b style="color:#e6edf3">{icon} {i}. {title}</b><br>
          <span style="color:#8b949e;font-size:.85rem">{text}</span>
        </div>""", unsafe_allow_html=True)
    ml_insight(
        "Recommendations are generated by a <b>rule-based decision engine</b>, "
        "not the ML model. Each rule maps a specific feature threshold "
        "(e.g. attendance below 3.0, exam score below 40%) to a predefined "
        "advisory action. This makes them auditable, transparent, and safe "
        "to act on — the advisor understands exactly why each recommendation "
        "was triggered.")

    # ── PDF download ───────────────────────────────────────────────────────
    sec("Export Assessment Report")
    d1, d2 = st.columns(2)
    with d1:
        with st.spinner("Generating PDF..."):
            pdf = generate_pdf(lp["sid"],lp["sname"],fac,lp["semester"],
                               pred,probs,feats,recs)
        if pdf:
            st.download_button(
                "Download PDF Report", data=pdf,
                file_name=f"risk_{lp['sid'] or 'student'}_{datetime.date.today()}.pdf",
                mime="application/pdf", use_container_width=True)
        else:
            st.info("Install `reportlab` to enable PDF export.")
    with d2:
        fd2 = pd.DataFrame([feats])
        fd2["risk_label"] = RISK_MAP[pred]
        fd2["prob_high"]  = round(float(probs[2]), 4)
        st.download_button(
            "Download Feature Data (CSV)",
            data=fd2.to_csv(index=False),
            file_name=f"features_{lp['sid'] or 'student'}.csv",
            mime="text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB — FACULTY ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
def tab_analytics(role: str):
    rcfg       = ROLES[role]
    fac_filter = rcfg["faculty"]   # None = Dean sees all

    sec("Faculty Analytics")
    st.markdown(
        '<p style="color:#8b949e;font-size:.85rem;margin-bottom:.8rem">'
        'Analytics are derived from the batch prediction results. '
        'Upload a CSV via the Batch Prediction tab to populate these charts.</p>',
        unsafe_allow_html=True)

    batch_df = st.session_state.get("batch_df")

    if batch_df is None:
        st.info("No batch data loaded. Use the Batch Prediction tab to upload "
                "a CSV and run predictions — the results will appear here.")
        # Show empty chart placeholders
        empty = pd.DataFrame(columns=["faculty","semester","semester_gpa",
                                       "risk_label"])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_gpa_trend(empty, fac_filter),
                            use_container_width=True, config=cfg)
        with c2:
            st.plotly_chart(chart_risk_trend(empty, fac_filter),
                            use_container_width=True, config=cfg)
        return

    # Filter data to role's faculty
    df = batch_df[batch_df["faculty"]==fac_filter].copy() \
         if fac_filter else batch_df.copy()

    if len(df) == 0:
        st.warning(f"No data available for {fac_filter}.")
        return

    # Faculty header
    if fac_filter:
        faculty_tag(fac_filter)
        st.markdown("<br>", unsafe_allow_html=True)

    n    = len(df)
    n_hr = (df["risk_label"]==2).sum() if "risk_label" in df.columns else 0
    n_mr = (df["risk_label"]==1).sum() if "risk_label" in df.columns else 0
    n_lr = (df["risk_label"]==0).sum() if "risk_label" in df.columns else 0

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    with k1: kpi(f"{n:,}",    "Students",   "blue",
                 f"in {fac_filter or 'all faculties'}")
    with k2: kpi(f"{n_hr:,}", "High Risk",  "red",
                 f"{n_hr/n*100:.1f}% of cohort")
    with k3: kpi(f"{n_mr:,}", "Medium Risk","amber",
                 f"{n_mr/n*100:.1f}% of cohort")
    with k4: kpi(f"{n_lr:,}", "Low Risk",   "green",
                 f"{n_lr/n*100:.1f}% of cohort")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk distribution ──────────────────────────────────────────────────
    sec("Risk Distribution")
    rd1, rd2 = st.columns([1, 1.5])
    with rd1:
        st.plotly_chart(chart_donut(n_lr,n_mr,n_hr),
                        use_container_width=True, config=cfg)
    with rd2:
        st.plotly_chart(chart_stacked(df, fac_filter),
                        use_container_width=True, config=cfg)
    ml_insight(
        "The <b>donut chart</b> shows the proportion of students in each risk class "
        "as predicted by the LightGBM model. "
        "The <b>stacked bar</b> breaks this down by faculty, enabling comparison "
        "of risk burden across departments. "
        "A faculty with a tall red segment has a disproportionate number of "
        "High Risk students and may require reallocation of advisory resources.")

    # ── Temporal trends ────────────────────────────────────────────────────
    sec("Semester-on-Semester Trends")
    t1, t2 = st.columns(2)
    with t1:
        st.plotly_chart(chart_gpa_trend(df, fac_filter),
                        use_container_width=True, config=cfg)
    with t2:
        st.plotly_chart(chart_risk_trend(df, fac_filter),
                        use_container_width=True, config=cfg)
    ml_insight(
        "The <b>GPA trend line</b> shows the average GPA per semester. "
        "The coloured risk bands (red = High Risk zone, amber = Medium Risk zone) "
        "show when the faculty average dropped into a risk threshold. "
        "The <b>High-Risk count chart</b> tracks how many students the model "
        "classified as High Risk each semester — a rising line signals that "
        "the model is detecting deteriorating academic conditions.")

    # ── Heatmap ────────────────────────────────────────────────────────────
    sec("Risk Heatmap — Faculty x Semester")
    st.plotly_chart(chart_heatmap(df, fac_filter),
                    use_container_width=True, config=cfg)
    ml_insight(
        "Each cell shows what percentage of students in that "
        "faculty-semester combination were classified as High Risk by the model. "
        "<b>Dark red cells</b> represent periods of acute academic risk. "
        "This helps faculty leadership identify which semesters and which departments "
        "experienced the highest model-predicted risk concentration.")

    # ── Top at-risk students ───────────────────────────────────────────────
    if "prob_high" in df.columns:
        sec("Highest-Risk Students — Top 15")
        show_cols = [c for c in ["student_id","name","faculty","gender",
                                  "semester","semester_gpa","risk_name",
                                  "prob_high"]
                     if c in df.columns]
        top = (df[df["risk_label"]==2][show_cols]
               .sort_values("prob_high", ascending=False).head(15))
        if len(top):
            def _hl(v):
                try:
                    f = float(v)
                    if f>=0.85: return "color:#f85149;font-weight:700"
                    if f>=0.70: return "color:#f0883e;font-weight:600"
                    return ""
                except: return ""
            st.dataframe(
                top.style.applymap(_hl, subset=["prob_high"]),
                use_container_width=True, hide_index=True)
            ml_insight(
                "This table lists High Risk students ranked by their "
                "<b>model-predicted probability</b> of High Risk (prob_high column). "
                "Students at the top of this list should be prioritised for "
                "advisory outreach. "
                "The probability value is the LightGBM model's softmax output "
                "for the High Risk class — a higher value means the model is "
                "more confident in the High Risk classification.")
        else:
            st.info("No High Risk students in the current dataset.")


# ══════════════════════════════════════════════════════════════════════════
# TAB — BATCH PREDICTION  (Dean only)
# ══════════════════════════════════════════════════════════════════════════
def tab_batch():
    sec("Batch Prediction — All Faculties")
    st.markdown(
        '<p style="color:#8b949e;font-size:.85rem;margin-bottom:.8rem">'
        'Upload a CSV of student records. The app will apply the same feature '
        'engineering as the training pipeline and run the LightGBM model across '
        'the entire cohort. Results feed the Faculty Analytics tab.</p>',
        unsafe_allow_html=True)

    with st.expander("Required CSV columns"):
        st.code(", ".join([
            "student_id","faculty","gender","semester","semester_gpa",
            "avg_attendance","avg_total_mark","avg_ca_score",
            "avg_exam_score","total_credits","num_courses"]))
        st.caption(
            "Optional: name, any other columns — retained in output. "
            "Each student needs at least 2 semester rows so that prev_gpa "
            "can be computed by shifting the GPA column within each student group.")

    # Sample template
    sample = pd.DataFrame({
        "student_id" : [100001]*2 + [100002]*2,
        "name"       : ["Alice Mensah"]*2 + ["Kofi Asante"]*2,
        "faculty"    : ["FESAC","FESAC","FBA","FBA"],
        "gender"     : ["Female","Female","Male","Male"],
        "semester"   : ["2021_S2","2022_S1","2021_S2","2022_S1"],
        "semester_gpa": [2.8, 2.5, 1.3, 0.9],
        "avg_attendance": [4.0, 3.5, 2.0, 1.5],
        "avg_total_mark": [65.0, 60.0, 42.0, 35.0],
        "avg_ca_score"  : [28.0, 25.0, 17.0, 14.0],
        "avg_exam_score": [37.0, 35.0, 25.0, 21.0],
        "total_credits" : [18, 18, 21, 21],
        "num_courses"   : [6,  6,  7,  7],
    })
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("Download CSV Template", buf.getvalue(),
                       "pu_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload student records CSV", type=["csv"])
    if not uploaded:
        return

    try:
        df_raw = pd.read_csv(uploaded)
        st.success(f"{len(df_raw):,} rows loaded.")

        with st.spinner("Running feature engineering pipeline and model inference..."):
            df = apply_pipeline_batch(df_raw)

        if len(df) == 0:
            st.error("No rows survived feature engineering. "
                     "Ensure every student has at least 2 semester rows.")
            return

        X     = df[FEATURE_COLS].fillna(0).values
        X_sc  = scaler.transform(X)
        probs = model.predict_proba(X_sc)
        preds = probs.argmax(axis=1)

        df["risk_label"] = preds
        df["risk_name"]  = [RISK_MAP[p] for p in preds]
        df["prob_low"]   = probs[:,0].round(4)
        df["prob_med"]   = probs[:,1].round(4)
        df["prob_high"]  = probs[:,2].round(4)

        st.session_state["batch_df"] = df

        n    = len(df)
        n_hr = (preds==2).sum()
        n_mr = (preds==1).sum()
        n_lr = (preds==0).sum()

        k1,k2,k3,k4 = st.columns(4)
        with k1: kpi(f"{n:,}",    "Records Processed")
        with k2: kpi(f"{n_hr:,}", "High Risk",  "red",   f"{n_hr/n*100:.1f}%")
        with k3: kpi(f"{n_mr:,}", "Medium Risk","amber",  f"{n_mr/n*100:.1f}%")
        with k4: kpi(f"{n_lr:,}", "Low Risk",  "green",  f"{n_lr/n*100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_donut(n_lr,n_mr,n_hr),
                            use_container_width=True, config=cfg)
        with c2:
            st.plotly_chart(chart_stacked(df, None),
                            use_container_width=True, config=cfg)

        ml_insight(
            "The model has been applied to all students using the same "
            "16-feature pipeline from the training notebook. "
            "Each student receives a predicted risk class (0=Low, 1=Medium, 2=High) "
            "and three probability values that sum to 1.0. "
            "These results are now available in the Faculty Analytics tab, "
            "where each HOD will see only their faculty's data.")

        sec("Prediction Results")
        show = [c for c in ["student_id","name","faculty","gender","semester",
                             "semester_gpa","risk_name","prob_high"]
                if c in df.columns]
        df_d = df[show].sort_values(
            "risk_name",
            key=lambda s: s.map({"High Risk":0,"Medium Risk":1,"Low Risk":2}))

        def _col(v):
            if v=="High Risk":   return "color:#f85149;font-weight:700"
            if v=="Medium Risk": return "color:#f0883e;font-weight:600"
            if v=="Low Risk":    return "color:#3fb950"
            return ""

        st.dataframe(
            df_d.style.applymap(_col, subset=["risk_name"]),
            use_container_width=True, height=440)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Full Results CSV",
                data=df_d.to_csv(index=False),
                file_name=f"predictions_{datetime.date.today()}.csv",
                mime="text/csv", use_container_width=True)
        with dl2:
            st.success("Faculty Analytics tab is now populated with these results.")

    except Exception as e:
        st.error(f"Processing error: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    if "auth" not in st.session_state:
        st.session_state["auth"] = False

    if not st.session_state["auth"]:
        login_page()
        return

    role = st.session_state["role"]
    rcfg = ROLES[role]
    tabs = rcfg["tabs"]

    render_sidebar(role)

    fac_display = rcfg["faculty"] or "All Faculties"
    st.markdown(f"""
    <div class="pu-header">
      <div style="font-size:1.9rem">🎓</div>
      <div>
        <h1>Pentecost University — Academic Performance Tracker</h1>
        <div class="sub">
          {rcfg['icon']} {role} &nbsp;·&nbsp;
          {fac_display} &nbsp;·&nbsp;
          {MODEL_NAME} (F1={BASELINE_F1:.4f}) &nbsp;·&nbsp;
          {datetime.date.today().strftime("%d %B %Y")}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    tab_labels = []
    if "predict"   in tabs: tab_labels.append("🔍 Individual Prediction")
    if "analytics" in tabs: tab_labels.append("📊 Faculty Analytics")
    if "batch"     in tabs: tab_labels.append("📂 Batch Prediction")

    tab_objs = st.tabs(tab_labels)
    tab_keys = (["predict"]   * ("predict"   in tabs) +
                ["analytics"] * ("analytics" in tabs) +
                ["batch"]     * ("batch"     in tabs))

    for tab_obj, key in zip(tab_objs, tab_keys):
        with tab_obj:
            if   key == "predict":   tab_predict(role)
            elif key == "analytics": tab_analytics(role)
            elif key == "batch":     tab_batch()


if __name__ == "__main__":
    main()
