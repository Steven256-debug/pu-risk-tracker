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
RISK_COLOR = {0:"#3fb950",     1:"#f0883e",      2:"#f85149"}
RISK_BG    = {0:"#0d2b1a",     1:"#2b1f0a",      2:"#2b0d0d"}
RISK_BORDER= {0:"#3fb950",     1:"#f0883e",      2:"#f85149"}

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
    "FESAC":"#58a6ff","FBA":"#f0883e","FEHAS":"#3fb950","PSTM":"#bc8cff"
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
    (1.50, 1.99, "Third Class",        "#f0883e", "📜"),
    (1.00, 1.49, "Pass",               "#8b949e", "📋"),
    (0.00, 0.99, "Fail",               "#f85149", "❌"),
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html,body,[class*="css"]{font-family:'Inter',sans-serif !important;background:#0d1117;}
.stApp{background:#0d1117;}
.block-container{padding:1.2rem 1.8rem !important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:#111820 !important;
    border-right:1px solid #21262d !important;}
[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
[data-testid="stSidebar"] hr{border-color:#21262d !important;}

/* ── Top banner ── */
.top-banner{background:linear-gradient(90deg,#1F4E79 0%,#2563a8 100%);
    border-radius:14px;padding:1.1rem 1.6rem;
    display:flex;align-items:center;gap:1.2rem;
    margin-bottom:1.4rem;}
.top-banner h1{font-size:1.25rem;font-weight:700;
    color:white !important;margin:0;letter-spacing:-.01em;}
.top-banner .sub{color:#93c5fd;font-size:.78rem;margin-top:.1rem;}

/* ── Step cards ── */
.step-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:1.4rem;text-align:center;}
.step-num{width:36px;height:36px;border-radius:50%;background:#1F4E79;
    color:white;font-size:.9rem;font-weight:700;
    display:flex;align-items:center;justify-content:center;margin:0 auto .7rem;}
.step-title{color:#e6edf3;font-weight:600;font-size:.95rem;margin-bottom:.3rem;}
.step-desc{color:#8b949e;font-size:.82rem;line-height:1.5;}

/* ── Summary cards ── */
.sum-card{background:#161b22;border:1px solid #30363d;
    border-radius:12px;padding:1.2rem 1.4rem;
    border-top:4px solid var(--c);text-align:center;}
.sum-val{font-size:2.4rem;font-weight:700;color:var(--c);line-height:1;}
.sum-lbl{font-size:.8rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:.07em;margin-top:.3rem;}
.sum-sub{font-size:.78rem;color:#484f58;margin-top:.2rem;}

/* ── Search box ── */
.stTextInput input{background:#161b22 !important;
    color:#e6edf3 !important;border:1px solid #30363d !important;
    border-radius:10px !important;font-size:.95rem !important;padding:.6rem 1rem !important;}
.stTextInput label{display:none !important;}

/* ── Student card ── */
.s-card{background:#161b22;border:1px solid #30363d;border-radius:12px;
    padding:1rem 1.2rem;margin:.45rem 0;
    border-left:4px solid var(--rc);
    transition:border-color .15s;}
.s-card:hover{border-color:var(--rc);background:#1c2128;}
.s-name{font-weight:600;color:#e6edf3;font-size:.95rem;}
.s-meta{color:#8b949e;font-size:.79rem;margin-top:.1rem;}
.s-badge{display:inline-flex;align-items:center;gap:.3rem;
    padding:.22rem .75rem;border-radius:20px;font-size:.8rem;
    font-weight:600;border:1px solid var(--rc);
    background:var(--rbg);color:var(--rc);}

/* ── Detail panel ── */
.detail-panel{background:#0d1117;border:1px solid #21262d;
    border-radius:10px;padding:1rem 1.2rem;margin:.5rem 0;}
.detail-row{display:flex;justify-content:space-between;
    padding:.35rem 0;border-bottom:1px solid #21262d;
    font-size:.85rem;}
.detail-row:last-child{border-bottom:none;}
.detail-lbl{color:#8b949e;}
.detail-val{color:#e6edf3;font-weight:500;}

/* ── Rec item ── */
.rec-item{background:#161b22;border-left:3px solid var(--rc);
    border-radius:0 8px 8px 0;padding:.65rem 1rem;margin:.3rem 0;
    font-size:.84rem;color:#c9d1d9;}

/* ── Upload area ── */
.stFileUploader{background:#161b22 !important;border-radius:12px !important;
    border:2px dashed #30363d !important;}

/* ── Buttons ── */
.stButton>button{background:#1F4E79 !important;color:white !important;
    border:none !important;border-radius:8px !important;
    font-weight:600 !important;padding:.45rem 1.2rem !important;
    transition:all .15s !important;}
.stButton>button:hover{background:#2563a8 !important;
    transform:translateY(-1px) !important;}

/* ── Filter chips ── */
.chip{display:inline-flex;align-items:center;gap:.3rem;
    padding:.25rem .75rem;border-radius:20px;font-size:.8rem;
    font-weight:500;cursor:pointer;border:1px solid #30363d;
    background:#161b22;color:#8b949e;margin:.2rem;}

/* ── Grad badge ── */
.grad-badge{display:inline-flex;align-items:center;gap:.4rem;
    padding:.3rem .9rem;border-radius:20px;font-size:.82rem;
    font-weight:600;border:1px solid;}

/* ── Progress bar ── */
.prog-bg{background:#21262d;border-radius:6px;height:8px;overflow:hidden;margin:.3rem 0;}
.prog-fill{height:100%;border-radius:6px;}

/* ── Expander ── */
.streamlit-expanderHeader{background:#161b22 !important;
    border:1px solid #30363d !important;border-radius:8px !important;
    color:#c9d1d9 !important;font-size:.9rem !important;}
.streamlit-expanderContent{background:#111820 !important;
    border:1px solid #21262d !important;border-radius:0 0 8px 8px !important;}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0d1117;}
::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px;}

/* ── Selectbox / multiselect ── */
.stSelectbox>div>div{background:#161b22 !important;
    border-color:#30363d !important;border-radius:8px !important;
    color:#c9d1d9 !important;}
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


def get_grad_class(cgpa: float):
    for low, high, label, color, emoji in GRAD_CLASSES:
        if low <= round(cgpa, 3) <= high:
            return label, color, emoji
    return "Fail", "#f85149", "❌"


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
        BLUE  = colors.HexColor("#1F4E79")
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
        marker=dict(colors=["#3fb950","#f0883e","#f85149"],
                    line=dict(color="#0d1117",width=3)),
        textinfo="label+percent",
        textfont=dict(size=11,color="#e6edf3"),
        pull=[0.02,0.02,0.05],
        hovertemplate="<b>%{label}</b><br>%{value:,} students<br>%{percent}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='color:#8b949e;font-size:11px'>Students</span>",
        x=0.5,y=0.5,showarrow=False,
        font=dict(size=20,color="#e6edf3"),align="center")
    fig.update_layout(
        paper_bgcolor="#0d1117",plot_bgcolor="#0d1117",height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(orientation="h",yanchor="bottom",y=-0.2,
                    xanchor="center",x=0.5,
                    font=dict(size=10,color="#8b949e"),
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
    for k,lbl,c in [(0,"Low Risk","#3fb950"),(1,"Medium Risk","#f0883e"),(2,"High Risk","#f85149")]:
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
        paper_bgcolor="#0d1117",plot_bgcolor="#161b22",
        height=280,margin=dict(l=40,r=10,t=10,b=40),
        font=dict(family="Inter",color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d",tickcolor="#8b949e"),
        yaxis=dict(gridcolor="#21262d",tickcolor="#8b949e"),
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
            clrs.append(FAC_COLOR.get(fac, "#58a6ff"))
    if not faculties:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0d1117", height=240,
            annotations=[dict(text="No GPA data available", x=0.5, y=0.5,
                showarrow=False, font=dict(color="#4a6a8a", size=13))])
        return fig
    fig = go.Figure(go.Bar(
        x=avgs, y=faculties, orientation="h",
        marker_color=clrs, marker_line_width=0,
        text=[f"{v:.2f}" for v in avgs], textposition="outside",
        textfont=dict(size=11, color="#e6edf3"),
        hovertemplate="<b>%{y}</b><br>Avg GPA: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=2.0, line_dash="dash", line_color="#f85149",
                  line_width=1.5, opacity=0.7,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#f85149",
                  annotation_position="top right")
    fig.add_vline(x=3.0, line_dash="dash", line_color="#f0883e",
                  line_width=1.5, opacity=0.7,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f0883e",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=260, margin=dict(l=10, r=60, t=20, b=30),
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(title="Average GPA", range=[0, 4.2],
                   gridcolor="#21262d", tickcolor="#8b949e"),
        yaxis=dict(gridcolor="#21262d", tickcolor="#8b949e"),
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
        fig.update_layout(paper_bgcolor="#0d1117", height=260)
        return fig
    fig = go.Figure()
    for vals, lbl, c in [
        (hr_pcts, "High Risk",   "#f85149"),
        (mr_pcts, "Medium Risk", "#f0883e"),
        (lr_pcts, "Low Risk",    "#3fb950"),
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
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=280, margin=dict(l=40, r=10, t=10, b=40),
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d", tickcolor="#8b949e"),
        yaxis=dict(title="Percentage of Students (%)",
                   gridcolor="#21262d", tickcolor="#8b949e", range=[0, 105]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
    )
    return fig


def chart_gender_gpa(df):
    """Average GPA by gender per faculty — grouped bars."""
    if "gender" not in df.columns or "semester_gpa" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0d1117", height=300,
            annotations=[dict(text="No gender data available",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#4a6a8a", size=13))])
        return fig
    genders = [g for g in ["Female","Male"]
               if g in df["gender"].str.strip().str.title().unique()]
    g_colors = {"Female":"#bc8cff", "Male":"#58a6ff"}
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
            marker_color=g_colors.get(g, "#58a6ff"),
            marker_line_width=0, opacity=0.88,
            text=[f"{v:.2f}" if v > 0 else "" for v in avgs],
            textposition="outside",
            textfont=dict(size=9, color="#e6edf3"),
            hovertemplate=f"<b>%{{x}}</b><br>{g}: %{{y:.2f}}<extra></extra>",
        ))
    fig.add_hline(y=2.0, line_dash="dash", line_color="#f85149",
                  line_width=1.2, opacity=0.6,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#f85149")
    fig.add_hline(y=3.0, line_dash="dash", line_color="#f0883e",
                  line_width=1.2, opacity=0.6,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f0883e",
                  annotation_position="bottom right")
    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=320, margin=dict(l=50, r=20, t=20, b=40),
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(gridcolor="#21262d", tickcolor="#8b949e"),
        yaxis=dict(title="Average GPA", gridcolor="#21262d",
                   tickcolor="#8b949e", range=[0, 4.5]),
        legend=dict(font=dict(size=10, color="#8b949e"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def chart_gender_risk_split(df):
    """Risk split by gender — side-by-side donut charts."""
    if "gender" not in df.columns or "risk_class" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0d1117", height=280)
        return fig
    genders = [g for g in ["Female","Male"]
               if g in df["gender"].str.strip().str.title().unique()]
    clrs = ["#3fb950","#f0883e","#f85149"]
    fig  = go.Figure()
    for i, g in enumerate(genders):
        sub  = df[df["gender"].str.strip().str.title()==g]
        vals = [(sub["risk_class"]==k).sum() for k in [0,1,2]]
        x0   = i * 0.52
        fig.add_trace(go.Pie(
            labels=["Low Risk","Medium Risk","High Risk"],
            values=vals, hole=0.55, name=g,
            domain={"x":[x0, x0+0.45], "y":[0,1]},
            marker=dict(colors=clrs, line=dict(color="#0d1117", width=2)),
            textinfo="percent", textfont=dict(size=10, color="#e6edf3"),
            hovertemplate=f"<b>{g}</b><br>%{{label}}: %{{value:,}}<br>%{{percent}}<extra></extra>",
            title=dict(text=g, font=dict(size=13, color="#e6edf3")),
        ))
    fig.update_layout(
        paper_bgcolor="#0d1117", height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                    xanchor="center", x=0.5,
                    font=dict(size=10, color="#8b949e"),
                    bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    return fig


def chart_gpa_trend(df):
    """Semester-on-semester average GPA trend lines per faculty."""
    if "semester" not in df.columns or "semester_gpa" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0d1117", height=340,
            annotations=[dict(
                text="Upload multi-semester data to see GPA trends",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#4a6a8a", size=13))])
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
        c = FAC_COLOR.get(fac, "#58a6ff")
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["semester_gpa"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color="#0d1117",width=1.5)),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.05)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>Avg GPA: %{{y:.2f}}<extra></extra>",
        ))
        drawn += 1
    if drawn == 0:
        fig.add_annotation(text="Need at least 2 semesters per faculty for trends",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#4a6a8a", size=13))
    fig.add_hrect(y0=0,   y1=2.0, fillcolor="rgba(248,81,73,0.05)",  line_width=0)
    fig.add_hrect(y0=2.0, y1=3.0, fillcolor="rgba(240,136,62,0.03)", line_width=0)
    fig.add_hline(y=2.0, line_dash="dash", line_color="#f85149",
                  line_width=1, opacity=0.5,
                  annotation_text="High Risk (2.0)",
                  annotation_font_color="#f85149")
    fig.add_hline(y=3.0, line_dash="dash", line_color="#f0883e",
                  line_width=1, opacity=0.5,
                  annotation_text="Medium Risk (3.0)",
                  annotation_font_color="#f0883e",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=340, margin=dict(l=50, r=20, t=20, b=50),
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(title="Semester", tickangle=-30,
                   gridcolor="#21262d", tickcolor="#8b949e"),
        yaxis=dict(title="Average GPA", range=[0, 4.2],
                   gridcolor="#21262d", tickcolor="#8b949e"),
        legend=dict(font=dict(size=10, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def chart_risk_trend(df):
    """High Risk count per semester per faculty — trend lines."""
    if "semester" not in df.columns or "risk_class" not in df.columns:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor="#0d1117", height=340,
            annotations=[dict(
                text="Upload multi-semester data to see risk trends",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#4a6a8a", size=13))])
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
        c = FAC_COLOR.get(fac, "#58a6ff")
        r,g,b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
        fig.add_trace(go.Scatter(
            x=trend["semester"], y=trend["count"],
            mode="lines+markers", name=fac,
            line=dict(color=c, width=2.5, shape="spline"),
            marker=dict(size=8, color=c, line=dict(color="#0d1117",width=1.5)),
            fill="tonexty", fillcolor=f"rgba({r},{g},{b},0.06)",
            hovertemplate=f"<b>{fac}</b><br>%{{x}}<br>High Risk: %{{y}}<extra></extra>",
        ))
        drawn += 1
    if drawn == 0:
        fig.add_annotation(text="Need at least 2 semesters per faculty for trends",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#4a6a8a", size=13))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        height=340, margin=dict(l=50, r=20, t=20, b=50),
        font=dict(family="Inter", color="#c9d1d9"),
        xaxis=dict(title="Semester", tickangle=-30,
                   gridcolor="#21262d", tickcolor="#8b949e"),
        yaxis=dict(title="High Risk Students",
                   gridcolor="#21262d", tickcolor="#8b949e"),
        legend=dict(font=dict(size=10, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
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



# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1.5rem">
          <div style="font-size:3.5rem">🎓</div>
          <h1 style="font-size:1.7rem;font-weight:700;color:#e6edf3;
                     letter-spacing:-.02em;margin:.4rem 0">
              Pentecost University</h1>
          <p style="color:#58a6ff;font-size:.82rem;font-weight:600;
                    letter-spacing:.1em;text-transform:uppercase;margin:0">
              Academic Performance Tracker</p>
          <p style="color:#8b949e;font-size:.84rem;margin-top:.5rem">
              Sign in to view and manage student academic risk predictions</p>
          <div style="height:2px;background:linear-gradient(90deg,
                      transparent,#58a6ff,transparent);
                      margin:1rem auto;width:60%"></div>
        </div>""", unsafe_allow_html=True)

        if not artefacts_ok:
            st.warning("**Setup required:** The prediction model files are not found. "
                       "Please upload `best_model.pkl`, `scaler.pkl`, "
                       "`feature_cols.json`, and `thresholds.json` to the repository.")

        st.markdown("""
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:1.6rem 1.8rem">""",
                    unsafe_allow_html=True)

        with st.form("login"):
            st.markdown('<p style="color:#8b949e;font-size:.78rem;'
                        'text-transform:uppercase;letter-spacing:.1em;'
                        'margin-bottom:.5rem">Who are you?</p>',
                        unsafe_allow_html=True)
            role = st.selectbox("Role", list(ROLES.keys()),
                                label_visibility="collapsed")
            st.markdown('<p style="color:#8b949e;font-size:.78rem;'
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
        <p style="text-align:center;color:#484f58;
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
        st.markdown(f"""
        <div style="padding:.8rem 0 .5rem;text-align:center">
          <div style="font-size:1.8rem">{cfg['icon']}</div>
          <div style="font-weight:700;font-size:.9rem;color:#e6edf3;margin:.2rem 0">
              {role.split(' — ')[0]}</div>
          {"" if not faculty else
           f'<div style="font-size:.72rem;color:#58a6ff">{faculty} — {FACULTY_FULL.get(faculty,"")[:25]}...</div>'}
        </div>
        <hr style="border-color:#21262d;margin:.4rem 0">
        """, unsafe_allow_html=True)

        # Dataset status
        if df is not None:
            n    = len(df)
            n_hr = (df["risk_class"]==2).sum()
            n_mr = (df["risk_class"]==1).sum()
            n_lr = (df["risk_class"]==0).sum()
            st.markdown(f"""
            <div style="background:#0d2b1a;border:1px solid #3fb950;
                        border-radius:8px;padding:.6rem .8rem;font-size:.82rem;
                        margin-bottom:.5rem">
              <div style="color:#3fb950;font-weight:600">Dataset loaded</div>
              <div style="color:#8b949e;margin-top:.2rem">{n:,} students analysed</div>
            </div>
            <div style="font-size:.8rem;margin:.4rem 0">
              🔴 <b style="color:#f85149">{n_hr}</b> need immediate attention<br>
              🟡 <b style="color:#f0883e">{n_mr}</b> need monitoring<br>
              🟢 <b style="color:#3fb950">{n_lr}</b> on track
            </div>""", unsafe_allow_html=True)
            st.markdown('<hr style="border-color:#21262d">',
                        unsafe_allow_html=True)

        # Quick guide
        with st.expander("How to use this system"):
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

        st.markdown('<hr style="border-color:#21262d">', unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            for k in ["auth","role","df"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("""
        <div style="font-size:.64rem;color:#484f58;
                    text-align:center;margin-top:.5rem">
          Ghana DPA 2012 Compliant
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# STUDENT DETAIL CARD
# ══════════════════════════════════════════════════════════════════════════════
def student_detail(row: dict, uid: str = ""):
    """Expandable detail panel for one student."""
    risk_class = int(row.get("risk_class", 0))
    rc         = RISK_COLOR[risk_class]
    recs       = get_recommendations(row)

    # ── Performance scores ────────────────────────────────────────────────
    st.markdown("**Academic Performance**")

    def score_bar(label, value, maximum, color):
        pct = min(value/maximum*100, 100) if maximum > 0 else 0
        ok  = "✅" if pct >= 50 else "⚠️" if pct >= 40 else "🚨"
        st.markdown(f"""
        <div style="margin:.3rem 0">
          <div style="display:flex;justify-content:space-between;
                      font-size:.8rem;color:#8b949e;margin-bottom:.15rem">
            <span>{ok} {label}</span>
            <span style="color:{color};font-weight:600">
                {value:.1f} / {maximum}</span>
          </div>
          <div class="prog-bg">
            <div class="prog-fill"
                 style="width:{pct:.0f}%;background:{color}"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        score_bar("Total Mark",  row.get("avg_total_mark",0), 100, "#58a6ff")
        score_bar("CA Score",    row.get("avg_ca_score",  0),  40, "#3fb950")
    with sc2:
        score_bar("Exam Score",  row.get("avg_exam_score",0),  60, "#f0883e")
        score_bar("Attendance",  row.get("avg_attendance",0),   5, "#bc8cff")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key facts ─────────────────────────────────────────────────────────
    st.markdown("**Key Facts**")
    gpa_trend = row.get("gpa_trend", 0)
    trend_txt = (f"📈 Improving ({gpa_trend:+.2f})" if gpa_trend > 0.05
                 else f"📉 Declining ({gpa_trend:+.2f})" if gpa_trend < -0.05
                 else "➡️ Stable")

    st.markdown(f"""
    <div class="detail-panel">
      <div class="detail-row">
        <span class="detail-lbl">Previous Semester GPA</span>
        <span class="detail-val">{row.get('prev_gpa',0):.2f} / 4.00</span>
      </div>
      <div class="detail-row">
        <span class="detail-lbl">GPA Direction</span>
        <span class="detail-val">{trend_txt}</span>
      </div>
      <div class="detail-row">
        <span class="detail-lbl">Consecutive Sems Below 1.5</span>
        <span class="detail-val">{int(row.get('consec_fails',0))}</span>
      </div>
      <div class="detail-row">
        <span class="detail-lbl">Credits This Semester</span>
        <span class="detail-val">{int(row.get('total_credits',0))}</span>
      </div>
      <div class="detail-row">
        <span class="detail-lbl">Courses Enrolled</span>
        <span class="detail-val">{int(row.get('num_courses',0))}</span>
      </div>
      <div class="detail-row">
        <span class="detail-lbl">Prediction Confidence</span>
        <span class="detail-val" style="color:{rc}">
          {max(row.get('prob_low',0),row.get('prob_med',0),row.get('prob_high',0)):.0%}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── What this means ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{RISK_BG[risk_class]};border:1px solid {rc};
                border-radius:8px;padding:.8rem 1rem;margin:.5rem 0;
                font-size:.86rem;color:#c9d1d9;line-height:1.6">
      <b style="color:{rc}">{RISK_ICON[risk_class]} What this means:</b><br>
      {RISK_MEANING[risk_class]}
    </div>""", unsafe_allow_html=True)

    # ── Recommended actions ───────────────────────────────────────────────
    st.markdown("**Recommended Actions**")
    for icon, text in recs:
        color = ("#f85149" if icon=="🔴" else
                 "#f0883e" if icon=="🟡" else "#3fb950")
        st.markdown(f"""
        <div class="rec-item" style="--rc:{color}">
          {icon} {text}
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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Graduation Classification Projection**")

        # Build upgrade hints cleanly (avoid nested f-string issues)
        upgrade_html = ""
        for t in grad["next"]:
            upgrade_html += (
                '<div style="color:#8b949e;font-size:.81rem;margin-top:.5rem">'
                'To reach <b style="color:' + t["color"] + '">'
                + t["class"] +
                '</b>, needs avg GPA of <b style="color:' + t["color"] + '">'
                + str(round(t["needed"], 2)) +
                '</b> in remaining semesters</div>'
            )

        grad_html = (
            '<div style="background:#161b22;border:1px solid #30363d;'
            'border-radius:10px;padding:1rem;border-top:3px solid '
            + grad["color"] + '">'
            '<div style="display:flex;align-items:center;gap:1rem">'
            '<div style="font-size:2rem">' + grad["emoji"] + '</div>'
            '<div>'
            '<div style="font-weight:700;color:#e6edf3;font-size:1rem">'
            + grad["label"] + '</div>'
            '<div style="color:#8b949e;font-size:.82rem">'
            'Projected Final CGPA: '
            '<b style="color:' + grad["color"] + '">'
            + str(grad["proj_cgpa"]) + '</b>'
            '&nbsp;·&nbsp;' + str(grad["remaining"]) +
            ' semester(s) remaining</div>'
            '</div></div>'
            + upgrade_html +
            '</div>'
        )
        st.markdown(grad_html, unsafe_allow_html=True)

    # ── PDF download — lazy generation with unique key ───────────────────
    st.markdown("<br>", unsafe_allow_html=True)
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
    st.markdown(f"""
    <div class="top-banner">
      <div style="font-size:1.8rem">🎓</div>
      <div>
        <h1>Academic Performance Tracker</h1>
        <div class="sub">
          {cfg['icon']} {role} &nbsp;·&nbsp;
          {FACULTY_FULL.get(faculty, "All Faculties") if faculty else "All Faculties"}
          &nbsp;·&nbsp; {datetime.date.today().strftime("%d %B %Y")}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not artefacts_ok:
        st.error("The prediction model is not set up yet. "
                 "Please contact your system administrator.")
        return

    df = st.session_state.get("df")

    # ══════════════════════════════════════════════════════════════════════
    # STATE 1 — No data uploaded yet
    # ══════════════════════════════════════════════════════════════════════
    if df is None:

        # How it works — 3 steps
        st.markdown('<p style="color:#e6edf3;font-size:1rem;font-weight:600;'
                    'margin-bottom:.8rem">Get started in 3 simple steps</p>',
                    unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        for col, num, title, desc in [
            (s1, "1", "Upload your student data",
             "Upload a CSV file with your students' records for the current semester."),
            (s2, "2", "View predictions instantly",
             "The system automatically identifies which students need attention."),
            (s3, "3", "Take action",
             "See recommended actions for each student and download reports."),
        ]:
            with col:
                st.markdown(f"""
                <div class="step-card">
                  <div class="step-num">{num}</div>
                  <div class="step-title">{title}</div>
                  <div class="step-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Upload section
        st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
                    'margin-bottom:.3rem">Upload your student records</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#8b949e;font-size:.84rem;margin-bottom:.8rem">'
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

        uploaded = st.file_uploader("", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            try:
                df_raw = pd.read_csv(uploaded)
                # Faculty filter for non-Dean roles
                if faculty and "faculty" in df_raw.columns:
                    df_raw = df_raw[df_raw["faculty"]==faculty].copy()

                with st.spinner(f"Analysing {len(df_raw):,} records... "
                                "This usually takes a few seconds."):
                    df_result = run_batch_pipeline(df_raw)

                if len(df_result) == 0:
                    st.warning("No results could be generated. "
                               "Make sure each student has at least "
                               "2 rows in the file.")
                else:
                    st.session_state["df"] = df_result
                    st.rerun()

            except Exception as e:
                st.error(f"There was a problem reading your file: {e}")

        # Required columns note
        with st.expander("What columns does my CSV file need?"):
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
    for col, val, lbl, sub, css_color in [
        (k1, n,    "Students Analysed", "Full cohort",         "#58a6ff"),
        (k2, n_hr, "High Risk",         "Need immediate attention", "#f85149"),
        (k3, n_mr, "Medium Risk",       "Need monitoring",          "#f0883e"),
        (k4, n_lr, "Low Risk",          "On track",                 "#3fb950"),
    ]:
        with col:
            st.markdown(f"""
            <div class="sum-card" style="--c:{css_color}">
              <div class="sum-val">{val:,}</div>
              <div class="sum-lbl">{lbl}</div>
              <div class="sum-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row ──────────────────────────────────────────────────────────
    ch1, ch2 = st.columns([1, 1.8])
    with ch1:
        st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.3rem">'
                    'Risk breakdown</p>', unsafe_allow_html=True)
        st.plotly_chart(donut_chart(n_lr, n_mr, n_hr),
                        use_container_width=True,
                        config={"displayModeBar": False})
    with ch2:
        st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.3rem">'
                    'Risk by faculty</p>', unsafe_allow_html=True)
        st.plotly_chart(faculty_bar(df),
                        use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYTICS — Faculty comparison, gender analysis, trends, graduation
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("📊 View Detailed Analytics (Faculty Comparison, Gender, Trends, Graduation)",
                     expanded=False):

        # ── Component 1: Faculty Performance Comparison ────────────────────
        st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">🏛️ Faculty Performance Comparison</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.6rem">'
                    'Compare average GPA and risk levels across all four faculties '
                    'to identify which departments need the most support.</p>',
                    unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'Average GPA by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_faculty_gpa(df),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with fc2:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'Risk Level Breakdown by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_faculty_risk_pct(df),
                            use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 2: Gender Performance Analysis ────────────────────────
        st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">👥 Male vs Female Performance</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.6rem">'
                    'Compare academic performance and risk distribution '
                    'between male and female students across faculties.</p>',
                    unsafe_allow_html=True)

        gc1, gc2 = st.columns([1.6, 1])
        with gc1:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'Average GPA by Gender per Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gender_gpa(df),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with gc2:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'Risk Split by Gender</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gender_risk_split(df),
                            use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 3: Semester Trend Lines ───────────────────────────────
        st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
                    'margin:.5rem 0">📈 Performance Trends Over Time</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.6rem">'
                    'See how each faculty\'s GPA and risk levels have changed '
                    'across semesters. Requires data with at least 2 semesters '
                    'per faculty.</p>', unsafe_allow_html=True)

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'GPA Trend by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_gpa_trend(df),
                            use_container_width=True,
                            config={"displayModeBar": False})
        with tc2:
            st.markdown('<p style="color:#8b949e;font-size:.8rem;margin-bottom:.2rem">'
                        'High Risk Count Trend by Faculty</p>', unsafe_allow_html=True)
            st.plotly_chart(chart_risk_trend(df),
                            use_container_width=True,
                            config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Component 4: Graduation Projection Summary ──────────────────────
        grad_df = graduation_summary(df)
        if len(grad_df) > 0:
            st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
                        'margin:.5rem 0">🎓 Graduation Classification Summary '
                        '(Level 300 &amp; 400)</p>', unsafe_allow_html=True)
            st.markdown('<p style="color:#8b949e;font-size:.82rem;margin-bottom:.6rem">'
                        'Projected final classification for final-year students '
                        'based on current CGPA and performance trend.</p>',
                        unsafe_allow_html=True)

            # Summary counts by classification
            class_counts = grad_df["Classification"].value_counts()
            cls_cols = st.columns(min(len(class_counts), 6))
            for col, (cls, count) in zip(cls_cols, class_counts.items()):
                with col:
                    st.markdown(f"""
                    <div style="background:#161b22;border:1px solid #30363d;
                                border-radius:10px;padding:.7rem;text-align:center">
                      <div style="font-size:1.4rem">{cls.split()[0]}</div>
                      <div style="font-size:1.3rem;font-weight:700;color:#e6edf3">
                        {count}</div>
                      <div style="font-size:.7rem;color:#8b949e">
                        {' '.join(cls.split()[1:])}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(grad_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background:#161b22;border:1px dashed #30363d;
                        border-radius:10px;padding:1rem;text-align:center;
                        color:#8b949e;font-size:.85rem">
              🎓 No graduation projections available.<br>
              <span style="font-size:.78rem">
                Add a <code>level</code> column (Level 300 / Level 400) and
                <code>cumulative_gpa</code>, <code>completed_credits</code>,
                <code>programme_credits</code> columns to your CSV to enable
                graduation classification predictions.
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action row: search + filter + download ─────────────────────────────
    st.markdown('<p style="color:#e6edf3;font-size:.95rem;font-weight:600;'
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
    st.markdown(f'<p style="color:#8b949e;font-size:.82rem;margin-bottom:.6rem">'
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

        with st.expander(expander_label, expanded=(risk_class == 2)):
            uid = str(row.get("student_id","")) + "_" + str(row.get("semester","")) + "_" + str(idx)
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
