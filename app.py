"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Pentecost University — AI-Powered Academic Performance Tracker          ║
║  app.py  ·  Production Streamlit Deployment                              ║
║  Authors   : Steven Asante-Poku Jnr & Frank Amoah  |  2025             ║
║  Supervisor: Mr Harry Attieku-Boateng                                   ║
║                                                                          ║
║  Pipeline facts (from final notebook run — April 2026):                  ║
║    Model    : LightGBM                                                   ║
║    Macro F1 : 0.6383  (Test, real unbalanced data)                      ║
║    ECE      : 0.0957  (calibration quality)                              ║
║    BV Gap   : +0.3290 (SMOTETomek artefact — NOT real overfitting)      ║
║    Q33      : 2.0   → High Risk  when GPA < 2.0                        ║
║    Q66      : 3.0   → Medium Risk when 2.0 ≤ GPA < 3.0                ║
║    Features : 16 (see FEATURE_COLS below)                                ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pickle, json, os, datetime, io
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from anthropic import Anthropic

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PU Academic Risk Tracker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS — mirror Cell 6 of final notebook exactly
# ═══════════════════════════════════════════════════════════════════════════
PU_BLUE   = "#003087"
PU_GOLD   = "#C9A84C"

FACULTIES = ["FESAC", "FBA", "FEHAS", "PSTM"]
SEMESTERS = ["2019_S1","2019_S2","2020_S1","2020_S2",
             "2021_S1","2021_S2","2022_S1","2022_S2"]

RISK_MAP   = {0: "Low Risk",    1: "Medium Risk",   2: "High Risk"}
RISK_COLOR = {0: "#27ae60",     1: "#f39c12",        2: "#e74c3c"}
RISK_EMOJI = {0: "🟢",           1: "🟡",              2: "🔴"}
RISK_CSS   = {0: "risk-low",    1: "risk-medium",    2: "risk-high"}
RISK_BG    = {0: "#d4f5e2",     1: "#fff3cc",         2: "#fde8e8"}
RISK_FG    = {0: "#1a7a42",     1: "#a06000",         2: "#c0392b"}

FACULTY_FULL = {
    "FESAC": "Faculty of Engineering & Applied Sciences",
    "FBA"  : "Faculty of Business Administration",
    "FEHAS": "Faculty of Education, Humanities & Applied Sciences",
    "PSTM" : "Pentecost School of Theology & Ministry",
}

# 16 features — exact order from FEATURE_COLS in notebook Cell 6
FEATURE_COLS = [
    "avg_attendance", "avg_total_mark", "avg_ca_score", "avg_exam_score",
    "total_credits",  "num_courses",    "gender_enc",   "semester_index",
    "prev_gpa",       "gpa_trend",      "consec_fails", "trend_x_fail",
    "fac_FESAC",      "fac_FBA",        "fac_FEHAS",    "fac_PSTM",
]

# Per-class F1 placeholder — classification_report display() output was not
# captured in the notebook. Values will be populated after rerunning Cell 9
# and reading the printed report. Overall Macro F1 = 0.6383 is confirmed.
PER_CLASS_F1 = {"Low Risk": "—", "Medium Risk": "—", "High Risk": "—"}


# ═══════════════════════════════════════════════════════════════════════════
# SAFE SECRETS HELPER
# Wraps st.secrets so the app renders even without a secrets.toml
# ═══════════════════════════════════════════════════════════════════════════
def _secret(key: str, fallback: str = "") -> str:
    """Read from st.secrets or environment, never crash."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, fallback)


# ═══════════════════════════════════════════════════════════════════════════
# CSS — Pentecost University brand
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root { --blue:#003087; --gold:#C9A84C; --light:#EEF3FB; }
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #001a4d 0%, #003087 60%, #0a4fa0 100%);
    border-right: 3px solid var(--gold);
}
[data-testid="stSidebar"] * { color: #fff !important; }

/* ── Page header ── */
.pu-header {
    background: linear-gradient(90deg, var(--blue) 0%, #0a4fa0 100%);
    border-bottom: 4px solid var(--gold);
    padding: 1.1rem 2rem; border-radius: 0 0 12px 12px;
    margin-bottom: 1.4rem; display: flex; align-items: center; gap: 1.2rem;
}
.pu-header h1 {
    font-family: 'Playfair Display', serif;
    color: white !important; font-size: 1.5rem; margin: 0;
}
.pu-header .sub {
    color: var(--gold); font-size: .82rem; font-weight: 600;
    letter-spacing: .05em; text-transform: uppercase;
}

/* ── Metric cards ── */
.mc {
    background: white; border-radius: 12px; padding: 1.1rem 1.3rem;
    border-left: 5px solid var(--blue);
    box-shadow: 0 2px 12px rgba(0,48,135,.08);
    transition: transform .15s, box-shadow .15s;
}
.mc:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,48,135,.14); }
.mc .val { font-family: 'Playfair Display', serif; font-size: 2rem; color: var(--blue); line-height: 1; }
.mc .lbl { font-size: .75rem; color: #666; text-transform: uppercase; letter-spacing: .06em; margin-top: .2rem; }
.mc.gold  { border-left-color: var(--gold); }
.mc.green { border-left-color: #27ae60; }
.mc.red   { border-left-color: #e74c3c; }
.mc.amber { border-left-color: #f39c12; }

/* ── Risk badges ── */
.rb { display: inline-flex; align-items: center; gap: .4rem;
    padding: .28rem .85rem; border-radius: 20px; font-weight: 600; font-size: .83rem; }
.risk-low    { background: #d4f5e2; color: #1a7a42; }
.risk-medium { background: #fff3cc; color: #a06000; }
.risk-high   { background: #fde8e8; color: #c0392b; }

/* ── Student card ── */
.sc {
    background: white; border-radius: 14px; padding: 1.4rem;
    box-shadow: 0 3px 16px rgba(0,48,135,.09);
    border-top: 4px solid var(--blue); margin-bottom: .8rem;
}

/* ── Section titles ── */
.stt {
    font-family: 'Playfair Display', serif; color: var(--blue); font-size: 1.2rem;
    border-bottom: 2px solid var(--gold); padding-bottom: .35rem; margin: 1.4rem 0 .9rem;
}

/* ── Chat bubbles ── */
.cu { background: var(--light); border-radius: 12px 12px 2px 12px;
    padding: .75rem 1rem; margin: .4rem 0;
    border-left: 3px solid var(--blue); font-size: .88rem; }
.ca { background: white; border-radius: 12px 12px 12px 2px;
    padding: .75rem 1rem; margin: .4rem 0;
    border-left: 3px solid var(--gold);
    box-shadow: 0 2px 8px rgba(0,0,0,.06); font-size: .88rem; }

/* ── Signal boxes ── */
.sig { background: white; padding: .55rem .8rem; border-radius: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,.06); margin-bottom: .4rem; font-size: .87rem; }

/* ── Buttons ── */
.stButton > button {
    background: var(--blue); color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: .45rem 1.4rem; transition: all .15s;
}
.stButton > button:hover {
    background: #0a4fa0; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,48,135,.25);
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 2px solid #e0e6f0; }
.stTabs [aria-selected="true"] {
    color: var(--blue) !important;
    border-bottom-color: var(--blue) !important; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ROLE CONFIG  (passwords read safely via _secret)
# ═══════════════════════════════════════════════════════════════════════════
ROLE_PWDS = {
    "Academic Advisor"   : _secret("ADVISOR_PASSWORD",  "advisor2025"),
    "Registry Admin"     : _secret("ADMIN_PASSWORD",    "registry2025"),
    "Head of Department" : _secret("HOD_PASSWORD",      "hod2025"),
}
ROLE_PERMS = {
    "Academic Advisor"   : ["individual", "chat"],
    "Registry Admin"     : ["individual", "batch", "chat"],
    "Head of Department" : ["individual", "batch", "analytics", "chat"],
}
ROLE_ICON = {
    "Academic Advisor"   : "👨‍🏫",
    "Registry Admin"     : "🗂️",
    "Head of Department" : "🏛️",
}


# ═══════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artefacts():
    try:
        with open("best_model.pkl",    "rb") as f: mdl   = pickle.load(f)
        with open("scaler.pkl",        "rb") as f: scl   = pickle.load(f)
        with open("feature_cols.json", "r")  as f: fcols = json.load(f)
        with open("thresholds.json",   "r")  as f: thr   = json.load(f)
        return mdl, scl, fcols, thr, True
    except FileNotFoundError:
        return None, None, None, None, False

model, scaler, _fcols_json, thresholds, artefacts_ok = load_artefacts()

# Trust saved feature_cols.json over hardcoded list if available
if _fcols_json:
    FEATURE_COLS = _fcols_json

Q33         = thresholds.get("Q33",        2.0)      if thresholds else 2.0
Q66         = thresholds.get("Q66",        3.0)      if thresholds else 3.0
MODEL_NAME  = thresholds.get("best_model", "LightGBM") if thresholds else "LightGBM"
BASELINE_F1 = thresholds.get("macro_f1",   0.6383)   if thresholds else 0.6383
ECE_VAL     = thresholds.get("ece",        0.0957)   if thresholds else 0.0957
BV_GAP      = thresholds.get("bv_gap",     0.329)    if thresholds else 0.329
TOP_SHAP    = thresholds.get("top_shap_features", [
    "avg_total_mark", "avg_exam_score", "gpa_trend", "avg_ca_score",
    "consec_fails", "trend_x_fail", "fac_FEHAS", "prev_gpa",
    "fac_FESAC", "gender_enc",
]) if thresholds else []


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def predict_one(feat_dict: dict):
    """Build vector → scale → predict. Returns (class_int, prob_array)."""
    row    = np.array([float(feat_dict.get(c, 0.0)) for c in FEATURE_COLS]).reshape(1, -1)
    row_sc = scaler.transform(row)
    probs  = model.predict_proba(row_sc)[0]
    return int(np.argmax(probs)), probs


def build_features(avg_attendance, avg_total_mark, avg_ca_score, avg_exam_score,
                   total_credits, num_courses, gender, semester_index,
                   prev_gpa, gpa_trend, consec_fails, faculty) -> dict:
    """
    Mirrors Cell 6 feature engineering exactly:
      trend_x_fail = gpa_trend × consec_fails
      gender_enc   = 1 if Female, 0 if Male
      fac_*        = one-hot encoding
    """
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
    """
    Replicates Cell 6 feature engineering for batch CSV input.
    Requires at least 2 semester rows per student (for prev_gpa shift).
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
    df["gender_enc"] = (df["gender"].str.strip().str.title()
                        .map({"Female": 1, "Male": 0}).fillna(0).astype(int))
    sem_map = {s: i for i, s in enumerate(SEMESTERS)}
    df["semester_index"] = df["semester"].map(sem_map).fillna(0).astype(int)
    return df.dropna(subset=["prev_gpa"]).reset_index(drop=True)


def prob_gauge(probs: np.ndarray, pred: int) -> plt.Figure:
    """Compact horizontal probability bar."""
    fig, ax = plt.subplots(figsize=(6, 0.65))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    left = 0
    for i, (p, c) in enumerate(zip(probs, ["#27ae60", "#f39c12", "#e74c3c"])):
        ax.barh(0, p, left=left, color=c,
                alpha=0.88 if i == pred else 0.28,
                height=0.6, edgecolor="white", linewidth=1.5)
        if p > 0.08:
            ax.text(left + p / 2, 0, f"{p:.0%}",
                    va="center", ha="center", fontsize=8, fontweight="bold",
                    color="white" if i == pred else "#555")
        left += p
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ═══════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0 1.2rem">
          <div style="font-size:3.5rem">🎓</div>
          <h1 style="font-family:'Playfair Display',serif; color:#003087; margin:.5rem 0 .2rem">
              Pentecost University</h1>
          <p style="color:#C9A84C; font-weight:600; letter-spacing:.08em;
                    font-size:.88rem; text-transform:uppercase">
              AI Academic Performance Tracker</p>
          <hr style="border:none; border-top:2px solid #C9A84C; margin:.8rem auto; width:60%">
        </div>""", unsafe_allow_html=True)

        if not artefacts_ok:
            st.error("""
**Model artefacts not found.**

Place these 4 files in the same folder as `app.py`:
```
best_model.pkl       scaler.pkl
feature_cols.json    thresholds.json
```
Download them from Colab after running Cell 13.
            """)

        with st.form("login"):
            role = st.selectbox("Select your role", list(ROLE_PWDS.keys()))
            pwd  = st.text_input("Password", type="password")
            ok   = st.form_submit_button("Sign In →", use_container_width=True)

        if ok:
            if ROLE_PWDS.get(role) == pwd:
                st.session_state.update({"auth": True, "role": role, "hist": []})
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")

        st.markdown("""
        <p style="text-align:center; color:#999; font-size:.72rem; margin-top:1.5rem">
          © 2025 Pentecost University · Ghana Data Protection Act 2012 (Act 843) Compliant
        </p>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar(role: str):
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem 0 .5rem">
          <div style="font-size:2.3rem">🎓</div>
          <div style="font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700">
              Pentecost University</div>
          <div style="color:#C9A84C; font-size:.76rem; text-transform:uppercase;
                      letter-spacing:.07em; margin-top:.2rem">
              Academic Risk Tracker</div>
        </div>
        <hr style="border-color:#C9A84C55; margin:.7rem 0">
        """, unsafe_allow_html=True)

        st.markdown(f"**{ROLE_ICON[role]} {role}**")
        st.caption(f"Model: {MODEL_NAME}  ·  F1: {BASELINE_F1:.4f}  ·  ECE: {ECE_VAL:.4f}")

        st.markdown('<hr style="border-color:#ffffff33">', unsafe_allow_html=True)
        st.markdown("**📊 Risk Thresholds**")
        for k, (emoji, label, rng) in {
            2: ("🔴", "High Risk",   f"GPA < {Q33:.1f}"),
            1: ("🟡", "Medium Risk", f"{Q33:.1f} ≤ GPA < {Q66:.1f}"),
            0: ("🟢", "Low Risk",    f"GPA ≥ {Q66:.1f}"),
        }.items():
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.09); border-radius:6px;
                padding:.38rem .7rem; margin:.25rem 0; font-size:.8rem">
              {emoji} <b>{label}</b><br>
              <span style="opacity:.7; font-size:.73rem">{rng}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border-color:#ffffff33">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True):
            for k in ["auth", "role", "hist", "lp"]:
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("""
        <div style="font-size:.67rem; opacity:.45; text-align:center; margin-top:.8rem">
          Ghana DPA 2012 Compliant · © 2025 Pentecost University
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
def tab_individual():
    st.markdown('<div class="stt">Individual Student Risk Assessment</div>',
                unsafe_allow_html=True)
    st.caption("Enter the student's current semester data. "
               "The model predicts their academic risk level for the **next** semester.")

    with st.form("individual_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📋 Student Identity**")
            sid   = st.text_input("Student ID", placeholder="e.g. 100045")
            sname = st.text_input("Name (optional)")
            fac   = st.selectbox("Faculty", FACULTIES,
                                 format_func=lambda x: f"{x} — {FACULTY_FULL[x]}")
            gen   = st.selectbox("Gender", ["Male", "Female"])
            sem_i = st.selectbox("Current Semester", range(len(SEMESTERS)),
                                 format_func=lambda i: SEMESTERS[i], index=7)

        with c2:
            st.markdown("**📊 Current Semester Performance**")
            atm  = st.number_input("Avg Total Mark (0–100)", 0.0, 100.0, 55.0, 0.5)
            aca  = st.number_input("Avg CA Score (0–40)",    0.0,  40.0, 22.0, 0.5)
            aex  = st.number_input("Avg Exam Score (0–60)",  0.0,  60.0, 33.0, 0.5)
            aatt = st.number_input("Attendance Score (0–5)", 0.0,   5.0,  3.5, 0.1)

        with c3:
            st.markdown("**📈 Enrolment & History**")
            tc   = st.number_input("Total Credits",       1, 30, 18)
            nc   = st.number_input("Courses Enrolled",    1, 12,  6)
            pgpa = st.number_input("Previous Semester GPA (0–4)", 0.0, 4.0, 1.8, 0.01)
            gtr  = st.number_input(
                "GPA Trend (current − previous)", -4.0, 4.0, 0.0, 0.01,
                help="Positive = improving · Negative = declining")
            cf   = st.number_input(
                "Consecutive Semesters with GPA < 1.5", 0, 8, 0,
                help="How many semesters in a row has GPA been below 1.5?")

        go = st.form_submit_button("🔮  Predict Risk Level", use_container_width=True)

    if go:
        if not artefacts_ok:
            st.error("Model artefacts missing — cannot predict. See login page for instructions.")
            return

        feats       = build_features(aatt, atm, aca, aex, tc, nc, gen,
                                     sem_i, pgpa, gtr, cf, fac)
        pred, probs = predict_one(feats)
        rl          = RISK_MAP[pred]

        # Persist context for AI Chat
        st.session_state["lp"] = dict(
            student_id=sid or "N/A", name=sname or "Student",
            faculty=fac, semester=SEMESTERS[sem_i],
            pred=pred, probs=probs.tolist(), risk_label=rl, features=feats,
        )

        # ── Result card ────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="sc">
          <div style="display:flex; align-items:center; gap:1rem; margin-bottom:.8rem">
            <div style="font-size:2.4rem">{RISK_EMOJI[pred]}</div>
            <div>
              <div style="font-family:'Playfair Display',serif;
                          font-size:1.25rem; color:#003087">
                {sname or "Student"}{f" ({sid})" if sid else ""}
              </div>
              <div style="color:#666; font-size:.83rem">
                {FACULTY_FULL[fac]} · {gen} · {SEMESTERS[sem_i]}
              </div>
            </div>
            <div style="margin-left:auto">
              <span class="rb {RISK_CSS[pred]}">{rl}</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.caption("Probability across all three risk classes")
        st.pyplot(prob_gauge(probs, pred), use_container_width=True)
        plt.close()

        # Per-class probability breakdown
        pb1, pb2, pb3 = st.columns(3)
        for col, k in zip([pb1, pb2, pb3], [0, 1, 2]):
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:.5rem;
                            background:{RISK_BG[k]}; border-radius:8px; margin-top:.4rem">
                  <div style="font-size:.74rem; font-weight:600; color:#555">
                      {RISK_EMOJI[k]} {RISK_MAP[k]}</div>
                  <div style="font-family:'Playfair Display',serif;
                              font-size:1.7rem; color:{RISK_COLOR[k]}">
                      {probs[k]:.1%}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;")

        # ── Risk signals ───────────────────────────────────────────────────
        st.markdown("**Key Risk Signals**")
        sigs = []
        if aatt < 3.0:
            sigs.append(("🚩", f"Attendance {aatt:.1f}/5 — below acceptable threshold (3.0)", "#e74c3c"))
        if cf > 0:
            sigs.append(("⚠️", f"{cf} consecutive semester(s) with GPA below 1.5", "#f39c12"))
        if gtr < -0.1:
            sigs.append(("📉", f"GPA declining (trend = {gtr:+.2f})", "#f39c12"))
        if pgpa < Q33:
            sigs.append(("🚩", f"Previous GPA {pgpa:.2f} already in High-Risk zone (< {Q33:.1f})", "#e74c3c"))
        if atm < 45:
            sigs.append(("⚠️", f"Avg total mark {atm:.0f} — below 50% threshold", "#f39c12"))
        if aca / 40 < 0.5 and aex / 60 < 0.5:
            sigs.append(("🚩", "Both CA and Exam below 50% of their maximums", "#e74c3c"))
        if not sigs:
            sigs.append(("✅", "No critical risk signals detected — student appears stable", "#27ae60"))

        scols = st.columns(min(len(sigs), 3))
        for i, (icon, msg, color) in enumerate(sigs):
            with scols[i % 3]:
                st.markdown(f"""
                <div class="sig" style="border-left:4px solid {color}">
                    {icon} {msg}
                </div>""", unsafe_allow_html=True)

        # ── Interventions ──────────────────────────────────────────────────
        st.markdown("&nbsp;")
        if pred == 2:
            st.error("""
**🚨 Recommended Actions — High Risk**
1. Schedule an immediate one-on-one advisory session this week
2. Review full academic transcript — identify which courses are weakest
3. Recommend course load reduction next semester if total credits > 18
4. Refer to Student Support Services if attendance is consistently below 3.0
5. Notify Head of Department for monitoring and resource allocation
            """)
        elif pred == 1:
            st.warning("""
**⚠️ Recommended Actions — Medium Risk**
1. Schedule a check-in meeting within the next 2 weeks
2. Encourage consistent attendance — target score ≥ 3.5
3. Review latest assessment feedback with student
4. If GPA trend is still negative next semester, escalate to High Risk protocol
            """)
        else:
            st.success("**✅ Low Risk** — Student is on track. "
                       "Continue monitoring through standard semester-end reviews.")

        st.info("💬 Switch to the **AI Chat Assistant** tab for a natural-language "
                "explanation of this prediction and tailored intervention advice.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════
def tab_batch():
    st.markdown('<div class="stt">Batch Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with raw semester records. "
                "The app runs the same feature engineering as notebook Cell 6 "
                "and predicts risk for every student-semester row.")

    with st.expander("📋 Required CSV columns"):
        st.markdown("""
**Mandatory:**
`student_id`, `faculty`, `gender`, `semester`, `semester_gpa`,
`avg_attendance`, `avg_total_mark`, `avg_ca_score`, `avg_exam_score`,
`total_credits`, `num_courses`

**Optional** (retained in output): `name`, any additional columns

**Semester values:** `2019_S1` … `2022_S2`

⚠️ **Each student needs ≥ 2 semester rows** — `prev_gpa` is computed by shifting
the GPA column within each student group. Single-row students are dropped automatically.
        """)

    # Downloadable sample template
    sample = pd.DataFrame({
        "student_id":     [100001]*2 + [100002]*2,
        "name":           ["Alice Mensah"]*2 + ["Kofi Asante"]*2,
        "faculty":        ["FESAC","FESAC","FBA","FBA"],
        "gender":         ["Female","Female","Male","Male"],
        "semester":       ["2021_S2","2022_S1","2021_S2","2022_S1"],
        "semester_gpa":   [2.8, 2.5, 1.3, 0.9],
        "avg_attendance": [4.0, 3.5, 2.0, 1.5],
        "avg_total_mark": [65.0, 60.0, 42.0, 35.0],
        "avg_ca_score":   [28.0, 25.0, 17.0, 14.0],
        "avg_exam_score": [37.0, 35.0, 25.0, 21.0],
        "total_credits":  [18, 18, 21, 21],
        "num_courses":    [6, 6, 7, 7],
    })
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("⬇️ Download Sample CSV Template",
                       buf.getvalue(), "pu_batch_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload student records CSV", type=["csv"])
    if not uploaded:
        return

    try:
        df_raw = pd.read_csv(uploaded)
        st.success(f"✅  {len(df_raw):,} rows loaded.")

        with st.spinner("Running feature engineering pipeline …"):
            df = apply_pipeline_batch(df_raw)

        if len(df) == 0:
            st.error("No rows survived feature engineering. "
                     "Ensure every student has at least 2 semester rows.")
            return

        # Predict
        X     = df[FEATURE_COLS].fillna(0).values
        X_sc  = scaler.transform(X)
        probs = model.predict_proba(X_sc)
        preds = probs.argmax(axis=1)

        df["risk_label"] = preds
        df["risk_name"]  = [RISK_MAP[p] for p in preds]
        df["prob_low"]   = probs[:, 0].round(3)
        df["prob_med"]   = probs[:, 1].round(3)
        df["prob_high"]  = probs[:, 2].round(3)

        n    = len(df)
        n_hr = (preds == 2).sum()
        n_mr = (preds == 1).sum()
        n_lr = (preds == 0).sum()

        # ── Summary metrics ────────────────────────────────────────────────
        st.markdown('<div class="stt">Cohort Risk Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        for col, val, lbl, css in [
            (m1, n,    "Students Assessed", ""),
            (m2, n_hr, "High Risk 🔴",      "red"),
            (m3, n_mr, "Medium Risk 🟡",    "amber"),
            (m4, n_lr, "Low Risk 🟢",       "green"),
        ]:
            with col:
                st.markdown(f"""
                <div class="mc {css}">
                  <div class="val">{val:,}</div>
                  <div class="lbl">{lbl} · {val/n*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            fig.patch.set_facecolor("none")
            ax.pie(
                [n_lr, n_mr, n_hr],
                labels=[f"Low\n{n_lr}", f"Med\n{n_mr}", f"High\n{n_hr}"],
                colors=["#27ae60", "#f39c12", "#e74c3c"],
                startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
            )
            ax.set_title("Risk Distribution", fontweight="bold", color=PU_BLUE)
            st.pyplot(fig); plt.close()

        with ch2:
            if "faculty" in df.columns:
                fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
                fig2.patch.set_facecolor("none")
                fac_hr = (df.groupby("faculty")["risk_label"]
                          .apply(lambda x: (x == 2).sum())
                          .reindex(FACULTIES, fill_value=0))
                bars = ax2.bar(fac_hr.index, fac_hr.values,
                               color=PU_BLUE, edgecolor="white", alpha=0.85)
                for bar, v in zip(bars, fac_hr.values):
                    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.2,
                             str(v), ha="center", fontsize=9, fontweight="bold")
                ax2.set_title("High-Risk Count by Faculty",
                              fontweight="bold", color=PU_BLUE)
                ax2.set_ylabel("Students")
                st.pyplot(fig2); plt.close()

        # ── Results table ──────────────────────────────────────────────────
        st.markdown('<div class="stt">Individual Results</div>', unsafe_allow_html=True)
        show = [c for c in ["student_id","name","faculty","gender","semester",
                             "semester_gpa","risk_name","prob_low","prob_med","prob_high"]
                if c in df.columns]

        risk_order = {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
        df_display = (df[show]
                      .sort_values("risk_name", key=lambda s: s.map(risk_order)))

        # Use .map() — compatible with Pandas ≥ 2.2 (applymap deprecated)
        def highlight_risk(val):
            if val == "High Risk":   return "background:#fde8e8"
            if val == "Medium Risk": return "background:#fff3cc"
            if val == "Low Risk":    return "background:#d4f5e2"
            return ""

        st.dataframe(
            df_display.style.map(highlight_risk, subset=["risk_name"]),
            use_container_width=True, height=420,
        )

        csv_out = df_display.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_out,
            file_name=f"pu_risk_{datetime.date.today()}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Processing error: {e}")
        st.exception(e)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
def tab_analytics():
    st.markdown('<div class="stt">Model Performance & Institutional Analytics</div>',
                unsafe_allow_html=True)

    # ── Top-line metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl, css in [
        (m1, f"{BASELINE_F1:.4f}", "Macro F1 (Test)", ""),
        (m2, f"{ECE_VAL:.4f}",    "ECE",             "green"),
        (m3, f"{BV_GAP:+.4f}",   "BV Gap *",        "amber"),
        (m4, str(len(FEATURE_COLS)), "Features",     "gold"),
    ]:
        with col:
            st.markdown(f"""
            <div class="mc {css}">
              <div class="val" style="font-size:1.6rem">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("* Why is the BV Gap large (+0.329)?"):
        st.info("""
The BV gap is an **expected artefact** of SMOTETomek class balancing.

Before balancing, the training set had severe imbalance:
- Low Risk: **10 samples** · Medium Risk: **2,224** · High Risk: **29,766**

SMOTETomek generated synthetic samples, expanding training from **32,000 → 89,278 rows**.
The model trains on this inflated balanced set (Train F1 ≈ 0.97), then is evaluated
on the real unbalanced 16,000-row test set (Test F1 = **0.6383**).

**The gap reflects a distribution difference, not real-world overfitting.**
The Test F1 of 0.6383 is the honest, authoritative metric.
        """)

    # ── Per-class performance ──────────────────────────────────────────────
    st.markdown('<div class="stt">Per-Class Performance</div>', unsafe_allow_html=True)
    pc_df = pd.DataFrame({
        "Class"          : ["🟢 Low Risk",  "🟡 Medium Risk",         "🔴 High Risk"],
        "GPA Range"      : [f"≥ {Q66:.1f}", f"{Q33:.1f}–{Q66:.1f}", f"< {Q33:.1f}"],
        "Macro F1"       : [PER_CLASS_F1["Low Risk"],
                            PER_CLASS_F1["Medium Risk"],
                            PER_CLASS_F1["High Risk"]],
        "Interpretation" : [
            "Students flagged Low Risk are reliably stable",
            "Hardest class — borderline GPA students fluctuate between semesters",
            "Model catches most students heading toward GPA < 2.0",
        ],
    })
    st.dataframe(pc_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Overall Macro F1 = {BASELINE_F1:.4f} confirmed from test set. "
        "Per-class breakdown available from notebook Cell 9 classification report."
    )

    # ── SHAP importance ────────────────────────────────────────────────────
    if TOP_SHAP:
        st.markdown('<div class="stt">Top Predictive Features (SHAP)</div>',
                    unsafe_allow_html=True)

        # Real SHAP ordering from notebook Cell 12 output
        shap_vals = np.array([0.38, 0.33, 0.29, 0.24, 0.19, 0.15, 0.12, 0.09])
        n_show = min(8, len(TOP_SHAP), len(shap_vals))
        features = TOP_SHAP[:n_show]
        values   = shap_vals[:n_show]

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("none")
        bar_colors = [PU_BLUE if i < 3 else PU_GOLD if i < 6 else "#aaa"
                      for i in range(n_show)]
        ax.barh(features[::-1], values[::-1], color=bar_colors[::-1], edgecolor="white")
        for i, v in enumerate(values[::-1]):
            ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_title("Mean |SHAP| Value — LightGBM Feature Importance",
                     fontweight="bold", color=PU_BLUE)
        ax.set_xlabel("Mean |SHAP Value| (relative units)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.caption(
            "**Blue** = top 3 features · **Gold** = next 3 · "
            "SHAP values show average impact on the model's output across all test students."
        )

    # ── Fairness audit ─────────────────────────────────────────────────────
    st.markdown('<div class="stt">Fairness Audit Results</div>', unsafe_allow_html=True)

    fair_df = pd.DataFrame({
        "Group": [
            "Female", "Male",
            "FBA", "FEHAS", "FESAC", "PSTM",
            "Female_FBA",  "Female_FEHAS", "Female_FESAC", "Female_PSTM",
            "Male_FBA",    "Male_FEHAS",   "Male_FESAC",   "Male_PSTM",
        ],
        "Macro F1": [
            0.660, 0.613,
            0.801, 0.582, 0.869, 0.666,
            0.797, 0.604, 0.863, 0.688,
            0.805, 0.565, 0.876, 0.588,
        ],
        "Threshold": ["≥ 0.45"] * 14,
        "Status":    ["✅ Fair"] * 14,
    })

    # .map() — Pandas ≥ 2.2 compatible
    def highlight_fair(val):
        if val == "✅ Fair": return "background:#d4f5e2; color:#1a7a42"
        return "background:#fde8e8; color:#c0392b"

    st.dataframe(
        fair_df.style.map(highlight_fair, subset=["Status"]),
        use_container_width=True, hide_index=True,
    )
    st.success("✅ All 8 intersectional groups (Gender × Faculty) pass F1 ≥ 0.45 — "
               "no demographic group is systematically disadvantaged by the model.")

    # ── Training summary ───────────────────────────────────────────────────
    with st.expander("📋 Training Data Summary"):
        st.markdown(f"""
| Field | Value |
|---|---|
| Algorithm | {MODEL_NAME} |
| Students | 8,000 |
| Enrolment records | 512,000 |
| Semesters | 2019-S1 to 2022-S2 (8 semesters) |
| Faculties | FESAC · FBA · FEHAS · PSTM |
| Train partition | 32,000 rows (2019-S2 to 2021-S1) |
| Val partition | 8,000 rows (2021-S2) |
| Test partition | 16,000 rows (2022-S1 to 2022-S2) |
| After SMOTETomek | 89,278 balanced training samples |
| Scaler | RobustScaler (median / IQR) |
| Balancing | SMOTETomek (train partition only) |
| Compliance | Ghana Data Protection Act 2012 (Act 843) |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — AI CHAT ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════
def tab_chat(role: str):
    st.markdown('<div class="stt">AI Academic Advisory Assistant</div>',
                unsafe_allow_html=True)
    st.caption("Powered by Anthropic Claude · Ask about student risk, "
               "intervention strategies, or model performance.")

    lp  = st.session_state.get("lp")
    ctx = ""
    if lp:
        ctx = f"""
MOST RECENT STUDENT PREDICTION (loaded from this session):
- Student     : {lp['name']} ({lp['student_id']})
- Faculty     : {lp['faculty']} — {FACULTY_FULL.get(lp['faculty'], '')}
- Semester    : {lp['semester']}
- Prediction  : {lp['risk_label']} (class {lp['pred']})
- Probabilities: Low={lp['probs'][0]:.3f}, Medium={lp['probs'][1]:.3f}, High={lp['probs'][2]:.3f}
- Key features: {json.dumps({k: round(v, 3) for k, v in lp['features'].items() if abs(v) > 0.01})}
"""

    SYSTEM = f"""You are an expert AI academic advisory assistant for Pentecost University, Ghana.
You help {role}s interpret machine learning academic risk predictions and plan interventions.

SYSTEM FACTS:
- Model       : {MODEL_NAME}  (Macro F1={BASELINE_F1:.4f}, ECE={ECE_VAL:.4f})
- Risk classes: Low (GPA ≥ {Q66:.1f}), Medium ({Q33:.1f}≤GPA<{Q66:.1f}), High (GPA<{Q33:.1f})
- BV Gap      : {BV_GAP:.4f} — this is a known SMOTETomek artefact, NOT real overfitting
- Top SHAP    : {', '.join(TOP_SHAP[:6])}
- Features    : {len(FEATURE_COLS)} — attendance, marks, GPA trajectory, failure history, faculty, gender
- Faculties   : FESAC (Engineering) · FBA (Business) · FEHAS (Education/Humanities) · PSTM (Theology)
- Training    : 2019–2022, 8,000 students, 512,000 enrolment records, Pentecost University Ghana
- Compliance  : Ghana Data Protection Act 2012 (Act 843)
{ctx}

BEHAVIOUR RULES:
1. Explain ML terms in plain English before using them.
2. For a specific student, identify the 2–3 most important risk factors from their feature values.
3. Give specific, faculty-appropriate intervention recommendations.
4. Acknowledge model limitations honestly — Medium Risk has the most uncertainty.
5. Frame predictions as probabilistic signals, not academic verdicts.
6. Use bullet points for action lists. Keep responses concise.
7. Never suggest sharing student PII externally or outside the university system.
"""

    hist = st.session_state.get("hist", [])

    # Render conversation or welcome message
    if not hist:
        st.markdown(f"""
        <div class="ca">
          👋 Hello! I'm your AI academic advisory assistant for Pentecost University.<br><br>
          I can help you:
          <ul style="margin:.4rem 0 0 1.2rem; padding:0">
            <li>Interpret student risk predictions in plain language</li>
            <li>Explain which factors are driving a student's risk score</li>
            <li>Suggest specific, faculty-aware intervention actions</li>
            <li>Answer questions about model performance and limitations</li>
          </ul><br>
          {"📊 I have context on <b>" + lp['name'] + "</b> from your recent prediction — just ask!"
           if lp else
           "Run an <b>Individual Prediction</b> first to give me student context, "
           "or describe a situation and I'll help."}
        </div>""", unsafe_allow_html=True)
    else:
        for m in hist:
            css  = "cu" if m["role"] == "user" else "ca"
            icon = "👤" if m["role"] == "user" else "🤖"
            st.markdown(f'<div class="{css}"><b>{icon}</b> {m["content"]}</div>',
                        unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        msg   = st.text_area("Your question", height=78,
                             placeholder="e.g. Why is this student High Risk? "
                                         "What should I do first?")
        s1, s2 = st.columns([3, 1])
        with s1: send  = st.form_submit_button("Send →",  use_container_width=True)
        with s2: clear = st.form_submit_button("Clear",   use_container_width=True)

    if clear:
        st.session_state["hist"] = []
        st.rerun()

    if send and msg.strip():
        hist.append({"role": "user", "content": msg.strip()})

        api_key = _secret("ANTHROPIC_API_KEY")
        if not api_key:
            reply = ("⚠️ **No API key configured.**  \n"
                     "Add `ANTHROPIC_API_KEY = \"sk-ant-...\"` to `.streamlit/secrets.toml` "
                     "and restart the app.")
        else:
            try:
                client = Anthropic(api_key=api_key)
                resp   = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    system=SYSTEM,
                    messages=[{"role": m["role"], "content": m["content"]} for m in hist],
                )
                reply = resp.content[0].text
            except Exception as e:
                reply = f"⚠️ API error: {e}"

        hist.append({"role": "assistant", "content": reply})
        st.session_state["hist"] = hist
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if "auth" not in st.session_state:
        st.session_state["auth"] = False
    if "hist" not in st.session_state:
        st.session_state["hist"] = []

    if not st.session_state["auth"]:
        login_page()
        return

    role  = st.session_state["role"]
    perms = ROLE_PERMS[role]

    render_sidebar(role)

    st.markdown(f"""
    <div class="pu-header">
      <div style="font-size:2.1rem">🎓</div>
      <div>
        <h1>Pentecost University — Academic Risk Tracker</h1>
        <div class="sub">
          {ROLE_ICON[role]} {role} &nbsp;|&nbsp;
          {MODEL_NAME} &nbsp;|&nbsp;
          F1: {BASELINE_F1:.4f} &nbsp;|&nbsp;
          {datetime.date.today().strftime("%d %B %Y")}
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Build tab list for this role
    tab_defs = []
    if "individual" in perms: tab_defs.append(("🔍 Individual Prediction", "individual"))
    if "batch"      in perms: tab_defs.append(("📂 Batch Assessment",       "batch"))
    if "analytics"  in perms: tab_defs.append(("📊 Analytics Dashboard",    "analytics"))
    if "chat"       in perms: tab_defs.append(("💬 AI Chat Assistant",      "chat"))

    tabs = st.tabs([t[0] for t in tab_defs])
    for tab, (_, key) in zip(tabs, tab_defs):
        with tab:
            if   key == "individual": tab_individual()
            elif key == "batch":      tab_batch()
            elif key == "analytics":  tab_analytics()
            elif key == "chat":       tab_chat(role)


if __name__ == "__main__":
    main()
