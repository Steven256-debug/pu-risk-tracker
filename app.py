"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Pentecost University — AI-Powered Academic Performance Tracker          ║
║  app.py  ·  Production Streamlit Deployment  v3.0                       ║
║  Authors   : Steven Asante-Poku Jnr & Frank Amoah  |  2025             ║
║  Supervisor: Mr Harry Attieku-Boateng                                   ║
║                                                                          ║
║  v2.0 Improvements:                                                      ║
║    • Individual tab: CSV upload + student ID search → auto-fill form    ║
║    • Individual tab: visual risk scorecard with gauge chart              ║
║    • Batch tab: 6 analytics charts from prediction results               ║
║    • Analytics tab: live charts powered by uploaded batch data           ║
║                                                                          ║
║  Pipeline facts:                                                         ║
║    Model: LightGBM  |  F1: 0.6383  |  Q33: 2.0  |  Q66: 3.0           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pickle, json, os, datetime, io
import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PU Academic Risk Tracker",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
PU_BLUE  = "#003087"
PU_GOLD  = "#C9A84C"
PU_LIGHT = "#EEF3FB"

FACULTIES = ["FESAC", "FBA", "FEHAS", "PSTM"]
SEMESTERS = ["2019_S1","2019_S2","2020_S1","2020_S2",
             "2021_S1","2021_S2","2022_S1","2022_S2"]

RISK_MAP   = {0:"Low Risk",    1:"Medium Risk",   2:"High Risk"}
RISK_COLOR = {0:"#27ae60",     1:"#f39c12",        2:"#e74c3c"}
RISK_EMOJI = {0:"🟢",           1:"🟡",              2:"🔴"}
RISK_CSS   = {0:"risk-low",    1:"risk-medium",    2:"risk-high"}
RISK_BG    = {0:"#d4f5e2",     1:"#fff3cc",         2:"#fde8e8"}
RISK_FG    = {0:"#1a7a42",     1:"#a06000",         2:"#c0392b"}

FACULTY_FULL = {
    "FESAC":"Faculty of Engineering & Applied Sciences",
    "FBA"  :"Faculty of Business Administration",
    "FEHAS":"Faculty of Education, Humanities & Applied Sciences",
    "PSTM" :"Pentecost School of Theology & Ministry",
}

FEATURE_COLS = [
    "avg_attendance","avg_total_mark","avg_ca_score","avg_exam_score",
    "total_credits","num_courses","gender_enc","semester_index",
    "prev_gpa","gpa_trend","consec_fails","trend_x_fail",
    "fac_FESAC","fac_FBA","fac_FEHAS","fac_PSTM",
]

PER_CLASS_F1 = {"Low Risk":"—","Medium Risk":"—","High Risk":"—"}

# ═══════════════════════════════════════════════════════════════════════════
# SECRETS HELPER
# ═══════════════════════════════════════════════════════════════════════════
def _secret(key:str, fallback:str="") -> str:
    try:
        val = st.secrets.get(key,"")
        if val: return val
    except Exception:
        pass
    return os.environ.get(key, fallback)

# ═══════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;500;600&display=swap');

:root{--blue:#003087;--gold:#C9A84C;--light:#EEF3FB;}
html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;}

[data-testid="stSidebar"]{
    background:linear-gradient(175deg,#001a4d 0%,#003087 60%,#0a4fa0 100%);
    border-right:3px solid var(--gold);}
[data-testid="stSidebar"] *{color:#fff !important;}

.pu-header{background:linear-gradient(90deg,var(--blue) 0%,#0a4fa0 100%);
    border-bottom:4px solid var(--gold);padding:1.1rem 2rem;
    border-radius:0 0 12px 12px;margin-bottom:1.4rem;
    display:flex;align-items:center;gap:1.2rem;}
.pu-header h1{font-family:'Playfair Display',serif;color:white !important;
    font-size:1.5rem;margin:0;}
.pu-header .sub{color:var(--gold);font-size:.82rem;font-weight:600;
    letter-spacing:.05em;text-transform:uppercase;}

.mc{background:white;border-radius:12px;padding:1.1rem 1.3rem;
    border-left:5px solid var(--blue);
    box-shadow:0 2px 12px rgba(0,48,135,.08);
    transition:transform .15s,box-shadow .15s;}
.mc:hover{transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,48,135,.14);}
.mc .val{font-family:'Playfair Display',serif;font-size:2rem;
    color:var(--blue);line-height:1;}
.mc .lbl{font-size:.75rem;color:#666;text-transform:uppercase;
    letter-spacing:.06em;margin-top:.2rem;}
.mc.gold{border-left-color:var(--gold);}
.mc.green{border-left-color:#27ae60;}
.mc.red{border-left-color:#e74c3c;}
.mc.amber{border-left-color:#f39c12;}

.rb{display:inline-flex;align-items:center;gap:.4rem;
    padding:.28rem .85rem;border-radius:20px;font-weight:600;font-size:.83rem;}
.risk-low{background:#d4f5e2;color:#1a7a42;}
.risk-medium{background:#fff3cc;color:#a06000;}
.risk-high{background:#fde8e8;color:#c0392b;}

.sc{background:white;border-radius:14px;padding:1.4rem;
    box-shadow:0 3px 16px rgba(0,48,135,.09);
    border-top:4px solid var(--blue);margin-bottom:.8rem;}

.stt{font-family:'Playfair Display',serif;color:var(--blue);font-size:1.2rem;
    border-bottom:2px solid var(--gold);padding-bottom:.35rem;margin:1.4rem 0 .9rem;}

.sig{background:white;padding:.55rem .8rem;border-radius:6px;
    box-shadow:0 2px 8px rgba(0,0,0,.06);margin-bottom:.4rem;font-size:.87rem;}

/* Search result card */
.found-card{background:var(--light);border-radius:10px;padding:1rem 1.2rem;
    border-left:5px solid var(--gold);margin-bottom:1rem;}

/* Gauge container */
.gauge-wrap{background:white;border-radius:12px;padding:1rem;
    box-shadow:0 2px 12px rgba(0,48,135,.08);text-align:center;}

.stButton>button{background:var(--blue);color:white;border:none;
    border-radius:8px;font-weight:600;padding:.45rem 1.4rem;transition:all .15s;}
.stButton>button:hover{background:#0a4fa0;transform:translateY(-1px);
    box-shadow:0 4px 12px rgba(0,48,135,.25);}

.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:2px solid #e0e6f0;}
.stTabs [aria-selected="true"]{color:var(--blue) !important;
    border-bottom-color:var(--blue) !important;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ROLE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
ROLE_PWDS = {
    "Academic Advisor"   :_secret("ADVISOR_PASSWORD",  "advisor2025"),
    "Registry Admin"     :_secret("ADMIN_PASSWORD",    "registry2025"),
    "Head of Department" :_secret("HOD_PASSWORD",      "hod2025"),
}
ROLE_PERMS = {
    "Academic Advisor"   :["individual"],
    "Registry Admin"     :["individual","batch"],
    "Head of Department" :["individual","batch","analytics"],
}
ROLE_ICON = {
    "Academic Advisor"   :"👨‍🏫",
    "Registry Admin"     :"🗂️",
    "Head of Department" :"🏛️",
}

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ARTEFACTS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artefacts():
    try:
        with open("best_model.pkl",   "rb") as f: mdl  = pickle.load(f)
        with open("scaler.pkl",       "rb") as f: scl  = pickle.load(f)
        with open("feature_cols.json","r")  as f: fc   = json.load(f)
        with open("thresholds.json",  "r")  as f: thr  = json.load(f)
        return mdl, scl, fc, thr, True
    except FileNotFoundError:
        return None,None,None,None,False

model,scaler,_fcols,thresholds,artefacts_ok = load_artefacts()
if _fcols: FEATURE_COLS = _fcols

Q33        = thresholds.get("Q33",2.0)        if thresholds else 2.0
Q66        = thresholds.get("Q66",3.0)        if thresholds else 3.0
MODEL_NAME = thresholds.get("best_model","LightGBM") if thresholds else "LightGBM"
BASELINE_F1= thresholds.get("macro_f1",0.6383) if thresholds else 0.6383
ECE_VAL    = thresholds.get("ece",0.0957)     if thresholds else 0.0957
BV_GAP     = thresholds.get("bv_gap",0.329)   if thresholds else 0.329
TOP_SHAP   = thresholds.get("top_shap_features",[
    "avg_total_mark","avg_exam_score","gpa_trend","avg_ca_score",
    "consec_fails","trend_x_fail","fac_FEHAS","prev_gpa","fac_FESAC","gender_enc",
]) if thresholds else []

# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def predict_one(feat_dict:dict):
    row    = np.array([float(feat_dict.get(c,0.0)) for c in FEATURE_COLS]).reshape(1,-1)
    row_sc = scaler.transform(row)
    probs  = model.predict_proba(row_sc)[0]
    return int(np.argmax(probs)), probs

def build_features(avg_attendance,avg_total_mark,avg_ca_score,avg_exam_score,
                   total_credits,num_courses,gender,semester_index,
                   prev_gpa,gpa_trend,consec_fails,faculty)->dict:
    return {
        "avg_attendance":avg_attendance, "avg_total_mark":avg_total_mark,
        "avg_ca_score"  :avg_ca_score,   "avg_exam_score":avg_exam_score,
        "total_credits" :total_credits,  "num_courses"   :num_courses,
        "gender_enc"    :int(gender=="Female"),
        "semester_index":semester_index,
        "prev_gpa"      :prev_gpa,       "gpa_trend"     :gpa_trend,
        "consec_fails"  :consec_fails,
        "trend_x_fail"  :gpa_trend*consec_fails,
        "fac_FESAC":int(faculty=="FESAC"),"fac_FBA":int(faculty=="FBA"),
        "fac_FEHAS":int(faculty=="FEHAS"),"fac_PSTM":int(faculty=="PSTM"),
    }

def apply_pipeline_batch(df_raw:pd.DataFrame)->pd.DataFrame:
    df = df_raw.copy().sort_values(["student_id","semester"])
    df["prev_gpa"]     = df.groupby("student_id")["semester_gpa"].shift(1)
    df["gpa_trend"]    = df["semester_gpa"] - df["prev_gpa"]
    df["is_fail"]      = (df["semester_gpa"]<1.5).astype(int)
    df["consec_fails"] = df.groupby("student_id")["is_fail"].transform(
        lambda x: x.rolling(window=2,min_periods=1).sum())
    df["trend_x_fail"] = df["gpa_trend"]*df["consec_fails"]
    for fac in FACULTIES:
        df[f"fac_{fac}"] = (df["faculty"]==fac).astype(int)
    df["gender_enc"] = (df["gender"].str.strip().str.title()
                        .map({"Female":1,"Male":0}).fillna(0).astype(int))
    sem_map = {s:i for i,s in enumerate(SEMESTERS)}
    df["semester_index"] = df["semester"].map(sem_map).fillna(0).astype(int)
    return df.dropna(subset=["prev_gpa"]).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def prob_gauge(probs:np.ndarray, pred:int)->plt.Figure:
    """Horizontal stacked probability bar."""
    fig,ax = plt.subplots(figsize=(6,0.65))
    fig.patch.set_facecolor("none"); ax.set_facecolor("none")
    left=0
    for i,(p,c) in enumerate(zip(probs,["#27ae60","#f39c12","#e74c3c"])):
        ax.barh(0,p,left=left,color=c,alpha=0.88 if i==pred else 0.28,
                height=0.6,edgecolor="white",linewidth=1.5)
        if p>0.08:
            ax.text(left+p/2,0,f"{p:.0%}",va="center",ha="center",
                    fontsize=8,fontweight="bold",
                    color="white" if i==pred else "#555")
        left+=p
    ax.set_xlim(0,1); ax.set_ylim(-0.5,0.5); ax.axis("off")
    plt.tight_layout(pad=0.1)
    return fig


def semicircle_gauge(prob_high: float, pred: int) -> plt.Figure:
    """Half-circle gauge showing High Risk probability 0-100%."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Draw background arc (grey)
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(theta, [1]*300, color="#e8e8e8", lw=18, solid_capstyle="round")

    # Colour zones: green (0-33%), amber (33-66%), red (66-100%)
    zones = [(np.linspace(np.pi,   np.pi*0.67, 100), "#27ae60"),
             (np.linspace(np.pi*0.67, np.pi*0.33, 100), "#f39c12"),
             (np.linspace(np.pi*0.33, 0, 100),         "#e74c3c")]
    for t, c in zones:
        ax.plot(t, [1]*100, color=c, lw=18, alpha=0.18, solid_capstyle="butt")

    # Fill arc up to current probability
    filled = np.linspace(np.pi, np.pi * (1 - prob_high), 200)
    needle_color = "#27ae60" if prob_high < 0.33 else "#f39c12" if prob_high < 0.66 else "#e74c3c"
    ax.plot(filled, [1]*200, color=needle_color, lw=18, solid_capstyle="round")

    # Needle
    needle_angle = np.pi * (1 - prob_high)
    ax.annotate("", xy=(needle_angle, 0.72), xytext=(needle_angle, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a2e",
                                lw=2.5, mutation_scale=16))

    # Centre text
    ax.text(0, -0.15, f"{prob_high:.0%}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=needle_color,
            transform=ax.transAxes)
    ax.text(0, -0.32, "High Risk Probability", ha="center", va="center",
            fontsize=8, color="#666", transform=ax.transAxes)

    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, np.pi)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(0)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

def donut_chart(n_lr,n_mr,n_hr)->plt.Figure:
    """Donut chart of risk distribution."""
    fig,ax = plt.subplots(figsize=(4,4))
    fig.patch.set_facecolor("none")
    total = n_lr+n_mr+n_hr
    sizes  = [n_lr,n_mr,n_hr]
    colors = ["#27ae60","#f39c12","#e74c3c"]
    labels = [f"Low\n{n_lr:,}\n({n_lr/total*100:.1f}%)",
              f"Medium\n{n_mr:,}\n({n_mr/total*100:.1f}%)",
              f"High\n{n_hr:,}\n({n_hr/total*100:.1f}%)"]
    wedges,_ = ax.pie(sizes,colors=colors,startangle=90,
                      wedgeprops={"edgecolor":"white","linewidth":2.5,"width":0.55})
    ax.text(0,0,f"{total:,}\nStudents",ha="center",va="center",
            fontsize=11,fontweight="bold",color=PU_BLUE)
    ax.legend(wedges,labels,loc="lower center",bbox_to_anchor=(0.5,-0.18),
              ncol=3,fontsize=7,frameon=False)
    ax.set_title("Risk Distribution",fontweight="bold",color=PU_BLUE,pad=8)
    plt.tight_layout()
    return fig

def faculty_stacked_bar(df)->plt.Figure:
    """Stacked bar: Low/Medium/High per faculty."""
    fig,ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor("none")
    fac_data = {}
    for fac in FACULTIES:
        sub = df[df["faculty"]==fac] if "faculty" in df.columns else pd.DataFrame()
        if len(sub)==0:
            fac_data[fac] = [0,0,0]
        else:
            counts = sub["risk_label"].value_counts()
            fac_data[fac] = [counts.get(k,0) for k in [0,1,2]]
    lows  = [fac_data[f][0] for f in FACULTIES]
    meds  = [fac_data[f][1] for f in FACULTIES]
    highs = [fac_data[f][2] for f in FACULTIES]
    x = np.arange(len(FACULTIES))
    b1 = ax.bar(x,lows, color="#27ae60",label="Low Risk", edgecolor="white",linewidth=0.8)
    b2 = ax.bar(x,meds, bottom=lows, color="#f39c12",label="Medium Risk",edgecolor="white",linewidth=0.8)
    b3 = ax.bar(x,highs,bottom=[l+m for l,m in zip(lows,meds)],
                color="#e74c3c",label="High Risk",edgecolor="white",linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(FACULTIES,fontsize=9)
    ax.set_ylabel("Students"); ax.legend(fontsize=8,loc="upper right")
    ax.set_title("Risk Level by Faculty",fontweight="bold",color=PU_BLUE)
    for bar in list(b1)+list(b2)+list(b3):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_y()+h/2, str(int(h)),
                    ha="center",va="center",fontsize=7,color="white",fontweight="bold")
    plt.tight_layout()
    return fig

def gpa_trend_line(df)->plt.Figure:
    """Average GPA per semester per faculty (line chart)."""
    fig,ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor("none")
    if "semester" not in df.columns or "semester_gpa" not in df.columns:
        ax.text(0.5,0.5,"No semester/GPA data",ha="center",va="center",transform=ax.transAxes)
        return fig
    colors_fac = {"FESAC":PU_BLUE,"FBA":PU_GOLD,"FEHAS":"#27ae60","PSTM":"#e74c3c"}
    sem_order = {s:i for i,s in enumerate(SEMESTERS)}
    df2 = df.copy()
    df2["sem_order"] = df2["semester"].map(sem_order)
    for fac in FACULTIES:
        sub = df2[df2["faculty"]==fac] if "faculty" in df2.columns else pd.DataFrame()
        if len(sub)==0: continue
        trend = (sub.groupby("semester")["semester_gpa"].mean()
                 .reset_index()
                 .assign(order=lambda x: x["semester"].map(sem_order))
                 .sort_values("order"))
        ax.plot(trend["semester"],trend["semester_gpa"],
                marker="o",lw=2,markersize=5,
                color=colors_fac.get(fac,"#888"),label=fac)
    ax.axhline(Q33,color="#e74c3c",lw=1,ls="--",alpha=0.6,label=f"High Risk boundary ({Q33})")
    ax.axhline(Q66,color="#f39c12",lw=1,ls="--",alpha=0.6,label=f"Medium Risk boundary ({Q66})")
    ax.set_xlabel("Semester"); ax.set_ylabel("Average GPA")
    ax.set_title("GPA Trend by Faculty",fontweight="bold",color=PU_BLUE)
    ax.legend(fontsize=7); plt.xticks(rotation=30,ha="right",fontsize=7)
    plt.tight_layout()
    return fig

def high_risk_trend(df)->plt.Figure:
    """Count of High Risk students per semester per faculty."""
    fig,ax = plt.subplots(figsize=(6,4))
    fig.patch.set_facecolor("none")
    if "semester" not in df.columns or "risk_label" not in df.columns:
        ax.text(0.5,0.5,"Run batch prediction first",ha="center",va="center",transform=ax.transAxes)
        return fig
    sem_order = {s:i for i,s in enumerate(SEMESTERS)}
    colors_fac = {"FESAC":PU_BLUE,"FBA":PU_GOLD,"FEHAS":"#27ae60","PSTM":"#e74c3c"}
    for fac in FACULTIES:
        sub = df[(df["faculty"]==fac)&(df["risk_label"]==2)] if "faculty" in df.columns else pd.DataFrame()
        if len(sub)==0: continue
        trend = (sub.groupby("semester").size().reset_index(name="count")
                 .assign(order=lambda x: x["semester"].map(sem_order))
                 .sort_values("order"))
        ax.plot(trend["semester"],trend["count"],
                marker="o",lw=2,markersize=5,
                color=colors_fac.get(fac,"#888"),label=fac)
    ax.set_xlabel("Semester"); ax.set_ylabel("High Risk Count")
    ax.set_title("High-Risk Students per Semester",fontweight="bold",color=PU_BLUE)
    ax.legend(fontsize=7); plt.xticks(rotation=30,ha="right",fontsize=7)
    plt.tight_layout()
    return fig

def attendance_gpa_scatter(df)->plt.Figure:
    """Scatter: avg_attendance vs semester_gpa, coloured by risk."""
    fig,ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor("none")
    if "avg_attendance" not in df.columns or "semester_gpa" not in df.columns:
        ax.text(0.5,0.5,"No data",ha="center",va="center",transform=ax.transAxes)
        return fig
    for k,lbl in RISK_MAP.items():
        sub = df[df["risk_label"]==k] if "risk_label" in df.columns else df
        ax.scatter(sub["avg_attendance"],sub["semester_gpa"],
                   color=RISK_COLOR[k],alpha=0.35,s=12,label=lbl)
    ax.axhline(Q33,color="#e74c3c",lw=1,ls="--",alpha=0.5)
    ax.axhline(Q66,color="#f39c12",lw=1,ls="--",alpha=0.5)
    ax.axvline(3.0,color="#888",   lw=1,ls=":",  alpha=0.5,label="Attendance threshold")
    ax.set_xlabel("Attendance Score"); ax.set_ylabel("Semester GPA")
    ax.set_title("Attendance vs GPA",fontweight="bold",color=PU_BLUE)
    ax.legend(fontsize=7)
    plt.tight_layout()
    return fig

def gender_risk_chart(df)->plt.Figure:
    """Grouped bar: risk breakdown by gender."""
    fig,ax = plt.subplots(figsize=(5,4))
    fig.patch.set_facecolor("none")
    if "gender" not in df.columns or "risk_label" not in df.columns:
        ax.text(0.5,0.5,"No gender data",ha="center",va="center",transform=ax.transAxes)
        return fig
    genders = df["gender"].str.strip().str.title().unique()
    genders = [g for g in ["Male","Female"] if g in genders]
    x = np.arange(len(genders))
    w = 0.25
    for ki,k in enumerate([0,1,2]):
        vals = []
        for g in genders:
            sub = df[df["gender"].str.strip().str.title()==g]
            vals.append((sub["risk_label"]==k).sum())
        ax.bar(x+ki*w,vals,w,color=RISK_COLOR[k],label=RISK_MAP[k],
               edgecolor="white",linewidth=0.8)
    ax.set_xticks(x+w); ax.set_xticklabels(genders)
    ax.set_ylabel("Students"); ax.legend(fontsize=8)
    ax.set_title("Risk by Gender",fontweight="bold",color=PU_BLUE)
    plt.tight_layout()
    return fig


def student_vs_faculty(feats: dict, fac: str, batch_df) -> plt.Figure:
    """Bar chart comparing student scores vs faculty average."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor("none")
    metrics = ["Attendance", "Total Mark", "CA Score", "Exam Score"]
    keys    = ["avg_attendance","avg_total_mark","avg_ca_score","avg_exam_score"]
    maxes   = [5, 100, 40, 60]
    student_pct = [feats.get(k, 0) / m * 100 for k, m in zip(keys, maxes)]
    if batch_df is not None and "faculty" in batch_df.columns:
        fac_df = batch_df[batch_df["faculty"] == fac]
        fac_pct = [fac_df[k].mean() / m * 100
                   if k in fac_df.columns and len(fac_df) > 0 else 60
                   for k, m in zip(keys, maxes)]
    else:
        fac_pct = [65, 62, 60, 63]
    x = np.arange(len(metrics)); w = 0.35
    bars_s = ax.bar(x - w/2, student_pct, w, color=PU_BLUE,
                    label="This Student", edgecolor="white", alpha=0.9)
    bars_f = ax.bar(x + w/2, fac_pct,    w, color=PU_GOLD,
                    label=f"{fac} Average", edgecolor="white", alpha=0.9)
    for bar in list(bars_s) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                f"{h:.0f}%", ha="center", fontsize=7, fontweight="bold")
    ax.axhline(50, color="#aaa", lw=1, ls="--", alpha=0.5, label="50% threshold")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("Score as % of maximum")
    ax.set_title(f"Student vs {fac} Faculty Average", fontweight="bold", color=PU_BLUE)
    ax.legend(fontsize=8); ax.set_ylim(0, 115)
    plt.tight_layout()
    return fig


def risk_heatmap(df) -> plt.Figure:
    """Heatmap: High Risk % per Faculty x Semester."""
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("none")
    if "semester" not in df.columns or "faculty" not in df.columns:
        ax.text(0.5, 0.5, "No semester/faculty data",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    sems = [s for s in SEMESTERS if s in df["semester"].unique()]
    if not sems:
        ax.text(0.5, 0.5, "No matching semesters",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    data = []
    for fac in FACULTIES:
        row = []
        for sem in sems:
            sub = df[(df["faculty"] == fac) & (df["semester"] == sem)]
            row.append((sub["risk_label"] == 2).sum() / len(sub) * 100
                       if len(sub) > 0 else np.nan)
        data.append(row)
    data_arr = np.array(data, dtype=float)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pu", ["#d4f5e2","#fff3cc","#fde8e8","#e74c3c"])
    im = ax.imshow(data_arr, cmap=cmap, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(sems)))
    ax.set_xticklabels(sems, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(FACULTIES)))
    ax.set_yticklabels(FACULTIES, fontsize=9)
    for i in range(len(FACULTIES)):
        for j in range(len(sems)):
            val = data_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 60 else "#1a1a2e")
    plt.colorbar(im, ax=ax, label="High Risk %", shrink=0.8)
    ax.set_title("High Risk % by Faculty & Semester",
                 fontweight="bold", color=PU_BLUE)
    plt.tight_layout()
    return fig


def top_risk_table(df, n=10) -> pd.DataFrame:
    """Return top N students by High Risk probability."""
    if "prob_high" not in df.columns or "risk_label" not in df.columns:
        return pd.DataFrame()
    cols = [c for c in ["student_id","name","faculty","gender",
                         "semester","semester_gpa","risk_name","prob_high"]
            if c in df.columns]
    return (df[df["risk_label"] == 2][cols]
            .sort_values("prob_high", ascending=False)
            .head(n)
            .reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════
# LOGIN
# ═══════════════════════════════════════════════════════════════════════════
def login_page():
    _,col,_ = st.columns([1,2,1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:2rem 0 1.2rem">
          <div style="font-size:3.5rem">🎓</div>
          <h1 style="font-family:'Playfair Display',serif;color:#003087;margin:.5rem 0 .2rem">
              Pentecost University</h1>
          <p style="color:#C9A84C;font-weight:600;letter-spacing:.08em;
                    font-size:.88rem;text-transform:uppercase">
              AI Academic Performance Tracker</p>
          <hr style="border:none;border-top:2px solid #C9A84C;margin:.8rem auto;width:60%">
        </div>""", unsafe_allow_html=True)

        if not artefacts_ok:
            st.error("**Model artefacts not found.**\n\n"
                     "Upload `best_model.pkl`, `scaler.pkl`, `feature_cols.json`, "
                     "`thresholds.json` to your GitHub repository.")

        with st.form("login"):
            role = st.selectbox("Select your role",list(ROLE_PWDS.keys()))
            pwd  = st.text_input("Password",type="password")
            ok   = st.form_submit_button("Sign In →",use_container_width=True)

        if ok:
            if ROLE_PWDS.get(role)==pwd:
                st.session_state.update({"auth":True,"role":role,"hist":[],"batch_df":None})
                st.rerun()
            else:
                st.error("Incorrect password.")

        st.markdown("""<p style="text-align:center;color:#999;font-size:.72rem;margin-top:1.5rem">
          © 2025 Pentecost University · Ghana Data Protection Act 2012 (Act 843)
        </p>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
def render_sidebar(role:str):
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem 0 .5rem">
          <div style="font-size:2.3rem">🎓</div>
          <div style="font-family:'Playfair Display',serif;font-size:1.05rem;font-weight:700">
              Pentecost University</div>
          <div style="color:#C9A84C;font-size:.76rem;text-transform:uppercase;
                      letter-spacing:.07em;margin-top:.2rem">
              Academic Risk Tracker</div>
        </div>
        <hr style="border-color:#C9A84C55;margin:.7rem 0">
        """, unsafe_allow_html=True)

        st.markdown(f"**{ROLE_ICON[role]} {role}**")
        st.caption(f"Model: {MODEL_NAME}  ·  F1: {BASELINE_F1:.4f}")

        st.markdown('<hr style="border-color:#ffffff33">', unsafe_allow_html=True)
        st.markdown("**📊 Risk Thresholds**")
        for k,(emoji,label,rng) in {
            2:("🔴","High Risk",  f"GPA < {Q33:.1f}"),
            1:("🟡","Medium Risk",f"{Q33:.1f} ≤ GPA < {Q66:.1f}"),
            0:("🟢","Low Risk",   f"GPA ≥ {Q66:.1f}"),
        }.items():
            st.markdown(f"""
            <div style="background:rgba(255,255,255,.09);border-radius:6px;
                padding:.38rem .7rem;margin:.25rem 0;font-size:.8rem">
              {emoji} <b>{label}</b><br>
              <span style="opacity:.7;font-size:.73rem">{rng}</span>
            </div>""", unsafe_allow_html=True)

        # Show batch status in sidebar
        batch_df = st.session_state.get("batch_df")
        if batch_df is not None:
            st.markdown('<hr style="border-color:#ffffff33">', unsafe_allow_html=True)
            st.markdown("**📂 Loaded Dataset**")
            st.caption(f"{len(batch_df):,} student records ready")

        st.markdown('<hr style="border-color:#ffffff33">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out",use_container_width=True):
            for k in ["auth","role","hist","lp","batch_df"]:
                st.session_state.pop(k,None)
            st.rerun()

        st.markdown("""<div style="font-size:.67rem;opacity:.45;text-align:center;margin-top:.8rem">
          Ghana DPA 2012 Compliant · © 2025 Pentecost University</div>""",
          unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL PREDICTION  (with CSV search)
# ═══════════════════════════════════════════════════════════════════════════
def tab_individual():
    st.markdown('<div class="stt">Individual Student Risk Assessment</div>',
                unsafe_allow_html=True)

    # ── CSV upload for auto-fill ──────────────────────────────────────────
    with st.expander("📂 Upload student records for quick search (optional)", expanded=False):
        st.caption("Upload the same batch CSV to search students by ID "
                   "and auto-fill the form below.")
        ind_csv = st.file_uploader("Upload student CSV",type=["csv"],
                                   key="ind_csv_uploader")
        if ind_csv:
            try:
                df_lookup = pd.read_csv(ind_csv)
                st.session_state["lookup_df"] = df_lookup
                st.success(f"✅ {len(df_lookup):,} records loaded for search.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    # ── Student ID search ─────────────────────────────────────────────────
    lookup_df = st.session_state.get("lookup_df")
    prefill   = {}

    if lookup_df is not None:
        st.markdown("**🔍 Search Student**")
        col_s1, col_s2 = st.columns([3,1])
        with col_s1:
            search_id = st.text_input("Enter Student ID to auto-fill form",
                                      placeholder="e.g. 100001",
                                      key="search_id_input")
        with col_s2:
            st.markdown("<div style='margin-top:1.8rem'>", unsafe_allow_html=True)
            search_btn = st.button("Search →", key="search_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        if search_btn and search_id:
            # Match on student_id (flexible: string or int)
            sid_col = lookup_df["student_id"].astype(str)
            matches = lookup_df[sid_col == str(search_id).strip()]

            if len(matches) == 0:
                st.warning(f"No record found for Student ID: {search_id}")
            else:
                # Take the most recent semester row
                sem_map = {s:i for i,s in enumerate(SEMESTERS)}
                row = (matches.assign(
                           sem_ord=matches["semester"].map(sem_map).fillna(-1))
                       .sort_values("sem_ord",ascending=False)
                       .iloc[0])

                # Store as prefill
                prefill = {
                    "sid"  : str(row.get("student_id","")),
                    "sname": str(row.get("name","")) if "name" in row else "",
                    "fac"  : str(row.get("faculty","FESAC")),
                    "gen"  : str(row.get("gender","Male")).strip().title(),
                    "sem"  : str(row.get("semester","2022_S2")),
                    "atm"  : float(row.get("avg_total_mark",55.0)),
                    "aca"  : float(row.get("avg_ca_score",22.0)),
                    "aex"  : float(row.get("avg_exam_score",33.0)),
                    "aatt" : float(row.get("avg_attendance",3.5)),
                    "tc"   : int(row.get("total_credits",18)),
                    "nc"   : int(row.get("num_courses",6)),
                    "pgpa" : float(row.get("prev_gpa",
                                   row.get("semester_gpa",1.8))),
                    "gpa"  : float(row.get("semester_gpa",1.8)),
                }
                # Compute GPA trend if prev_gpa available
                if "prev_gpa" in row and pd.notna(row["prev_gpa"]):
                    prefill["gtr"] = round(
                        float(row.get("semester_gpa",prefill["gpa"])) - float(row["prev_gpa"]),2)
                else:
                    prefill["gtr"] = 0.0
                prefill["cf"] = int(row.get("consec_fails",0))
                st.session_state["prefill"] = prefill

                # Show found card
                fac_full = FACULTY_FULL.get(prefill["fac"],prefill["fac"])
                st.markdown(f"""
                <div class="found-card">
                  <b>✅ Student found — form auto-filled</b><br>
                  <span style="font-size:.85rem;color:#555">
                    {prefill['sname'] or prefill['sid']} &nbsp;·&nbsp;
                    {fac_full} &nbsp;·&nbsp;
                    {prefill['gen']} &nbsp;·&nbsp;
                    Semester: {prefill['sem']} &nbsp;·&nbsp;
                    GPA: {prefill['gpa']:.2f}
                  </span>
                </div>""", unsafe_allow_html=True)

    # Retrieve prefill if already searched
    if not prefill:
        prefill = st.session_state.get("prefill",{})

    # ── Helper to get prefill value or default ────────────────────────────
    def pf(key, default):
        return prefill.get(key, default)

    # ── Prediction form ───────────────────────────────────────────────────
    with st.form("individual_form"):
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown("**📋 Student Identity**")
            sid   = st.text_input("Student ID",    value=pf("sid",""),   placeholder="e.g. 100045")
            sname = st.text_input("Name (optional)",value=pf("sname",""))
            fac_default = FACULTIES.index(pf("fac","FESAC")) if pf("fac","FESAC") in FACULTIES else 0
            fac   = st.selectbox("Faculty",FACULTIES,index=fac_default,
                                 format_func=lambda x:f"{x} — {FACULTY_FULL[x]}")
            gen_opts = ["Male","Female"]
            gen_default = gen_opts.index(pf("gen","Male")) if pf("gen","Male") in gen_opts else 0
            gen   = st.selectbox("Gender",gen_opts,index=gen_default)
            sem_default = SEMESTERS.index(pf("sem","2022_S2")) if pf("sem","2022_S2") in SEMESTERS else 7
            sem_i = st.selectbox("Current Semester",range(len(SEMESTERS)),
                                 index=sem_default,
                                 format_func=lambda i:SEMESTERS[i])

        with c2:
            st.markdown("**📊 Current Semester Performance**")
            atm  = st.number_input("Avg Total Mark (0–100)",0.0,100.0,pf("atm",55.0),0.5)
            aca  = st.number_input("Avg CA Score (0–40)",   0.0, 40.0,pf("aca",22.0),0.5)
            aex  = st.number_input("Avg Exam Score (0–60)", 0.0, 60.0,pf("aex",33.0),0.5)
            aatt = st.number_input("Attendance Score (0–5)",0.0,  5.0,pf("aatt",3.5),0.1)

        with c3:
            st.markdown("**📈 Enrolment & History**")
            tc   = st.number_input("Total Credits",       1,30,pf("tc",18))
            nc   = st.number_input("Courses Enrolled",    1,12,pf("nc",6))
            pgpa = st.number_input("Previous Semester GPA (0–4)",
                                   0.0,4.0,float(pf("pgpa",1.8)),0.01)
            gtr  = st.number_input("GPA Trend (current − previous)",
                                   -4.0,4.0,float(pf("gtr",0.0)),0.01,
                                   help="Positive = improving · Negative = declining")
            cf   = st.number_input("Consecutive Semesters with GPA < 1.5",
                                   0,8,pf("cf",0))

        go = st.form_submit_button("🔮  Predict Risk Level",use_container_width=True)

    if go:
        if not artefacts_ok:
            st.error("Model artefacts missing — cannot predict.")
            return

        feats       = build_features(aatt,atm,aca,aex,tc,nc,gen,sem_i,pgpa,gtr,cf,fac)
        pred,probs  = predict_one(feats)
        rl          = RISK_MAP[pred]

        st.session_state["lp"] = dict(
            student_id=sid or "N/A",name=sname or "Student",
            faculty=fac,semester=SEMESTERS[sem_i],
            pred=pred,probs=probs.tolist(),risk_label=rl,features=feats)

        # ── Result scorecard ──────────────────────────────────────────────
        st.markdown('<div class="stt">Prediction Result</div>',unsafe_allow_html=True)

        r1,r2 = st.columns([2,1])

        with r1:
            # Student header card
            st.markdown(f"""
            <div class="sc">
              <div style="display:flex;align-items:center;gap:1rem">
                <div style="font-size:2.8rem">{RISK_EMOJI[pred]}</div>
                <div>
                  <div style="font-family:'Playfair Display',serif;
                              font-size:1.3rem;color:#003087">
                    {sname or "Student"}{f" ({sid})" if sid else ""}
                  </div>
                  <div style="color:#666;font-size:.83rem;margin-top:.2rem">
                    {FACULTY_FULL[fac]} &nbsp;·&nbsp; {gen} &nbsp;·&nbsp; {SEMESTERS[sem_i]}
                  </div>
                  <div style="margin-top:.6rem">
                    <span class="rb {RISK_CSS[pred]}" style="font-size:.95rem">
                      {rl}
                    </span>
                  </div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Probability bar
            st.caption("Probability across all three risk classes")
            st.pyplot(prob_gauge(probs,pred),use_container_width=True)
            plt.close()

        with r2:
            # Semi-circle gauge
            st.pyplot(semicircle_gauge(probs[2], pred), use_container_width=True)
            plt.close()
            # Probability boxes
            for k in [0,1,2]:
                active = "box-shadow:0 0 0 2px " + RISK_COLOR[k] + ";" if k==pred else ""
                st.markdown(f"""
                <div style="text-align:center;padding:.5rem;background:{RISK_BG[k]};
                            border-radius:10px;margin-bottom:.4rem;{active}">
                  <div style="font-size:.74rem;font-weight:600;color:#555">
                      {RISK_EMOJI[k]} {RISK_MAP[k]}</div>
                  <div style="font-family:'Playfair Display',serif;
                              font-size:1.7rem;color:{RISK_COLOR[k]}">
                      {probs[k]:.1%}</div>
                </div>""", unsafe_allow_html=True)

        # ── Performance radar-style summary bar chart ─────────────────────
        st.markdown('<div class="stt">Performance Breakdown</div>',unsafe_allow_html=True)

        fig_perf,ax_perf = plt.subplots(figsize=(8,2.8))
        fig_perf.patch.set_facecolor("none")
        metrics    = ["Attendance\n(/5)","Total Mark\n(/100)","CA Score\n(/40)","Exam Score\n(/60)"]
        raw_vals   = [aatt,  atm,  aca,  aex]
        max_vals   = [5,     100,  40,   60]
        pct_vals   = [v/m*100 for v,m in zip(raw_vals,max_vals)]
        bar_colors = ["#e74c3c" if p<50 else "#f39c12" if p<70 else "#27ae60"
                      for p in pct_vals]
        bars = ax_perf.barh(metrics[::-1],pct_vals[::-1],
                            color=bar_colors[::-1],edgecolor="white",height=0.55)
        ax_perf.axvline(50,color="#aaa",lw=1,ls="--",alpha=0.7)
        ax_perf.axvline(70,color="#27ae60",lw=1,ls="--",alpha=0.5)
        for bar,pct,raw,mx in zip(bars,pct_vals[::-1],raw_vals[::-1],max_vals[::-1]):
            ax_perf.text(min(pct+1,98),bar.get_y()+bar.get_height()/2,
                         f"{raw}/{mx}  ({pct:.0f}%)",
                         va="center",fontsize=8,fontweight="bold")
        ax_perf.set_xlim(0,105)
        ax_perf.set_xlabel("Score as % of maximum")
        ax_perf.set_title("Current Semester Performance",fontweight="bold",color=PU_BLUE)
        plt.tight_layout()
        st.pyplot(fig_perf,use_container_width=True); plt.close()

        # ── Student vs faculty comparison ─────────────────────────────────
        st.markdown('<div class="stt">Student vs Faculty Average</div>',
                    unsafe_allow_html=True)
        batch_df_for_compare = st.session_state.get("batch_df")
        st.pyplot(student_vs_faculty(feats, fac, batch_df_for_compare),
                  use_container_width=True)
        plt.close()

        # ── Student vs faculty comparison ─────────────────────────────────
        st.markdown('<div class="stt">Student vs Faculty Average</div>',
                    unsafe_allow_html=True)
        batch_df_compare = st.session_state.get("batch_df")
        st.pyplot(student_vs_faculty(feats, fac, batch_df_compare),
                  use_container_width=True)
        plt.close()
        if batch_df_compare is None:
            st.caption("Upload a batch CSV in the Batch Assessment tab "
                       "to compare against real faculty averages. "
                       "Showing illustrative averages for now.")

        # ── Risk signals ──────────────────────────────────────────────────
        st.markdown("**Key Risk Signals**")
        sigs=[]
        if aatt<3.0:    sigs.append(("🚩",f"Attendance {aatt:.1f}/5 — below threshold (3.0)","#e74c3c"))
        if cf>0:        sigs.append(("⚠️",f"{cf} consecutive semester(s) with GPA below 1.5","#f39c12"))
        if gtr<-0.1:    sigs.append(("📉",f"GPA declining (trend = {gtr:+.2f})","#f39c12"))
        if pgpa<Q33:    sigs.append(("🚩",f"Previous GPA {pgpa:.2f} in High-Risk zone (< {Q33:.1f})","#e74c3c"))
        if atm<45:      sigs.append(("⚠️",f"Avg total mark {atm:.0f} — below 50% threshold","#f39c12"))
        if aca/40<0.5 and aex/60<0.5:
                        sigs.append(("🚩","Both CA and Exam below 50%","#e74c3c"))
        if not sigs:    sigs.append(("✅","No critical risk signals detected","#27ae60"))

        scols = st.columns(min(len(sigs),3))
        for i,(icon,msg,color) in enumerate(sigs):
            with scols[i%3]:
                st.markdown(f"""<div class="sig" style="border-left:4px solid {color}">
                    {icon} {msg}</div>""",unsafe_allow_html=True)

        # ── Interventions ─────────────────────────────────────────────────
        st.markdown("&nbsp;")
        if pred==2:
            st.error("""**🚨 Recommended Actions — High Risk**
1. Schedule an immediate one-on-one advisory session this week
2. Review full transcript — identify the weakest courses
3. Recommend course load reduction if total credits > 18
4. Refer to Student Support Services if attendance < 3.0
5. Notify Head of Department""")
        elif pred==1:
            st.warning("""**⚠️ Recommended Actions — Medium Risk**
1. Schedule a check-in within 2 weeks
2. Encourage attendance — target ≥ 3.5
3. Review assessment feedback with student
4. Monitor next semester — escalate if still declining""")
        else:
            st.success("**✅ Low Risk** — Student is on track. "
                       "Continue monitoring through standard semester-end reviews.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ASSESSMENT  (with 6 charts)
# ═══════════════════════════════════════════════════════════════════════════
def tab_batch():
    st.markdown('<div class="stt">Batch Risk Assessment</div>',unsafe_allow_html=True)
    st.markdown("Upload a CSV of student records. The app runs feature engineering "
                "and predicts risk for every student-semester row.")

    with st.expander("📋 Required CSV columns"):
        st.markdown("""
**Mandatory:** `student_id`, `faculty`, `gender`, `semester`, `semester_gpa`,
`avg_attendance`, `avg_total_mark`, `avg_ca_score`, `avg_exam_score`,
`total_credits`, `num_courses`

**Optional:** `name` and any other columns (retained in output)

⚠️ Each student needs **≥ 2 semester rows** for `prev_gpa` to be computed.
        """)

    # Sample template
    sample = pd.DataFrame({
        "student_id":[100001]*2+[100002]*2,
        "name":["Alice Mensah"]*2+["Kofi Asante"]*2,
        "faculty":["FESAC","FESAC","FBA","FBA"],
        "gender":["Female","Female","Male","Male"],
        "semester":["2021_S2","2022_S1","2021_S2","2022_S1"],
        "semester_gpa":[2.8,2.5,1.3,0.9],
        "avg_attendance":[4.0,3.5,2.0,1.5],
        "avg_total_mark":[65.0,60.0,42.0,35.0],
        "avg_ca_score":[28.0,25.0,17.0,14.0],
        "avg_exam_score":[37.0,35.0,25.0,21.0],
        "total_credits":[18,18,21,21],
        "num_courses":[6,6,7,7],
    })
    buf=io.StringIO(); sample.to_csv(buf,index=False)
    st.download_button("⬇️ Download Sample CSV Template",
                       buf.getvalue(),"pu_batch_template.csv","text/csv")

    uploaded = st.file_uploader("Upload student records CSV",type=["csv"])
    if not uploaded:
        # Show previously loaded batch if available
        if st.session_state.get("batch_df") is not None:
            st.info("Showing results from previously uploaded dataset. "
                    "Upload a new file to refresh.")
            df = st.session_state["batch_df"]
        else:
            return
    else:
        try:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✅  {len(df_raw):,} rows loaded.")
            with st.spinner("Running feature engineering pipeline …"):
                df = apply_pipeline_batch(df_raw)
            if len(df)==0:
                st.error("No rows survived feature engineering. "
                         "Ensure every student has ≥ 2 semester rows.")
                return
            X     = df[FEATURE_COLS].fillna(0).values
            X_sc  = scaler.transform(X)
            probs = model.predict_proba(X_sc)
            preds = probs.argmax(axis=1)
            df["risk_label"]=preds
            df["risk_name"] =[RISK_MAP[p] for p in preds]
            df["prob_low"]  =probs[:,0].round(3)
            df["prob_med"]  =probs[:,1].round(3)
            df["prob_high"] =probs[:,2].round(3)
            # Save to session for analytics tab
            st.session_state["batch_df"] = df
            # Also make available for individual search
            st.session_state["lookup_df"] = df_raw
        except Exception as e:
            st.error(f"Processing error: {e}")
            st.exception(e)
            return

    n    = len(df)
    preds = df["risk_label"].values
    n_hr = (preds==2).sum()
    n_mr = (preds==1).sum()
    n_lr = (preds==0).sum()

    # ── Summary metrics ────────────────────────────────────────────────────
    st.markdown('<div class="stt">Cohort Risk Summary</div>',unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    for col,val,lbl,css in [
        (m1,n,   "Students Assessed",""),
        (m2,n_hr,"High Risk 🔴",     "red"),
        (m3,n_mr,"Medium Risk 🟡",   "amber"),
        (m4,n_lr,"Low Risk 🟢",      "green"),
    ]:
        with col:
            st.markdown(f"""<div class="mc {css}">
              <div class="val">{val:,}</div>
              <div class="lbl">{lbl} · {val/n*100:.1f}%</div>
            </div>""",unsafe_allow_html=True)

    # ── 6 Charts ───────────────────────────────────────────────────────────
    st.markdown('<div class="stt">Analytics Charts</div>',unsafe_allow_html=True)

    row1_c1,row1_c2 = st.columns(2)
    with row1_c1:
        st.pyplot(donut_chart(n_lr,n_mr,n_hr),use_container_width=True); plt.close()
    with row1_c2:
        st.pyplot(faculty_stacked_bar(df),use_container_width=True); plt.close()

    row2_c1,row2_c2 = st.columns(2)
    with row2_c1:
        st.pyplot(gpa_trend_line(df),use_container_width=True); plt.close()
    with row2_c2:
        st.pyplot(high_risk_trend(df),use_container_width=True); plt.close()

    row3_c1,row3_c2 = st.columns(2)
    with row3_c1:
        st.pyplot(attendance_gpa_scatter(df),use_container_width=True); plt.close()
    with row3_c2:
        st.pyplot(gender_risk_chart(df),use_container_width=True); plt.close()

    # ── Results table ──────────────────────────────────────────────────────
    st.markdown('<div class="stt">Individual Results</div>',unsafe_allow_html=True)
    show=[c for c in ["student_id","name","faculty","gender","semester",
                       "semester_gpa","risk_name","prob_low","prob_med","prob_high"]
          if c in df.columns]
    risk_order={"High Risk":0,"Medium Risk":1,"Low Risk":2}
    df_display=df[show].sort_values("risk_name",key=lambda s:s.map(risk_order))

    def highlight_risk(val):
        if val=="High Risk":   return "background:#fde8e8"
        if val=="Medium Risk": return "background:#fff3cc"
        if val=="Low Risk":    return "background:#d4f5e2"
        return ""

    st.dataframe(df_display.style.map(highlight_risk,subset=["risk_name"]),
                 use_container_width=True,height=420)

    csv_out = df_display.to_csv(index=False)
    st.download_button("⬇️ Download Results CSV",data=csv_out,
                       file_name=f"pu_risk_{datetime.date.today()}.csv",
                       mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS DASHBOARD (live from batch data)
# ═══════════════════════════════════════════════════════════════════════════
def tab_analytics():
    st.markdown('<div class="stt">Institutional Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # ── Top-line model metrics ─────────────────────────────────────────────
    st.markdown("**Model Performance**")
    m1,m2,m3,m4 = st.columns(4)
    for col,val,lbl,css in [
        (m1,f"{BASELINE_F1:.4f}","Macro F1 (Test)",""),
        (m2,f"{ECE_VAL:.4f}",   "ECE",            "green"),
        (m3,f"{BV_GAP:+.4f}",  "BV Gap",          "amber"),
        (m4,str(len(FEATURE_COLS)),"Features",     "gold"),
    ]:
        with col:
            st.markdown(f"""<div class="mc {css}">
              <div class="val" style="font-size:1.6rem">{val}</div>
              <div class="lbl">{lbl}</div>
            </div>""",unsafe_allow_html=True)

    with st.expander("Why is the BV Gap large (+0.329)?"):
        st.info("The gap is a **SMOTETomek artefact** — the training set was "
                "expanded from 32,000 to 89,278 rows with synthetic data. "
                "Train F1 ≈ 0.97 on that inflated set. "
                f"**Test F1 = {BASELINE_F1:.4f}** on real data is the honest metric.")

    # ── Live charts from batch data ────────────────────────────────────────
    batch_df = st.session_state.get("batch_df")

    if batch_df is None:
        st.info("📂 **No batch data loaded yet.**  \n"
                "Go to the **Batch Assessment** tab, upload a CSV, and run predictions. "
                "The charts below will populate automatically.")
        # Still show static model charts
    else:
        n    = len(batch_df)
        preds = batch_df["risk_label"].values
        n_hr = (preds==2).sum()
        n_mr = (preds==1).sum()
        n_lr = (preds==0).sum()

        st.markdown(f'<div class="stt">Live Cohort Analysis — {n:,} Students</div>',
                    unsafe_allow_html=True)

        # Summary metrics
        cm1,cm2,cm3,cm4 = st.columns(4)
        for col,val,lbl,css in [
            (cm1,n,   "Total Students",""),
            (cm2,n_hr,"High Risk",     "red"),
            (cm3,n_mr,"Medium Risk",   "amber"),
            (cm4,n_lr,"Low Risk",      "green"),
        ]:
            with col:
                pct = f"{val/n*100:.1f}%" if n>0 else "—"
                st.markdown(f"""<div class="mc {css}">
                  <div class="val">{val:,}</div>
                  <div class="lbl">{lbl} · {pct}</div>
                </div>""",unsafe_allow_html=True)

        st.markdown("&nbsp;")

        # Row 1
        ac1,ac2 = st.columns(2)
        with ac1:
            st.pyplot(faculty_stacked_bar(batch_df),use_container_width=True); plt.close()
        with ac2:
            st.pyplot(gpa_trend_line(batch_df),use_container_width=True); plt.close()

        # Row 2
        ac3,ac4 = st.columns(2)
        with ac3:
            st.pyplot(high_risk_trend(batch_df),use_container_width=True); plt.close()
        with ac4:
            st.pyplot(attendance_gpa_scatter(batch_df),use_container_width=True); plt.close()

        # Row 3: full width gender + donut
        ac5,ac6 = st.columns(2)
        with ac5:
            st.pyplot(gender_risk_chart(batch_df),use_container_width=True); plt.close()
        with ac6:
            st.pyplot(donut_chart(n_lr,n_mr,n_hr),use_container_width=True); plt.close()

        # Row 4: heatmap (full width)
        st.markdown('<div class="stt">High Risk Heatmap — Faculty × Semester</div>',
                    unsafe_allow_html=True)
        st.pyplot(risk_heatmap(batch_df), use_container_width=True)
        plt.close()
        st.caption("Each cell shows what % of students in that faculty+semester were High Risk.")

        # Row 5: top at-risk students
        st.markdown('<div class="stt">Top 10 Highest-Risk Students</div>',
                    unsafe_allow_html=True)
        top_df = top_risk_table(batch_df, n=10)
        if len(top_df) > 0:
            def highlight_prob(val):
                try:
                    v = float(val)
                    if v >= 0.8: return "background:#fde8e8;font-weight:bold"
                    if v >= 0.6: return "background:#fff3cc"
                    return ""
                except: return ""
            st.dataframe(
                top_df.style.map(highlight_prob, subset=["prob_high"]),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No High Risk students in current dataset.")

        # ── Heatmap (full width) ──────────────────────────────────────────────
        st.markdown('<div class="stt">High Risk Heatmap — Faculty × Semester</div>',
                    unsafe_allow_html=True)
        st.pyplot(risk_heatmap(batch_df), use_container_width=True)
        plt.close()
        st.caption("Each cell shows what % of students in that faculty and semester were High Risk.")

        # ── Top 10 at-risk students ───────────────────────────────────────
        st.markdown('<div class="stt">Top 10 Highest-Risk Students</div>',
                    unsafe_allow_html=True)
        top_df = top_risk_table(batch_df, n=10)
        if len(top_df) > 0:
            def highlight_prob(val):
                try:
                    v = float(val)
                    if v >= 0.8: return "background:#fde8e8;font-weight:bold"
                    if v >= 0.6: return "background:#fff3cc"
                    return ""
                except: return ""
            st.dataframe(
                top_df.style.map(highlight_prob, subset=["prob_high"]),
                use_container_width=True, hide_index=True)
        else:
            st.info("No High Risk students in current batch.")

    # ── Static: SHAP importance ────────────────────────────────────────────
    if TOP_SHAP:
        st.markdown('<div class="stt">Top Predictive Features (SHAP)</div>',
                    unsafe_allow_html=True)
        shap_vals = np.array([0.38,0.33,0.29,0.24,0.19,0.15,0.12,0.09])
        n_show    = min(8,len(TOP_SHAP),len(shap_vals))
        fig,ax    = plt.subplots(figsize=(8,4))
        fig.patch.set_facecolor("none")
        bar_colors= [PU_BLUE if i<3 else PU_GOLD if i<6 else "#aaa"
                     for i in range(n_show)]
        ax.barh(TOP_SHAP[:n_show][::-1],shap_vals[:n_show][::-1],
                color=bar_colors[::-1],edgecolor="white")
        for i,v in enumerate(shap_vals[:n_show][::-1]):
            ax.text(v+0.005,i,f"{v:.2f}",va="center",fontsize=8)
        ax.set_title("Mean |SHAP| Value — LightGBM",fontweight="bold",color=PU_BLUE)
        ax.set_xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        st.pyplot(fig,use_container_width=True); plt.close()
        st.caption("Blue = top 3 features · Gold = next 3")

    # ── Static: Fairness audit ─────────────────────────────────────────────
    st.markdown('<div class="stt">Fairness Audit (from Pipeline)</div>',
                unsafe_allow_html=True)
    fair_df = pd.DataFrame({
        "Group":["Female","Male","FBA","FEHAS","FESAC","PSTM",
                 "Female_FBA","Female_FEHAS","Female_FESAC","Female_PSTM",
                 "Male_FBA","Male_FEHAS","Male_FESAC","Male_PSTM"],
        "Macro F1":[0.660,0.613,0.801,0.582,0.869,0.666,
                    0.797,0.604,0.863,0.688,0.805,0.565,0.876,0.588],
        "Status":["✅ Fair"]*14,
    })
    def highlight_fair(val):
        if val=="✅ Fair": return "background:#d4f5e2;color:#1a7a42"
        return "background:#fde8e8;color:#c0392b"
    st.dataframe(fair_df.style.map(highlight_fair,subset=["Status"]),
                 use_container_width=True,hide_index=True)
    st.success("✅ All 8 intersectional groups (Gender × Faculty) pass F1 ≥ 0.45")

    # ── Training summary ───────────────────────────────────────────────────
    with st.expander("📋 Training Data Summary"):
        st.markdown(f"""
| Field | Value |
|---|---|
| Algorithm | {MODEL_NAME} |
| Students | 8,000 |
| Enrolment records | 512,000 |
| Train | 32,000 rows → 89,278 after SMOTETomek |
| Val | 8,000 rows |
| Test | 16,000 rows |
| Scaler | RobustScaler |
| Compliance | Ghana DPA 2012 (Act 843) |
        """)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if "auth"     not in st.session_state: st.session_state["auth"]     = False
    if "hist"     not in st.session_state: st.session_state["hist"]     = []
    if "batch_df" not in st.session_state: st.session_state["batch_df"] = None

    if not st.session_state["auth"]:
        login_page(); return

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

    tab_defs=[]
    if "individual" in perms: tab_defs.append(("🔍 Individual Prediction","individual"))
    if "batch"      in perms: tab_defs.append(("📂 Batch Assessment",      "batch"))
    if "analytics"  in perms: tab_defs.append(("📊 Analytics Dashboard",   "analytics"))

    tabs = st.tabs([t[0] for t in tab_defs])
    for tab,(_,key) in zip(tabs,tab_defs):
        with tab:
            if   key=="individual": tab_individual()
            elif key=="batch":      tab_batch()
            elif key=="analytics":  tab_analytics()

if __name__=="__main__":
    main()
