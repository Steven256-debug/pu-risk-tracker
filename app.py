
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Academic Risk System", layout="wide")

# =========================
# LOAD ASSETS
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_cols.json") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols

model, scaler, FEATURE_COLS = load_assets()

# =========================
# VALIDATION
# =========================
def validate_csv(df):
    required = [
        "student_id","faculty","gender","semester","semester_gpa",
        "avg_attendance","avg_total_mark","avg_ca_score",
        "avg_exam_score","total_credits","num_courses"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    return df

def align_features(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing model features: {missing}")
        st.stop()
    return df[FEATURE_COLS].fillna(0)

# =========================
# SIDEBAR ROLE SELECTION
# =========================
role = st.sidebar.selectbox("Select Role", ["Admin", "Faculty"])

st.title("Academic Risk Intelligence System")

# =========================================================
# ADMIN DASHBOARD
# =========================================================
if role == "Admin":

    st.header("Admin Dashboard - Institutional Overview")

    uploaded = st.file_uploader("Upload Institutional Dataset", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        df = validate_csv(df)

        X = align_features(df)
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        df["risk_level"] = preds

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Students", len(df))
        col2.metric("High Risk Students", (df["risk_level"] == 2).sum())
        col3.metric("Average GPA", round(df["semester_gpa"].mean(), 2))

        st.subheader("Risk Distribution Across Institution")
        fig = px.histogram(df, x="risk_level", color="faculty")
        st.plotly_chart(fig, use_container_width=True)

        st.info("This chart shows how risk levels are distributed across faculties.")

        st.subheader("Faculty Performance Comparison")
        fig2 = px.box(df, x="faculty", y="semester_gpa")
        st.plotly_chart(fig2, use_container_width=True)

        st.info("This boxplot compares GPA distributions across faculties.")

# =========================================================
# FACULTY DASHBOARD
# =========================================================
elif role == "Faculty":

    st.header("Faculty Dashboard")

    uploaded = st.file_uploader("Upload Faculty Dataset", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        df = validate_csv(df)

        # Select faculty
        faculties = df["faculty"].unique()
        selected_faculty = st.selectbox("Select Faculty", faculties)

        df = df[df["faculty"] == selected_faculty]

        X = align_features(df)
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        df["risk_level"] = preds

        col1, col2, col3 = st.columns(3)

        col1.metric("Students in Faculty", len(df))
        col2.metric("High Risk", (df["risk_level"] == 2).sum())
        col3.metric("Average GPA", round(df["semester_gpa"].mean(), 2))

        st.subheader("Risk Distribution")
        fig = px.histogram(df, x="risk_level")
        st.plotly_chart(fig, use_container_width=True)

        st.info("Shows risk distribution within selected faculty.")

        st.subheader("High Risk Students")
        st.dataframe(df[df["risk_level"] == 2].head(10))

        st.subheader("Download Results")
        st.download_button(
            "Download Faculty Predictions",
            df.to_csv(index=False),
            file_name="faculty_predictions.csv"
        )

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("AI Academic Risk System | Pentecost University")
