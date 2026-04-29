import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import shap
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Academic Risk AI", layout="wide")

st.title("🎓 AI Academic Risk Intelligence System")

# =========================
# LOAD MODEL + FILES
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with open("feature_cols.json") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols

model, scaler, feature_cols = load_assets()

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio("Navigation", ["Student", "Faculty", "Batch Upload"])

# =========================
# PDF GENERATOR
# =========================
def generate_pdf(student_id, risk, prob, recommendations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Student ID: {student_id}", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))
    content.append(Paragraph(f"Risk Probability: {round(prob*100,2)}%", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Recommendations:", styles["Heading2"]))

    for rec in recommendations:
        content.append(Paragraph(f"- {rec}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

# =========================
# STUDENT PAGE
# =========================
if page == "Student":

    st.header("🔍 Individual Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 70)
        total_mark = st.slider("Total Mark (%)", 0, 100, 55)

    with col2:
        ca_score = st.slider("CA Score", 0, 40, 22)
        exam_score = st.slider("Exam Score", 0, 60, 33)

    if st.button("Predict Risk"):

        # =========================
        # BUILD INPUT
        # =========================
        input_dict = {
            "attendance": attendance,
            "total_mark": total_mark,
            "ca": ca_score,
            "exam": exam_score
        }

        X = pd.DataFrame([input_dict])

        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0

        X = X[feature_cols]
        X_scaled = scaler.transform(X)

        probs = model.predict_proba(X_scaled)[0]
        pred = np.argmax(probs)

        labels = ["Low", "Medium", "High"]
        risk = labels[pred]

        # =========================
        # RISK DISPLAY
        # =========================
        st.subheader(f"Risk Level: {risk}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probs[2]*100,
            title={'text': "High Risk Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 60], 'color': "orange"},
                    {'range': [60, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # ALERT SYSTEM
        # =========================
        st.subheader("🚨 Alerts")

        alerts = []
        if probs[2] > 0.6:
            alerts.append("High academic risk detected")
        if attendance < 60:
            alerts.append("Low attendance")
        if exam_score < 40:
            alerts.append("Poor exam performance")

        if alerts:
            for alert in alerts:
                st.error(alert)
        else:
            st.success("No critical alerts")

        # =========================
        # RECOMMENDATIONS
        # =========================
        st.subheader("💡 Recommendations")

        recommendations = []

        if attendance < 70:
            recommendations.append("Increase attendance")
        if exam_score < 50:
            recommendations.append("Focus on exam preparation")
        if ca_score < 25:
            recommendations.append("Improve CA performance")
        if total_mark < 60:
            recommendations.append("Seek academic support")

        if not recommendations:
            recommendations.append("Maintain performance")

        for r in recommendations:
            st.write("✔️", r)

        # =========================
        # SHAP EXPLAINABILITY
        # =========================
        st.subheader("🧠 Model Explanation")

        explainer = shap.Explainer(model)
        shap_values = explainer(X_scaled)

        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Impact": shap_values.values[0]
        })

        fig2 = px.bar(
            shap_df,
            x="Impact",
            y="Feature",
            orientation="h",
            color="Impact",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # PDF DOWNLOAD
        # =========================
        st.subheader("📄 Report")

        pdf = generate_pdf("Student_001", risk, probs[2], recommendations)

        st.download_button(
            "Download PDF Report",
            pdf,
            file_name="student_report.pdf"
        )

# =========================
# FACULTY DASHBOARD
# =========================
elif page == "Faculty":

    st.header("📊 Faculty Dashboard")

    file = st.file_uploader("Upload Faculty Data", type="csv")

    if file:
        df = pd.read_csv(file)

        st.subheader("Risk Distribution")
        fig = px.histogram(df, x="risk_level", color="risk_level")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🚨 High Risk Students")
        high = df[df["risk_level"] == "High"]

        if len(high) > 0:
            st.warning(f"{len(high)} high-risk students")
            st.dataframe(high.head(10))
        else:
            st.success("No high-risk students")

        st.subheader("GPA Trends")
        fig2 = px.line(df, x="semester", y="gpa", color="student_id")
        st.plotly_chart(fig2, use_container_width=True)

# =========================
# BATCH PREDICTION
# =========================
elif page == "Batch Upload":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_cols]
        X_scaled = scaler.transform(X)

        probs = model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)

        labels = ["Low", "Medium", "High"]
        df["risk_level"] = [labels[p] for p in preds]

        st.dataframe(df)

        st.download_button(
            "Download Predictions",
            df.to_csv(index=False),
            file_name="predictions.csv"
        )

        fig = px.histogram(df, x="risk_level", color="risk_level")
        st.plotly_chart(fig, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("AI Academic Risk System | Pentecost University")
