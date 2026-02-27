import streamlit as st
import pandas as pd
import pickle
import json

# Page Setup

st.set_page_config(page_title="Early Mental Health Risk Prediction System", layout="centered")

st.title("Early Mental Health Risk Prediction System")
st.write("This application predicts early mental health risk level of a student.")

# Load Trained Model

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load model comparison metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Feature List 

FEATURES = [
    "age",
    "cgpa",
    "average_sleep",
    "academic_workload",
    "academic_pressure",
    "financial_concerns",
    "social_relationships",
    "campus_discrimination",
    "sports_engagement",
    "study_satisfaction",
    "stress_relief_activities"
]

# User Input

st.subheader("Enter Student Details")

academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
social_relationships = st.slider("Social Relationships", 1, 5, 3)
financial_concerns = st.slider("Financial Concerns", 1, 5, 3)
study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
cgpa = st.slider("CGPA", 2.0, 4.0, 3.0, step=0.1)
academic_workload = st.slider("Academic Workload", 1, 5, 3)

# Create input dataframe
input_df = pd.DataFrame([{
    "age": 21,  # default value
    "cgpa": cgpa,
    "average_sleep": 7,  # default value
    "academic_workload": academic_workload,
    "academic_pressure": academic_pressure,
    "financial_concerns": financial_concerns,
    "social_relationships": social_relationships,
    "campus_discrimination": 0,  # default value
    "sports_engagement": 2,  # default value
    "study_satisfaction": study_satisfaction,
    "stress_relief_activities": 1  # default value
}])

input_df = input_df[FEATURES]

# Prediction

if st.button("Predict Risk"):

    # model predicts and gives output in array containing (0/1) ,to obtain the element [0] used
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] # gives array with probabilty of second class(high risk)

    if prediction == 1:
        st.error("High Risk: The student may need support.")
    else:
        st.success("Low Risk: The student appears stable.")

    st.write(f"Risk Probability: {probability:.2f}")

# Feature Importance

st.subheader("Top Influencing Features")

if hasattr(model, "feature_importances_"):
    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

elif hasattr(model, "coef_"):
    importances = pd.Series(
        abs(model.coef_[0]),
        index=FEATURES
    ).sort_values(ascending=False)

else:
    importances = None

if importances is not None:
    st.bar_chart(importances.head(6))
else:
    st.write("Feature importance not available.")

# Model Comparison

st.subheader("Model Comparison")
st.write("Best Model:", metrics["best_model"])

comparison_df = pd.DataFrame(metrics["model_comparison"]).T
st.dataframe(comparison_df)

st.info("This tool does not replace professional mental health diagnosis.")