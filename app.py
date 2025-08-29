import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("bankruptcy_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Bankruptcy Prediction App")
st.write("Enter company risk factors to check bankruptcy possibility.")

# Input fields
industrial_risk = st.selectbox("Industrial Risk", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])
management_risk = st.selectbox("Management Risk", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])
credibility = st.selectbox("Credibility", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])
competitiveness = st.selectbox("Competitiveness", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])
operating_risk = st.selectbox("Operating Risk", [0, 0.5, 1], format_func=lambda x: {0:"Low", 0.5:"Medium", 1:"High"}[x])

# Convert inputs into numpy array
features = np.array([[industrial_risk, management_risk, financial_flexibility,
                      credibility, competitiveness, operating_risk]])

# Predict button
if st.button("🔍 Predict Bankruptcy"):
    prediction = model.predict(features)[0]
    result = "Bankruptcy" if prediction == 1 else "Non Bankruptcy"

    # Show results in table
    results_df = pd.DataFrame({
        "Feature": ["Industrial Risk", "Management Risk", "Financial Flexibility",
                    "Credibility", "Competitiveness", "Operating Risk", "Prediction"],
        "Value": [industrial_risk, management_risk, financial_flexibility,
                  credibility, competitiveness, operating_risk, result]
    })

    st.table(results_df)
