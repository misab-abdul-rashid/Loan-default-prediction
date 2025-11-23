# streamlit_app/app.py
import streamlit as st
import joblib
import numpy as np
import os

# Correct model path
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "model_v1.pkl")
)

# Load model ONCE
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model not found or failed to load.\nPath: {model_path}\nError: {e}")
    model = None

st.title('Loan Default Prediction')

age = st.number_input('Age', 18, 100, 30)
income = st.number_input('Income', 0.0, 1e7, 50000.0)
loan_amount = st.number_input('Loan amount', 0.0, 1e7, 5000.0)
term_months = st.number_input('Term (months)', 6, 360, 36)
credit_score = st.number_input('Credit score', 300, 850, 650)

if st.button('Predict'):
    if model is None:
        st.error('Model not available. Check file path.')
    else:
        dti = loan_amount / (income + 1e-6)
        X = np.array([[age, income, loan_amount, term_months, credit_score, dti]])

        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, 'predict_proba') else None

        st.success(f"Prediction (1 = default): {pred}")
        st.info(f"Probability of default: {prob}")
