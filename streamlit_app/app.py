# streamlit_app/app.py
import streamlit as st
import joblib
import numpy as np
import os

st.title('Loan Default Prediction (Debug)')

# Build absolute model path (based on this file)
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model_v1.pkl"))

# Show debug info in the app UI (safe)
st.write("**DEBUG:** model_path =", model_path)
st.write("**DEBUG:** file exists =", os.path.exists(model_path))

# Try to load once and show clear error if it fails
model = None
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully.")
except Exception as e:
    st.error("Model load failed.")
    st.code(str(e))

# Inputs
age = st.number_input('Age', 18, 100, 30)
income = st.number_input('Income', 0.0, 1e7, 50000.0)
loan_amount = st.number_input('Loan amount', 0.0, 1e7, 5000.0)
term_months = st.number_input('Term (months)', 6, 360, 36)
credit_score = st.number_input('Credit score', 300, 850, 650)

if st.button('Predict'):
    if model is None:
        st.error('Model not available. See debug info above.')
    else:
        dti = loan_amount / (income + 1e-6)
        X = np.array([[age, income, loan_amount, term_months, credit_score, dti]])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, 'predict_proba') else None
        st.success(f"Prediction (1 = default): {pred}")
        st.info(f"Probability of default: {prob}")
