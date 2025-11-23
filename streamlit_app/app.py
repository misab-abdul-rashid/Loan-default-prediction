# streamlit_app/app.py
import streamlit as st
import joblib
import numpy as np
import os

# Build path to model file (relative to this file)
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_v1.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Model load failed: {e}")
    model = None


st.title('Loan Default Prediction (Demo)')

age = st.number_input('Age', 18, 100, 30)
income = st.number_input('Income', 0.0, 1e7, 50000.0)
loan_amount = st.number_input('Loan amount', 0.0, 1e7, 5000.0)
term_months = st.number_input('Term (months)', 6, 360, 36)
credit_score = st.number_input('Credit score', 300, 850, 650)

model = None
try:
    # This line must be indented under the try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'models', 'model_v1.pkl'))
except Exception:
    st.warning('Model not found. Run training and add models/model_v1.pkl')

if st.button('Predict'):
    # Ensure dti calculation and input converted to correct shape
    dti = loan_amount / (income + 1e-6)
    X = np.array([[age, income, loan_amount, term_months, credit_score, dti]])
    if model is None:
        st.error('Model not available')
    else:
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, 'predict_proba') else None
        st.write('Prediction (1 = default):', pred)
        st.write('Probability of default:', prob)
