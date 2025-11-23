# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import traceback
import numpy as np
import joblib
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_v1.pkl')


app = FastAPI(title='Loan Default Predictor')


class Applicant(BaseModel):
age: int
income: float
loan_amount: float
term_months: int
credit_score: int


@app.on_event('startup')
def load_model():
global model
try:
model = joblib.load(MODEL_PATH)
except Exception:
model = None


@app.get('/')
def root():
return {'status': 'ok'}


@app.post('/predict')
def predict(a: Applicant):
try:
if model is None:
return {'error': 'model not available'}
# construct feature vector
dti = a.loan_amount / (a.income + 1e-6)
X = np.array([[a.age, a.income, a.loan_amount, a.term_months, a.credit_score, dti]])
pred = int(model.predict(X)[0])
prob = float(model.predict_proba(X)[0,1]) if hasattr(model, 'predict_proba') else None
return {'prediction': pred, 'probability': prob}
except Exception as e:
traceback.print_exc()
return {'error': str(e)}
