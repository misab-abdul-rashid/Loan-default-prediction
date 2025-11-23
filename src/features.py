# src/features.py
import pandas as pd
from sklearn.model_selection import train_test_split


NUMERIC_COLS = ['age','income','loan_amount','term_months','credit_score']


def preprocess(df: pd.DataFrame):
df = df.copy()
# basic cleaning
df = df.dropna()
# ensure numeric types
for c in NUMERIC_COLS:
df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna()
# feature: debt_to_income_ratio
df['dti'] = df['loan_amount'] / (df['income'] + 1e-6)
features = NUMERIC_COLS + ['dti']
X = df[features]
y = df['default'] if 'default' in df.columns else None
return X, y


def train_test_split_xy(X, y, test_size=0.2, random_state=42):
return train_test_split(X, y, test_size=test_size, random_state=random_state)
