# src/data.py
import pandas as pd


def load_data(path='data/sample_loan.csv'):
df = pd.read_csv(path)
return df
