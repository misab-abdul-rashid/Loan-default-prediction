# src/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
return model


def save_model(model, path='models/model_v1.pkl'):
joblib.dump(model, path)


def load_model(path='models/model_v1.pkl'):
return joblib.load(path)
