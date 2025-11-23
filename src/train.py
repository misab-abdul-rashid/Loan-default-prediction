# src/train.py
from src.data import load_data
from src.features import preprocess, train_test_split_xy
from src.model import train_model, save_model
from sklearn.metrics import classification_report, roc_auc_score




def run_training():
df = load_data()
X, y = preprocess(df)
X_train, X_test, y_train, y_test = train_test_split_xy(X, y)


model = train_model(X_train, y_train)
save_model(model)


preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, preds))
try:
print('AUC:', roc_auc_score(y_test, probs))
except Exception:
pass


if __name__ == '__main__':
run_training()
