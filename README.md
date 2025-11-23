# Loan Default Prediction — End-to-end ML (FastAPI + Streamlit)


**Purpose:** Build, evaluate, and deploy a model to predict loan default risk. This repo demonstrates an end-to-end ML workflow: data → features → model → API + UI.


## Quick links
- FastAPI endpoint: `/predict` (see `app/main.py`)
- Streamlit demo: `streamlit_app/app.py` (interactive UI)


## How to run (no terminal required for demos)
### Option A — Run Streamlit demo on Streamlit Cloud / locally
1. Open `streamlit_app/app.py` or deploy via share.streamlit.io
2. Or run locally (requires terminal):
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
