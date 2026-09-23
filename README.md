# LAND-MINE GUARD AI
Software-only prototype for landslide and mine-subsidence early-warning research.

## SIH 2026 alignment
- SIH26001: AI-Based early warning and landslide Risk Monitoring System in NER (Software).
- SIH26025: AI-enabled low-cost real-time mine subsidence monitoring, prediction and early warning (Hardware).
This student project is a software-only simulation/prototype inspired by the software-relevant portions of these challenges.

## Run
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. `python -m venv .venv`
4. Activate the environment.
5. `pip install -r requirements.txt`
6. `python app/train_models.py`
7. `streamlit run app/app.py`

## Demo flow
Use the sidebar:
Normal -> Heavy rainfall -> Rapid ground movement.
Show the risk score, anomaly detection, classifier, trend chart and node status.

## Important
The included dataset is synthetic and intended for demonstration. It is not a validated geological dataset and the output is not an operational safety warning.
