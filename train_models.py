
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "sample_ground_data.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "rainfall_mm", "soil_moisture_pct", "ground_tilt_deg",
    "displacement_mm", "vibration_g", "crack_growth_mm"
]

df = pd.read_csv(DATA)
X = df[FEATURES]
y = df["status"]

# 1) Unsupervised anomaly detector: learns the normal envelope.
normal_X = X[y == "SAFE"]
iso = IsolationForest(
    n_estimators=200,
    contamination=0.12,
    random_state=42
)
iso.fit(normal_X)
joblib.dump(iso, MODEL_DIR / "isolation_forest.joblib")

# 2) Transparent classifier for the demo.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))
joblib.dump(rf, MODEL_DIR / "risk_classifier.joblib")
print("Models saved to", MODEL_DIR)
