
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "sample_ground_data.csv"
MODEL_DIR = BASE / "models"

FEATURES = [
    "rainfall_mm", "soil_moisture_pct", "ground_tilt_deg",
    "displacement_mm", "vibration_g", "crack_growth_mm"
]

st.set_page_config(page_title="Land-Mine Guard AI", page_icon="⛰️", layout="wide")

st.title("⛰️ LAND-MINE GUARD AI")
st.caption("Software-only prototype for landslide & mine-subsidence early-warning research")

st.warning(
    "Educational prototype: results are simulated/algorithmic indicators, not a "
    "validated geological prediction or an operational safety warning."
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA, parse_dates=["timestamp"])

@st.cache_resource
def load_models():
    iso = joblib.load(MODEL_DIR / "isolation_forest.joblib")
    rf = joblib.load(MODEL_DIR / "risk_classifier.joblib")
    return iso, rf

df = load_data()

if not (MODEL_DIR/"isolation_forest.joblib").exists():
    st.error("Models are missing. Run: python app/train_models.py")
    st.stop()

iso, rf = load_models()

# Sidebar simulation
st.sidebar.header("Scenario Simulator")
mode = st.sidebar.selectbox("Scenario", ["Latest data", "Normal", "Heavy rainfall", "Rapid ground movement", "Custom"])

if mode == "Latest data":
    row = df.iloc[-1].copy()
elif mode == "Normal":
    row = pd.Series({
        "rainfall_mm": 12.0, "soil_moisture_pct": 42.0, "ground_tilt_deg": 0.8,
        "displacement_mm": 1.2, "vibration_g": 0.12, "crack_growth_mm": 0.2
    })
elif mode == "Heavy rainfall":
    row = pd.Series({
        "rainfall_mm": 150.0, "soil_moisture_pct": 88.0, "ground_tilt_deg": 3.2,
        "displacement_mm": 8.5, "vibration_g": 0.35, "crack_growth_mm": 1.7
    })
elif mode == "Rapid ground movement":
    row = pd.Series({
        "rainfall_mm": 110.0, "soil_moisture_pct": 82.0, "ground_tilt_deg": 7.0,
        "displacement_mm": 22.0, "vibration_g": 0.95, "crack_growth_mm": 4.5
    })
else:
    row = pd.Series({
        "rainfall_mm": st.sidebar.slider("Rainfall (mm)", 0.0, 220.0, 80.0),
        "soil_moisture_pct": st.sidebar.slider("Soil moisture (%)", 0.0, 100.0, 60.0),
        "ground_tilt_deg": st.sidebar.slider("Ground tilt (°)", 0.0, 15.0, 2.0),
        "displacement_mm": st.sidebar.slider("Displacement (mm)", 0.0, 40.0, 5.0),
        "vibration_g": st.sidebar.slider("Vibration (g)", 0.0, 2.0, 0.2),
        "crack_growth_mm": st.sidebar.slider("Crack growth (mm)", 0.0, 8.0, 0.5)
    })

x = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

# Transparent engineering risk index
score = (
    0.20*min(row["rainfall_mm"]/160, 1) +
    0.15*(row["soil_moisture_pct"]/100) +
    0.20*min(row["ground_tilt_deg"]/8, 1) +
    0.20*min(row["displacement_mm"]/20, 1) +
    0.10*min(row["vibration_g"]/1.0, 1) +
    0.15*min(row["crack_growth_mm"]/4, 1)
) * 100
score = float(np.clip(score, 0, 100))

anomaly = iso.predict(x)[0] == -1
ml_status = rf.predict(x)[0]
final_status = "CRITICAL" if score >= 70 else ("WARNING" if score >= 40 else "SAFE")

st.subheader("Current Risk Assessment")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Risk score", f"{score:.1f}/100")
c2.metric("AI class", ml_status)
c3.metric("Anomaly detected", "YES" if anomaly else "NO")
c4.metric("System state", final_status)

if final_status == "CRITICAL":
    st.error("🔴 CRITICAL — early-warning condition indicated by the prototype.")
elif final_status == "WARNING":
    st.warning("🟡 WARNING — abnormal or elevated ground-risk indicators.")
else:
    st.success("🟢 SAFE — no elevated condition indicated in this simulation.")

st.subheader("Sensor/Indicator Values")
m1,m2,m3 = st.columns(3)
m1.metric("Rainfall", f'{row["rainfall_mm"]:.1f} mm')
m1.metric("Soil moisture", f'{row["soil_moisture_pct"]:.1f} %')
m2.metric("Ground tilt", f'{row["ground_tilt_deg"]:.2f}°')
m2.metric("Displacement", f'{row["displacement_mm"]:.1f} mm')
m3.metric("Vibration", f'{row["vibration_g"]:.2f} g')
m3.metric("Crack growth", f'{row["crack_growth_mm"]:.1f} mm')

st.subheader("Historical Trend")
plot_df = df.tail(120).set_index("timestamp")
st.line_chart(plot_df[["risk_score", "rainfall_mm", "displacement_mm"]])

st.subheader("Risk Explanation")
components = {
    "Rainfall": min(row["rainfall_mm"]/160, 1)*20,
    "Moisture": (row["soil_moisture_pct"]/100)*15,
    "Tilt": min(row["ground_tilt_deg"]/8, 1)*20,
    "Displacement": min(row["displacement_mm"]/20, 1)*20,
    "Vibration": min(row["vibration_g"]/1, 1)*10,
    "Crack growth": min(row["crack_growth_mm"]/4, 1)*15,
}
explain = pd.DataFrame({"Contribution": components}).sort_values("Contribution", ascending=False)
st.bar_chart(explain)

st.subheader("Node View (Software Simulation)")
nodes = pd.DataFrame({
    "Node": ["N1 - Slope top", "N2 - Slope middle", "N3 - Mine panel"],
    "Status": [final_status, "WARNING" if score >= 40 else "SAFE", final_status if row["displacement_mm"] > 10 else "SAFE"]
})
st.dataframe(nodes, use_container_width=True, hide_index=True)

st.caption(
    "Prototype architecture: simulated observations → anomaly detection → transparent risk index "
    "→ classifier → dashboard/alert. Real deployment requires field calibration, geological "
    "validation, reliable sensors, communications and safety-authority approval."
)
