# ==========================================
# Seismic Traffic Light Dashboard v4
# Render Deployment Version
# ==========================================

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import pickle
from datetime import timedelta

# -----------------------------
# PATHS (RENDER SAFE)
# -----------------------------
model_event_path = "models/seismic_event_occurrence_model_v2.cbm"
model_magnitude_path = "models/seismic_magnitude_model_v2.cbm"
model_traffic_path = "models/seismic_traffic_light_3class_model_v2.cbm"
medians_path = "data/train_medians_v2.pkl"
threshold_path = "data/optimal_event_threshold_v2.txt"
operational_data_path = "data/operational_seismic_linear_decay121.csv"

# -----------------------------
# LOAD MODELS
# -----------------------------
model_event = CatBoostClassifier()
model_event.load_model(model_event_path)

model_magnitude = CatBoostRegressor()
model_magnitude.load_model(model_magnitude_path)

model_traffic = CatBoostClassifier()
model_traffic.load_model(model_traffic_path)

with open(medians_path, "rb") as f:
    train_medians = pickle.load(f)

with open(threshold_path, "r") as f:
    optimal_threshold = float(f.read().strip())

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(operational_data_path, low_memory=False)

# =============================
# PREPROCESSING (UNCHANGED)
# =============================
sentinel_cols = ['pgv_max', 'magnitude', 'hourly_seismicity_rate']
for col in sentinel_cols:
    if col in df.columns:
        df.loc[df[col] == -999.0, col] = 0

datetime_cols = [
    'recorded_at', 'phase_started_at',
    'phase_production_ended_at', 'phase_ended_at', 'occurred_at'
]
for col in datetime_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df = df.sort_values("recorded_at").reset_index(drop=True)
min_date = df["recorded_at"].min()
max_date = df["recorded_at"].max()
has_timestamp = True

# Ground truth
has_ground_truth = False
if 'magnitude' in df.columns and 'hourly_seismicity_rate' in df.columns:
    df['event_occurs'] = ((df['magnitude'] >= 0.17) |
                          (df['hourly_seismicity_rate'] > 0)).astype(int)

    def classify_traffic(m):
        if m >= 1.0:
            return 2
        elif m >= 0.17:
            return 1
        return 0

    df['traffic_light_actual'] = df['magnitude'].apply(classify_traffic)
    df['magnitude_actual'] = df['magnitude']
    has_ground_truth = True

# =============================
# FEATURE ENGINEERING
# =============================
df['hour'] = df['recorded_at'].dt.hour
df['day_of_week'] = df['recorded_at'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['recorded_at'].dt.month

# (KEEP YOUR FULL FEATURE ENGINEERING HERE — UNCHANGED)
# ⬇️⬇️⬇️
# 🔴 PASTE YOUR EXISTING FEATURE ENGINEERING CODE HERE
# ⬆️⬆️⬆️

# =============================
# FEATURE SELECTION
# =============================
exclude_cols = [
    'recorded_at', 'event_occurs', 'traffic_light_actual',
    'magnitude', 'magnitude_actual', 'hourly_seismicity_rate'
]

feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]

numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

X = X.replace([np.inf, -np.inf], np.nan)
X[numeric_cols] = X[numeric_cols].fillna(train_medians)
X[categorical_cols] = X[categorical_cols].astype(str).fillna("missing")

cat_features = [i for i, c in enumerate(X.columns) if c in categorical_cols]
pool = Pool(X, cat_features=cat_features)

# =============================
# PREDICTIONS
# =============================
event_prob = model_event.predict_proba(pool)[:, 1]
event_pred = (event_prob >= optimal_threshold).astype(int)

magnitude_pred = np.zeros(len(X))
if event_pred.sum() > 0:
    mag_pool = Pool(X[event_pred == 1], cat_features=cat_features)
    magnitude_pred[event_pred == 1] = model_magnitude.predict(mag_pool)

traffic_pred = model_traffic.predict(pool).flatten()

# =============================
# DASH DATA
# =============================
df_dash = pd.DataFrame({
    "timestamp": df["recorded_at"],
    "event_probability": event_prob,
    "event_predicted": event_pred,
    "magnitude_predicted": magnitude_pred,
    "traffic_light_pred": traffic_pred
})

# =============================
# DASH APP
# =============================
app = Dash(__name__)
server = app.server  # REQUIRED FOR RENDER

app.layout = html.Div([
    html.H1("🚦 Seismic Monitoring Dashboard", style={"textAlign": "center"}),
    dcc.DatePickerRange(
        id="date_picker",
        min_date_allowed=min_date,
        max_date_allowed=max_date,
        start_date=min_date,
        end_date=max_date
    ),
    dcc.Graph(id="main_plot")
])

@app.callback(
    Output("main_plot", "figure"),
    Input("date_picker", "start_date"),
    Input("date_picker", "end_date")
)
def update_plot(start, end):
    mask = (df_dash["timestamp"] >= start) & (df_dash["timestamp"] <= end)
    dff = df_dash[mask]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=dff["timestamp"],
            y=dff["magnitude_predicted"],
            mode="markers",
            marker=dict(color="red", size=6),
            name="Predicted Events"
        ),
        secondary_y=True
    )
    fig.update_layout(height=600)
    return fig

# =============================
# RUN (RENDER SAFE)
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
