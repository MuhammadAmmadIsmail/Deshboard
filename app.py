import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ========================================== 
# PAGE CONFIG
# ========================================== 
st.set_page_config(
    page_title="Seismic Monitoring Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================== 
# PATHS (LOCAL FILES ONLY)
# ========================================== 
model_event_path = "model/seismic_event_occurrence_model_v2.cbm"
model_magnitude_path = "model/seismic_magnitude_model_v2.cbm"
model_traffic_path = "model/seismic_traffic_light_3class_model_v2.cbm"
medians_path = "model/train_medians_v2.pkl"
threshold_path = "model/optimal_event_threshold_v2.txt"
operational_data_path = "operational_seismic_linear_decay121.csv"

# ========================================== 
# LOAD MODELS WITH FALLBACK
# ========================================== 
@st.cache_resource
def load_models_with_fallback():
    """
    Try to load CatBoost models.
    If CatBoost unavailable, use cached predictions.
    """
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
        
        model_event = CatBoostClassifier()
        model_event.load_model(model_event_path)
        
        model_magnitude = CatBoostRegressor()
        model_magnitude.load_model(model_magnitude_path)
        
        model_traffic = CatBoostClassifier()
        model_traffic.load_model(model_traffic_path)
        
        with open(medians_path, 'rb') as f:
            train_medians = pickle.load(f)
        
        with open(threshold_path, 'r') as f:
            optimal_threshold = float(f.read().strip())
        
        return {
            'event': model_event,
            'magnitude': model_magnitude,
            'traffic': model_traffic,
            'medians': train_medians,
            'threshold': optimal_threshold,
            'mode': 'catboost'
        }
    
    except Exception as e:
        st.warning("⚠️ CatBoost not available. Using cached predictions...")
        
        predictions_cache = "predictions_cache.pkl"
        try:
            with open(predictions_cache, 'rb') as f:
                cache = pickle.load(f)
            cache['mode'] = 'cached'
            return cache
        except:
            st.error("❌ Failed to load models!")
            return None

@st.cache_data
def load_data():
    """Load operational data from local CSV file"""
    try:
        df = pd.read_csv(operational_data_path, low_memory=False)
        return df
    except FileNotFoundError:
        st.error(f"❌ Data file not found: {operational_data_path}")
        return None

# Load models and data
models_dict = load_models_with_fallback()
df = load_data()
data_loaded = (models_dict is not None and df is not None)

if data_loaded:
    # ========================================== 
    # DATA PREPROCESSING
    # ========================================== 
    df_processed = df.copy()
    
    # Replace sentinel values
    sentinel_cols = ['pgv_max', 'magnitude', 'hourly_seismicity_rate']
    for col in sentinel_cols:
        if col in df_processed.columns:
            mask = df_processed[col] == -999.0
            df_processed.loc[mask, col] = 0
    
    # Parse datetime columns
    datetime_cols = ['recorded_at', 'phase_started_at', 'phase_production_ended_at', 'phase_ended_at', 'occurred_at']
    for col in datetime_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
    
    # Sort chronologically
    if 'recorded_at' in df_processed.columns:
        df_processed = df_processed.sort_values('recorded_at').reset_index(drop=True)
        has_timestamp = True
        min_date = df_processed['recorded_at'].min()
        max_date = df_processed['recorded_at'].max()
    else:
        df_processed = df_processed.reset_index(drop=True)
        has_timestamp = False
    
    # Create ground truth targets
    has_ground_truth = False
    if 'magnitude' in df_processed.columns and 'hourly_seismicity_rate' in df_processed.columns:
        df_processed['event_occurs'] = ((df_processed['magnitude'] >= 0.17) | 
                                       (df_processed['hourly_seismicity_rate'] > 0)).astype(int)
        
        def classify_traffic_light_3class(magnitude):
            if magnitude >= 1.0:
                return 2
            elif magnitude >= 0.17:
                return 1
            else:
                return 0
        
        df_processed['traffic_light_actual'] = df_processed['magnitude'].apply(classify_traffic_light_3class)
        df_processed['magnitude_actual'] = df_processed['magnitude'].copy()
        has_ground_truth = True
    
    # ========================================== 
    # FEATURE ENGINEERING
    # ========================================== 
    if 'recorded_at' in df_processed.columns:
        df_processed['hour'] = df_processed['recorded_at'].dt.hour
        df_processed['day_of_week'] = df_processed['recorded_at'].dt.dayofweek
        df_processed['is_weekend'] = (df_processed['day_of_week'] >= 5).astype(int)
        df_processed['month'] = df_processed['recorded_at'].dt.month
    
    if 'phase_started_at' in df_processed.columns and 'recorded_at' in df_processed.columns:
        df_processed['phase_duration_hours'] = (df_processed['recorded_at'] - 
                                               df_processed['phase_started_at']).dt.total_seconds() / 3600
    
    # Rolling statistics
    for window in [6, 12, 24]:
        if 'inj_temp' in df_processed.columns:
            df_processed[f'inj_temp_rolling_mean_{window}h'] = df_processed['inj_temp'].rolling(window, min_periods=1).mean()
            df_processed[f'inj_temp_rolling_std_{window}h'] = df_processed['inj_temp'].rolling(window, min_periods=1).std()
        if 'inj_whp' in df_processed.columns:
            df_processed[f'inj_whp_rolling_mean_{window}h'] = df_processed['inj_whp'].rolling(window, min_periods=1).mean()
        if 'prod_flow' in df_processed.columns:
            df_processed[f'prod_flow_rolling_max_{window}h'] = df_processed['prod_flow'].rolling(window, min_periods=1).max()
    
    # Rate of change
    for c in ['inj_temp', 'inj_whp', 'cum_inj_energy', 'prod_temp']:
        if c in df_processed.columns:
            df_processed[f'{c}_change'] = df_processed[c].diff()
    
    # Pressure and temperature differences
    if 'inj_whp' in df_processed.columns and 'prod_whp' in df_processed.columns:
        df_processed['pressure_diff'] = df_processed['inj_whp'] - df_processed['prod_whp']
    if 'inj_temp' in df_processed.columns and 'prod_temp' in df_processed.columns:
        df_processed['temp_diff'] = df_processed['inj_temp'] - df_processed['prod_temp']
    
    # Energy efficiency metrics
    if 'inj_energy' in df_processed.columns and 'inj_flow' in df_processed.columns:
        df_processed['inj_energy_per_flow'] = df_processed['inj_energy'] / (df_processed['inj_flow'] + 1e-6)
    if 'cooling_energy' in df_processed.columns and 'inj_energy' in df_processed.columns:
        df_processed['cooling_efficiency'] = df_processed['cooling_energy'] / (df_processed['inj_energy'] + 1e-6)
    
    # Cumulative stress indicators
    if 'cum_inj_energy' in df_processed.columns and 'cum_volume' in df_processed.columns:
        df_processed['cum_energy_normalized'] = df_processed['cum_inj_energy'] / (df_processed['cum_volume'] + 1e-6)
    
    # Interaction features
    if 'inj_temp' in df_processed.columns and 'inj_whp' in df_processed.columns:
        df_processed['temp_pressure_interaction'] = df_processed['inj_temp'] * df_processed['inj_whp']
    if 'inj_flow' in df_processed.columns and 'inj_whp' in df_processed.columns:
        df_processed['flow_pressure_interaction'] = df_processed['inj_flow'] * df_processed['inj_whp']
    
    # ========================================== 
    # PREPARE FEATURES
    # ========================================== 
    exclude_cols = [
        'recorded_at', 'phase_started_at', 'phase_production_ended_at', 'phase_ended_at', 'occurred_at',
        'event_occurs', 'event_magnitude', 'traffic_light', 'traffic_light_actual', 'magnitude',
        'magnitude_actual', 'hourly_seismicity_rate', 'rounded', 'adjusted'
    ]
    feature_cols = [c for c in df_processed.columns if c not in exclude_cols]
    X_operational = df_processed[feature_cols].copy()
    
    # Imputation
    numeric_cols = X_operational.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_operational.select_dtypes(exclude=[np.number]).columns.tolist()
    
    X_operational = X_operational.replace([np.inf, -np.inf], np.nan)
    X_operational.loc[:, numeric_cols] = X_operational[numeric_cols].fillna(models_dict['medians'])
    for col in categorical_cols:
        X_operational.loc[:, col] = X_operational[col].astype(str).fillna('missing')
    
    # ========================================== 
    # MAKE PREDICTIONS
    # ========================================== 
    if models_dict['mode'] == 'catboost':
        from catboost import Pool
        
        cat_features = [i for i, col in enumerate(X_operational.columns) if col in categorical_cols]
        operational_pool = Pool(X_operational, cat_features=cat_features)
        
        y_event_prob = models_dict['event'].predict_proba(operational_pool)[:, 1]
        y_magnitude_pred = np.zeros(len(X_operational))
        
        y_event_pred_base = (y_event_prob >= models_dict['threshold']).astype(int)
        if y_event_pred_base.sum() > 0:
            X_predicted_events = X_operational.iloc[y_event_pred_base == 1].reset_index(drop=True)
            event_pool = Pool(X_predicted_events, cat_features=cat_features)
            y_magnitude_pred[y_event_pred_base == 1] = models_dict['magnitude'].predict(event_pool)
        
        y_traffic_pred = models_dict['traffic'].predict(operational_pool).flatten()
    
    else:  # cached mode
        y_event_prob = models_dict['event_prob']
        y_magnitude_pred = models_dict['magnitude_pred']
        y_traffic_pred = models_dict['traffic_pred']
        y_event_pred_base = (y_event_prob >= models_dict['threshold']).astype(int)
    
    # ========================================== 
    # PREPARE DASHBOARD DATA
    # ========================================== 
    df_dashboard = pd.DataFrame({
        'event_probability': y_event_prob,
        'event_predicted': y_event_pred_base,
        'magnitude_predicted': y_magnitude_pred,
        'traffic_light_pred': y_traffic_pred
    })
    
    if has_timestamp:
        df_dashboard['timestamp'] = df_processed['recorded_at'].values
    
    operational_vars = {
        'inj_flow': 'Injection Flow (m³/h)',
        'inj_whp': 'Injection Pressure (bar)',
        'inj_temp': 'Injection Temperature (°C)',
        'inj_ap': 'Injection Annular Pressure (bar)',
        'prod_temp': 'Production Temperature (°C)',
        'prod_whp': 'Production Pressure (bar)',
        'gt03_whp': 'GT03 Wellhead Pressure (bar)',
        'hedh_thpwr': 'Thermal Power (kW)',
        'basin_flow': 'Basin Flow (m³/h)',
        'prod_flow': 'Production Flow (m³/h)',
        'volume': 'Injected Volume (m³)',
        'cum_volume': 'Cumulative Volume (m³)',
        'inj_energy': 'Injected Energy (MWh)',
        'cum_inj_energy': 'Cumulative Energy (MWh)',
        'cooling_energy': 'Cooling Energy (MWh)',
        'cum_cooling_energy': 'Cumulative Cooling Energy (MWh)',
        'heat_exch_energy': 'Heat Exchanger Energy (MWh)',
    }
    
    for col, label in operational_vars.items():
        if col in df_processed.columns:
            df_dashboard[col] = df_processed[col].values
    
    if has_ground_truth:
        df_dashboard['event_actual'] = df_processed['event_occurs'].values
        df_dashboard['magnitude_actual'] = df_processed['magnitude_actual'].values
        df_dashboard['traffic_light_actual'] = df_processed['traffic_light_actual'].values
    
    traffic_labels = {0: '🟢 GREEN', 1: '🟡 YELLOW', 2: '🔴 RED'}
    traffic_colors = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c'}
    df_dashboard['traffic_label'] = df_dashboard['traffic_light_pred'].map(traffic_labels)
    df_dashboard['traffic_color'] = df_dashboard['traffic_light_pred'].map(traffic_colors)
    
    # ========================================== 
    # STREAMLIT UI
    # ========================================== 
    st.title("🚦 Advanced Seismic Monitoring Dashboard")
    st.markdown("Real-time Operational Monitoring | Event Detection | Risk Assessment")
    if models_dict['mode'] == 'cached':
        st.info("ℹ️ Running in cached prediction mode (CatBoost unavailable on this platform)")
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    if has_timestamp:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date.date(),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=(min_date + timedelta(days=30)).date(),
            min_value=min_date.date(),
            max_value=max_date.date()
        )
    
    threshold = st.sidebar.slider(
        "Event Threshold",
        min_value=0.001,
        max_value=0.99,
        value=models_dict['threshold'],
        step=0.001,
        format="%.3f"
    )
    
    show_actual = st.sidebar.checkbox("Show Actual Events", value=True) if has_ground_truth else False
    
    # Variable selection
    available_vars = [(col, label) for col, label in operational_vars.items() if col in df_dashboard.columns]
    selected_vars = st.multiselect(
        "📊 Select Operational Variables to Display",
        options=[col for col, _ in available_vars],
        default=['inj_whp', 'prod_whp', 'inj_temp'] if len(available_vars) > 0 else [],
        format_func=lambda x: operational_vars.get(x, x)
    )
    
    # Filter data
    if has_timestamp:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (df_dashboard['timestamp'] >= start_dt) & (df_dashboard['timestamp'] <= end_dt)
        df_filtered = df_dashboard[mask].copy()
    else:
        df_filtered = df_dashboard.copy()
    
    # Update predictions with new threshold
    df_filtered['event_pred_dynamic'] = (df_filtered['event_probability'] >= threshold).astype(int)
    
    # Create main plot
    if len(df_filtered) > 0:
        fig = go.Figure()
        
        var_colors = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22', '#34495e', '#16a085', '#d35400']
        
        for idx, var in enumerate(selected_vars):
            if var in df_filtered.columns:
                var_label = operational_vars.get(var, var)
                fig.add_trace(go.Scatter(
                    x=df_filtered['timestamp'] if has_timestamp else df_filtered.index,
                    y=df_filtered[var],
                    mode='lines',
                    name=var_label,
                    line=dict(color=var_colors[idx % len(var_colors)], width=1.5),
                    yaxis='y' if idx == 0 else f'y{idx+1}',
                    hovertemplate=f'{var_label}: %{{y:.2f}}'
                ))
        
        # Add predicted events
        events_df = df_filtered[df_filtered['event_pred_dynamic'] == 1].copy()
        if len(events_df) > 0:
            hover_text = []
            for _, row in events_df.iterrows():
                text = f"⚠️ SEISMIC EVENT\n"
                text += f"Probability: {row['event_probability']:.4f}\n"
                text += f"Magnitude: {row['magnitude_predicted']:.3f}\n"
                text += f"Risk: {row['traffic_label']}"
                hover_text.append(text)
            
            fig.add_trace(go.Scatter(
                x=events_df['timestamp'] if has_timestamp else events_df.index,
                y=events_df['magnitude_predicted'],
                mode='markers',
                name='Predicted Events',
                marker=dict(
                    size=12,
                    color=events_df['traffic_color'],
                    line=dict(width=2, color='white'),
                    symbol='diamond'
                ),
                text=hover_text,
                hovertemplate='%{text}',
                yaxis='y2'
            ))
        
        # Add actual events
        if has_ground_truth and show_actual:
            actual_events = df_filtered[df_filtered['event_actual'] == 1].copy()
            if len(actual_events) > 0:
                fig.add_trace(go.Scatter(
                    x=actual_events['timestamp'] if has_timestamp else actual_events.index,
                    y=actual_events['magnitude_actual'],
                    mode='markers',
                    name='Actual Events',
                    marker=dict(size=10, color='black', symbol='x', line=dict(width=2)),
                    yaxis='y2',
                    hovertemplate='Actual Magnitude: %{y:.3f}'
                ))
        
        # Update layout
        fig.update_xaxes(title_text="Time", showgrid=True, gridcolor='lightgray')
        if selected_vars:
            first_var_label = operational_vars.get(selected_vars[0], selected_vars[0])
            fig.update_yaxes(title_text=first_var_label, secondary_y=False, showgrid=True)
        fig.update_yaxes(title_text="Event Magnitude", secondary_y=True, yaxis='y2')
        
        fig.update_layout(
            title=f"Seismic Monitoring" + (f": {start_date} to {end_date}" if has_timestamp else ""),
            hovermode='closest',
            plot_bgcolor='white',
            height=700,
            margin=dict(r=150)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    n_events = df_filtered['event_pred_dynamic'].sum()
    pct_events = (n_events / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    
    with col1:
        st.metric("Total Samples", f"{len(df_filtered):,}")
    with col2:
        st.metric("⚠️ Events Detected", f"{n_events:,}")
    with col3:
        st.metric("Detection Rate", f"{pct_events:.2f}%")
    with col4:
        if n_events > 0:
            avg_mag = df_filtered[df_filtered['event_pred_dynamic'] == 1]['magnitude_predicted'].mean()
            st.metric("Avg Magnitude", f"{avg_mag:.3f}")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        green_count = (df_filtered['traffic_light_pred'] == 0).sum()
        yellow_count = (df_filtered['traffic_light_pred'] == 1).sum()
        red_count = (df_filtered['traffic_light_pred'] == 2).sum()
        
        fig_risk = go.Figure(data=[go.Pie(
            labels=['🟢 GREEN', '🟡 YELLOW', '🔴 RED'],
            values=[green_count, yellow_count, red_count],
            marker=dict(colors=['#27ae60', '#f39c12', '#e74c3c'])
        )])
        fig_risk.update_layout(title="Risk Distribution", height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Confusion matrix
    if has_ground_truth:
        with col2:
            y_true = df_filtered['event_actual'].values
            y_pred = df_filtered['event_pred_dynamic'].values
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Pred: No', 'Pred: Event'],
                y=['Actual: No', 'Actual: Event'],
                text=cm,
                texttemplate='%{text}',
                colorscale='Blues'
            ))
            fig_cm.update_layout(title="Confusion Matrix", height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Metrics
            st.subheader("🎯 Performance Metrics")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            with col_m1:
                st.metric("Accuracy", f"{accuracy*100:.2f}%")
            with col_m2:
                st.metric("Precision", f"{precision*100:.2f}%")
            with col_m3:
                st.metric("Recall", f"{recall*100:.2f}%")
            with col_m4:
                st.metric("F1-Score", f"{f1*100:.2f}%")
    
    # Event table
    st.subheader("⚠️ Detected Events")
    
    if n_events > 0:
        events_display = df_filtered[df_filtered['event_pred_dynamic'] == 1].copy()
        events_display['Time'] = events_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') if has_timestamp else events_display.index
        events_display['Probability'] = events_display['event_probability'].round(4)
        events_display['Magnitude'] = events_display['magnitude_predicted'].round(3)
        events_display['Risk Level'] = events_display['traffic_label']
        
        table_cols = ['Time', 'Probability', 'Magnitude', 'Risk Level']
        for var in selected_vars:
            if var in events_display.columns:
                var_label = operational_vars.get(var, var)
                events_display[var_label] = events_display[var].round(2)
                table_cols.append(var_label)
        
        if has_ground_truth:
            events_display['Actual Magnitude'] = events_display['magnitude_actual'].round(3)
            table_cols.append('Actual Magnitude')
        
        st.dataframe(events_display[table_cols], use_container_width=True)
    else:
        st.success("✅ No events detected in this period")

else:
    st.error("❌ Failed to load models or data. Please check file paths and dependencies.")