import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                           precision_score, recall_score, f1_score,
                           roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import PartialDependenceDisplay
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Sepsis Clinical Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'model_details' not in st.session_state:
    st.session_state.model_details = {
        "model": None,
        "model_type": None,
        "features": None
    }

# --- Data Loading and Cleaning ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/nadimkawkabani/fuck/main/ICU_Sepsis.csv"
    try:
        df = pd.read_csv(url)
        if df.empty:
            st.error("The loaded CSV file from the URL is empty.")
            return None
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('’', '')
        rename_map = {
            'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
            'Direnç_Durumu': 'Resistance_Status',
            'Mortalite': 'Mortality',
            'KOAH_Asthım': 'COPD_Asthma'
        }
        df.rename(columns=rename_map, inplace=True, errors='ignore')

        if 'Mortality' not in df.columns:
            st.error("Error: The required target column 'Mortality' was not found.")
            return None
            
        if 'Gender' in df.columns:
            if pd.api.types.is_object_dtype(df['Gender']):
                gender_map = {'male': 1, 'erkek': 1, 'm': 1, 'female': 0, 'kadın': 0, 'f': 0}
                df['Gender'] = df['Gender'].astype(str).str.lower().map(gender_map).fillna(0)
            elif pd.api.types.is_numeric_dtype(df['Gender']):
                df['Gender'] = df['Gender'].replace(2, 0)

        mappings = {
            'Comorbidity': {'var': 1},
            'Systemic_Inflammatory_Response_Syndrome_SIRS_presence': {'var': 1}
        }
        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().map(mapping).fillna(0).astype(int)

        for col in ['Gender', 'Mortality', 'Comorbidity', 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence']:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        numeric_cols = ['Age', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure',
                       'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        if 'Age' in df.columns:
            bins = [0, 18, 40, 50, 60, 70, 80, 120]
            labels = ['<18', '18-39', '40-49', '50-59', '60-69', '70-79', '80+']
            df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
            
        return df
    except Exception as e:
        st.error(f"Failed to load or process data. Error: {str(e)}")
        return None

sepsis_df = load_data()

# --- Visualization Functions ---
def plot_interactive_distribution(df, column, hue=None):
    if hue:
        fig = px.histogram(df, x=column, color=hue, marginal='box', nbins=30, barmode='overlay', title=f'Distribution of {column} by {hue}', opacity=0.7)
    else:
        fig = px.histogram(df, x=column, marginal='box', nbins=30, title=f'Distribution of {column}')
    fig.update_layout(legend_title_text=hue if hue else '')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns):
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# --- EDA Dashboard Function ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset with interactive visualizations.")
    st.sidebar.header("🔍 EDA Filters")
    filtered_df = df.copy()
    if 'Age_Group' in df.columns and isinstance(df['Age_Group'].dtype, pd.CategoricalDtype):
        age_options = list(df['Age_Group'].cat.categories)
        selected_age = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age)]
    if 'Gender' in df.columns:
        selected_gender_str = st.sidebar.selectbox("Filter by Gender", options=['All', 'Male', 'Female'], index=0)
        if selected_gender_str != 'All':
            gender_code = 1 if selected_gender_str == 'Male' else 0
            filtered_df = filtered_df[filtered_df['Gender'] == gender_code]
    if 'Mortality' in df.columns:
        mortality_filter = st.sidebar.selectbox("Filter by Mortality Status", options=['All', 'Survived', 'Died'], index=0)
        if mortality_filter != 'All':
            mortality_code = 1 if mortality_filter == 'Died' else 0
            filtered_df = filtered_df[filtered_df['Mortality'] == mortality_code]
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return
    vital_cols = [col for col in ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation'] if col in df.columns]
    lab_cols = [col for col in ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 'Creatinine'] if col in df.columns]
    comorbidity_cols = [col for col in ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Others'] if col in df.columns]
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "🩸 Vitals & Labs", "⚠️ Risk Factors", "📈 Correlations"])
    with tab1:
        st.header("Demographic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patient Age Distribution")
            if 'Age' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                plot_interactive_distribution(filtered_df, 'Age', 'Mortality')
        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns:
                gender_map = {1: 'Male', 0: 'Female'}
                gender_counts = filtered_df['Gender'].map(gender_map).value_counts()
                fig = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index, title='Gender Distribution')
                st.plotly_chart(fig, use_container_width=True)
        st.subheader("Age vs. Mortality")
        if 'Age' in filtered_df.columns and 'Mortality' in filtered_df.columns:
            fig = px.box(filtered_df, x='Mortality', y='Age', color='Mortality', points='all', title='Age Distribution by Mortality Status')
            fig.update_xaxes(title_text='Mortality Status', tickvals=[0, 1], ticktext=['Survived', 'Died'])
            st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.header("Vitals & Lab Results Analysis")
        with st.expander("📈 Vital Signs Analysis", expanded=True):
            if vital_cols and 'Mortality' in filtered_df.columns:
                selected_vital = st.selectbox("Select Vital Sign to Visualize", options=vital_cols)
                plot_interactive_distribution(filtered_df, selected_vital, 'Mortality')
            else: st.warning("Vital sign or Mortality columns not found.")
        with st.expander("🧪 Lab Results Analysis"):
            if lab_cols and 'Mortality' in filtered_df.columns:
                selected_lab = st.selectbox("Select Lab Value", options=lab_cols, key="lab_select")
                plot_interactive_distribution(filtered_df, selected_lab, 'Mortality')
            else: st.warning("Lab or Mortality columns not found.")
    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        if comorbidity_cols and 'Mortality' in filtered_df.columns:
            st.subheader("Comorbidity Prevalence")
            comorbidity_counts = filtered_df[comorbidity_cols].sum().sort_values(ascending=False)
            fig = px.bar(comorbidity_counts, orientation='h', title='Prevalence of Comorbidities', labels={'value': 'Count', 'index': 'Comorbidity'})
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Comorbidity or Mortality columns not found.")
    with tab4:
        st.header("Feature Correlations")
        all_numeric_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col])]
        if len(all_numeric_cols) > 1:
            plot_correlation_matrix(filtered_df, all_numeric_cols)
        else: st.warning("Not enough numeric columns for correlation analysis.")

# --- ML Model Functions ---
@st.cache_resource
def train_model(_X_train, _y_train, model_type='Random Forest', **params):
    if model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', None), class_weight='balanced', random_state=42, n_jobs=-1)
    elif model_type == 'Logistic Regression':
        model = make_pipeline(StandardScaler(), LogisticRegression(penalty=params.get('penalty', 'l2'), C=params.get('C', 1.0), class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear'))
    elif model_type == 'XGBoost':
        model = XGBClassifier(n_estimators=params.get('n_estimators', 100), max_depth=params.get('max_depth', 3), learning_rate=params.get('learning_rate', 0.1), objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(_X_train, _y_train)
    return model

# --- SIMPLIFIED PREDICTION DASHBOARD ---
def display_prediction_dashboard(df):
    st.title("🧮 Sepsis Mortality Risk Calculator")
    st.markdown("Enter patient data below to calculate the predicted risk of mortality based on a pre-trained Random Forest model.")

    features = ['Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP', 'The_National_Early_Warning_Score_NEWS', 'qSOFA_Score']
    target = 'Mortality'
    available_features = [f for f in features if f in df.columns]

    if not available_features or target not in df.columns:
        st.error("❌ Essential columns for prediction are missing from the source data.")
        return

    df_model = df[available_features + [target]].dropna()
    if df_model.empty:
        st.error("❌ No data available for model training after handling missing values.")
        return

    X = df_model[available_features]
    y = df_model[target]

    if y.nunique() < 2:
        st.error(
            "❌ **Cannot Build Model:** The source data only contains one outcome and cannot be used for prediction."
        )
        return

    # --- Automatic Model Training (if not already trained) ---
    if st.session_state.model_details["model"] is None:
        with st.spinner("Initializing the predictive model... Please wait."):
            # We use default parameters for the automatic training
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model = train_model(X_train, y_train, model_type='Random Forest')
            st.session_state.model_details = {
                "model": model,
                "model_type": 'Random Forest',
                "features": X_train.columns.tolist()
            }
        st.success("✅ Predictive model is ready.")

    # Get the trained model from session state
    model = st.session_state.model_details["model"]
    model_features = st.session_state.model_details["features"]

    # --- Display the Risk Calculator Form ---
    st.header("Patient Risk Calculator")
    with st.form("prediction_form"):
        input_data = {}
        col1, col2, col3 = st.columns(3)
        
        # Dynamically create input fields
        for i, feature in enumerate(model_features):
            with [col1, col2, col3][i % 3]:
                if feature == 'Age':
                    input_data[feature] = st.slider("Age (years)", 18, 100, 65)
                elif feature == 'Gender':
                    selected_gender = st.selectbox("Gender", ['Male', 'Female'])
                    input_data[feature] = 1 if selected_gender == 'Male' else 0
                elif feature in ['Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma']:
                    input_data[feature] = 1 if st.checkbox(f"Has {feature.replace('_', ' ')}", False) else 0
                else:
                    # Use reasonable defaults for sliders if data is available
                    min_val = float(X[feature].min())
                    max_val = float(X[feature].max())
                    mean_val = float(X[feature].mean())
                    input_data[feature] = st.slider(feature.replace('_', ' '), min_val, max_val, mean_val)

        submitted = st.form_submit_button("Calculate Mortality Risk")
        
        if submitted:
            input_df = pd.DataFrame([input_data])[model_features]
            risk_percent = 0.0
            if hasattr(model, "predict_proba") and len(model.classes_) == 2:
                risk_percent = model.predict_proba(input_df)[0][1] * 100
            
            st.subheader("Prediction Results")
            colA, colB = st.columns([1, 2])
            with colA:
                st.metric(label="Predicted Mortality Risk", value=f"{risk_percent:.1f}%")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_percent,
                    title={'text': "Risk Level"},
                    gauge={'axis': {'range': [None, 100]},
                           'steps': [
                               {'range': [0, 20], 'color': "lightgreen"},
                               {'range': [20, 50], 'color': "orange"},
                               {'range': [50, 100], 'color': "red"}],
                           'bar': {'color': "darkblue"}}
                ))
                fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with colB:
                if risk_percent > 50:
                    st.error("🔴 HIGH RISK", icon="🚨")
                    st.markdown("**Recommendations:** Consider immediate ICU admission, aggressive fluid resuscitation, broad-spectrum antibiotics, and frequent monitoring.")
                elif risk_percent > 20:
                    st.warning("🟠 MODERATE RISK", icon="⚠️")
                    st.markdown("**Recommendations:** Consider hospital admission, initiate sepsis protocol, perform frequent vital sign checks, and evaluate need for antibiotics.")
                else:
                    st.success("🟢 LOW RISK", icon="✅")
                    st.markdown("**Recommendations:** Continue observation, consider outpatient follow-up, and educate patient on when to seek further care.")

# --- Main App Logic ---
def main():
    st.sidebar.title("🩺 Sepsis Analytics Suite")
    if sepsis_df is not None:
        st.sidebar.markdown("---")
        app_mode = st.sidebar.selectbox(
            "Choose Dashboard",
            ["Comprehensive EDA", "Mortality Risk Calculator"], # Changed label
            index=0,
            key="app_mode_select"
        )
        st.sidebar.markdown("---")
        if app_mode == "Comprehensive EDA":
            display_eda_dashboard(sepsis_df)
        elif app_mode == "Mortality Risk Calculator": # Changed label
            display_prediction_dashboard(sepsis_df)
    else:
        st.title("Welcome to the Sepsis Clinical Analytics Dashboard")
        st.error("🚨 Could not load the dataset. Please ensure the URL in the script is correct and the file is publicly accessible on GitHub.")
        st.image("https://www.sccm.org/SCCM/media/images/sepsis-rebranded-logo.jpg", width=400)

if __name__ == "__main__":
    main()
