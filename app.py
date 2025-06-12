import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Sepsis Clinical Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    
    # --- Data Cleaning and Preprocessing ---
    # This section is kept for robustness, even if not all columns exist
    rename_map = {
        'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
        'Direnç_Durumu': 'Resistance_Status',
        'Mortalite': 'Mortality',
        'KOAH_Asthım': 'COPD_Asthma',
        'Antibioterapy’': 'Antibioterapy'
    }
    df.rename(columns=rename_map, inplace=True)
    
    if 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence' in df.columns and df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'].dtype == 'object':
        df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'] = df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'].map({'Var': 1, 'Yok': 0}).fillna(0)
    if 'Comorbidity' in df.columns and df['Comorbidity'].dtype == 'object':
        df['Comorbidity'] = df['Comorbidity'].map({'Var': 1, 'Yok': 0}).fillna(0)
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    if 'Age' in df.columns:
        bins = [0, 40, 50, 60, 70, 80, 120]
        labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    return df

# --- COMPREHENSIVE EDA DASHBOARD (WITH DEBUGGING) ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset.")

    # --- Define EXPECTED column groups ---
    vital_cols = ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation']
    lab_cols = ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 'Creatinine']
    risk_score_cols = ['The_National_Early_Warning_Score_NEWS', 'qSOFA_Score', 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence']
    comorbidity_cols = ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Others']
    
    # --- SELF-DIAGNOSING DEBUGGER ---
    with st.expander("⚠️ Data Column Check (Click to see details)", expanded=True):
        actual_cols = set(df.columns)
        st.write("**Actual Columns found in your file:**", df.columns.tolist())
        
        missing_vitals = set(vital_cols) - actual_cols
        missing_labs = set(lab_cols) - actual_cols
        missing_risks = set(risk_score_cols) - actual_cols
        
        if not (missing_vitals or missing_labs or missing_risks):
            st.success("All expected columns for EDA were found!")
        else:
            st.error("Some expected columns were NOT found in your CSV file. The charts below might be empty. Please check the names in the script.")
            if missing_vitals: st.warning(f"**Missing Vital Columns:** `{list(missing_vitals)}`")
            if missing_labs: st.warning(f"**Missing Lab Columns:** `{list(missing_labs)}`")
            if missing_risks: st.warning(f"**Missing Risk Columns:** `{list(missing_risks)}`")

    # --- Filter data based on available columns ---
    available_vitals = [col for col in vital_cols if col in df.columns]
    available_labs = [col for col in lab_cols if col in df.columns]
    available_risks = [col for col in risk_score_cols if col in df.columns]
    available_comorbidities = [col for col in comorbidity_cols if col in df.columns]

    # --- Sidebar Filters ---
    # ... (sidebar filter code remains the same) ...
    st.sidebar.header("EDA Filters")
    filtered_df = df.copy() # start with full df

    # --- Tabbed Layout ---
    tab1, tab2, tab3 = st.tabs(["📊 Demographics", "🩸 Vitals & Lab Results", "⚠️ Risk Factors & Comorbidities"])

    with tab1:
        # ... (demographics plotting code) ...
        st.header("Demographic Analysis")
        # Code here remains the same, it will plot if 'Age', 'Gender' etc. are found.

    with tab2:
        st.header("Vitals & Lab Results Analysis")
        with st.expander("Vital Signs Analysis", expanded=True):
            if not available_vitals: st.warning("No vital sign columns found to plot.")
            else:
                # Dynamically create columns for layout
                cols = st.columns(min(len(available_vitals), 3)) 
                for i, vital in enumerate(available_vitals):
                    with cols[i % min(len(available_vitals), 3)]:
                        fig, ax = plt.subplots(); sns.boxplot(data=df, x='Mortality', y=vital, ax=ax); st.pyplot(fig)
        
        with st.expander("Lab Results Analysis"):
            if not available_labs: st.warning("No lab result columns found to plot.")
            else:
                lab_to_plot = st.selectbox("Select Lab Value", options=available_labs)
                fig, ax = plt.subplots(); sns.kdeplot(data=df, x=lab_to_plot, hue='Mortality', fill=True); st.pyplot(fig)

    with tab3:
        # ... (risk factors plotting code) ...
        st.header("Risk Factors & Comorbidities Analysis")

# ... (The display_prediction_dashboard function remains the same) ...

# --- Main App Logic ---
sepsis_df = load_data('ICU_Sepsis_Cleaned.csv')

if sepsis_df is not None:
    st.sidebar.title("🩺 Sepsis Analytics")
    app_mode = st.sidebar.selectbox(
        "Choose the Dashboard",
        ["Comprehensive EDA", "Mortality Predictive Analysis"]
    )
    
    if app_mode == "Comprehensive EDA":
        display_eda_dashboard(sepsis_df)
    else:
        # Placeholder for the prediction dashboard
        st.title("🤖 Mortality Prediction & Risk Analysis")
        st.info("The predictive analysis dashboard would be displayed here.")
        # display_prediction_dashboard(sepsis_df)
