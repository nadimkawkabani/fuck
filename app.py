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
    """
    Loads and preprocesses the sepsis data from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the correct directory.")
        return None

    # --- Data Cleaning ---
    # Standardize column names for robustness
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('’', '')
    
    rename_map = {
        'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
        'Direnç_Durumu': 'Resistance_Status',
        'Mortalite': 'Mortality',
        'KOAH_Asthım': 'COPD_Asthma'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    
    # Convert binary/categorical columns
    if 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence' in df.columns and df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'].dtype == 'object':
        df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'] = df['Systemic_Inflammatory_Response_Syndrome_SIRS_presence'].map({'Var': 1, 'Yok': 0}).fillna(0)
    if 'Comorbidity' in df.columns and df['Comorbidity'].dtype == 'object':
        df['Comorbidity'] = df['Comorbidity'].map({'Var': 1, 'Yok': 0}).fillna(0)
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    if 'Age' in df.columns:
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        bins = [0, 40, 50, 60, 70, 80, 120]
        labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    return df

# --- COMPREHENSIVE EDA DASHBOARD (FULL, UNABRIDGED VERSION) ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset, exploring all key factors.")

    # --- Sidebar Filters ---
    st.sidebar.header("EDA Filters")
    filtered_df = df.copy()
    if 'Age_Group' in df.columns:
        age_options = sorted(list(df['Age_Group'].dropna().unique()))
        selected_age = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age)]

    if 'Gender' in df.columns:
        gender_map = {0: 'Female', 1: 'Male'}
        selected_gender_str = st.sidebar.selectbox("Filter by Gender", options=['All', 'Male', 'Female'], index=0, key="eda_gender_filter")
        if selected_gender_str != 'All':
            gender_code = 1 if selected_gender_str == 'Male' else 0
            filtered_df = filtered_df[filtered_df['Gender'] == gender_code]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- Define column groups based on what's available ---
    vital_cols = [col for col in ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation'] if col in df.columns]
    lab_cols = [col for col in ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 'Creatinine'] if col in df.columns]
    risk_score_cols = [col for col in ['The_National_Early_Warning_Score_NEWS', 'qSOFA_Score', 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence'] if col in df.columns]
    comorbidity_cols = [col for col in ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Others'] if col in df.columns]

    # --- Tabbed Layout ---
    tab1, tab2, tab3 = st.tabs(["📊 Demographics", "🩸 Vitals & Lab Results", "⚠️ Risk Factors & Comorbidities"])

    with tab1:
        st.header("Demographic Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patient Age Distribution")
            if 'Age' in filtered_df.columns:
                fig, ax = plt.subplots(); sns.histplot(filtered_df['Age'].dropna(), kde=True, ax=ax, bins=20); ax.set_title("Distribution of Patient Ages"); st.pyplot(fig)
        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                gender_counts = filtered_df.dropna(subset=['Gender'])['Gender'].map(gender_map).value_counts()
                fig, ax = plt.subplots(); gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['skyblue', 'lightcoral']); ax.set_ylabel(''); st.pyplot(fig)

    with tab2:
        st.header("Vitals & Lab Results Analysis")
        st.markdown("Comparing measurements between survivors and non-survivors.")
        with st.expander("Vital Signs Analysis", expanded=True):
            if not vital_cols or 'Mortality' not in filtered_df.columns: st.warning("Vital sign or Mortality columns not found.")
            else:
                cols = st.columns(min(len(vital_cols), 3))
                for i, vital in enumerate(vital_cols):
                    with cols[i % len(cols)]:
                        fig, ax = plt.subplots(); sns.boxplot(data=filtered_df, x='Mortality', y=vital, ax=ax, palette='viridis'); ax.set_xticklabels(['Survived', 'Died']); ax.set_title(vital); st.pyplot(fig)
        
        with st.expander("Lab Results Analysis"):
            if not lab_cols: st.warning("No lab columns found.")
            else:
                lab_to_plot = st.selectbox("Select Lab Value", options=lab_cols)
                fig, ax = plt.subplots(); sns.kdeplot(data=filtered_df, x=lab_to_plot, hue='Mortality', fill=True, common_norm=False, palette='coolwarm'); ax.set_title(f"Distribution of {lab_to_plot}"); st.pyplot(fig)

    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        if not comorbidity_cols or 'Mortality' not in filtered_df.columns: st.warning("Comorbidity or Mortality columns not found.")
        else:
            mortality_rates = {}
            for col in comorbidity_cols:
                if filtered_df[col].sum() > 0:
                    rate = filtered_df[filtered_df[col] == 1]['Mortality'].mean()
                    if pd.notna(rate): mortality_rates[col] = rate * 100
            
            if mortality_rates:
                mortality_df = pd.DataFrame(list(mortality_rates.items()), columns=['Comorbidity', 'Mortality Rate (%)']).sort_values('Mortality Rate (%)', ascending=False)
                fig, ax = plt.subplots(figsize=(10, 8)); sns.barplot(data=mortality_df, x='Mortality Rate (%)', y='Comorbidity', ax=ax, palette='rocket'); ax.set_title("Mortality Rate by Comorbidity"); st.pyplot(fig)

# --- PREDICTIVE ANALYSIS DASHBOARD (FULL, UNABRIDGED VERSION) ---
def display_prediction_dashboard(df):
    st.title("🤖 Mortality Prediction & Risk Analysis")
    st.markdown("Using machine learning to predict patient mortality.")

    features = [
        'Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus',
        'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Pulse_rate', 'Respiratory_Rate',
        'Systolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP', 
        'The_National_Early_Warning_Score_NEWS', 'qSOFA_Score'
    ]
    target = 'Mortality'

    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns: st.error("Essential columns for prediction are missing."); return
    
    df_model = df[available_features + [target]].dropna()
    if df_model.empty: st.error("No data for model training after handling missing values."); return
        
    X = df_model[available_features]
    y = df_model[target]

    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'); model.fit(X_train, y_train); return model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Performance", "🔑 Key Risk Factors", "🎯 Patient Risk Stratification", "🧮 Live Prediction Calculator"])

    with tab1:
        st.header("Model Performance Evaluation"); y_pred = model.predict(X_test); accuracy = accuracy_score(y_test, y_pred)
        col1, col2 = st.columns(2)
        with col1: st.metric("Model Accuracy on Test Data", f"{accuracy:.2%}")
        with col2:
            st.subheader("Confusion Matrix"); cm = confusion_matrix(y_test, y_pred); fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Predicted Survived', 'Predicted Died'], yticklabels=['Actual Survived', 'Actual Died']); plt.ylabel('Actual Outcome'); plt.xlabel('Predicted Outcome'); st.pyplot(fig)

    with tab2:
        st.header("Top Clinical Predictors of Mortality"); importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        fig, ax = plt.subplots(figsize=(10, 8)); sns.barplot(x='importance', y='feature', data=importances.head(15), palette='mako', ax=ax); ax.set_title("Top 15 Most Important Features"); st.pyplot(fig)

    with tab3:
        st.header("Patient Risk Stratification"); df_model['risk_score'] = model.predict_proba(X)[:, 1]
        top_features = importances['feature'].tolist(); x_axis_feat = top_features[0]; y_axis_feat = top_features[1]
        fig = px.scatter(df_model, x=x_axis_feat, y=y_axis_feat, color='risk_score', color_continuous_scale=px.colors.sequential.OrRd, hover_name=df_model.index, hover_data={'risk_score': ':.2f}', 'Mortality': True}, title=f"Patient Risk Map: {x_axis_feat} vs. {y_axis_feat}")
        fig.update_layout(xaxis_title=x_axis_feat.replace('_', ' '), yaxis_title=y_axis_feat.replace('_', ' '), coloraxis_colorbar=dict(title="Mortality Risk")); st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Live Patient Risk Calculator")
        with st.form("prediction_form"):
            input_data = {}; col1, col2, col3 = st.columns(3)
            for i, feature in enumerate(available_features):
                current_col = [col1, col2, col3][i % 3]
                with current_col:
                    if 'Age' in feature: input_data[feature] = st.slider(feature, 18, 100, 65)
                    elif 'Gender' in feature: input_data[feature] = 1 if st.selectbox(feature, ['Female', 'Male'], key=f"pred_{feature}") == 'Male' else 0
                    elif df_model[feature].max() <= 1 and df_model[feature].min() >= 0: input_data[feature] = st.checkbox(feature, value=False)
                    else:
                        min_val = float(df_model[feature].min()); max_val = float(df_model[feature].max()); mean_val = float(df_model[feature].mean())
                        input_data[feature] = st.slider(feature, min_val, max_val, mean_val)
            submitted = st.form_submit_button("Calculate Mortality Risk")
        if submitted:
            input_df = pd.DataFrame([input_data])[X_train.columns]; prediction_proba = model.predict_proba(input_df)[0][1]; risk_percent = prediction_proba * 100
            st.progress(prediction_proba); st.metric(label="Predicted Risk of Mortality", value=f"{risk_percent:.1f}%")
            if risk_percent > 50: st.error("HIGH RISK")
            elif risk_percent > 20: st.warning("MODERATE RISK")
            else: st.success("LOW RISK")

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
        display_prediction_dashboard(sepsis_df)
