import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # --- Data Cleaning and Preprocessing ---
    df.rename(columns={
        'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
        'Direnç_Durumu': 'Resistance_Status',
        'Mortalite': 'Mortality',
        'KOAH_Asthım': 'COPD_Asthma'
    }, inplace=True)

    if "Antibioterapy’" in df.columns:
        df.rename(columns={"Antibioterapy’": "Antibioterapy"}, inplace=True)
    
    binary_map_var = {'Var': 1, 'Yok': 0}
    for col in ['Systemic_Inflammatory_Response_Syndrome_SIRS_presence', 'Comorbidity']:
         if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map(binary_map_var).fillna(0)
            
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    if 'Age' in df.columns:
        bins = [0, 40, 50, 60, 70, 80, 120]
        labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    return df

# --- NEW: Comprehensive EDA Dashboard ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset, exploring all key factors.")

    # --- Sidebar Filters for EDA ---
    st.sidebar.header("EDA Filters")
    if 'Age_Group' not in df.columns:
        st.warning("Age_Group column not found. Age-based filtering is disabled.")
        filtered_df = df.copy()
    else:
        age_options = sorted(list(df['Age_Group'].dropna().unique()))
        selected_age = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)
        filtered_df = df[df['Age_Group'].isin(selected_age)]

    selected_gender = st.sidebar.selectbox("Filter by Gender", options=['All', 'Male', 'Female'], index=0)

    if selected_gender != 'All':
        gender_code = 1 if selected_gender == 'Male' else 0
        filtered_df = filtered_df[filtered_df['Gender'] == gender_code]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- Define column groups for plotting ---
    demographic_cols = ['Age_Group', 'Gender']
    vital_cols = ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation']
    lab_cols = ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 'Creatinine']
    risk_score_cols = ['The_National_Early_Warning_Score_NEWS', 'qSOFA_Score', 'Systemic_Inflammatory_Response_Syndrome_SIRS_presence']
    comorbidity_cols = ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Others']
    
    # Check for available columns
    available_vitals = [col for col in vital_cols if col in df.columns]
    available_labs = [col for col in lab_cols if col in df.columns]
    available_risks = [col for col in risk_score_cols if col in df.columns]
    available_comorbidities = [col for col in comorbidity_cols if col in df.columns]

    # --- Tabbed Layout ---
    tab1, tab2, tab3 = st.tabs(["📊 Demographics", "🩸 Vitals & Lab Results", "⚠️ Risk Factors & Comorbidities"])

    with tab1:
        st.header("Demographic Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Age Distribution")
            if 'Age' in filtered_df.columns:
                fig, ax = plt.subplots()
                sns.histplot(filtered_df['Age'], kde=True, ax=ax, bins=20)
                ax.set_title("Distribution of Patient Ages")
                st.pyplot(fig)

            st.subheader("Mortality Rate by Age Group")
            if 'Age_Group' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                age_mortality = filtered_df.groupby('Age_Group', observed=True)['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
                if 1 in age_mortality.columns:
                    fig, ax = plt.subplots()
                    age_mortality[1].plot(kind='barh', ax=ax, color='salmon')
                    ax.set_xlabel("Mortality Rate (%)")
                    st.pyplot(fig)

        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns:
                gender_counts = filtered_df['Gender'].map({0: 'Female', 1: 'Male'}).value_counts()
                st.dataframe(gender_counts)
                fig, ax = plt.subplots()
                gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['skyblue', 'lightcoral'])
                ax.set_ylabel('')
                ax.set_title("Patient Gender Breakdown")
                st.pyplot(fig)

            st.subheader("Mortality Rate by Gender")
            if 'Gender' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                gender_mortality = filtered_df.groupby('Gender')['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
                gender_mortality.index = gender_mortality.index.map({0: 'Female', 1: 'Male'})
                if 1 in gender_mortality.columns:
                    fig, ax = plt.subplots()
                    gender_mortality[1].plot(kind='bar', ax=ax, color='lightcoral')
                    ax.set_ylabel("Mortality Rate (%)")
                    ax.tick_params(axis='x', rotation=0)
                    st.pyplot(fig)

    with tab2:
        st.header("Vitals & Lab Results Analysis")
        st.markdown("Comparing measurements between patients who survived and those who did not.")
        
        # --- Vitals Section ---
        with st.expander("Vital Signs Analysis", expanded=True):
            if not available_vitals:
                st.warning("No vital sign columns found in the dataset.")
            else:
                num_vitals_cols = 3
                cols = st.columns(num_vitals_cols)
                for i, vital in enumerate(available_vitals):
                    with cols[i % num_vitals_cols]:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.boxplot(data=filtered_df, x='Mortality', y=vital, ax=ax, palette='viridis')
                        ax.set_xticklabels(['Survived', 'Died'])
                        ax.set_title(vital.replace('_', ' ').title())
                        st.pyplot(fig)

        # --- Lab Results Section ---
        with st.expander("Lab Results Analysis"):
            if not available_labs:
                st.warning("No lab result columns found in the dataset.")
            else:
                lab_to_plot = st.selectbox("Select a Lab Value to Visualize", options=available_labs)
                fig, ax = plt.subplots()
                sns.kdeplot(data=filtered_df, x=lab_to_plot, hue='Mortality', fill=True, palette='coolwarm', common_norm=False)
                ax.set_title(f"Distribution of {lab_to_plot} for Survivors vs. Non-Survivors")
                st.pyplot(fig)
                
                # Correlation Heatmap for all numerical values
                st.subheader("Correlation of Vitals and Labs")
                corr_df = filtered_df[available_vitals + available_labs].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_df, cmap='coolwarm', ax=ax, annot=False) # Annot=False for readability
                st.pyplot(fig)


    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Risk Score Distributions")
            if not available_risks:
                st.warning("No risk score columns found.")
            else:
                for score in available_risks:
                    if score in filtered_df.columns: # Check again
                        fig, ax = plt.subplots()
                        sns.boxplot(data=filtered_df, x='Mortality', y=score, ax=ax, palette='mako')
                        ax.set_xticklabels(['Survived', 'Died'])
                        ax.set_title(f"Impact of {score.replace('_', ' ')}")
                        st.pyplot(fig)
        
        with col2:
            st.subheader("Impact of Specific Comorbidities")
            if not available_comorbidities:
                st.warning("No comorbidity columns found.")
            else:
                comorbidity_data = []
                for comorbidity in available_comorbidities:
                    if comorbidity in filtered_df.columns:
                        rate = filtered_df[filtered_df[comorbidity] == 1]['Mortality'].mean() * 100
                        if not pd.isna(rate):
                           comorbidity_data.append({'Comorbidity': comorbidity, 'Mortality Rate (%)': rate})
                
                if comorbidity_data:
                    mortality_df = pd.DataFrame(comorbidity_data).sort_values('Mortality Rate (%)', ascending=False)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(data=mortality_df, y='Comorbidity', x='Mortality Rate (%)', ax=ax, palette='rocket')
                    ax.set_title("Mortality Rate for Patients with Specific Comorbidities")
                    st.pyplot(fig)
                else:
                    st.info("No comorbidity data to display.")

# ... (The display_prediction_dashboard function remains the same) ...
def display_prediction_dashboard(df):
    st.title("🤖 Mortality Prediction Analysis")
    st.markdown("Training a model to predict patient mortality based on clinical data.")
    features = [
        'Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus',
        'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Pulse_rate', 'Respiratory_Rate',
        'Systolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP', 
        'The_National_Early_Warning_Score_NEWS', 'qSOFA_Score'
    ]
    target = 'Mortality'

    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns:
        st.error("The dataset is missing essential columns for prediction.")
        return
    
    df_model = df[available_features + [target]].dropna()
    if df_model.empty:
        st.error("No data available for model training after removing rows with missing values.")
        return
        
    X = df_model[available_features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with st.spinner("Training the model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    st.success("Model trained successfully!")

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Survived', 'Died'], yticklabels=['Survived', 'Died'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    st.subheader("Top Clinical Predictors of Mortality")
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances, palette='mako', ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    st.sidebar.header("Real-Time Risk Prediction")
    st.sidebar.markdown("Enter hypothetical patient data to predict mortality risk.")
    inputs = {}
    if 'Age' in available_features: inputs['Age'] = st.sidebar.slider("Age", 18, 100, 65)
    if 'Gender' in available_features: inputs['Gender'] = 1 if st.sidebar.selectbox("Gender", ['Female', 'Male'], key="pred_gender") == 'Male' else 0
    if 'The_National_Early_Warning_Score_NEWS' in available_features: inputs['NEWS Score'] = st.sidebar.slider("NEWS Score", 0, 20, 5)
    
    if st.sidebar.button("Predict Risk"):
        if inputs.get('NEWS Score', 0) > 7 or inputs.get('Age', 0) > 75:
            st.sidebar.error("High Risk of Mortality")
        else:
            st.sidebar.success("Lower Risk of Mortality")
        st.sidebar.caption("Note: This is an illustrative prediction.")

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
