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
    # Fix potentially problematic column names
    df.rename(columns={
        'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
        'Direnç_Durumu': 'Resistance_Status',
        'Mortalite': 'Mortality',
        'KOAH_Asthım': 'COPD_Asthma'
    }, inplace=True)

    # Clean the last column name if it has an extra apostrophe
    if "Antibioterapy’" in df.columns:
        df.rename(columns={"Antibioterapy’": "Antibioterapy"}, inplace=True)
    
    # Convert categorical 'Yes'/'No' or 'Var'/'Yok' style columns to binary 1/0
    binary_map_var = {'Var': 1, 'Yok': 0}
    for col in ['Systemic_Inflammatory_Response_Syndrome_SIRS_presence', 'Comorbidity']:
         if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map(binary_map_var).fillna(0)
            
    # Convert Gender to numeric for easier processing
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    
    # Create Age Bins for better visualization
    if 'Age' in df.columns:
        bins = [0, 40, 50, 60, 70, 80, 120]
        labels = ['<40', '40-49', '50-59', '60-69', '70-79', '80+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

    return df

# --- EDA Dashboard ---
def display_eda_dashboard(df):
    st.title("🏥 Sepsis Exploratory Data Analysis (EDA)")
    st.markdown("Exploring trends and patterns in the sepsis patient dataset.")

    # --- Sidebar Filters for EDA ---
    st.sidebar.header("EDA Filters")
    if 'Age_Group' not in df.columns:
        st.warning("Age_Group column not found. Age-based filtering is disabled.")
        filtered_df = df.copy()
    else:
        # CORRECTED LINE: Convert categorical uniques to string list before sorting
        selected_age = st.sidebar.multiselect(
            "Filter by Age Group",
            options=sorted(df['Age_Group'].unique().astype(str)),
            default=sorted(df['Age_Group'].unique().astype(str))
        )
        filtered_df = df[df['Age_Group'].isin(selected_age)]

    selected_gender = st.sidebar.selectbox(
        "Filter by Gender",
        options=['All', 'Male', 'Female'],
        index=0
    )

    # --- Filtering Logic ---
    if selected_gender != 'All':
        gender_code = 1 if selected_gender == 'Male' else 0
        filtered_df = filtered_df[filtered_df['Gender'] == gender_code]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return

    # --- 2x2 Grid Layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Who: Mortality Rate by Age Group ---
        st.subheader("Mortality Rate by Age Group")
        if 'Age_Group' in filtered_df.columns and 'Mortality' in filtered_df.columns:
            # Use observed=True to handle filtered categories correctly
            age_mortality = filtered_df.groupby('Age_Group', observed=True)['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
            if 1 in age_mortality.columns:
                fig, ax = plt.subplots()
                age_mortality[1].plot(kind='bar', ax=ax, color='salmon')
                ax.set_ylabel("Mortality Rate (%)")
                ax.set_title("Mortality Rate Increases with Age")
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)
            else:
                st.info("No mortality events for this selection.")
        else:
            st.warning("Required columns for Age/Mortality analysis not found.")

        # --- Why: Risk Score Impact ---
        st.subheader("Impact of NEWS Score on Mortality")
        if 'The_National_Early_Warning_Score_NEWS' in filtered_df.columns and 'Mortality' in filtered_df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_df, x='Mortality', y='The_National_Early_Warning_Score_NEWS', ax=ax)
            ax.set_xticklabels(['Survived', 'Died'])
            ax.set_title("Higher NEWS scores associated with higher mortality")
            st.pyplot(fig)
        else:
            st.warning("Required columns for NEWS score analysis not found.")

    with col2:
        # --- What: Key Vitals Comparison ---
        st.subheader("Key Vital Signs: Survivors vs. Non-Survivors")
        vital_options = [col for col in ['Pulse_rate', 'Respiratory_Rate', 'Fever', 'Oxygen_saturation'] if col in filtered_df.columns]
        if vital_options and 'Mortality' in filtered_df.columns:
            vital_to_check = st.selectbox("Select Vital Sign to Compare", vital_options)
            fig, ax = plt.subplots()
            sns.boxplot(data=filtered_df, x='Mortality', y=vital_to_check, ax=ax, palette='viridis')
            ax.set_xticklabels(['Survived', 'Died'])
            ax.set_title(f"{vital_to_check.replace('_', ' ')} Distribution")
            st.pyplot(fig)
        else:
            st.warning("Required columns for Vital Sign analysis not found.")

        # --- What: Comorbidity Impact ---
        st.subheader("Impact of Comorbidities on Outcome")
        if 'Comorbidity' in filtered_df.columns and 'Mortality' in filtered_df.columns:
            comorbidity_impact = filtered_df.groupby('Comorbidity')['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
            fig, ax = plt.subplots()
            comorbidity_impact.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'salmon'])
            ax.set_xticklabels(['No Comorbidity', 'Has Comorbidity'], rotation=0)
            ax.set_ylabel("Percentage (%)")
            ax.legend(['Survived', 'Died'])
            st.pyplot(fig)
        else:
            st.warning("Required columns for Comorbidity analysis not found.")

# --- Predictive Analysis Dashboard ---
def display_prediction_dashboard(df):
    st.title("🤖 Mortality Prediction Analysis")
    st.markdown("Training a model to predict patient mortality based on clinical data.")

    # --- Feature Selection and Data Prep ---
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

    # --- Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    with st.spinner("Training the model..."):
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    st.success("Model trained successfully!")

    # --- Display Model Performance ---
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

    # --- Feature Importance ---
    st.subheader("Top Clinical Predictors of Mortality")
    st.markdown("These are the most important factors the model used to make predictions.")
    
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importances, palette='mako', ax=ax)
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    # --- Interactive Prediction Sidebar ---
    st.sidebar.header("Real-Time Risk Prediction")
    st.sidebar.markdown("Enter hypothetical patient data to predict mortality risk.")
    
    inputs = {}
    if 'Age' in available_features: inputs['Age'] = st.sidebar.slider("Age", 18, 100, 65)
    if 'Gender' in available_features: inputs['Gender'] = 1 if st.sidebar.selectbox("Gender", ['Female', 'Male']) == 'Male' else 0
    if 'The_National_Early_Warning_Score_NEWS' in available_features: inputs['NEWS Score'] = st.sidebar.slider("NEWS Score", 0, 20, 5)
    
    if st.sidebar.button("Predict Risk"):
        if inputs.get('NEWS Score', 0) > 7 or inputs.get('Age', 0) > 75:
            prediction_result = "High Risk of Mortality"
            st.sidebar.error(prediction_result)
        else:
            prediction_result = "Lower Risk of Mortality"
            st.sidebar.success(prediction_result)
        st.sidebar.caption("Note: This is an illustrative prediction.")


# --- Main App Logic ---
sepsis_df = load_data('ICU_Sepsis_Cleaned.csv')

if sepsis_df is not None:
    st.sidebar.title("🩺 Sepsis Analytics")
    app_mode = st.sidebar.selectbox(
        "Choose the Dashboard",
        ["Sepsis EDA Dashboard", "Mortality Predictive Analysis"]
    )
    
    if app_mode == "Sepsis EDA Dashboard":
        display_eda_dashboard(sepsis_df)
    else:
        display_prediction_dashboard(sepsis_df)
