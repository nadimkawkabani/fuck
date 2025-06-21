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
import warnings
import sys
import os

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress XGBoost warnings (useful for some environments)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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
    """
    Loads, cleans, and preprocesses the sepsis dataset from a URL.
    - Cleans column names.
    - Handles target variable encoding.
    - Encodes categorical and binary features.
    - Converts columns to appropriate numeric types.
    - Creates age groups for EDA.
    """
    url = "https://raw.githubusercontent.com/nadimkawkabani/fuck/main/ICU_Sepsis.csv"
    try:
        df = pd.read_csv(url)
        if df.empty:
            st.error("The loaded CSV file from the URL is empty.")
            return None
            
        # Clean column names
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
            
        # Clean and convert target variable to binary (0/1)
        if df['Mortality'].nunique() > 2:
            df['Mortality'] = (df['Mortality'] > 0).astype(int)
            
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

# --- Visualization Functions ---
def plot_interactive_distribution(df, column, hue=None):
    """Generates an interactive histogram with an optional hue."""
    title = f'Distribution of {column}'
    if hue:
        fig = px.histogram(df, x=column, color=hue, marginal='box', nbins=30,
                           barmode='overlay', title=f'{title} by {hue}',
                           opacity=0.7, histnorm='probability density')
        fig.update_yaxes(title_text="Density")
    else:
        fig = px.histogram(df, x=column, marginal='box', nbins=30, title=title)
    
    fig.update_layout(legend_title_text=hue if hue else '')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns):
    """Generates an interactive correlation matrix heatmap."""
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# --- EDA Dashboard Function ---
def display_eda_dashboard(df):
    """Renders the entire Exploratory Data Analysis dashboard."""
    st.title("🏥 Unveiling Sepsis Mortality: An Exploratory Analysis of Clinical Risk Factors.")
    st.markdown("A deep dive into the sepsis patient dataset with interactive visualizations.")
    st.sidebar.header("🔍 EDA Filters")
    filtered_df = df.copy()
    
    # --- Sidebar Filters ---
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
        
    # --- Define column groups ---
    vital_cols = [col for col in ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation'] if col in df.columns]
    lab_cols = [col for col in ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 'Creatinine'] if col in df.columns]
    comorbidity_cols = [col for col in ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Others'] if col in df.columns]
    
    # --- EDA Tabs ---
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
        all_vitals_labs = vital_cols + lab_cols
        if all_vitals_labs:
             selected_feature = st.selectbox("Select Vital Sign or Lab Value to Visualize", options=all_vitals_labs)
             if selected_feature in filtered_df.columns and 'Mortality' in filtered_df.columns:
                plot_interactive_distribution(filtered_df, selected_feature, 'Mortality')
        else: st.warning("No vital sign or lab columns found.")
        
    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        if comorbidity_cols and 'Mortality' in filtered_df.columns:
            st.subheader("Comorbidity Rate per 100,000 Patients")
            total_patients = len(filtered_df)
            if total_patients > 0:
                comorbidity_counts = filtered_df[comorbidity_cols].sum()
                comorbidity_rates = (comorbidity_counts / total_patients) * 100000
                comorbidity_rates = comorbidity_rates.sort_values(ascending=False)
                fig = px.bar(
                    comorbidity_rates,
                    orientation='h',
                    title='Comorbidity Rate per 100,000 Patients',
                    labels={'value': 'Rate per 100,000', 'index': 'Comorbidity'},
                    text=comorbidity_rates.round(1)
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No patient data to calculate rates.")
        else:
            st.warning("Comorbidity or Mortality columns not found.")
            
    with tab4:
        st.header("Feature Correlations")
        all_numeric_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].nunique() > 1]
        if len(all_numeric_cols) > 1:
            plot_correlation_matrix(filtered_df, all_numeric_cols)
        else: st.warning("Not enough numeric columns with variance for correlation analysis.")

# --- ML Model Functions ---
@st.cache_resource
def train_model(_X_train, _y_train, model_type='XGBoost', **params):
    """Trains a specified machine learning model."""
    if model_type == 'XGBoost':
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, **params)
    elif model_type == 'Logistic Regression':
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear', **params))
    else: # Default to Random Forest
        model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **params)
    model.fit(_X_train, _y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns performance metrics, ROC data, and confusion matrix.
    This function is robust to different model types and potential data issues.
    """
    y_pred = model.predict(X_test)
    y_proba = np.zeros(len(y_test)) 
    
    # Safely get prediction probabilities
    if hasattr(model, "predict_proba"):
        try:
            proba_results = model.predict_proba(X_test)
            if proba_results.shape[1] == 2: 
                y_proba = proba_results[:, 1]
            elif proba_results.shape[1] == 1:
                y_proba = proba_results[:, 0]
        except Exception as e:
            st.warning(f"Could not get probability estimates: {e}")
    
    # Calculate metrics, handling potential errors
    try: 
        roc_auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5
    except ValueError: 
        roc_auc = 0.5
        
    # Handle metrics for binary or multiclass cases
    pos_label = 1 if len(np.unique(y_test)) <= 2 else None
    average_method = 'binary' if pos_label is not None else 'weighted'
    
    precision = precision_score(y_test, y_pred, pos_label=pos_label, average=average_method, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=pos_label, average=average_method, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=pos_label, average=average_method, zero_division=0)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [accuracy_score(y_test, y_pred), precision, recall, f1, roc_auc]
    })
    
    # Safely generate ROC curve data
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc_val = auc(fpr, tpr)
    except (ValueError, IndexError):
        fpr, tpr, roc_auc_val = [0, 1], [0, 1], 0.5
        
    # Safely generate confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
    except:
        cm = np.zeros((2, 2))
    
    return metrics_df, (fpr, tpr, roc_auc_val), cm

# --- Prediction Dashboard Function ---
def display_prediction_dashboard(df):
    """Renders the entire machine learning and prediction dashboard."""
    st.title("🤖 Enhanced Mortality Prediction & Risk Analysis")
    st.markdown("Use this dashboard to train, evaluate, and use machine learning models for sepsis mortality prediction.")
    
    all_possible_features = ['Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 
                           'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 
                           'COPD_Asthma', 'The_National_Early_Warning_Score_NEWS', 'qSOFA_Score', 
                           'WBC', 'CRP', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure',
                           'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation']
    target = 'Mortality'
    
    # --- Data Preparation for Modeling ---
    available_features = [f for f in all_possible_features if f in df.columns]
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
        st.error("❌ **Cannot Build Model:** The source data only contains one outcome and cannot be used for prediction.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- ML Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "📈 Performance", "🔍 Interpretability", "🧮 Risk Calculator"])

    with tab1:
        st.header("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Select Algorithm", ["XGBoost", "Random Forest", "Logistic Regression"], key="model_type_select")
            params = {}
            if model_type == "XGBoost":
                params = {
                    'n_estimators': st.slider("Number of trees", 50, 500, 100, key="xgb_n"), 
                    'max_depth': st.slider("Max depth", 2, 10, 3, key="xgb_d"), 
                    'learning_rate': st.slider("Learning rate", 0.01, 0.5, 0.1, key="xgb_lr")
                }
                if y_train.value_counts().get(1, 0) > 0 and y_train.value_counts().get(0, 0) > 0:
                    params['scale_pos_weight'] = y_train.value_counts()[0] / y_train.value_counts()[1]
            elif model_type == "Random Forest":
                params = {
                    'n_estimators': st.slider("Number of trees", 50, 500, 100, key="rf_n"), 
                    'max_depth': st.slider("Max depth", 2, 20, 10, key="rf_d")
                }
            elif model_type == "Logistic Regression":
                 params = {
                     'penalty': st.selectbox("Regularization", ["l2", "l1"], index=0, key="lr_p"), 
                     'C': st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0, key="lr_c")
                 }
        with col2:
            st.subheader("Class Distribution in Training Set")
            st.bar_chart(y_train.value_counts(normalize=True) * 100)
            st.markdown("Note: XGBoost model will automatically handle class imbalance using `scale_pos_weight`. Other models use `class_weight='balanced'`.")
            
        if st.button("Train Model", key="train_button"):
            with st.spinner(f"Training {model_type} model..."):
                try:
                    model = train_model(X_train, y_train, model_type=model_type, **params)
                    st.session_state.model_details = {
                        "model": model, 
                        "model_type": model_type, 
                        "features": X_train.columns.tolist()
                    }
                    st.success(f"✅ {model_type} model trained successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to train model: {str(e)}")

    if st.session_state.model_details["model"] is None:
        st.info("Please train a model using the 'Model Training' tab to see performance and make predictions.", icon="👈")
        return

    model = st.session_state.model_details["model"]
    model_type = st.session_state.model_details["model_type"]
    model_features = st.session_state.model_details["features"]

    with tab2:
        st.header(f"Performance Evaluation: {model_type}")
        metrics_df, roc_data, cm = evaluate_model(model, X_test[model_features], y_test)
        fpr, tpr, roc_auc_val = roc_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Metrics")
            st.dataframe(metrics_df.style.format({'Value': '{:.2%}'}), use_container_width=True)
            
            st.subheader("Confusion Matrix")
            fig = px.imshow(
                cm, 
                text_auto=True, 
                aspect="auto", 
                labels=dict(x="Predicted Label", y="True Label", color="Count"),
                color_continuous_scale=px.colors.sequential.Blues
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("ROC Curve")
            fig = px.area(
                x=fpr, 
                y=tpr, 
                title=f'ROC Curve (AUC = {roc_auc_val:.2f})', 
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header(f"Model Interpretability: {model_type}")
        st.subheader("Feature Importance")
        
        try:
            if model_type in ['XGBoost', 'Random Forest']:
                importances = model.feature_importances_
            elif model_type == 'Logistic Regression':
                importances = np.abs(model.named_steps['logisticregression'].coef_[0])
            else:
                st.warning("Feature importance not directly available for this model type.")
                importances = None
            
            if importances is not None:
                importance_df = pd.DataFrame({'Feature': model_features, 'Importance': importances})
                fig = px.bar(
                    importance_df.sort_values('Importance', ascending=True).tail(15), 
                    x='Importance', 
                    y='Feature', 
                    orientation='h', 
                    title='Top 15 Important Features'
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display feature importance: {e}")
        
        st.subheader("Partial Dependence Plots (PDP)")
        pdp_feature = st.selectbox("Select feature for PDP", options=model_features, key="pdp_feature")
        if X_train[pdp_feature].nunique() < 2:
            st.warning(f"⚠️ **Could not generate PDP for `{pdp_feature}`.** This feature has only one unique value.")
        else:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
                # PDP expects target class index for classification
                PartialDependenceDisplay.from_estimator(model, X_train, [pdp_feature], target=1, ax=ax)
                ax.set_title(f"Partial Dependence Plot for {pdp_feature} on Mortality Risk")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"An error occurred while generating the PDP: {e}")

    with tab4:
        st.header("Patient Risk Calculator")
        with st.form("prediction_form"):
            input_data = {}
            col1, col2, col3 = st.columns(3)
            
            feature_stats = X.describe().loc[['min', 'max', 'mean']].to_dict()
            
            for i, feature in enumerate(model_features):
                current_col = [col1, col2, col3][i % 3]
                with current_col:
                    if feature == 'Age':
                        input_data[feature] = st.slider("Age (years)", 0, 120, 50)
                    elif feature == 'Gender':
                        input_data[feature] = 1 if st.selectbox("Gender", ['Male', 'Female']) == 'Male' else 0
                    elif feature in ['Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma']:
                        input_data[feature] = 1 if st.checkbox(f"Has {feature.replace('_', ' ')}", False) else 0
                    else:
                        min_val = float(feature_stats.get(feature, {}).get('min', 0))
                        max_val = float(feature_stats.get(feature, {}).get('max', 100))
                        mean_val = float(feature_stats.get(feature, {}).get('mean', 50))
                        input_data[feature] = st.slider(feature.replace('_', ' '), min_val, max_val, mean_val)
                            
            submitted = st.form_submit_button("Calculate Mortality Risk")
            
            if submitted:
                input_df = pd.DataFrame([input_data])[model_features]
                
                try:
                    risk_percent = model.predict_proba(input_df)[0][1] * 100
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    risk_percent = 0.0
                
                st.subheader("Prediction Results")
                colA, colB = st.columns([1, 2])
                with colA:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", 
                        value=risk_percent, 
                        title={'text': "Mortality Risk"},
                        gauge={
                            'axis': {'range': [None, 100], 'tickprefix': "%"},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"}, 
                                {'range': [20, 50], 'color': "orange"}, 
                                {'range': [50, 100], 'color': "red"}
                            ],
                            'bar': {'color': "darkblue"}
                        }))
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


# --- Main Application Execution Logic ---
def main():
    """
    Main function to run the app. It handles the password check on every run
    and decides whether to show the dashboard or the lock screen.
    """
    CORRECT_PASSWORD = "msba"

    # --- Password Input in Sidebar ---
    password_attempt = st.sidebar.text_input("Enter Password to View Dashboard", type="password")

    # --- Check Password and Display Corresponding View ---
    if password_attempt == CORRECT_PASSWORD:
        # If password is correct, build the main dashboard
        st.sidebar.image("https://www.aub.edu.lb/osb/125/PublishingImages/OSB125.png", use_container_width=True)
        sepsis_df = load_data()
        st.sidebar.title("🩺 Sepsis Analytics Suite")

        if sepsis_df is not None:
            st.sidebar.markdown("---")
            app_mode = st.sidebar.selectbox(
                "Choose Dashboard",
                ["Mortality Predictive Analysis", "Comprehensive EDA"],
                index=0,
                key="app_mode_select"
            )
            st.sidebar.markdown("---")
            if app_mode == "Comprehensive EDA":
                display_eda_dashboard(sepsis_df)
            elif app_mode == "Mortality Predictive Analysis":
                display_prediction_dashboard(sepsis_df)
        else:
            st.title("Welcome to the Sepsis Clinical Analytics Dashboard")
            st.error("🚨 Could not load the dataset. Please check the data source and try again.")

    else:
        # If password is not correct, show a centered logo and title.
        # The password prompt remains in the sidebar.
        _ , mid_col, _ = st.columns([1, 2, 1])
        with mid_col:
            st.image("https://www.aub.edu.lb/osb/125/PublishingImages/OSB125.png")

        st.markdown(
            "<h2 style='text-align: center;'>A Data-Driven Approach to Sepsis Mortality Prediction and Risk Factor Analysis</h2>", 
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
