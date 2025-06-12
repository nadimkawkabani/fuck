import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Sepsis Clinical Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}
if 'model' not in st.session_state:
    st.session_state.model = None

# --- Data Loading and Caching with Enhanced Error Handling ---
@st.cache_data
def load_data(file_path):
    """
    Loads and preprocesses the sepsis data from a CSV file with robust error handling.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            st.error("Error: The file is empty.")
            return None
            
        # Data validation
        required_columns = ['Age', 'Gender', 'Mortality']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"Error: Missing required columns - {', '.join(missing_cols)}")
            return None
            
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None

    # --- Data Cleaning ---
    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('’', '')
    
    rename_map = {
        'Ek_Hastalık_isimlerş': 'Comorbidity_Names',
        'Direnç_Durumu': 'Resistance_Status',
        'Mortalite': 'Mortality',
        'KOAH_Asthım': 'COPD_Asthma'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    
    # Convert binary/categorical columns with enhanced handling
    binary_mappings = {
        'Systemic_Inflammatory_Response_Syndrome_SIRS_presence': {'Var': 1, 'Yok': 0},
        'Comorbidity': {'Var': 1, 'Yok': 0},
        'Gender': {'Female': 0, 'Male': 1, 'F': 0, 'M': 1}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping).fillna(0).astype(int)
    
    # Numeric conversion with better handling
    numeric_cols = ['Age', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 
                   'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Age grouping with better binning
    if 'Age' in df.columns:
        bins = [0, 18, 40, 50, 60, 70, 80, 120]
        labels = ['<18', '18-39', '40-49', '50-59', '60-69', '70-79', '80+']
        df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    
    # Optimize data types
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].nunique() < 10:
            df[col] = df[col].astype('category')
    
    return df

# --- Enhanced Visualization Functions ---
def plot_interactive_distribution(df, column, hue=None):
    """Create interactive distribution plot with Plotly."""
    if hue:
        fig = px.histogram(df, x=column, color=hue, marginal='box',
                          nbins=30, barmode='overlay',
                          title=f'Distribution of {column} by {hue}',
                          opacity=0.7)
    else:
        fig = px.histogram(df, x=column, marginal='box',
                          nbins=30, title=f'Distribution of {column}')
    
    fig.update_layout(legend_title_text=hue if hue else '')
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df, columns):
    """Plot interactive correlation matrix."""
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect='auto',
                   color_continuous_scale='RdBu_r',
                   title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# --- COMPREHENSIVE EDA DASHBOARD (ENHANCED) ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset with interactive visualizations.")

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 EDA Filters")
    filtered_df = df.copy()
    
    # Age filter
    if 'Age_Group' in df.columns:
        age_options = sorted(list(df['Age_Group'].cat.categories))
        selected_age = st.sidebar.multiselect(
            "Filter by Age Group", 
            options=age_options, 
            default=age_options
        )
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age)]

    # Gender filter
    if 'Gender' in df.columns:
        gender_map = {0: 'Female', 1: 'Male'}
        selected_gender_str = st.sidebar.selectbox(
            "Filter by Gender", 
            options=['All', 'Male', 'Female'], 
            index=0
        )
        if selected_gender_str != 'All':
            gender_code = 1 if selected_gender_str == 'Male' else 0
            filtered_df = filtered_df[filtered_df['Gender'] == gender_code]
    
    # Mortality filter
    if 'Mortality' in df.columns:
        mortality_filter = st.sidebar.selectbox(
            "Filter by Mortality Status",
            options=['All', 'Survived', 'Died'],
            index=0
        )
        if mortality_filter != 'All':
            mortality_code = 1 if mortality_filter == 'Died' else 0
            filtered_df = filtered_df[filtered_df['Mortality'] == mortality_code]
    
    if filtered_df.empty:
        st.warning("⚠️ No data available for the selected filters.")
        return

    # --- Define column groups ---
    vital_cols = [col for col in ['Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 
                                'Diastolic_blood_pressure', 'Fever', 'Oxygen_saturation'] 
                 if col in df.columns]
    lab_cols = [col for col in ['Albumin', 'CRP', 'Glukoz', 'Eosinophil_count', 'HCT', 'Hemoglobin', 
                              'Lymphocyte_count', 'Monocyte_count', 'Neutrophil_count', 'PLT', 'RBC', 'WBC', 
                              'Creatinine'] if col in df.columns]
    risk_score_cols = [col for col in ['The_National_Early_Warning_Score_NEWS', 'qSOFA_Score', 
                                     'Systemic_Inflammatory_Response_Syndrome_SIRS_presence'] 
                      if col in df.columns]
    comorbidity_cols = [col for col in ['Comorbidity', 'Solid_organ_cancer', 'Hematological_Diseases', 
                                      'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 
                                      'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 
                                      'Others'] if col in df.columns]

    # --- Tabbed Layout ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Demographics", "🩸 Vitals & Labs", "⚠️ Risk Factors", "📈 Correlations"])

    with tab1:
        st.header("Demographic Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Patient Age Distribution")
            if 'Age' in filtered_df.columns:
                plot_interactive_distribution(filtered_df, 'Age', 'Mortality')
        
        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                gender_counts = filtered_df['Gender'].map(gender_map).value_counts()
                fig = px.pie(gender_counts, values=gender_counts.values, 
                            names=gender_counts.index, 
                            title='Gender Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Age vs. Mortality")
        if 'Age' in filtered_df.columns and 'Mortality' in filtered_df.columns:
            fig = px.box(filtered_df, x='Mortality', y='Age', color='Mortality',
                        points='all', title='Age Distribution by Mortality Status')
            fig.update_xaxes(title_text='Mortality Status', tickvals=[0, 1], ticktext=['Survived', 'Died'])
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Vitals & Lab Results Analysis")
        
        with st.expander("📈 Vital Signs Analysis", expanded=True):
            if not vital_cols or 'Mortality' not in filtered_df.columns: 
                st.warning("Vital sign or Mortality columns not found.")
            else:
                selected_vital = st.selectbox("Select Vital Sign to Visualize", options=vital_cols)
                plot_interactive_distribution(filtered_df, selected_vital, 'Mortality')
                
                # Trend analysis by age group
                if 'Age_Group' in filtered_df.columns:
                    st.subheader(f"{selected_vital} Trend by Age Group")
                    fig = px.box(filtered_df, x='Age_Group', y=selected_vital, color='Mortality',
                                title=f'{selected_vital} by Age Group and Mortality')
                    st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("🧪 Lab Results Analysis"):
            if not lab_cols: 
                st.warning("No lab columns found.")
            else:
                selected_lab = st.selectbox("Select Lab Value", options=lab_cols)
                plot_interactive_distribution(filtered_df, selected_lab, 'Mortality')
                
                # Correlation with vitals
                if vital_cols:
                    st.subheader(f"{selected_lab} vs. Vital Signs")
                    x_axis = st.selectbox("Select X-axis variable", options=vital_cols)
                    fig = px.scatter(filtered_df, x=x_axis, y=selected_lab, color='Mortality',
                                   trendline='lowess', title=f'{selected_lab} vs. {x_axis}')
                    st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        
        if not comorbidity_cols or 'Mortality' not in filtered_df.columns: 
            st.warning("Comorbidity or Mortality columns not found.")
        else:
            # Comorbidity prevalence
            st.subheader("Comorbidity Prevalence")
            comorbidity_counts = filtered_df[comorbidity_cols].sum().sort_values(ascending=False)
            fig = px.bar(comorbidity_counts, orientation='h', 
                        title='Prevalence of Comorbidities',
                        labels={'value': 'Count', 'index': 'Comorbidity'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Mortality rates by comorbidity
            st.subheader("Mortality Rates by Comorbidity")
            mortality_rates = {}
            for col in comorbidity_cols:
                if filtered_df[col].sum() > 0:
                    rate = filtered_df[filtered_df[col] == 1]['Mortality'].mean()
                    if pd.notna(rate): 
                        mortality_rates[col] = rate * 100
            
            if mortality_rates:
                mortality_df = pd.DataFrame(list(mortality_rates.items()), 
                                          columns=['Comorbidity', 'Mortality Rate (%)']
                                          ).sort_values('Mortality Rate (%)', ascending=False)
                
                fig = px.bar(mortality_df, x='Mortality Rate (%)', y='Comorbidity',
                            title="Mortality Rate by Comorbidity",
                            color='Mortality Rate (%)',
                            color_continuous_scale='OrRd')
                st.plotly_chart(fig, use_container_width=True)
                
                # Comorbidity combinations
                st.subheader("Comorbidity Combinations Analysis")
                top_comorbidities = mortality_df.head(5)['Comorbidity'].tolist()
                selected_comorbidities = st.multiselect(
                    "Select comorbidities to analyze combinations",
                    options=top_comorbidities,
                    default=top_comorbidities[:2]
                )
                
                if len(selected_comorbidities) >= 2:
                    combo_filter = filtered_df[selected_comorbidities].sum(axis=1)
                    combo_mortality = filtered_df.loc[combo_filter >= 1, 'Mortality'].mean() * 100
                    st.metric(f"Mortality Rate for Patients with {' + '.join(selected_comorbidities)}", 
                             f"{combo_mortality:.1f}%")

    with tab4:
        st.header("Feature Correlations")
        
        all_numeric_cols = [col for col in filtered_df.columns 
                           if pd.api.types.is_numeric_dtype(filtered_df[col])]
        
        if len(all_numeric_cols) > 1:
            plot_correlation_matrix(filtered_df, all_numeric_cols)
        else:
            st.warning("Not enough numeric columns for correlation analysis.")

# --- Enhanced Model Training with Multiple Algorithms ---
@st.cache_resource
def train_model(X_train, y_train, model_type='Random Forest', **params):
    """Train a model with the specified algorithm and parameters."""
    if model_type == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'Logistic Regression':
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty=params.get('penalty', 'l2'),
                C=params.get('C', 1.0),
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        )
    elif model_type == 'XGBoost':
        model = XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            objective='binary:logistic',
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model

# --- Enhanced Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """Generate comprehensive model evaluation metrics and plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [accuracy, precision, recall, f1, roc_auc]
    })
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return metrics_df, (fpr, tpr, roc_auc), confusion_matrix(y_test, y_pred)

# --- PREDICTIVE ANALYSIS DASHBOARD (ENHANCED) ---
def display_prediction_dashboard(df):
    st.title("🤖 Enhanced Mortality Prediction & Risk Analysis")
    st.markdown("Advanced machine learning for sepsis mortality prediction with multiple algorithms.")
    
    # Define features and target
    features = [
        'Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 
        'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 
        'COPD_Asthma', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 
        'Fever', 'Oxygen_saturation', 'WBC', 'CRP', 'The_National_Early_Warning_Score_NEWS', 
        'qSOFA_Score'
    ]
    target = 'Mortality'
    
    # Check for required columns
    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns: 
        st.error("❌ Essential columns for prediction are missing.")
        return
    
    # Prepare data
    df_model = df[available_features + [target]].dropna()
    if df_model.empty: 
        st.error("❌ No data for model training after handling missing values.")
        return
    
    X = df_model[available_features]
    y = df_model[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model selection and training
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Training", 
        "📈 Performance", 
        "🔍 Interpretability", 
        "🧮 Risk Calculator"
    ])
    
    with tab1:
        st.header("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Algorithm",
                ["Random Forest", "Logistic Regression", "XGBoost"],
                index=0
            )
            
            # Algorithm-specific parameters
            if model_type == "Random Forest":
                n_estimators = st.slider("Number of trees", 50, 500, 100)
                max_depth = st.slider("Max depth", 2, 20, 10)
                params = {'n_estimators': n_estimators, 'max_depth': max_depth}
                
            elif model_type == "Logistic Regression":
                penalty = st.selectbox("Regularization", ["l2", "l1"], index=0)
                C = st.slider("Inverse regularization strength", 0.01, 10.0, 1.0)
                params = {'penalty': penalty, 'C': C}
                
            elif model_type == "XGBoost":
                n_estimators = st.slider("Number of trees", 50, 500, 100)
                max_depth = st.slider("Max depth", 2, 10, 3)
                learning_rate = st.slider("Learning rate", 0.01, 0.5, 0.1)
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate
                }
        
        with col2:
            st.subheader("Class Distribution")
            st.write(pd.DataFrame({
                'Count': y.value_counts(),
                'Percentage': y.value_counts(normalize=True) * 100
            }))
            
            if st.button("Train Model"):
                with st.spinner(f"Training {model_type} model..."):
                    model = train_model(X_train, y_train, model_type, **params)
                    st.session_state.model = model
                    st.success(f"{model_type} model trained successfully!")
    
    if st.session_state.model is None:
        st.warning("Please train a model first using the 'Model Training' tab.")
        return
    
    model = st.session_state.model
    
    with tab2:
        st.header("Model Performance Evaluation")
        
        metrics_df, roc_data, cm = evaluate_model(model, X_test, y_test)
        fpr, tpr, roc_auc = roc_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics")
            st.dataframe(metrics_df.style.format({'Value': '{:.2%}'}))
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Predicted Survived', 'Predicted Died'],
                       yticklabels=['Actual Survived', 'Actual Died'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
        
        with col2:
            st.subheader("ROC Curve")
            fig = px.area(x=fpr, y=tpr, 
                          title=f'ROC Curve (AUC = {roc_auc:.2f})',
                          labels={'x': 'False Positive Rate', 
                                 'y': 'True Positive Rate'})
            fig.add_shape(type='line', line=dict(dash='dash'),
                         x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("Feature Importance")
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importances.head(10), x='Importance', y='Feature',
                            orientation='h', title='Top 10 Important Features')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")
    
    with tab3:
        st.header("Model Interpretability")
        
        if hasattr(model, 'feature_importances_'):
            # SHAP values explanation
            st.subheader("SHAP Value Analysis")
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
                st.pyplot(fig)
                
                st.subheader("Individual SHAP Explanations")
                sample_idx = st.slider("Select sample to explain", 0, len(X_test)-1, 0)
                fig, ax = plt.subplots()
                shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], 
                               X_test.iloc[sample_idx,:], matplotlib=True, show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.warning(f"SHAP explanation not available: {str(e)}")
        
        # Partial dependence plots
        st.subheader("Partial Dependence Plots")
        selected_feature = st.selectbox("Select feature for PDP", options=X.columns)
        
        try:
            from sklearn.inspection import PartialDependenceDisplay
            fig, ax = plt.subplots()
            PartialDependenceDisplay.from_estimator(model, X_test, [selected_feature], ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"PDP not available: {str(e)}")
    
    with tab4:
        st.header("Patient Risk Calculator")
        
        with st.form("prediction_form"):
            st.subheader("Patient Characteristics")
            input_data = {}
            
            # Organize inputs into columns
            col1, col2, col3 = st.columns(3)
            
            for i, feature in enumerate(available_features):
                current_col = [col1, col2, col3][i % 3]
                
                with current_col:
                    if feature == 'Age':
                        input_data[feature] = st.slider(
                            "Age (years)", 
                            min_value=18, 
                            max_value=100, 
                            value=50
                        )
                    elif feature == 'Gender':
                        input_data[feature] = st.selectbox(
                            "Gender", 
                            options=['Female', 'Male']
                        )
                        input_data[feature] = 1 if input_data[feature] == 'Male' else 0
                    elif feature in ['Comorbidity', 'Hypertension', 'Heart_Diseases',
                                   'Diabetes_mellitus', 'Chronic_Renal_Failure',
                                   'Neurological_Diseases', 'COPD_Asthma']:
                        input_data[feature] = st.checkbox(
                            f"Has {feature.replace('_', ' ')}", 
                            value=False
                        )
                    else:
                        # For numeric features
                        min_val = float(X[feature].min())
                        max_val = float(X[feature].max())
                        mean_val = float(X[feature].mean())
                        
                        input_data[feature] = st.slider(
                            feature.replace('_', ' '),
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=(max_val - min_val)/100
                        )
            
            submitted = st.form_submit_button("Calculate Mortality Risk")
        
        if submitted:
            input_df = pd.DataFrame([input_data])[X_train.columns]
            
            # Get prediction
            try:
                prediction_proba = model.predict_proba(input_df)[0][1]
                risk_percent = prediction_proba * 100
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Predicted Mortality Risk", 
                        value=f"{risk_percent:.1f}%"
                    )
                    
                    # Risk gauge
                    fig = px.indicator(
                        mode="gauge+number",
                        value=risk_percent,
                        title="Risk Level",
                        gauge={
                            'axis': {'range': [0, 100]},
                            'steps': [
                                {'range': [0, 20], 'color': "green"},
                                {'range': [20, 50], 'color': "orange"},
                                {'range': [50, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_percent
                            }
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk interpretation
                    if risk_percent > 50:
                        st.error("HIGH RISK - Immediate intervention recommended")
                        st.markdown("""
                        **Clinical Actions:**
                        - ICU admission consideration
                        - Aggressive fluid resuscitation
                        - Broad-spectrum antibiotics
                        - Frequent monitoring
                        """)
                    elif risk_percent > 20:
                        st.warning("MODERATE RISK - Close monitoring needed")
                        st.markdown("""
                        **Clinical Actions:**
                        - Consider hospital admission
                        - Initiate sepsis protocol
                        - Frequent vital sign checks
                        - Consider antibiotics
                        """)
                    else:
                        st.success("LOW RISK - Continue observation")
                        st.markdown("""
                        **Clinical Actions:**
                        - Outpatient follow-up
                        - Reassess if condition changes
                        - Patient education
                        """)
                
                # Store prediction in session state
                st.session_state.input_data = input_data
                st.session_state.last_prediction = {
                    'risk': risk_percent,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

# --- Main App Logic ---
def main():
    st.sidebar.title("🩺 Sepsis Analytics Suite")
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sepsis data (CSV)", 
        type=['csv'],
        key='data_uploader'
    )
    
    if uploaded_file is not None:
        sepsis_df = load_data(uploaded_file)
    else:
        # Try to load default file
        default_file = 'ICU_Sepsis_Cleaned.csv'
        if os.path.exists(default_file):
            sepsis_df = load_data(default_file)
        else:
            st.error("Please upload a data file or ensure 'ICU_Sepsis_Cleaned.csv' exists.")
            return
    
    if sepsis_df is not None:
        app_mode = st.sidebar.selectbox(
            "Choose Dashboard",
            ["Comprehensive EDA", "Mortality Predictive Analysis"],
            index=0
        )
        
        if app_mode == "Comprehensive EDA":
            display_eda_dashboard(sepsis_df)
        else:
            display_prediction_dashboard(sepsis_df)

if __name__ == "__main__":
    main()
