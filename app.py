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

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("/content/ICU_Sepsis_Cleaned.csv")
    df['suicides_no'] = pd.to_numeric(df['suicides_no'], errors='coerce')
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df['suicide_rate'] = np.where(
        df['population'] > 0,
        (df['suicides_no'] / df['population']) * 100000, 0
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['age'] = df['age'].astype(str)
    return df

df_original = load_data()

            
        # --- Data Cleaning ---
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
        
        binary_mappings = {
            'Systemic_Inflammatory_Response_Syndrome_SIRS_presence': {'Var': 1, 'Yok': 0},
            'Comorbidity': {'Var': 1, 'Yok': 0},
            'Gender': {'Female': 0, 'Male': 1, 'F': 0, 'M': 1, 'Kadın': 0, 'Erkek': 1},
            'Mortality': {'Mortal': 1, 'Mortal Değil': 0, 1: 1, 0: 0}
        }
        
        for col, mapping in binary_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(df[col]) 
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
        
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 10:
                df[col] = df[col].astype('category')
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load or process data from URL. Please check the URL and file format. Error: {str(e)}")
        return None

# Load the data once when the app starts
sepsis_df = load_data()


# --- Visualization Functions (No Changes) ---
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

# --- EDA Dashboard Function (No Changes) ---
def display_eda_dashboard(df):
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset with interactive visualizations.")
    st.sidebar.header("🔍 EDA Filters")
    filtered_df = df.copy()
    
    if 'Age_Group' in df.columns and pd.api.types.is_categorical_dtype(df['Age_Group']):
        age_options = list(df['Age_Group'].cat.categories)
        selected_age = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)
        filtered_df = filtered_df[filtered_df['Age_Group'].isin(selected_age)]

    if 'Gender' in df.columns:
        gender_map = {0: 'Female', 1: 'Male'}
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
            if 'Age' in filtered_df.columns: plot_interactive_distribution(filtered_df, 'Age', 'Mortality')
        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns and 'Mortality' in filtered_df.columns:
                gender_map = {0: 'Female', 1: 'Male'}
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
            if lab_cols:
                selected_lab = st.selectbox("Select Lab Value", options=lab_cols, key="lab_select")
                plot_interactive_distribution(filtered_df, selected_lab, 'Mortality')
            else: st.warning("No lab columns found.")
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

# --- ML Model Functions (No Changes) ---
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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics_df = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'], 'Value': [accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc_score(y_test, y_proba)]})
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    return metrics_df, (fpr, tpr, roc_auc_val), confusion_matrix(y_test, y_pred)

def display_prediction_dashboard(df):
    st.title("🤖 Enhanced Mortality Prediction & Risk Analysis")
    st.markdown("Advanced machine learning for sepsis mortality prediction.")
    
    features = ['Age', 'Gender', 'Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma', 'Pulse_rate', 'Respiratory_Rate', 'Systolic_blood_pressure', 'Fever', 'Oxygen_saturation', 'WBC', 'CRP', 'The_National_Early_Warning_Score_NEWS', 'qSOFA_Score']
    target = 'Mortality'
    
    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns:
        st.error("❌ Essential columns for prediction are missing from the data.")
        return
    
    df_model = df[available_features + [target]].dropna()
    if df_model.empty:
        st.error("❌ No data available for model training after handling missing values.")
        return
    
    X = df_model[available_features]
    y = df_model[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "📈 Performance", "🔍 Interpretability", "🧮 Risk Calculator"])
    
    with tab1:
        st.header("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Select Algorithm", ["Random Forest", "XGBoost", "Logistic Regression"], key="model_type_select")
            params = {}
            if model_type == "Random Forest": params = {'n_estimators': st.slider("Number of trees", 50, 500, 100, key="rf_n"), 'max_depth': st.slider("Max depth", 2, 20, 10, key="rf_d")}
            elif model_type == "Logistic Regression": params = {'penalty': st.selectbox("Regularization", ["l2", "l1"], index=0, key="lr_p"), 'C': st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0, key="lr_c")}
            elif model_type == "XGBoost": params = {'n_estimators': st.slider("Number of trees", 50, 500, 100, key="xgb_n"), 'max_depth': st.slider("Max depth", 2, 10, 3, key="xgb_d"), 'learning_rate': st.slider("Learning rate", 0.01, 0.5, 0.1, key="xgb_lr")}
        with col2:
            st.subheader("Class Distribution in Training Set")
            st.bar_chart(y_train.value_counts(normalize=True) * 100)
        if st.button("Train Model", key="train_button"):
            with st.spinner(f"Training {model_type} model..."):
                model = train_model(X_train, y_train, model_type, **params)
                st.session_state.model_details = {"model": model, "model_type": model_type, "features": X_train.columns.tolist()}
                st.success(f"✅ {model_type} model trained successfully!")
    
    if st.session_state.model_details["model"] is None:
        st.info("Please train a model using the 'Model Training' tab to see performance and make predictions.", icon="👈")
        return

    model = st.session_state.model_details["model"]
    model_type = st.session_state.model_details["model_type"]

    with tab2:
        st.header(f"Performance Evaluation: {model_type}")
        metrics_df, roc_data, cm = evaluate_model(model, X_test, y_test)
        fpr, tpr, roc_auc_val = roc_data
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Key Metrics")
            st.dataframe(metrics_df.style.format({'Value': '{:.2%}'}), use_container_width=True)
            st.subheader("Confusion Matrix")
            fig = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted Label", y="True Label", color="Count"), x=['Survived', 'Died'], y=['Survived', 'Died'], color_continuous_scale=px.colors.sequential.Blues)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("ROC Curve")
            fig = px.area(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc_val:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            st.plotly_chart(fig, use_container_width=True)
    with tab3:
        st.header(f"Model Interpretability: {model_type}")
        st.subheader("Feature Importance")
        importance_df = None
        if model_type in ["Random Forest", "XGBoost"]: importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        elif model_type == "Logistic Regression": importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(model.named_steps['logisticregression'].coef_[0])})
        if importance_df is not None:
            fig = px.bar(importance_df.sort_values('Importance', ascending=False).head(15), x='Importance', y='Feature', orientation='h', title='Top 15 Important Features')
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Partial Dependence Plots (PDP)")
        pdp_feature = st.selectbox("Select feature for PDP", options=X.columns, key="pdp_feature")
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            PartialDependenceDisplay.from_estimator(model, X_train, [pdp_feature], ax=ax)
            ax.set_title(f"Partial Dependence Plot for {pdp_feature}")
            st.pyplot(fig)
        except Exception as e: st.warning(f"Could not generate PDP: {e}")
    with tab4:
        st.header("Patient Risk Calculator")
        with st.form("prediction_form"):
            input_data, col1, col2, col3 = {}, *st.columns(3)
            model_features = st.session_state.model_details["features"]
            for i, feature in enumerate(model_features):
                with [col1, col2, col3][i % 3]:
                    if feature == 'Age': input_data[feature] = st.slider("Age (years)", 18, 100, 65)
                    elif feature == 'Gender': input_data[feature] = 1 if st.selectbox("Gender", ['Female', 'Male']) == 'Male' else 0
                    elif feature in ['Comorbidity', 'Hypertension', 'Heart_Diseases', 'Diabetes_mellitus', 'Chronic_Renal_Failure', 'Neurological_Diseases', 'COPD_Asthma']: input_data[feature] = 1 if st.checkbox(f"Has {feature.replace('_', ' ')}", False) else 0
                    else: min_val, max_val, mean_val = float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()); input_data[feature] = st.slider(feature.replace('_', ' '), min_val, max_val, mean_val)
            if st.form_submit_button("Calculate Mortality Risk"):
                risk_percent = model.predict_proba(pd.DataFrame([input_data])[model_features])[0][1] * 100
                st.subheader("Prediction Results")
                colA, colB = st.columns(2)
                with colA:
                    st.metric(label="Predicted Mortality Risk", value=f"{risk_percent:.1f}%")
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_percent, title={'text': "Risk Level"}, gauge={'axis': {'range': [None, 100]}, 'steps': [{'range': [0, 20], 'color': "lightgreen"}, {'range': [20, 50], 'color': "orange"}, {'range': [50, 100], 'color': "red"}], 'bar': {'color': "darkblue"}}))
                    st.plotly_chart(fig, use_container_width=True)
                with colB:
                    if risk_percent > 50: st.error("🔴 HIGH RISK", icon="🚨"); st.markdown("**Recommendations:** Consider immediate ICU admission, aggressive fluid resuscitation, broad-spectrum antibiotics, and frequent monitoring.")
                    elif risk_percent > 20: st.warning("🟠 MODERATE RISK", icon="⚠️"); st.markdown("**Recommendations:** Consider hospital admission, initiate sepsis protocol, perform frequent vital sign checks, and evaluate need for antibiotics.")
                    else: st.success("🟢 LOW RISK", icon="✅"); st.markdown("**Recommendations:** Continue observation, consider outpatient follow-up, and educate patient on when to seek further care.")

# --- Main App Logic ---
def main():
    st.sidebar.title("🩺 Sepsis Analytics Suite")
    
    # Check if data was loaded successfully at startup
    if sepsis_df is not None:
        st.sidebar.markdown("---")
        app_mode = st.sidebar.selectbox(
            "Choose Dashboard",
            ["Comprehensive EDA", "Mortality Predictive Analysis"],
            index=0,
            key="app_mode_select"
        )
        st.sidebar.markdown("---")
        
        if app_mode == "Comprehensive EDA":
            display_eda_dashboard(sepsis_df)
        elif app_mode == "Mortality Predictive Analysis":
            display_prediction_dashboard(sepsis_df)
    else:
        # This message will show if the data loading failed
        st.title("Welcome to the Sepsis Clinical Analytics Dashboard")
        st.error("🚨 Could not load the dataset. Please ensure the URL in the script is correct and the file is publicly accessible on GitHub.")
        st.image("https://www.sccm.org/SCCM/media/images/sepsis-rebranded-logo.jpg", width=400)

if __name__ == "__main__":
    main()
