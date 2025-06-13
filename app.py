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
    # --- FIX: Replaced the broken URL with a reliable, public sepsis dataset ---
    url= "https://raw.githubusercontent.com/nadimkawkabani/fuck/main/ICU_Sepsis.csv"
    try:
        df = pd.read_csv(url)
        if df.empty:
            st.error("The loaded CSV file from the URL is empty.")
            return None

        # --- ADAPTATION: Adjusted cleaning for the new dataset ---
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        df.rename(columns={'mortal_in_hospital': 'Mortality', 'age': 'Age', 'gender': 'Gender'}, inplace=True, errors='ignore')

        if 'Mortality' not in df.columns:
            st.error("Error: The required target column 'Mortality' was not found.")
            return None

        # Handle NaNs and data types for key columns
        numeric_cols = ['Age', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'WBC']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic imputation: forward fill within each patient's stay, then drop remaining NaNs
        df[numeric_cols] = df.groupby('icustay_id')[numeric_cols].ffill()
        df.dropna(subset=numeric_cols + ['Mortality', 'Gender'], inplace=True)

        # Ensure target and categorical features are integers
        df['Mortality'] = df['Mortality'].astype(int)
        df['Gender'] = df['Gender'].astype(int)

        if 'Age' in df.columns:
            bins = [0, 18, 40, 50, 60, 70, 80, 120]
            labels = ['<18', '18-39', '40-49', '50-59', '60-69', '70-79', '80+']
            df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

        # Reduce dataset size for faster dashboard performance
        df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df

        return df
    except Exception as e:
        st.error(f"Failed to load or process data. Error: {str(e)}")
        return None

sepsis_df = load_data()

# --- Visualization Functions (No changes needed) ---
def plot_interactive_distribution(df, column, hue=None):
    title = f'Distribution of {column}'
    if hue and hue in df.columns:
        fig = px.histogram(df, x=column, color=hue, marginal='box', nbins=30,
                           barmode='overlay', title=f'{title} by {hue}',
                           opacity=0.7, histnorm='probability density')
        fig.update_yaxes(title_text="Density")
    else:
        fig = px.histogram(df, x=column, marginal='box', nbins=30, title=title)
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

    # --- ADAPTATION: Updated column lists for the new dataset ---
    vital_cols = [col for col in ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp'] if col in df.columns]
    lab_cols = [col for col in ['WBC'] if col in df.columns]

    # --- ADAPTATION: Simplified tabs as new data doesn't have comorbidities ---
    tab1, tab2, tab3 = st.tabs(["📊 Demographics", "🩸 Vitals & Labs", "📈 Correlations"])
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
    with tab2:
        st.header("Vitals & Lab Results Analysis")
        all_vitals_labs = vital_cols + lab_cols
        if all_vitals_labs:
             selected_feature = st.selectbox("Select Vital Sign or Lab Value to Visualize", options=all_vitals_labs)
             if selected_feature in filtered_df.columns and 'Mortality' in filtered_df.columns:
                plot_interactive_distribution(filtered_df, selected_feature, 'Mortality')
        else: st.warning("No vital sign or lab columns found.")
    with tab3:
        st.header("Feature Correlations")
        all_numeric_cols = [col for col in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].nunique() > 1]
        if len(all_numeric_cols) > 1:
            plot_correlation_matrix(filtered_df, all_numeric_cols)
        else: st.warning("Not enough numeric columns with variance for correlation analysis.")


# --- ML Model Functions (No changes needed) ---
@st.cache_resource
def train_model(_X_train, _y_train, model_type='XGBoost', **params):
    if model_type == 'XGBoost':
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42, **params)
    elif model_type == 'Logistic Regression':
        model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000, solver='liblinear', **params))
    else:
        model = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, **params)
    model.fit(_X_train, _y_train)
    return model

# --- ROBUST MODEL EVALUATION FUNCTION (No changes needed) ---
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = np.zeros(len(y_test))
    if hasattr(model, "predict_proba"):
        try:
            proba_results = model.predict_proba(X_test)
            if proba_results.shape[1] == 2: y_proba = proba_results[:, 1]
        except Exception: pass
    try: roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError: roc_auc = 0.5
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted', zero_division=0),
            recall_score(y_test, y_pred, average='weighted', zero_division=0),
            f1_score(y_test, y_pred, average='weighted', zero_division=0),
            roc_auc
        ]
    })
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
        roc_auc_val = auc(fpr, tpr)
    except (ValueError, IndexError):
        fpr, tpr, roc_auc_val = [0, 1], [0, 1], roc_auc
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return metrics_df, (fpr, tpr, roc_auc_val), cm

# --- Prediction Dashboard ---
def display_prediction_dashboard(df):
    st.title("🤖 Enhanced Mortality Prediction & Risk Analysis")
    st.markdown("Use this dashboard to train, evaluate, and use machine learning models for sepsis mortality prediction.")

    # --- ADAPTATION: Updated feature list for the new dataset ---
    features = ['Age', 'Gender', 'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'WBC']
    target = 'Mortality'

    available_features = [f for f in features if f in df.columns]
    if not available_features or target not in df.columns:
        st.error("❌ Essential columns for prediction are missing from the source data.")
        return

    df_model = df[available_features + [target]].dropna()
    X = df_model[available_features]
    y = df_model[target]

    if y.nunique() < 2:
        st.error("❌ **Cannot Build Model:** The target variable only contains one outcome.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Training", "📈 Performance", "🔍 Interpretability", "🧮 Risk Calculator"])

    with tab1:
        st.header("Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox("Select Algorithm", ["XGBoost", "Random Forest", "Logistic Regression"], key="model_type_select")
            params = {}
            if model_type == "XGBoost":
                params = {'n_estimators': st.slider("Number of trees", 50, 500, 100, key="xgb_n"), 'max_depth': st.slider("Max depth", 2, 10, 3, key="xgb_d"), 'learning_rate': st.slider("Learning rate", 0.01, 0.5, 0.1, key="xgb_lr")}
            elif model_type == "Random Forest":
                params = {'n_estimators': st.slider("Number of trees", 50, 500, 100, key="rf_n"), 'max_depth': st.slider("Max depth", 2, 20, 10, key="rf_d")}
            elif model_type == "Logistic Regression":
                 params = {'penalty': st.selectbox("Regularization", ["l2", "l1"], index=0, key="lr_p"), 'C': st.slider("Inverse regularization strength (C)", 0.01, 10.0, 1.0, key="lr_c")}
        with col2:
            st.subheader("Class Distribution in Training Set")
            st.bar_chart(y_train.value_counts(normalize=True) * 100)
            if 'XGBoost' in model_type:
                 st.markdown("Note: XGBoost model will automatically handle class imbalance using `scale_pos_weight`.")

        if st.button("Train Model", key="train_button"):
            with st.spinner(f"Training {model_type} model..."):
                if model_type == 'XGBoost':
                    if y_train.value_counts().get(1, 0) > 0 and y_train.value_counts().get(0, 0) > 0:
                        params['scale_pos_weight'] = y_train.value_counts()[0] / y_train.value_counts()[1]
                model = train_model(X_train, y_train, model_type=model_type, **params)
                st.session_state.model_details = {"model": model, "model_type": model_type, "features": X_train.columns.tolist()}
                st.success(f"✅ {model_type} model trained successfully!")

    # --- State Validation (No changes needed) ---
    if st.session_state.model_details["model"] is not None:
        model_features_list = st.session_state.model_details.get("features")
        current_features_list = X.columns.tolist()
        if set(model_features_list) != set(current_features_list):
            st.warning("⚠️ The feature set has changed. Please retrain the model.")
            st.session_state.model_details = {"model": None, "model_type": None, "features": None}
            st.stop()
    else:
        st.info("👈 Please train a model using the 'Model Training' tab to see performance.", icon="👈")
        st.stop()

    model = st.session_state.model_details["model"]
    model_type = st.session_state.model_details["model_type"]

    with tab2:
        # Code remains the same...
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
        # Code remains the same...
        st.header(f"Model Interpretability: {model_type}")
        st.subheader("Feature Importance")
        importance_df = None
        if hasattr(model, 'named_steps') and 'logisticregression' in model.named_steps:
             importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(model.named_steps['logisticregression'].coef_[0])})
        elif hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        if importance_df is not None:
            fig = px.bar(importance_df.sort_values('Importance', ascending=False).head(15), x='Importance', y='Feature', orientation='h', title='Top 15 Important Features')
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Partial Dependence Plots (PDP)")
        pdp_feature = st.selectbox("Select feature for PDP", options=X.columns, key="pdp_feature")
        if X_train[pdp_feature].nunique() < 2:
            st.warning(f"⚠️ **Could not generate PDP for `{pdp_feature}`.** This feature has only one unique value.")
        else:
            try:
                fig, ax = plt.subplots(figsize=(8, 5))
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
            model_features = st.session_state.model_details["features"]
            # --- ADAPTATION: Simplified form for the new dataset's features ---
            for i, feature in enumerate(model_features):
                with [col1, col2, col3][i % 3]:
                    if feature == 'Age':
                        input_data[feature] = st.number_input("Age (years)", 18, 100, 65)
                    elif feature == 'Gender':
                        selected_gender = st.selectbox("Gender", ['Male', 'Female'])
                        input_data[feature] = 1 if selected_gender == 'Male' else 0
                    else: # For all other numeric features
                        min_val = float(X[feature].min())
                        max_val = float(X[feature].max())
                        mean_val = float(X[feature].mean())
                        input_data[feature] = st.slider(feature.replace('_', ' '), min_val, max_val, mean_val)
            submitted = st.form_submit_button("Calculate Mortality Risk")
            if submitted:
                input_df = pd.DataFrame([input_data])[model_features]
                risk_percent = model.predict_proba(input_df)[0][1] * 100
                st.subheader("Prediction Results")
                # Rest of the calculator code remains the same...
                colA, colB = st.columns([1, 2])
                with colA:
                    st.metric(label="Predicted Mortality Risk", value=f"{risk_percent:.1f}%")
                    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_percent, title={'text': "Risk Level"},
                                                 gauge={'axis': {'range': [None, 100]},
                                                        'steps': [{'range': [0, 20], 'color': "lightgreen"}, {'range': [20, 50], 'color': "orange"}, {'range': [50, 100], 'color': "red"}],
                                                        'bar': {'color': "darkblue"}}))
                    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                with colB:
                    if risk_percent > 50:
                        st.error("🔴 HIGH RISK", icon="🚨")
                        st.markdown("**Recommendations:** Consider immediate ICU admission, aggressive fluid resuscitation, broad-spectrum antibiotics, and frequent monitoring.")
                    elif risk_percent > 20:
                        st.warning("🟠 MODERATE RISK", icon="⚠️")
                        st.markdown("**Recommendations:** Consider hospital admission, initiate sepsis protocol, and perform frequent vital sign checks.")
                    else:
                        st.success("🟢 LOW RISK", icon="✅")
                        st.markdown("**Recommendations:** Continue observation and educate patient on when to seek further care.")

# --- Main App Logic ---
def main():
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
        st.error("🚨 Could not load the dataset. Please check the data source URL and your internet connection.")
        st.image("https://www.sccm.org/SCCM/media/images/sepsis-rebranded-logo.jpg", width=400)

if __name__ == "__main__":
    main()
