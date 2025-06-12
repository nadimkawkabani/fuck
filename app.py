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

# --- EDA Dashboard (remains the same) ---
def display_eda_dashboard(df):
    # This function is unchanged from the previous version.
    # It contains the comprehensive EDA with tabs for Demographics, Vitals, etc.
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset, exploring all key factors.")

    # ... (code for the comprehensive EDA dashboard is here) ...
    # To keep this response clean, I'm omitting the full code for this function as it's not the part being changed.
    # Just imagine the previous 'comprehensive EDA' function is here. 
    # If you need it again, please let me know.
    st.info("The full EDA dashboard would be displayed here.")


# --- NEW: Predictive Analysis Dashboard with Tabs ---
def display_prediction_dashboard(df):
    st.title("🤖 Mortality Prediction & Risk Analysis")
    st.markdown("Using machine learning to predict patient mortality and identify key risk factors.")

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
    # We train the model once and reuse it across tabs.
    @st.cache_resource
    def train_model(X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        return model

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train)

    # --- Predictive Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Model Performance", 
        "🔑 Key Risk Factors", 
        "🎯 Patient Risk Stratification", 
        "🧮 Live Prediction Calculator"
    ])

    with tab1:
        st.header("Model Performance Evaluation")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy on Test Data", f"{accuracy:.2%}")
            st.info("Accuracy measures the overall correctness of the model's predictions.")
        
        with col2:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                        xticklabels=['Predicted Survived', 'Predicted Died'], 
                        yticklabels=['Actual Survived', 'Actual Died'])
            plt.ylabel('Actual Outcome')
            plt.xlabel('Predicted Outcome')
            st.pyplot(fig)

    with tab2:
        st.header("Top Clinical Predictors of Mortality")
        st.markdown("These are the most important factors the model used to distinguish between high-risk and low-risk patients.")
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=importances.head(15), palette='mako', ax=ax)
        ax.set_title("Top 15 Most Important Features in Predicting Mortality")
        st.pyplot(fig)
        st.info("A higher importance score means the model relies on this feature more heavily when making a prediction.")

    with tab3:
        st.header("Patient Risk Stratification")
        st.markdown("This plot visualizes the model's predicted risk for every patient in the dataset, segmented by two key drivers.")

        # Get prediction probabilities for the entire dataset
        df_model['risk_score'] = model.predict_proba(X)[:, 1] # Probability of mortality

        # Select two of the most important features to plot against
        top_features = importances['feature'].tolist()
        x_axis_feat = top_features[0]
        y_axis_feat = top_features[1]
        
        fig = px.scatter(
            df_model,
            x=x_axis_feat,
            y=y_axis_feat,
            color='risk_score',
            color_continuous_scale=px.colors.sequential.OrRd,
            hover_name=df_model.index,
            hover_data={'risk_score': ':.2f}', 'Mortality': True},
            title=f"Patient Risk Map: {x_axis_feat} vs. {y_axis_feat}"
        )
        fig.update_layout(
            xaxis_title=x_axis_feat.replace('_', ' ').title(),
            yaxis_title=y_axis_feat.replace('_', ' ').title(),
            coloraxis_colorbar=dict(title="Mortality Risk")
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Each dot is a patient. Yellow dots indicate a high predicted risk of mortality. Hover over a dot to see details.")

    with tab4:
        st.header("Live Patient Risk Calculator")
        st.markdown("Enter a new patient's data to get an instant risk prediction from the trained model.")

        # Create a form for user inputs
        with st.form("prediction_form"):
            input_data = {}
            # Create columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            
            # Use a loop to dynamically create sliders/inputs for all features
            for i, feature in enumerate(available_features):
                current_col = [col1, col2, col3][i % 3]
                with current_col:
                    # Use reasonable defaults and ranges
                    if 'Age' in feature:
                        input_data[feature] = st.slider(feature, 18, 100, 65)
                    elif 'Gender' in feature:
                        input_data[feature] = 1 if st.selectbox(feature, ['Female', 'Male']) == 'Male' else 0
                    elif df_model[feature].max() <= 1: # Binary features
                        input_data[feature] = st.checkbox(feature, value=False)
                    else: # Other numerical features
                        min_val = int(df_model[feature].min())
                        max_val = int(df_model[feature].max())
                        mean_val = int(df_model[feature].mean())
                        input_data[feature] = st.slider(feature, min_val, max_val, mean_val)
            
            submitted = st.form_submit_button("Calculate Mortality Risk")

        if submitted:
            # Prepare the input for the model
            input_df = pd.DataFrame([input_data])
            
            # Ensure the order of columns matches the training data
            input_df = input_df[X_train.columns]
            
            # Get prediction probability
            prediction_proba = model.predict_proba(input_df)[0][1] # Probability of class 1 (Died)
            
            st.subheader("Prediction Result")
            risk_percent = prediction_proba * 100
            
            # Display the result with a progress bar and text
            st.progress(prediction_proba)
            st.metric(label="Predicted Risk of Mortality", value=f"{risk_percent:.1f}%")

            if risk_percent > 50:
                st.error("This patient is classified as HIGH RISK. Immediate senior review is recommended.")
            elif risk_percent > 20:
                st.warning("This patient is classified as MODERATE RISK. Close monitoring is advised.")
            else:
                st.success("This patient is classified as LOW RISK.")

# --- Main App Logic ---
sepsis_df = load_data('ICU_Sepsis_Cleaned.csv')

if sepsis_df is not None:
    # Use the comprehensive EDA function if you have it, otherwise the original one.
    # For this example, let's assume the comprehensive one exists.
    # Replace display_eda_dashboard with your full EDA function.
    
    st.sidebar.title("🩺 Sepsis Analytics")
    app_mode = st.sidebar.selectbox(
        "Choose the Dashboard",
        ["Comprehensive EDA", "Mortality Predictive Analysis"]
    )
    
    if app_mode == "Comprehensive EDA":
        # Call your full EDA dashboard function here
        st.header("Comprehensive EDA Dashboard")
        st.info("This is where the detailed EDA visualizations would be shown.")
        # display_eda_dashboard(sepsis_df) # Uncomment this line
    else:
        display_prediction_dashboard(sepsis_df)
