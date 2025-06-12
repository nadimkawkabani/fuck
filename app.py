import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display_eda_dashboard(df):
    """
    Renders a comprehensive multi-tab dashboard for Exploratory Data Analysis.
    """
    st.title("🏥 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("A deep dive into the sepsis patient dataset, exploring all key factors.")

    # --- Sidebar Filters ---
    st.sidebar.header("EDA Filters")
    filtered_df = df.copy() # Start with the full dataframe

    if 'Age_Group' in df.columns:
        # Create a clean list of options by dropping NaNs before sorting
        age_options = sorted(list(df['Age_Group'].dropna().unique()))
        selected_age = st.sidebar.multiselect("Filter by Age Group", options=age_options, default=age_options)
        if selected_age:
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

    # --- Define column groups based on what's available in the dataframe ---
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
                fig, ax = plt.subplots()
                sns.histplot(filtered_df['Age'].dropna(), kde=True, ax=ax, bins=20)
                ax.set_title("Distribution of Patient Ages")
                st.pyplot(fig)
        
        with col2:
            st.subheader("Gender Distribution")
            if 'Gender' in filtered_df.columns:
                gender_counts = filtered_df.dropna(subset=['Gender'])['Gender'].map(gender_map).value_counts()
                fig, ax = plt.subplots()
                gender_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['skyblue', 'lightcoral'])
                ax.set_ylabel('')
                ax.set_title("Patient Gender Breakdown")
                st.pyplot(fig)

        st.subheader("Mortality Analysis by Demographics")
        if 'Mortality' in filtered_df.columns:
            col1, col2 = st.columns(2)
            with col1:
                if 'Age_Group' in filtered_df.columns:
                    age_mortality = filtered_df.groupby('Age_Group', observed=True)['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
                    if 1 in age_mortality.columns:
                        fig, ax = plt.subplots()
                        age_mortality[1].plot(kind='barh', ax=ax, color='salmon')
                        ax.set_xlabel("Mortality Rate (%)")
                        ax.set_title("Mortality Rate by Age Group")
                        st.pyplot(fig)

            with col2:
                if 'Gender' in filtered_df.columns:
                    gender_mortality = filtered_df.groupby('Gender')['Mortality'].value_counts(normalize=True).unstack().fillna(0) * 100
                    gender_mortality.index = gender_mortality.index.map(gender_map)
                    if 1 in gender_mortality.columns:
                        fig, ax = plt.subplots()
                        gender_mortality[1].plot(kind='bar', ax=ax, color='lightcoral', width=0.4)
                        ax.set_ylabel("Mortality Rate (%)")
                        ax.set_xlabel("Gender")
                        ax.tick_params(axis='x', rotation=0)
                        ax.set_title("Mortality Rate by Gender")
                        st.pyplot(fig)
        else:
            st.warning("'Mortality' column not found for detailed demographic analysis.")


    with tab2:
        st.header("Vitals & Lab Results Analysis")
        st.markdown("Comparing measurements between patients who survived and those who did not.")
        
        with st.expander("Vital Signs Analysis", expanded=True):
            if not vital_cols or 'Mortality' not in filtered_df.columns:
                st.warning("Vital sign or Mortality columns not found in the dataset.")
            else:
                cols = st.columns(min(len(vital_cols), 3))
                for i, vital in enumerate(vital_cols):
                    with cols[i % len(cols)]:
                        fig, ax = plt.subplots()
                        sns.boxplot(data=filtered_df, x='Mortality', y=vital, ax=ax, palette='viridis')
                        ax.set_xticklabels(['Survived', 'Died'])
                        ax.set_title(vital.replace('_', ' ').title())
                        st.pyplot(fig)

        with st.expander("Lab Results Analysis"):
            if not lab_cols or 'Mortality' not in filtered_df.columns:
                st.warning("Lab result or Mortality columns not found in the dataset.")
            else:
                lab_to_plot = st.selectbox("Select a Lab Value to Visualize", options=lab_cols)
                fig, ax = plt.subplots()
                sns.kdeplot(data=filtered_df, x=lab_to_plot, hue='Mortality', fill=True, common_norm=False, palette='coolwarm')
                ax.set_title(f"Distribution of {lab_to_plot} for Survivors vs. Non-Survivors")
                st.pyplot(fig)

    with tab3:
        st.header("Risk Factors & Comorbidities Analysis")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Risk Score Distributions")
            if not risk_score_cols or 'Mortality' not in filtered_df.columns:
                st.warning("Risk score or Mortality columns not found.")
            else:
                for score in risk_score_cols:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=filtered_df, x='Mortality', y=score, ax=ax, palette='mako')
                    ax.set_xticklabels(['Survived', 'Died'])
                    ax.set_title(f"Impact of {score.replace('_', ' ')}")
                    st.pyplot(fig)
        
        with col2:
            st.subheader("Impact of Specific Comorbidities")
            if not comorbidity_cols or 'Mortality' not in filtered_df.columns:
                st.warning("Comorbidity or Mortality columns not found.")
            else:
                mortality_rates = {}
                for col in comorbidity_cols:
                    # Check if the comorbidity exists and if there are patients with it
                    if filtered_df[col].sum() > 0:
                        rate = filtered_df[filtered_df[col] == 1]['Mortality'].mean()
                        if pd.notna(rate):
                            mortality_rates[col.replace('_', ' ')] = rate * 100
                
                if mortality_rates:
                    mortality_df = pd.DataFrame(list(mortality_rates.items()), columns=['Comorbidity', 'Mortality Rate (%)']).sort_values('Mortality Rate (%)', ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(data=mortality_df, x='Mortality Rate (%)', y='Comorbidity', ax=ax, palette='rocket')
                    ax.set_title("Mortality Rate for Patients with Specific Comorbidities")
                    ax.set_xlabel("Mortality Rate (%)")
                    ax.set_ylabel("Comorbidity")
                    st.pyplot(fig)
                else:
                    st.info("No patients with the specified comorbidities in the filtered data.")
