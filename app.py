import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Helper function for age sorting
def age_sort_key(age_str):
    """Helper function to sort age groups in a logical order"""
    age_order = {
        '5-14 years': 0,
        '15-24 years': 1,
        '25-34 years': 2,
        '35-54 years': 3,
        '55-74 years': 4,
        '75+ years': 5
    }
    return age_order.get(age_str, 6)  # Default for unexpected values

# --- Load Data ---
@st.cache_data
def load_data():
    file_path = "who_suicide_statistics.csv" 
    
    try:
        df_loaded = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Data file '{file_path}' not found.")
        return pd.DataFrame()

    # Data cleaning
    try:
        df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
        df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
        
        df_loaded['suicide_rate'] = np.where(
            (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()),
            (df_loaded['suicides_no'] / df_loaded['population']) * 100000,
            np.nan
        )
        
        df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_loaded['age'] = df_loaded['age'].astype(str)
        
        return df_loaded
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# --- Main Application ---
st.set_page_config(layout="wide", page_title="Suicide Statistics Dashboard")

df_original = load_data()

if df_original.empty:
    st.error("Application cannot start because data loading failed.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")

# Year slider
min_year_global = int(df_original['year'].min())
max_year_global = int(df_original['year'].max())
selected_year_range_global = st.sidebar.slider(
    "Year Range:",
    min_year_global, max_year_global,
    (max(min_year_global, max_year_global - 5), max_year_global),
    key="year_range_slider_main"
)

# Sex multiselect
if 'sex' in df_original.columns:
    unique_sex_global = df_original['sex'].unique()
    selected_sex_global = st.sidebar.multiselect(
        "Sex:",
        options=unique_sex_global, 
        default=unique_sex_global,
        key="sex_multiselect_main"
    )
else:
    st.sidebar.warning("Column 'sex' not found in data.")
    selected_sex_global = []

# Age multiselect
if 'age' in df_original.columns:
    df_original['age'] = df_original['age'].astype(str)
    unique_ages_global = sorted(df_original['age'].dropna().unique(), key=age_sort_key)
    selected_age_global = st.sidebar.multiselect(
        "Age Groups:",
        options=unique_ages_global,
        default=unique_ages_global,
        key="age_multiselect_main"
    )
else:
    st.sidebar.warning("Column 'age' not found in data.")
    selected_age_global = []

st.sidebar.markdown("---")
st.sidebar.caption("Data: WHO via Kaggle.")

# --- Apply Filters ---
year_condition = (df_original['year'] >= selected_year_range_global[0]) & \
                 (df_original['year'] <= selected_year_range_global[1])

sex_condition = df_original['sex'].isin(selected_sex_global) if selected_sex_global else pd.Series(True, index=df_original.index)
age_condition = df_original['age'].isin(selected_age_global) if selected_age_global else pd.Series(True, index=df_original.index)

df_filtered_global = df_original[year_condition & sex_condition & age_condition].copy()

# --- Main Dashboard ---
st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")

# 2x2 Grid Layout
row1_col1, row1_col2 = st.columns(2)

# Geographic Distribution (Map)
with row1_col1:
    st.markdown("<h6>🗺️ Geographic Distribution</h6>", unsafe_allow_html=True)
    map_year_options = sorted(df_original['year'].unique(), reverse=True)
    selected_map_year = st.selectbox(
        "Map Year:", 
        map_year_options,
        key="map_year_select"
    )
    
    map_data = df_original[
        (df_original['year'] == selected_map_year) & 
        (df_original['sex'].isin(selected_sex_global) if selected_sex_global else True) &
        (df_original['age'].isin(selected_age_global) if selected_age_global else True)
    ].groupby('country').agg(
        suicides=('suicides_no', 'sum'),
        population=('population', 'sum')
    ).reset_index()
    
    map_data['rate'] = (map_data['suicides'] / map_data['population']) * 100000
    
    if not map_data.empty:
        fig_map = px.choropleth(
            map_data, 
            locations="country", 
            locationmode="country names",
            color="rate",
            hover_name="country",
            color_continuous_scale=px.colors.sequential.OrRd
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No data available for the selected map filters.")

# Country Comparison
with row1_col2:
    st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
    available_countries = sorted(df_filtered_global['country'].unique())
    selected_countries = st.multiselect(
        "Select countries to compare:",
        options=available_countries,
        default=available_countries[:2] if len(available_countries) >= 2 else available_countries,
        key="country_comparison_select"
    )
    
    if selected_countries:
        comparison_data = df_filtered_global[df_filtered_global['country'].isin(selected_countries)]
        comparison_agg = comparison_data.groupby(['country', 'year']).agg(
            suicides=('suicides_no', 'sum'),
            population=('population', 'sum')
        ).reset_index()
        comparison_agg['rate'] = (comparison_agg['suicides'] / comparison_agg['population']) * 100000
        
        fig_comparison = px.line(
            comparison_agg,
            x='year',
            y='rate',
            color='country',
            markers=True,
            title='Suicide Rate Over Time by Country'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

# Second Row
row2_col1, row2_col2 = st.columns(2)

# Demographic Breakdown
with row2_col1:
    st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["By Sex", "By Age", "Combined"])
    
    with tab1:
        sex_data = df_filtered_global.groupby(['year', 'sex']).agg(
            suicides=('suicides_no', 'sum'),
            population=('population', 'sum')
        ).reset_index()
        sex_data['rate'] = (sex_data['suicides'] / sex_data['population']) * 100000
        
        fig_sex = px.line(
            sex_data,
            x='year',
            y='rate',
            color='sex',
            title='Suicide Rate by Sex Over Time'
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    
    with tab2:
        age_data = df_filtered_global.groupby(['year', 'age']).agg(
            suicides=('suicides_no', 'sum'),
            population=('population', 'sum')
        ).reset_index()
        age_data['rate'] = (age_data['suicides'] / age_data['population']) * 100000
        age_data['age'] = pd.Categorical(age_data['age'], categories=unique_ages_global, ordered=True)
        
        fig_age = px.line(
            age_data.sort_values('age'),
            x='year',
            y='rate',
            color='age',
            title='Suicide Rate by Age Group Over Time'
        )
        st.plotly_chart(fig_age, use_container_width=True)

# Age Group Distribution
with row2_col2:
    st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
    age_dist = df_filtered_global.groupby('age').agg(
        suicides=('suicides_no', 'sum'),
        population=('population', 'sum')
    ).reset_index()
    age_dist['rate'] = (age_dist['suicides'] / age_dist['population']) * 100000
    age_dist['age'] = pd.Categorical(age_dist['age'], categories=unique_ages_global, ordered=True)
    
    fig_age_dist = px.bar(
        age_dist.sort_values('age'),
        x='age',
        y='rate',
        title='Average Suicide Rate by Age Group'
    )
    st.plotly_chart(fig_age_dist, use_container_width=True)
