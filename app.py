import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os # Still good to have, though not strictly needed for the simplest path

# --- Page Config ---
st.set_page_config(
    page_title="Focused Suicide Stats Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function for Age Sorting ---
def age_sort_key(age_str):
    if pd.isna(age_str):
        return -1
    age_str = str(age_str)
    return int(age_str.replace(' years', '').split('-')[0].replace('+', ''))

# --- Load Data ---
@st.cache_data
def load_data():
    # CORRECTED FILE PATH: CSV is in the SAME directory as app.py
    file_path = "who_suicide_statistics.csv" 
    
    try:
        df_loaded = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Data file '{file_path}' not found. "
                 f"Please ensure 'who_suicide_statistics.csv' is in the ROOT of your GitHub repository, "
                 f"alongside app.py, and that the repository has been cloned by Streamlit Cloud.")
        return pd.DataFrame() 

    df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
    df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
    df_loaded['suicide_rate'] = np.where(
        (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()),
        (df_loaded['suicides_no'] / df_loaded['population']) * 100000,
        0 
    )
    df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_loaded['age'] = df_loaded['age'].astype(str)
    
    return df_loaded

# --- Main Application Logic ---
df_original = load_data()

if df_original.empty:
    st.error("Application cannot start because data loading failed. Please check the error message above and your file paths.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")
min_year_global, max_year_global = int(df_original['year'].min()), int(df_original['year'].max())
selected_year_range_global = st.sidebar.slider(
    "Year Range:",
    min_year_global, max_year_global,
    (max(min_year_global, max_year_global - 5), max_year_global),
    key="global_year_slider_sidebar"
)

if 'sex' in df_original.columns:
    unique_sex_global = df_original['sex'].unique()
    selected_sex_global = st.sidebar.multiselect(
        "Sex:",
        options=unique_sex_global, default=unique_sex_global,
        key="global_sex_multiselect_sidebar"
    )
else:
    st.sidebar.warning("Column 'sex' not found in data.")
    selected_sex_global = []

if 'age' in df_original.columns:
    unique_ages_global = sorted(df_original['age'].dropna().unique(), key=age_sort_key)
    selected_age_global = st.sidebar.multiselect(
        "Age Groups:",
        options=unique_ages_global, default=unique_ages_global,
        key="global_age_multiselect_sidebar"
    )
else:
    st.sidebar.warning("Column 'age' not found in data.")
    selected_age_global = []

st.sidebar.markdown("---")
st.sidebar.caption("Data: WHO via Kaggle.")

# Apply global filters
year_condition = (df_original['year'] >= selected_year_range_global[0]) & \
                 (df_original['year'] <= selected_year_range_global[1])

sex_condition = pd.Series(True, index=df_original.index)
if selected_sex_global and 'sex' in df_original.columns:
    sex_condition = df_original['sex'].isin(selected_sex_global)

age_condition = pd.Series(True, index=df_original.index)
if selected_age_global and 'age' in df_original.columns:
    age_condition = df_original['age'].isin(selected_age_global)

df_filtered_global = df_original[year_condition & sex_condition & age_condition].copy()

# --- Main Dashboard Area ---
st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")
filter_summary_sex = ', '.join(selected_sex_global) if (selected_sex_global and len(selected_sex_global) < (len(unique_sex_global) if 'sex' in df_original.columns else 0)) else 'All'
filter_summary_age = f"{len(selected_age_global)} groups" if (selected_age_global and len(selected_age_global) < (len(unique_ages_global) if 'age' in df_original.columns else 0)) else "All Ages"
st.caption(f"Displaying data for: Years {selected_year_range_global[0]}-{selected_year_range_global[1]} | Sex: {filter_summary_sex} | Ages: {filter_summary_age}")

# --- 2x2 Grid for Visuals ---
if not df_filtered_global.empty:
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown("<h6>🗺️ Geographic Distribution</h6>", unsafe_allow_html=True)
        map_year_options = sorted(df_original['year'].unique(), reverse=True)
        default_map_year_index = 0
        if selected_year_range_global[1] in map_year_options:
            default_map_year_index = map_year_options.index(selected_year_range_global[1])
        
        selected_map_year = st.selectbox(
            "Map Year:", map_year_options, index=default_map_year_index,
            key="map_year_select_main_compact", label_visibility="collapsed"
        )
        st.caption(f"Map showing year: {selected_map_year}")

        map_data_source = df_original[
            (df_original['year'] == selected_map_year) & sex_condition & age_condition
        ].copy()

        country_map_data = map_data_source.groupby('country').agg(
            s_no=('suicides_no', 'sum'), pop=('population', 'sum')
        ).reset_index()
        country_map_data['rate'] = np.where(country_map_data['pop'] > 0, (country_map_data['s_no'] / country_map_data['pop']) * 100000, 0)
        country_map_data.dropna(subset=['rate'], inplace=True)
        country_name_mapping = {
            "United States of America": "United States", "Russian Federation": "Russia",
            "Republic of Korea": "South Korea", "Iran (Islamic Rep of)": "Iran", "Czech Republic": "Czechia"
        }
        country_map_data['country_std'] = country_map_data['country'].replace(country_name_mapping)
        if not country_map_data.empty:
            fig_map = px.choropleth(
                country_map_data, locations="country_std", locationmode="country names", color="rate",
                hover_name="country", hover_data={"rate": ":.1f", "s_no": True, "country_std": False},
                color_continuous_scale=px.colors.sequential.OrRd
            )
            fig_map.update_layout(
                margin={"r":5,"t":5,"l":5,"b":5}, height=280,
                geo=dict(bgcolor= 'rgba(0,0,0,0)'),
                coloraxis_colorbar=dict(title="Rate", thickness=10, len=0.7, yanchor="middle", y=0.5, tickfont=dict(size=8))
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else: st.caption(f"No map data for {selected_map_year} with current filters.")

    with row1_col2:
        st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
        available_countries = sorted(df_filtered_global['country'].unique())
        if available_countries:
            default_countries = available_countries[:min(2, len(available_countries))]
            selected_countries_compare = st.multiselect(
                "Compare Countries:", available_countries, default=default_countries,
                key="country_compare_compact", label_visibility="collapsed"
            )
            st.caption(f"Comparing: {', '.join(selected_countries_compare) if selected_countries_compare else 'None'}")
            if selected_countries_compare:
                country_comp_data = df_filtered_global[df_filtered_global['country'].isin(selected_countries_compare)]
                country_comp_agg = country_comp_data.groupby(['country', 'year']).agg(
                    s_no=('suicides_no', 'sum'), pop=('population', 'sum')
                ).reset_index()
                country_comp_agg['rate'] = np.where(country_comp_agg['pop'] > 0, (country_comp_agg['s_no'] / country_comp_agg['pop']) * 100000, 0)
                if not country_comp_agg.empty:
                    fig_cc, ax_cc = plt.subplots(figsize=(5.5, 2.7))
                    sns.lineplot(data=country_comp_agg, x='year', y='rate', hue='country', ax=ax_cc, marker='o', markersize=3)
                    ax_cc.set_ylabel('Rate', fontsize=8)
                    ax_cc.set_xlabel('Year', fontsize=8)
                    ax_cc.tick_params(axis='both', which='major', labelsize=7)
                    ax_cc.legend(title='', fontsize='xx-small', loc='best', frameon=False)
                    ax_cc.grid(True, linestyle=':', linewidth=0.5)
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig_cc)
                else: st.caption("No data for selected countries.")
        else: st.caption("No countries available with current filters.")

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
        dem_df = df_filtered_global.copy()
        tab_titles = ["Sex", "Age", "Sex&Age"]
        tab_sex, tab_age, tab_sex_age = st.tabs(tab_titles)

        common_fig_size = (5.5, 2.5)
        common_font_size = 7
        legend_font_size = 'xx-small'

        with tab_sex:
            sex_data = dem_df.groupby(['year', 'sex']).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
            sex_data['rate'] = np.where(sex_data['p'] > 0, (sex_data['s'] / sex_data['p']) * 100000, 0)
            if not sex_data.empty and sex_data['rate'].notna().any():
                fig_s, ax_s = plt.subplots(figsize=common_fig_size)
                sns.lineplot(data=sex_data, x='year', y='rate', hue='sex', ax=ax_s, marker='o', markersize=3)
                ax_s.set_title('Rate by Sex', fontsize=common_font_size+1)
                ax_s.grid(True, linestyle=':', linewidth=0.5); ax_s.set_ylabel('Rate', fontsize=common_font_size); ax_s.set_xlabel('Year', fontsize=common_font_size)
                ax_s.tick_params(labelsize=common_font_size-1); ax_s.legend(fontsize=legend_font_size, title='', frameon=False, loc='best')
                plt.tight_layout(pad=0.5); st.pyplot(fig_s)
            else: st.caption("No Sex data for these filters.")
        with tab_age:
            if 'age' in dem_df.columns and not dem_df['age'].empty:
                age_order = sorted(dem_df['age'].dropna().unique(), key=age_sort_key)
                dem_df['age_cat'] = pd.Categorical(dem_df['age'], categories=age_order, ordered=True)
                age_data = dem_df.groupby(['year', 'age_cat'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                age_data['rate'] = np.where(age_data['p'] > 0, (age_data['s'] / age_data['p']) * 100000, 0)
                if not age_data.empty and age_data['rate'].notna().any():
                    fig_a, ax_a = plt.subplots(figsize=common_fig_size)
                    sns.lineplot(data=age_data, x='year', y='rate', hue='age_cat', ax=ax_a, marker='o', markersize=3)
                    ax_a.set_title('Rate by Age', fontsize=common_font_size+1)
                    ax_a.legend(title='', fontsize=legend_font_size, loc='best', frameon=False, ncol=2)
                    ax_a.grid(True, linestyle=':', linewidth=0.5); ax_a.set_ylabel('Rate', fontsize=common_font_size); ax_a.set_xlabel('Year', fontsize=common_font_size)
                    ax_a.tick_params(labelsize=common_font_size-1); plt.tight_layout(pad=0.5); st.pyplot(fig_a)
                else: st.caption("No Age data for these filters.")
            else: st.caption("No Age data available.")
        with tab_sex_age:
            if 'age' in dem_df.columns and not dem_df['age'].empty and 'sex' in dem_df.columns:
                age_order_sa = sorted(dem_df['age'].dropna().unique(), key=age_sort_key)
                dem_df['age_cat_sa'] = pd.Categorical(dem_df['age'], categories=age_order_sa, ordered=True)
                sa_data = dem_df.groupby(['year', 'sex', 'age_cat_sa'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                sa_data['rate'] = np.where(sa_data['p'] > 0, (sa_data['s'] / sa_data['p']) * 100000, 0)
                if not sa_data.empty and sa_data['rate'].notna().any():
                    g = sns.relplot(data=sa_data, x='year', y='rate', hue='age_cat_sa', style='sex', kind='line', 
                                    markers=True, height=2.2, aspect=1.5,
                                    facet_kws=dict(margin_titles=True), legend=False)
                    g.set_axis_labels("Year", "Rate", fontsize=common_font_size)
                    g.set_titles(col_template="{col_name}", row_template="{row_name}", size=common_font_size)
                    g.tick_params(labelsize=common_font_size-1)
                    plt.grid(True, linestyle=':', linewidth=0.5)
                    g.tight_layout(pad=0.5)
                    st.pyplot(g.fig)
                else: st.caption("No Sex&Age data for these filters.")
            else: st.caption("No Sex&Age data available.")

    with row2_col2:
        st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
        if 'age' in df_filtered_global.columns and not df_filtered_global['age'].empty:
            age_dist = df_filtered_global.groupby('age').agg(s_total=('suicides_no', 'sum'), p_total=('population', 'sum')).reset_index()
            age_dist['rate'] = np.where(age_dist['p_total'] > 0, (age_dist['s_total'] / age_dist['p_total']) * 100000, 0)
            age_dist.dropna(subset=['rate'], inplace=True)
            if not age_dist.empty:
                age_order_bar = sorted(age_dist['age'].dropna().unique(), key=age_sort_key)
                age_dist['age_cat'] = pd.Categorical(age_dist['age'], categories=age_order_bar, ordered=True)
                age_dist = age_dist.sort_values('age_cat')
                fig_ad, ax_ad = plt.subplots(figsize=(5.5, 2.7))
                sns.barplot(data=age_dist, x='age_cat', y='rate', ax=ax_ad, palette="coolwarm_r")
                ax_ad.set_xlabel('Age Group', fontsize=common_font_size)
                ax_ad.set_ylabel('Rate (per 100k)', fontsize=common_font_size)
                plt.xticks(rotation=45, ha="right")
                ax_ad.tick_params(axis='both', which='major', labelsize=common_font_size-1)
                ax_ad.grid(True, axis='y', linestyle=':', linewidth=0.5)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig_ad)
            else: st.caption("No Age Dist. data for these filters.")
        else: st.caption("No Age data available.")
else:
    st.warning("No data available for the selected global filters. Please adjust filters in the sidebar.")
