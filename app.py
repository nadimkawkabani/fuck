import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import os # Not needed for the current simple file path strategy

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
    age_str = str(age_str) # Ensure it's a string
    try:
        # Attempt to extract the first number for sorting
        return int(age_str.replace(' years', '').split('-')[0].replace('+', ''))
    except ValueError:
        # Handle cases like "unknown" or other non-standard age strings if they exist
        return 999 # Or some other value to push them to the end/beginning

# --- Load Data ---
@st.cache_data
def load_data():
    # CSV is in the SAME directory (root) as app.py in your GitHub repository.
    file_path = "who_suicide_statistics.csv" 
    
    df_loaded = None # Initialize to None
    try:
        df_loaded = pd.read_csv(file_path)
        # st.success(f"Successfully loaded data from '{file_path}'. Shape: {df_loaded.shape}") # For debugging
    except FileNotFoundError:
        st.error(f"FATAL ERROR (load_data): Data file '{file_path}' not found. "
                 f"Ensure 'who_suicide_statistics.csv' is in the ROOT of your GitHub repository, "
                 f"alongside app.py.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Error loading data (load_data): {e}")
        return pd.DataFrame()

    # Proceed with cleaning only if data was loaded successfully
    if df_loaded is not None and not df_loaded.empty:
        try:
            # Check if columns exist before processing
            required_cols = ['suicides_no', 'population', 'year', 'sex', 'age', 'country']
            missing_cols = [col for col in required_cols if col not in df_loaded.columns]
            if missing_cols:
                st.error(f"FATAL ERROR (load_data): Missing required columns in CSV: {', '.join(missing_cols)}")
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
            
        except Exception as e:
            st.error(f"Error during data cleaning (load_data): {e}")
            return pd.DataFrame()
    elif df_loaded is None: # Should have been caught by try-except pd.read_csv
        return pd.DataFrame() # Should not happen if logic above is correct

    return df_loaded

# --- Main Application Logic ---
# 1. Load the data
df_original = load_data()

# 2. Check if data loading was successful; if not, stop the app.
if df_original.empty:
    st.error("Application cannot start because data loading or initial processing failed. Please check error messages above and your file paths.")
    st.stop() 

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")

# Year Filter - More robust initialization
min_year_global = 1900 # Default fallback
max_year_global = 2025 # Default fallback
default_year_range_start = max_year_global - 5
can_create_year_slider = False

if 'year' in df_original.columns and not df_original['year'].empty:
    # Drop NaNs before min/max for safety, though 'year' should ideally not have NaNs
    year_series_cleaned = df_original['year'].dropna()
    if not year_series_cleaned.empty:
        min_val_series = year_series_cleaned.min()
        max_val_series = year_series_cleaned.max()
        # Ensure min_val_series and max_val_series are not NaN themselves
        if pd.notna(min_val_series) and pd.notna(max_val_series):
            min_year_global = int(min_val_series)
            max_year_global = int(max_val_series)
            if min_year_global <= max_year_global:
                can_create_year_slider = True
                default_year_range_start = max(min_year_global, max_year_global - 5)
            else:
                st.sidebar.error("Min year is greater than max year in data. Check data integrity.")
        else:
            st.sidebar.error("Year column min/max calculation resulted in NaN. Cannot create year filter.")
    else:
        st.sidebar.error("Year column contains only NaNs. Cannot create year filter.")
else:
    st.sidebar.error("Year column missing or completely empty. Cannot create year filter.")

if can_create_year_slider:
    selected_year_range_global = st.sidebar.slider(
        "Year Range:",
        min_year_global, 
        max_year_global, 
        (default_year_range_start, max_year_global),
        key="global_year_slider_sidebar"
    )
else:
    st.sidebar.warning("Year filter cannot be displayed. Using a default placeholder range.")
    selected_year_range_global = (1980, 2015) # Fallback if slider can't be made
    # Critical: If year filter is essential, stop the app
    # st.error("Critical year filter failed to initialize. App cannot continue.")
    # st.stop()


# Sex Filter
options_sex = []
default_sex = []
selected_sex_global = [] # Initialize to ensure it's always defined
if 'sex' in df_original.columns and df_original['sex'].notna().any():
    options_sex = sorted(list(df_original['sex'].dropna().unique()))
    default_sex = list(options_sex) 
selected_sex_global = st.sidebar.multiselect(
    "Sex:", options=options_sex, default=default_sex,
    key="global_sex_multiselect_sidebar",
    disabled=not bool(options_sex)
)
if not options_sex and 'sex' in df_original.columns: st.sidebar.info("No distinct sex values found for filtering after NaNs removed.")
elif 'sex' not in df_original.columns: st.sidebar.warning("Sex data unavailable for filtering (column missing).")

# Age Filter
options_ages = []
default_ages = []
selected_age_global = [] # Initialize to ensure it's always defined
if 'age' in df_original.columns and df_original['age'].notna().any():
    options_ages = sorted(list(df_original['age'].astype(str).dropna().unique()), key=age_sort_key)
    default_ages = list(options_ages) 
selected_age_global = st.sidebar.multiselect(
    "Age Groups:", options=options_ages, default=default_ages,
    key="global_age_multiselect_sidebar",
    disabled=not bool(options_ages)
)
if not options_ages and 'age' in df_original.columns: st.sidebar.info("No distinct age values found for filtering after NaNs removed.")
elif 'age' not in df_original.columns: st.sidebar.warning("Age data unavailable for filtering (column missing).")


st.sidebar.markdown("---")
st.sidebar.caption("Data: WHO via Kaggle.")

# Apply global filters
conditions = []
if 'year' in df_original.columns: # Check if year column exists for filtering
    conditions.append(
        (df_original['year'] >= selected_year_range_global[0]) & \
        (df_original['year'] <= selected_year_range_global[1])
    )
if selected_sex_global and 'sex' in df_original.columns:
    conditions.append(df_original['sex'].isin(selected_sex_global))
if selected_age_global and 'age' in df_original.columns:
    conditions.append(df_original['age'].isin(selected_age_global))

if conditions: # Only filter if there are valid conditions
    final_condition = pd.Series(True, index=df_original.index)
    for cond in conditions: 
        final_condition &= cond # Use bitwise AND
    df_filtered_global = df_original[final_condition].copy()
else: # If no valid conditions (e.g., year column was missing)
    df_filtered_global = df_original.copy() # Or an empty DataFrame if that's preferred
    st.warning("Filtering could not be fully applied due to missing essential columns (e.g., 'year'). Displaying broader data.")


# --- Main Dashboard Area ---
st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")

sex_summary_text = 'All'
if options_sex: 
    if selected_sex_global and len(selected_sex_global) < len(options_sex):
        sex_summary_text = ', '.join(selected_sex_global)
elif 'sex' not in df_original.columns: sex_summary_text = 'N/A (col missing)'

age_summary_text = 'All Ages'
if options_ages: 
    if selected_age_global and len(selected_age_global) < len(options_ages):
        age_summary_text = f"{len(selected_age_global)} groups"
elif 'age' not in df_original.columns: age_summary_text = 'N/A (col missing)'

st.caption(f"Displaying data for: Years {selected_year_range_global[0]}-{selected_year_range_global[1]} | Sex: {sex_summary_text} | Ages: {age_summary_text}") # THIS IS THE LINE THAT WAS problematic (around 81)

# --- 2x2 Grid for Visuals ---
if not df_filtered_global.empty:
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1: # Map
        st.markdown("<h6>🗺️ Geographic Distribution</h6>", unsafe_allow_html=True)
        if 'year' in df_original.columns:
            map_year_options = sorted(df_original['year'].dropna().unique(), reverse=True)
            default_map_year_index = 0
            if selected_year_range_global[1] in map_year_options: 
                default_map_year_index = map_year_options.index(selected_year_range_global[1])
            
            selected_map_year = st.selectbox(
                "Map Year:", map_year_options, index=default_map_year_index,
                key="map_year_select_main_compact", label_visibility="collapsed"
            )
            st.caption(f"Map showing year: {selected_map_year}")
            
            map_sex_cond_plot = pd.Series(True, index=df_original.index) # Plot specific conditions
            if selected_sex_global and 'sex' in df_original.columns: map_sex_cond_plot = df_original['sex'].isin(selected_sex_global)
            map_age_cond_plot = pd.Series(True, index=df_original.index)
            if selected_age_global and 'age' in df_original.columns: map_age_cond_plot = df_original['age'].isin(selected_age_global)

            map_data_source = df_original[
                (df_original['year'] == selected_map_year) & map_sex_cond_plot & map_age_cond_plot
            ].copy()

            if 'country' in map_data_source.columns:
                country_map_data = map_data_source.groupby('country').agg(s_no=('suicides_no', 'sum'), pop=('population', 'sum')).reset_index()
                country_map_data['rate'] = np.where(country_map_data['pop'] > 0, (country_map_data['s_no'] / country_map_data['pop']) * 100000, 0)
                country_map_data.dropna(subset=['rate'], inplace=True)
                country_name_mapping = {"United States of America": "United States", "Russian Federation": "Russia", "Republic of Korea": "South Korea", "Iran (Islamic Rep of)": "Iran", "Czech Republic": "Czechia"}
                country_map_data['country_std'] = country_map_data['country'].replace(country_name_mapping)
                
                if not country_map_data.empty:
                    fig_map = px.choropleth(country_map_data, locations="country_std", locationmode="country names", color="rate", hover_name="country", hover_data={"rate": ":.1f", "s_no": True, "country_std": False}, color_continuous_scale=px.colors.sequential.OrRd)
                    fig_map.update_layout(margin={"r":5,"t":5,"l":5,"b":5}, height=280, geo=dict(bgcolor= 'rgba(0,0,0,0)'), coloraxis_colorbar=dict(title="Rate", thickness=10, len=0.7, yanchor="middle", y=0.5, tickfont=dict(size=8)))
                    st.plotly_chart(fig_map, use_container_width=True)
                else: st.caption(f"No map data for {selected_map_year} with filters.")
            else: st.caption("Country data missing for map.")
        else: st.caption("Year data missing for map.")


    with row1_col2: # Country Comparison
        st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
        available_countries_options_plot = []
        if 'country' in df_filtered_global.columns: available_countries_options_plot = sorted(df_filtered_global['country'].unique())
        
        selected_countries_compare_plot = []
        if available_countries_options_plot:
            default_countries_plot = available_countries_options_plot[:min(2, len(available_countries_options_plot))]
            selected_countries_compare_plot = st.multiselect("Compare:", available_countries_options_plot, default=default_countries_plot, key="country_compare_compact_v3", label_visibility="collapsed")
            st.caption(f"Comparing: {', '.join(selected_countries_compare_plot) if selected_countries_compare_plot else 'None'}")
        else:
            st.caption("No countries in filtered data.")

        if selected_countries_compare_plot: 
            country_comp_data = df_filtered_global[df_filtered_global['country'].isin(selected_countries_compare_plot)]
            country_comp_agg = country_comp_data.groupby(['country', 'year']).agg(s_no=('suicides_no', 'sum'), pop=('population', 'sum')).reset_index()
            country_comp_agg['rate'] = np.where(country_comp_agg['pop'] > 0, (country_comp_agg['s_no'] / country_comp_agg['pop']) * 100000, 0)
            if not country_comp_agg.empty:
                fig_cc, ax_cc = plt.subplots(figsize=(5.5, 2.7))
                sns.lineplot(data=country_comp_agg, x='year', y='rate', hue='country', ax=ax_cc, marker='o', markersize=3)
                ax_cc.set_ylabel('Rate', fontsize=8); ax_cc.set_xlabel('Year', fontsize=8)
                ax_cc.tick_params(axis='both', which='major', labelsize=7)
                ax_cc.legend(title='', fontsize='xx-small', loc='best', frameon=False); ax_cc.grid(True, linestyle=':', linewidth=0.5)
                plt.tight_layout(pad=0.5); st.pyplot(fig_cc)
            else: st.caption("No data for selected countries.")
        elif available_countries_options_plot : st.caption("Select countries to compare.")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1: # Demographic Breakdown
        st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
        dem_df_plot = df_filtered_global.copy() # Use a plot-specific copy
        tab_titles = ["Sex", "Age", "Sex&Age"]
        tab_sex, tab_age, tab_sex_age = st.tabs(tab_titles)
        common_fig_size_plot = (5.5, 2.5); common_font_size_plot = 7; legend_font_size_plot = 'xx-small'

        with tab_sex:
            if 'sex' in dem_df_plot.columns:
                sex_data_plot = dem_df_plot.groupby(['year', 'sex']).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                sex_data_plot['rate'] = np.where(sex_data_plot['p'] > 0, (sex_data_plot['s'] / sex_data_plot['p']) * 100000, 0)
                if not sex_data_plot.empty and sex_data_plot['rate'].notna().any():
                    fig_s, ax_s = plt.subplots(figsize=common_fig_size_plot)
                    sns.lineplot(data=sex_data_plot, x='year', y='rate', hue='sex', ax=ax_s, marker='o', markersize=3)
                    ax_s.set_title('Rate by Sex', fontsize=common_font_size_plot+1)
                    ax_s.grid(True, linestyle=':', linewidth=0.5); ax_s.set_ylabel('Rate', fontsize=common_font_size_plot); ax_s.set_xlabel('Year', fontsize=common_font_size_plot)
                    ax_s.tick_params(labelsize=common_font_size_plot-1); ax_s.legend(fontsize=legend_font_size_plot, title='', frameon=False, loc='best')
                    plt.tight_layout(pad=0.5); st.pyplot(fig_s)
                else: st.caption("No Sex data for these filters.")
            else: st.caption("Sex data unavailable.")
        with tab_age:
            if 'age' in dem_df_plot.columns and not dem_df_plot['age'].empty:
                age_order_plot = sorted(dem_df_plot['age'].astype(str).dropna().unique(), key=age_sort_key)
                dem_df_plot['age_cat'] = pd.Categorical(dem_df_plot['age'], categories=age_order_plot, ordered=True)
                age_data_plot = dem_df_plot.groupby(['year', 'age_cat'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                age_data_plot['rate'] = np.where(age_data_plot['p'] > 0, (age_data_plot['s'] / age_data_plot['p']) * 100000, 0)
                if not age_data_plot.empty and age_data_plot['rate'].notna().any():
                    fig_a, ax_a = plt.subplots(figsize=common_fig_size_plot)
                    sns.lineplot(data=age_data_plot, x='year', y='rate', hue='age_cat', ax=ax_a, marker='o', markersize=3)
                    ax_a.set_title('Rate by Age', fontsize=common_font_size_plot+1)
                    ax_a.legend(title='', fontsize=legend_font_size_plot, loc='best', frameon=False, ncol=2)
                    ax_a.grid(True, linestyle=':', linewidth=0.5); ax_a.set_ylabel('Rate', fontsize=common_font_size_plot); ax_a.set_xlabel('Year', fontsize=common_font_size_plot)
                    ax_a.tick_params(labelsize=common_font_size_plot-1); plt.tight_layout(pad=0.5); st.pyplot(fig_a)
                else: st.caption("No Age data for these filters.")
            else: st.caption("Age data unavailable.")
        with tab_sex_age:
            if 'age' in dem_df_plot.columns and not dem_df_plot['age'].empty and 'sex' in dem_df_plot.columns:
                age_order_sa_plot = sorted(dem_df_plot['age'].astype(str).dropna().unique(), key=age_sort_key)
                dem_df_plot['age_cat_sa'] = pd.Categorical(dem_df_plot['age'], categories=age_order_sa_plot, ordered=True)
                sa_data_plot = dem_df_plot.groupby(['year', 'sex', 'age_cat_sa'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                sa_data_plot['rate'] = np.where(sa_data_plot['p'] > 0, (sa_data_plot['s'] / sa_data_plot['p']) * 100000, 0)
                if not sa_data_plot.empty and sa_data_plot['rate'].notna().any():
                    g = sns.relplot(data=sa_data_plot, x='year', y='rate', hue='age_cat_sa', style='sex', kind='line', markers=True, height=2.2, aspect=1.5, facet_kws=dict(margin_titles=True), legend=False)
                    g.set_axis_labels("Year", "Rate", fontsize=common_font_size_plot); g.set_titles(col_template="{col_name}", row_template="{row_name}", size=common_font_size_plot)
                    g.tick_params(labelsize=common_font_size_plot-1); plt.grid(True, linestyle=':', linewidth=0.5); g.tight_layout(pad=0.5); st.pyplot(g.fig)
                else: st.caption("No Sex&Age data for these filters.")
            else: st.caption("Sex&Age data unavailable.")

    with row2_col2: # Age Group Rate (Overall)
        st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
        if 'age' in df_filtered_global.columns and not df_filtered_global['age'].empty:
            age_dist_plot = df_filtered_global.groupby('age').agg(s_total=('suicides_no', 'sum'), p_total=('population', 'sum')).reset_index()
            age_dist_plot['rate'] = np.where(age_dist_plot['p_total'] > 0, (age_dist_plot['s_total'] / age_dist_plot['p_total']) * 100000, 0)
            age_dist_plot.dropna(subset=['rate'], inplace=True)
            if not age_dist_plot.empty:
                age_order_bar_plot = sorted(age_dist_plot['age'].astype(str).dropna().unique(), key=age_sort_key)
                age_dist_plot['age_cat'] = pd.Categorical(age_dist_plot['age'], categories=age_order_bar_plot, ordered=True)
                age_dist_plot = age_dist_plot.sort_values('age_cat')
                fig_ad, ax_ad = plt.subplots(figsize=(5.5, 2.7))
                sns.barplot(data=age_dist_plot, x='age_cat', y='rate', ax=ax_ad, palette="coolwarm_r")
                ax_ad.set_xlabel('Age Group', fontsize=common_font_size_plot); ax_ad.set_ylabel('Rate (per 100k)', fontsize=common_font_size_plot)
                plt.xticks(rotation=45, ha="right"); ax_ad.tick_params(axis='both', which='major', labelsize=common_font_size_plot-1)
                ax_ad.grid(True, axis='y', linestyle=':', linewidth=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig_ad)
            else: st.caption("No Age Dist. data for these filters.")
        else: st.caption("Age data unavailable.")
else:
    st.warning("No data available for the selected global filters. Please adjust filters in the sidebar.")
