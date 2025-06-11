import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- Page Config ---
st.set_page_config(
    page_title="Focused Suicide Stats Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Function for Age Sorting ---
def age_sort_key(age_str):
    if pd.isna(age_str): return -1
    age_str = str(age_str)
    return int(age_str.replace(' years', '').split('-')[0].replace('+', ''))

# --- Load Data ---
@st.cache_data
def load_data():
    file_path = "who_suicide_statistics.csv" 
    try:
        df_loaded = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Data file '{file_path}' not found.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

    if not df_loaded.empty:
        try:
            if 'suicides_no' in df_loaded.columns: df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
            if 'population' in df_loaded.columns: df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
            
            if 'population' in df_loaded.columns and 'suicides_no' in df_loaded.columns:
                df_loaded['suicide_rate'] = np.where(
                    (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()),
                    (df_loaded['suicides_no'] / df_loaded['population']) * 100000, 0 
                )
            else: df_loaded['suicide_rate'] = 0 
            
            df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
            if 'age' in df_loaded.columns: df_loaded['age'] = df_loaded['age'].astype(str)
        except Exception as e:
            st.error(f"Error during data cleaning: {e}")
            return pd.DataFrame()
    return df_loaded

# --- Main Application Logic ---
df_original = load_data()

if df_original.empty:
    st.error("App cannot start: data loading/processing failed.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.markdown("### Filters")

# Year Filter - More robust initialization
min_year_global = None
max_year_global = None
can_create_year_slider = False

if 'year' in df_original.columns and not df_original['year'].empty:
    min_val_series = df_original['year'].min()
    max_val_series = df_original['year'].max()
    if pd.notna(min_val_series) and pd.notna(max_val_series):
        min_year_global = int(min_val_series)
        max_year_global = int(max_val_series)
        if min_year_global <= max_year_global: # Ensure min is not greater than max
            can_create_year_slider = True
        else:
            st.sidebar.error("Min year is greater than max year. Check data.")
    else:
        st.sidebar.error("Year column contains NaNs affecting min/max. Cannot create year filter.")
else:
    st.sidebar.error("Year column missing or empty. Cannot create year filter.")

if can_create_year_slider:
    selected_year_range_global = st.sidebar.slider( # THIS IS WHERE LINE 81 LIKELY IS
        "Year Range:",
        min_year_global, # Now guaranteed to be an int or slider won't be created
        max_year_global, # Now guaranteed to be an int
        (max(min_year_global, max_year_global - 5), max_year_global),
        key="global_year_slider_sidebar"
    )
else:
    st.sidebar.error("Year filter cannot be displayed due to data issues.")
    # Provide default valid values for selected_year_range_global so the rest of the script doesn't break
    # or, more drastically, st.stop() here as well if year range is absolutely critical.
    # For now, let's provide a placeholder; the df_filtered_global will likely be empty.
    st.warning("Using placeholder year range due to filter error. Data display may be affected.")
    selected_year_range_global = (1900, 2000) # Placeholder
    # If year is absolutely critical and the app can't run without it, uncomment next line:
    # st.stop()


# Sex Filter
options_sex = []
default_sex = []
if 'sex' in df_original.columns and df_original['sex'].notna().any():
    options_sex = list(df_original['sex'].dropna().unique())
    default_sex = list(options_sex) 
selected_sex_global = st.sidebar.multiselect(
    "Sex:", options=options_sex, default=default_sex,
    key="global_sex_multiselect_sidebar",
    disabled=not bool(options_sex)
)
if not options_sex: st.sidebar.warning("Sex data unavailable for filtering.")

# Age Filter
options_ages = []
default_ages = []
if 'age' in df_original.columns and df_original['age'].notna().any():
    options_ages = sorted(list(df_original['age'].astype(str).dropna().unique()), key=age_sort_key)
    default_ages = list(options_ages) 
selected_age_global = st.sidebar.multiselect(
    "Age Groups:", options=options_ages, default=default_ages,
    key="global_age_multiselect_sidebar",
    disabled=not bool(options_ages)
)
if not options_ages: st.sidebar.warning("Age data unavailable for filtering.")

st.sidebar.markdown("---")
st.sidebar.caption("Data: WHO via Kaggle.")

# Apply global filters
# Ensure selected_year_range_global is defined before this point
conditions = [
    (df_original['year'] >= selected_year_range_global[0]) & \
    (df_original['year'] <= selected_year_range_global[1])
]
if selected_sex_global and 'sex' in df_original.columns:
    conditions.append(df_original['sex'].isin(selected_sex_global))
if selected_age_global and 'age' in df_original.columns:
    conditions.append(df_original['age'].isin(selected_age_global))

final_condition = pd.Series(True, index=df_original.index)
for cond in conditions: final_condition &= cond
df_filtered_global = df_original[final_condition].copy()

# --- Main Dashboard Area ---
st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")

sex_summary_text = 'All'
if options_sex: 
    if selected_sex_global and len(selected_sex_global) < len(options_sex):
        sex_summary_text = ', '.join(selected_sex_global)
elif 'sex' not in df_original.columns: sex_summary_text = 'N/A (column missing)'
age_summary_text = 'All Ages'
if options_ages: 
    if selected_age_global and len(selected_age_global) < len(options_ages):
        age_summary_text = f"{len(selected_age_global)} groups"
elif 'age' not in df_original.columns: age_summary_text = 'N/A (column missing)'

st.caption(f"Displaying data for: Years {selected_year_range_global[0]}-{selected_year_range_global[1]} | Sex: {sex_summary_text} | Ages: {age_summary_text}")

# --- 2x2 Grid for Visuals (Keep the rest of the plotting code as it was) ---
if not df_filtered_global.empty:
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1: # Map
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
        
        map_sex_cond = pd.Series(True, index=df_original.index)
        if selected_sex_global and 'sex' in df_original.columns: map_sex_cond = df_original['sex'].isin(selected_sex_global)
        map_age_cond = pd.Series(True, index=df_original.index)
        if selected_age_global and 'age' in df_original.columns: map_age_cond = df_original['age'].isin(selected_age_global)

        map_data_source = df_original[
            (df_original['year'] == selected_map_year) & map_sex_cond & map_age_cond
        ].copy()

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

    with row1_col2: # Country Comparison
        st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
        available_countries_options = []
        if 'country' in df_filtered_global.columns: available_countries_options = sorted(df_filtered_global['country'].unique())
        
        selected_countries_compare = []
        if available_countries_options:
            default_countries = available_countries_options[:min(2, len(available_countries_options))]
            selected_countries_compare = st.multiselect("Compare:", available_countries_options, default=default_countries, key="country_compare_compact_v2", label_visibility="collapsed")
            st.caption(f"Comparing: {', '.join(selected_countries_compare) if selected_countries_compare else 'None'}")
        else:
            st.caption("No countries in filtered data.")

        if selected_countries_compare: 
            country_comp_data = df_filtered_global[df_filtered_global['country'].isin(selected_countries_compare)]
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
        elif available_countries_options : st.caption("Select countries to compare.")


    row2_col1, row2_col2 = st.columns(2)
    with row2_col1: # Demographic Breakdown
        st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
        dem_df = df_filtered_global.copy()
        tab_titles = ["Sex", "Age", "Sex&Age"]
        tab_sex, tab_age, tab_sex_age = st.tabs(tab_titles)
        common_fig_size = (5.5, 2.5); common_font_size = 7; legend_font_size = 'xx-small'

        with tab_sex:
            if 'sex' in dem_df.columns:
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
            else: st.caption("Sex data unavailable.")
        with tab_age:
            if 'age' in dem_df.columns and not dem_df['age'].empty:
                age_order = sorted(dem_df['age'].astype(str).dropna().unique(), key=age_sort_key)
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
            else: st.caption("Age data unavailable.")
        with tab_sex_age:
            if 'age' in dem_df.columns and not dem_df['age'].empty and 'sex' in dem_df.columns:
                age_order_sa = sorted(dem_df['age'].astype(str).dropna().unique(), key=age_sort_key)
                dem_df['age_cat_sa'] = pd.Categorical(dem_df['age'], categories=age_order_sa, ordered=True)
                sa_data = dem_df.groupby(['year', 'sex', 'age_cat_sa'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                sa_data['rate'] = np.where(sa_data['p'] > 0, (sa_data['s'] / sa_data['p']) * 100000, 0)
                if not sa_data.empty and sa_data['rate'].notna().any():
                    g = sns.relplot(data=sa_data, x='year', y='rate', hue='age_cat_sa', style='sex', kind='line', markers=True, height=2.2, aspect=1.5, facet_kws=dict(margin_titles=True), legend=False)
                    g.set_axis_labels("Year", "Rate", fontsize=common_font_size); g.set_titles(col_template="{col_name}", row_template="{row_name}", size=common_font_size)
                    g.tick_params(labelsize=common_font_size-1); plt.grid(True, linestyle=':', linewidth=0.5); g.tight_layout(pad=0.5); st.pyplot(g.fig)
                else: st.caption("No Sex&Age data for these filters.")
            else: st.caption("Sex&Age data unavailable.")

    with row2_col2: # Age Group Rate (Overall)
        st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
        if 'age' in df_filtered_global.columns and not df_filtered_global['age'].empty:
            age_dist = df_filtered_global.groupby('age').agg(s_total=('suicides_no', 'sum'), p_total=('population', 'sum')).reset_index()
            age_dist['rate'] = np.where(age_dist['p_total'] > 0, (age_dist['s_total'] / age_dist['p_total']) * 100000, 0)
            age_dist.dropna(subset=['rate'], inplace=True)
            if not age_dist.empty:
                age_order_bar = sorted(age_dist['age'].astype(str).dropna().unique(), key=age_sort_key)
                age_dist['age_cat'] = pd.Categorical(age_dist['age'], categories=age_order_bar, ordered=True)
                age_dist = age_dist.sort_values('age_cat')
                fig_ad, ax_ad = plt.subplots(figsize=(5.5, 2.7))
                sns.barplot(data=age_dist, x='age_cat', y='rate', ax=ax_ad, palette="coolwarm_r")
                ax_ad.set_xlabel('Age Group', fontsize=common_font_size); ax_ad.set_ylabel('Rate (per 100k)', fontsize=common_font_size)
                plt.xticks(rotation=45, ha="right"); ax_ad.tick_params(axis='both', which='major', labelsize=common_font_size-1)
                ax_ad.grid(True, axis='y', linestyle=':', linewidth=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig_ad)
            else: st.caption("No Age Dist. data for these filters.")
        else: st.caption("Age data unavailable.")
else:
    st.warning("No data available for the selected global filters. Please adjust filters in the sidebar.")
