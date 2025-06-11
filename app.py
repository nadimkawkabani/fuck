import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# import os # Not needed for the current simple file path strategy

# --- Page Config (apply once at the top) ---
st.set_page_config(
    page_title="Global Suicide Insights", # Updated title
    layout="wide",
    initial_sidebar_state="collapsed" # Start collapsed, will expand after login
)

# --- Password (VERY BASIC - NOT FOR PRODUCTION) ---
# In a real app, use st.secrets or environment variables
CORRECT_PASSWORD = "msba" # Change this to your desired password

# --- Helper Function for Age Sorting ---
def age_sort_key(age_str):
    if pd.isna(age_str): return -1
    age_str = str(age_str)
    try:
        return int(age_str.replace(' years', '').split('-')[0].replace('+', ''))
    except ValueError: return 999

# --- Load Data (Cached) ---
@st.cache_data
def load_data():
    file_path = "who_suicide_statistics.csv" 
    try:
        df_loaded = pd.read_csv(file_path)
    except FileNotFoundError:
        # This error will be shown on the page where load_data is called if it fails
        return None # Indicate failure
    except Exception:
        return None # Indicate failure

    if df_loaded is not None and not df_loaded.empty:
        try:
            required_cols = ['suicides_no', 'population', 'year', 'sex', 'age', 'country']
            missing_cols = [col for col in required_cols if col not in df_loaded.columns]
            if missing_cols:
                # This warning will be shown where load_data is called
                return f"Missing columns: {', '.join(missing_cols)}" 

            df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
            df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
            
            df_loaded['suicide_rate'] = np.where(
                (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()),
                (df_loaded['suicides_no'] / df_loaded['population']) * 100000, 0 
            )
            df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
            if 'age' in df_loaded.columns: df_loaded['age'] = df_loaded['age'].astype(str)
        except Exception:
            return "Data cleaning error" # Indicate failure
    elif df_loaded is None:
        return None
        
    return df_loaded

# --- Function to display the login page ---
def show_login_page():
    st.image("https://www.middleeasteye.net/sites/default/files/styles/free_desktop_article_page_horizontal/public/images/MEA%20Logo.png?itok=080V0XH-", width=200) # Replace with your logo URL or path
    st.title("MEA - Passenger Surveys Data Explorer") # Example Title
    st.subheader("Dashboard Access")

    password_attempt = st.text_input("Please enter the password:", type="password", key="password_input")
    
    if st.button("Login", key="login_button"):
        if password_attempt == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            st.session_state["current_page"] = "Dashboard" # Default page after login
            # Clear the password input by rerunning, which resets widget state if key changes or not present
            st.rerun() 
        else:
            st.error("Password incorrect. Please try again.")
    st.caption("Hint: The password is 'msba'")


# --- Function to display the main dashboard content ---
def display_dashboard(df_original):
    st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")
    
    # --- Sidebar Filters (Specific to this "page") ---
    st.sidebar.markdown("### Dashboard Filters")
    min_year_global, max_year_global = int(df_original['year'].min()), int(df_original['year'].max())
    selected_year_range_global = st.sidebar.slider(
        "Year Range:", min_year_global, max_year_global,
        (max(min_year_global, max_year_global - 5), max_year_global),
        key="dash_year_slider"
    )

    options_sex, default_sex, selected_sex_global = [], [], []
    if 'sex' in df_original.columns and df_original['sex'].notna().any():
        options_sex = sorted(list(df_original['sex'].dropna().unique()))
        default_sex = list(options_sex)
    selected_sex_global = st.sidebar.multiselect(
        "Sex:", options=options_sex, default=default_sex,
        key="dash_sex_multi", disabled=not bool(options_sex)
    )

    options_ages, default_ages, selected_age_global = [], [], []
    if 'age' in df_original.columns and df_original['age'].notna().any():
        options_ages = sorted(list(df_original['age'].astype(str).dropna().unique()), key=age_sort_key)
        default_ages = list(options_ages)
    selected_age_global = st.sidebar.multiselect(
        "Age Groups:", options=options_ages, default=default_ages,
        key="dash_age_multi", disabled=not bool(options_ages)
    )
    
    # Apply filters
    conditions = [(df_original['year'] >= selected_year_range_global[0]) & (df_original['year'] <= selected_year_range_global[1])]
    if selected_sex_global and 'sex' in df_original.columns: conditions.append(df_original['sex'].isin(selected_sex_global))
    if selected_age_global and 'age' in df_original.columns: conditions.append(df_original['age'].isin(selected_age_global))
    final_condition = pd.Series(True, index=df_original.index); 
    for cond in conditions: final_condition &= cond
    df_filtered_global = df_original[final_condition].copy()

    # Filter summary caption
    sex_summary = 'All' if not selected_sex_global or len(selected_sex_global) == len(options_sex) else ', '.join(selected_sex_global)
    age_summary = 'All Ages' if not selected_age_global or len(selected_age_global) == len(options_ages) else f"{len(selected_age_global)} groups"
    st.caption(f"Displaying data for: Years {selected_year_range_global[0]}-{selected_year_range_global[1]} | Sex: {sex_summary} | Ages: {age_summary}")

    if df_filtered_global.empty:
        st.warning("No data matches the current filter selection.")
        return # Exit if no data to plot

    # --- 2x2 Grid for Visuals ---
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1: # Map
        st.markdown("<h6>🗺️ Geographic Distribution</h6>", unsafe_allow_html=True)
        map_year_options = sorted(df_original['year'].dropna().unique(), reverse=True)
        default_map_idx = map_year_options.index(selected_year_range_global[1]) if selected_year_range_global[1] in map_year_options else 0
        selected_map_year = st.selectbox("Map Year:", map_year_options, index=default_map_idx, key="dash_map_year_sb", label_visibility="collapsed")
        st.caption(f"Map showing year: {selected_map_year}")

        map_sex_cond = pd.Series(True, index=df_original.index)
        if selected_sex_global and 'sex' in df_original.columns: map_sex_cond = df_original['sex'].isin(selected_sex_global)
        map_age_cond = pd.Series(True, index=df_original.index)
        if selected_age_global and 'age' in df_original.columns: map_age_cond = df_original['age'].isin(selected_age_global)
        
        map_data_source = df_original[(df_original['year'] == selected_map_year) & map_sex_cond & map_age_cond].copy()
        if 'country' in map_data_source.columns:
            country_map_data = map_data_source.groupby('country').agg(s_no=('suicides_no', 'sum'), pop=('population', 'sum')).reset_index()
            country_map_data['rate'] = np.where(country_map_data['pop'] > 0, (country_map_data['s_no'] / country_map_data['pop']) * 100000, 0)
            country_map_data.dropna(subset=['rate'], inplace=True)
            country_name_mapping = {"United States of America": "United States", "Russian Federation": "Russia", "Republic of Korea": "South Korea", "Iran (Islamic Rep of)": "Iran", "Czech Republic": "Czechia"}
            country_map_data['country_std'] = country_map_data['country'].replace(country_name_mapping)
            if not country_map_data.empty:
                fig_map = px.choropleth(country_map_data, locations="country_std", locationmode="country names", color="rate", hover_name="country", hover_data={"rate": ":.1f", "s_no": True}, color_continuous_scale=px.colors.sequential.OrRd)
                fig_map.update_layout(margin={"r":5,"t":5,"l":5,"b":5}, height=280, geo=dict(bgcolor= 'rgba(0,0,0,0)'), coloraxis_colorbar=dict(title="Rate", thickness=10, len=0.7, yanchor="middle", y=0.5, tickfont=dict(size=8)))
                st.plotly_chart(fig_map, use_container_width=True)
            else: st.caption("No map data.")
        else: st.caption("Country data missing for map.")

    with row1_col2: # Country Comparison
        st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
        plot_countries_options = sorted(df_filtered_global['country'].unique()) if 'country' in df_filtered_global.columns else []
        selected_plot_countries = []
        if plot_countries_options:
            default_plot_countries = plot_countries_options[:min(2, len(plot_countries_options))]
            selected_plot_countries = st.multiselect("Compare:", plot_countries_options, default=default_plot_countries, key="dash_country_multi", label_visibility="collapsed")
            st.caption(f"Comparing: {', '.join(selected_plot_countries) if selected_plot_countries else 'None'}")
        if selected_plot_countries:
            plot_data = df_filtered_global[df_filtered_global['country'].isin(selected_plot_countries)].groupby(['country', 'year']).agg(s_no=('suicides_no', 'sum'), pop=('population', 'sum')).reset_index()
            plot_data['rate'] = np.where(plot_data['pop'] > 0, (plot_data['s_no'] / plot_data['pop']) * 100000, 0)
            if not plot_data.empty:
                fig_cc, ax_cc = plt.subplots(figsize=(5.5, 2.7)); sns.lineplot(data=plot_data, x='year', y='rate', hue='country', ax=ax_cc, marker='o', markersize=3)
                ax_cc.set_ylabel('Rate', fontsize=8); ax_cc.set_xlabel('Year', fontsize=8); ax_cc.tick_params(labelsize=7); ax_cc.legend(fontsize='xx-small', loc='best', frameon=False); ax_cc.grid(True, ls=':', lw=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig_cc)
            else: st.caption("No data for country plot.")
        elif plot_countries_options: st.caption("Select countries.")
        else: st.caption("No countries to compare in filtered data.")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1: # Demographics
        st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
        dem_df = df_filtered_global.copy()
        tabs = st.tabs(["Sex", "Age", "Sex & Age"])
        sml_fig, sml_fs, leg_fs = (5.5, 2.5), 7, 'xx-small'
        with tabs[0]: # Sex
            if 'sex' in dem_df.columns:
                data = dem_df.groupby(['year', 'sex']).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                data['rate'] = np.where(data['p'] > 0, (data['s'] / data['p']) * 100000, 0)
                if not data.empty and data['rate'].notna().any():
                    fig, ax = plt.subplots(figsize=sml_fig); sns.lineplot(data=data, x='year', y='rate', hue='sex', ax=ax, marker='o', markersize=3)
                    ax.set_title('Rate by Sex', fontsize=sml_fs+1); ax.grid(True, ls=':', lw=0.5); ax.set_ylabel('Rate', fontsize=sml_fs); ax.set_xlabel('Year', fontsize=sml_fs); ax.tick_params(labelsize=sml_fs-1); ax.legend(fontsize=leg_fs, title='', frameon=False, loc='best'); plt.tight_layout(pad=0.5); st.pyplot(fig)
                else: st.caption("No Sex data.")
            else: st.caption("Sex data unavailable.")
        with tabs[1]: # Age
            if 'age' in dem_df.columns and not dem_df['age'].empty:
                order = sorted(dem_df['age'].astype(str).dropna().unique(), key=age_sort_key)
                dem_df['age_cat'] = pd.Categorical(dem_df['age'], categories=order, ordered=True)
                data = dem_df.groupby(['year', 'age_cat'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                data['rate'] = np.where(data['p'] > 0, (data['s'] / data['p']) * 100000, 0)
                if not data.empty and data['rate'].notna().any():
                    fig, ax = plt.subplots(figsize=sml_fig); sns.lineplot(data=data, x='year', y='rate', hue='age_cat', ax=ax, marker='o', markersize=3)
                    ax.set_title('Rate by Age', fontsize=sml_fs+1); ax.legend(title='', fontsize=leg_fs, loc='best', frameon=False, ncol=2); ax.grid(True, ls=':', lw=0.5); ax.set_ylabel('Rate', fontsize=sml_fs); ax.set_xlabel('Year', fontsize=sml_fs); ax.tick_params(labelsize=sml_fs-1); plt.tight_layout(pad=0.5); st.pyplot(fig)
                else: st.caption("No Age data.")
            else: st.caption("Age data unavailable.")
        with tabs[2]: # Sex & Age
            if 'age' in dem_df.columns and not dem_df['age'].empty and 'sex' in dem_df.columns:
                order = sorted(dem_df['age'].astype(str).dropna().unique(), key=age_sort_key)
                dem_df['age_cat_sa'] = pd.Categorical(dem_df['age'], categories=order, ordered=True)
                data = dem_df.groupby(['year', 'sex', 'age_cat_sa'], observed=False).agg(s=('suicides_no', 'sum'), p=('population', 'sum')).reset_index()
                data['rate'] = np.where(data['p'] > 0, (data['s'] / data['p']) * 100000, 0)
                if not data.empty and data['rate'].notna().any():
                    g = sns.relplot(data=data, x='year', y='rate', hue='age_cat_sa', style='sex', kind='line', markers=True, height=2.2, aspect=1.5, facet_kws=dict(margin_titles=True), legend=False)
                    g.set_axis_labels("Year","Rate",fontsize=sml_fs); g.set_titles(size=sml_fs); g.tick_params(labelsize=sml_fs-1); plt.grid(True,ls=':',lw=0.5); g.tight_layout(pad=0.5); st.pyplot(g.fig)
                else: st.caption("No Sex&Age data.")
            else: st.caption("Sex&Age data unavailable.")

    with row2_col2: # Age Group Overall
        st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
        if 'age' in df_filtered_global.columns and not df_filtered_global['age'].empty:
            data = df_filtered_global.groupby('age').agg(s_total=('suicides_no', 'sum'), p_total=('population', 'sum')).reset_index()
            data['rate'] = np.where(data['p_total'] > 0, (data['s_total'] / data['p_total']) * 100000, 0)
            data.dropna(subset=['rate'], inplace=True)
            if not data.empty:
                order = sorted(data['age'].astype(str).dropna().unique(), key=age_sort_key)
                data['age_cat'] = pd.Categorical(data['age'], categories=order, ordered=True)
                data = data.sort_values('age_cat')
                fig, ax = plt.subplots(figsize=(5.5, 2.7)); sns.barplot(data=data, x='age_cat', y='rate', ax=ax, palette="coolwarm_r")
                ax.set_xlabel('Age Group',fontsize=sml_fs); ax.set_ylabel('Rate',fontsize=sml_fs); plt.xticks(rotation=45,ha="right"); ax.tick_params(labelsize=sml_fs-1); ax.grid(True,axis='y',ls=':',lw=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig)
            else: st.caption("No Age Dist. data.")
        else: st.caption("Age data unavailable.")

# --- Function for another "page" (Example) ---
def display_data_overview(df_original):
    st.title("📄 Data Overview & Exploration")
    st.markdown("Explore the raw dataset and its structure.")
    
    if st.checkbox("Show Raw Data Sample", key="overview_raw_cb"):
        st.dataframe(df_original.sample(min(100, len(df_original))))
    
    if st.checkbox("Show Data Summary (describe)", key="overview_describe_cb"):
        st.write(df_original.describe(include='all'))
        
    if st.checkbox("Show Missing Value Counts", key="overview_missing_cb"):
        missing_counts = df_original.isnull().sum()
        missing_percent = (missing_counts / len(df_original)) * 100
        missing_df = pd.DataFrame({'Missing Values': missing_counts, '% Missing': missing_percent})
        st.dataframe(missing_df[missing_df['Missing Values'] > 0].sort_values(by='% Missing', ascending=False))

# --- Main app flow ---
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

if not st.session_state["password_correct"]:
    show_login_page()
else:
    # Load data only after successful login (or ensure it's loaded once)
    df_data = load_data() # Call it again or ensure it was loaded globally
    
    if df_data is None:
        st.error("Critical error: Data could not be loaded. Please check file path and integrity.")
        st.stop()
    elif isinstance(df_data, str): # Means load_data returned an error message
        st.error(f"Data Loading/Processing Error: {df_data}")
        st.stop()
    elif df_data.empty:
        st.error("Data loaded but is empty. Cannot proceed.")
        st.stop()

    # Sidebar Navigation
    st.sidebar.image("https://www.middleeasteye.net/sites/default/files/styles/free_desktop_article_page_horizontal/public/images/MEA%20Logo.png?itok=080V0XH-", width=100) # Smaller logo in sidebar
    st.sidebar.title("Navigation")
    
    # Initialize current_page if not set (e.g., after login)
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"

    # Radio buttons for navigation
    page_options = ["Dashboard", "Data Overview"] # Add more page names here
    st.session_state["current_page"] = st.sidebar.radio(
        "Go to", 
        page_options, 
        index=page_options.index(st.session_state["current_page"]), # Keep current page selected
        key="navigation_radio"
    )
    
    if st.sidebar.button("Logout", key="logout_main_button"):
        st.session_state["password_correct"] = False
        st.session_state["current_page"] = None # Reset page
        if "password_input" in st.session_state: # Attempt to clear password field
            del st.session_state["password_input"]
        st.rerun()

    # Display the selected page
    if st.session_state["current_page"] == "Dashboard":
        display_dashboard(df_data)
    elif st.session_state["current_page"] == "Data Overview":
        display_data_overview(df_data)
    # Add more elif for other pages
    # elif st.session_state["current_page"] == "Another Page":
    #     display_another_page_function(df_data)
