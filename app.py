import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
print("hello")
# --- Page Config (apply once at the top) ---
st.set_page_config(
    page_title="Global Suicide Insights", 
    layout="wide",
    initial_sidebar_state="collapsed" # Start collapsed, will expand after login if needed by dashboard
)

# --- Password (VERY BASIC - NOT FOR PRODUCTION) ---
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
    df_loaded = None
    try:
        df_loaded = pd.read_csv(file_path)
    except FileNotFoundError:
        # Error will be handled by the main app logic after this function returns
        return None 
    except Exception:
        return None

    if df_loaded is not None and not df_loaded.empty:
        try:
            required_cols = ['suicides_no', 'population', 'year', 'sex', 'age', 'country']
            missing_cols = [col for col in required_cols if col not in df_loaded.columns]
            if missing_cols:
                # Return error string to be handled by main app
                return f"FATAL ERROR (load_data): Missing required columns in CSV: {', '.join(missing_cols)}"

            df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
            df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
            
            df_loaded['suicide_rate'] = np.where(
                (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()),
                (df_loaded['suicides_no'] / df_loaded['population']) * 100000, 0 
            )
            df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
            if 'age' in df_loaded.columns: df_loaded['age'] = df_loaded['age'].astype(str)
        except Exception as e:
            return f"Error during data cleaning (load_data): {e}" # Return error string
    elif df_loaded is None:
        return None # Should be caught by initial try-except
        
    return df_loaded

# --- Function to display the login elements ---
def show_login_interface():
    # Simulate the MEA login page style within a column for centering
    # You might need to adjust widths and add more CSS for exact replication
    
    # Attempt to hide the default Streamlit hamburger menu and footer -
    # This is a bit of a hack and might not always work or be desirable.
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #f0f2f6; /* Light grey background like MEA example */
            }
            .login-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 80vh; /* Adjust height as needed */
            }
            .login-box {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                width: 350px; /* Adjust width */
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True
    )

    # Centering content hack for login page
    # col1, centered_col, col3 = st.columns([1,1.5,1]) # Adjust ratios for centering
    # with centered_col:
    # Using markdown for centering as columns can be tricky with full page feel
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<div class="login-box">', unsafe_allow_html=True)

    # Placeholder for a logo if you have one
    # st.image("your_logo.png", width=150) # Replace with your logo
    st.markdown("## Suicide Statistics Dashboard") # Main title for login page
    st.markdown("Access Panel")
    st.markdown("---")


    password_attempt = st.text_input("Password:", type="password", key="password_input_login", label_visibility="collapsed", placeholder="Enter password")
    
    login_button_pressed = st.button("Login", key="login_button_main", use_container_width=True)

    if login_button_pressed:
        if password_attempt == CORRECT_PASSWORD:
            st.session_state["password_correct"] = True
            # No need to set current_page, the main app will just render
            st.rerun() 
        else:
            st.error("Password incorrect.")
    
    st.caption("Hint: msba")
    st.markdown("</div>", unsafe_allow_html=True) # Close login-box
    st.markdown("</div>", unsafe_allow_html=True) # Close login-container


# --- Function to display the main dashboard content (your existing 2x2 grid) ---
def display_main_dashboard(df_original_data):
    # Re-enable sidebar for filters for the dashboard page
    st.set_page_config(initial_sidebar_state="expanded") # Show sidebar for dashboard

    # --- Sidebar Filters ---
    st.sidebar.markdown("### Filters")
    min_year, max_year = int(df_original_data['year'].min()), int(df_original_data['year'].max())
    selected_years = st.sidebar.slider(
        "Year Range:", min_year, max_year, (max(min_year, max_year - 5), max_year),
        key="main_dash_year_slider"
    )

    options_sex, default_sex, selected_sex = [], [], []
    if 'sex' in df_original_data.columns and df_original_data['sex'].notna().any():
        options_sex = sorted(list(df_original_data['sex'].dropna().unique()))
        default_sex = list(options_sex)
    selected_sex = st.sidebar.multiselect("Sex:", options_sex, default_sex, key="main_dash_sex_multi", disabled=not bool(options_sex))

    options_ages, default_ages, selected_ages = [], [], []
    if 'age' in df_original_data.columns and df_original_data['age'].notna().any():
        options_ages = sorted(list(df_original_data['age'].astype(str).dropna().unique()), key=age_sort_key)
        default_ages = list(options_ages)
    selected_ages = st.sidebar.multiselect("Age Groups:", options_ages, default_ages, key="main_dash_age_multi", disabled=not bool(options_ages))
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Suicide Data Insights")
    if st.sidebar.button("Logout", key="dashboard_logout_button"):
        st.session_state["password_correct"] = False
        if "password_input_login" in st.session_state:
            del st.session_state["password_input_login"] # Attempt to clear if key exists
        st.rerun()


    # Apply filters
    conds = [(df_original_data['year'] >= selected_years[0]) & (df_original_data['year'] <= selected_years[1])]
    if selected_sex and 'sex' in df_original_data.columns: conds.append(df_original_data['sex'].isin(selected_sex))
    if selected_ages and 'age' in df_original_data.columns: conds.append(df_original_data['age'].isin(selected_ages))
    final_cond = pd.Series(True, index=df_original_data.index); 
    for c in conds: final_cond &= c
    df_filtered = df_original_data[final_cond].copy()

    # --- Main Dashboard Area ---
    st.markdown("#### 🎯 Focused Suicide Statistics Dashboard")
    sex_sum = 'All' if not selected_sex or len(selected_sex) == len(options_sex) else ','.join(selected_sex)
    age_sum = 'All' if not selected_ages or len(selected_ages) == len(options_ages) else f"{len(selected_ages)} groups"
    st.caption(f"Years: {selected_years[0]}-{selected_years[1]} | Sex: {sex_sum} | Ages: {age_sum}")

    if df_filtered.empty:
        st.warning("No data matches current filters. Please adjust sidebar filters.")
        return

    # --- 2x2 Grid for Visuals ---
    # (The plotting code from your previous working version goes here)
    # For brevity, I'll use placeholders, but you should paste your full 2x2 grid code
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1: # Map
        st.markdown("<h6>🗺️ Geographic Distribution</h6>", unsafe_allow_html=True)
        map_yr_opts = sorted(df_original_data['year'].dropna().unique(), reverse=True)
        def_map_idx = map_yr_opts.index(selected_years[1]) if selected_years[1] in map_yr_opts else 0
        sel_map_yr = st.selectbox("Map Year:", map_yr_opts, index=def_map_idx, key="main_dash_map_year", label_visibility="collapsed")
        st.caption(f"Map for year: {sel_map_yr}")

        map_sex_c = pd.Series(True, index=df_original_data.index)
        if selected_sex and 'sex' in df_original_data.columns: map_sex_c = df_original_data['sex'].isin(selected_sex)
        map_age_c = pd.Series(True, index=df_original_data.index)
        if selected_ages and 'age' in df_original_data.columns: map_age_c = df_original_data['age'].isin(selected_ages)
        
        map_src = df_original_data[(df_original_data['year'] == sel_map_yr) & map_sex_c & map_age_c].copy()
        if 'country' in map_src.columns:
            country_map = map_src.groupby('country').agg(s=('suicides_no','sum'),p=('population','sum')).reset_index()
            country_map['rate'] = np.where(country_map['p']>0,(country_map['s']/country_map['p'])*100000,0)
            country_map.dropna(subset=['rate'],inplace=True)
            name_map = {"United States of America":"US","Russian Federation":"Russia","Republic of Korea":"South Korea"} # Keep short
            country_map['country_std'] = country_map['country'].replace(name_map)
            if not country_map.empty:
                fig = px.choropleth(country_map,locations="country_std",locationmode="country names",color="rate",hover_name="country",color_continuous_scale=px.colors.sequential.OrRd)
                fig.update_layout(margin={"r":5,"t":5,"l":5,"b":5},height=280,geo=dict(bgcolor='rgba(0,0,0,0)'),coloraxis_colorbar=dict(title="Rate",thickness=10,len=0.7,tickfont=dict(size=8)))
                st.plotly_chart(fig, use_container_width=True)
            else: st.caption("No map data.")
        else: st.caption("Country data missing.")

    with row1_c2: # Country Comparison
        st.markdown("<h6>🆚 Country Comparison</h6>", unsafe_allow_html=True)
        country_opts = sorted(df_filtered['country'].unique()) if 'country' in df_filtered.columns else []
        sel_countries = []
        if country_opts:
            def_countries = country_opts[:min(2,len(country_opts))]
            sel_countries = st.multiselect("Compare:",country_opts,default=def_countries,key="main_dash_country_comp",label_visibility="collapsed")
            st.caption(f"Comparing: {', '.join(sel_countries) if sel_countries else 'None'}")
        if sel_countries:
            data = df_filtered[df_filtered['country'].isin(sel_countries)].groupby(['country','year']).agg(s=('suicides_no','sum'),p=('population','sum')).reset_index()
            data['rate'] = np.where(data['p']>0,(data['s']/data['p'])*100000,0)
            if not data.empty:
                fig,ax=plt.subplots(figsize=(5.5,2.7)); sns.lineplot(data=data,x='year',y='rate',hue='country',ax=ax,marker='o',ms=3)
                ax.set_ylabel('Rate',fontsize=8); ax.set_xlabel('Year',fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize='xx-small',loc='best',frameon=False); ax.grid(True,ls=':',lw=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig)
            else: st.caption("No data.")
        elif country_opts: st.caption("Select countries.")
        else: st.caption("No countries in filtered data.")
        
    row2_c1, row2_c2 = st.columns(2)
    with row2_c1: # Demographics
        st.markdown("<h6>🧑‍🤝‍🧑 Demographic Breakdown</h6>", unsafe_allow_html=True)
        dem_df = df_filtered.copy(); tabs = st.tabs(["Sex", "Age", "Sex&Age"]); fs, lfs = (5.5,2.5), 7
        with tabs[0]: # Sex
            if 'sex' in dem_df.columns:
                data=dem_df.groupby(['year','sex']).agg(s=('suicides_no','sum'),p=('population','sum')).reset_index(); data['rate']=np.where(data['p']>0,(data['s']/data['p'])*100000,0)
                if not data.empty and data['rate'].notna().any():
                    fig,ax=plt.subplots(figsize=fs); sns.lineplot(data=data,x='year',y='rate',hue='sex',ax=ax,marker='o',ms=3); ax.set_title('Rate by Sex',fontsize=lfs+1); ax.grid(True,ls=':',lw=0.5); ax.set_ylabel('Rate',fontsize=lfs); ax.set_xlabel('Year',fontsize=lfs); ax.tick_params(labelsize=lfs-1); ax.legend(fontsize='xx-small',title='',frameon=False,loc='best'); plt.tight_layout(pad=0.5); st.pyplot(fig)
                else: st.caption("No Sex data.")
            else: st.caption("Sex data unavailable.")
        with tabs[1]: # Age
            if 'age' in dem_df.columns and not dem_df['age'].empty:
                order=sorted(dem_df['age'].astype(str).dropna().unique(),key=age_sort_key); dem_df['age_cat']=pd.Categorical(dem_df['age'],categories=order,ordered=True)
                data=dem_df.groupby(['year','age_cat'],observed=False).agg(s=('suicides_no','sum'),p=('population','sum')).reset_index(); data['rate']=np.where(data['p']>0,(data['s']/data['p'])*100000,0)
                if not data.empty and data['rate'].notna().any():
                    fig,ax=plt.subplots(figsize=fs); sns.lineplot(data=data,x='year',y='rate',hue='age_cat',ax=ax,marker='o',ms=3); ax.set_title('Rate by Age',fontsize=lfs+1); ax.legend(title='',fontsize='xx-small',loc='best',frameon=False,ncol=2); ax.grid(True,ls=':',lw=0.5); ax.set_ylabel('Rate',fontsize=lfs); ax.set_xlabel('Year',fontsize=lfs); ax.tick_params(labelsize=lfs-1); plt.tight_layout(pad=0.5); st.pyplot(fig)
                else: st.caption("No Age data.")
            else: st.caption("Age data unavailable.")
        with tabs[2]: # Sex & Age
            if 'age' in dem_df.columns and not dem_df['age'].empty and 'sex' in dem_df.columns:
                order=sorted(dem_df['age'].astype(str).dropna().unique(),key=age_sort_key); dem_df['age_cat_sa']=pd.Categorical(dem_df['age'],categories=order,ordered=True)
                data=dem_df.groupby(['year','sex','age_cat_sa'],observed=False).agg(s=('suicides_no','sum'),p=('population','sum')).reset_index(); data['rate']=np.where(data['p']>0,(data['s']/data['p'])*100000,0)
                if not data.empty and data['rate'].notna().any():
                    g=sns.relplot(data=data,x='year',y='rate',hue='age_cat_sa',style='sex',kind='line',markers=True,height=2.2,aspect=1.5,facet_kws=dict(margin_titles=True),legend=False)
                    g.set_axis_labels("Year","Rate",fontsize=lfs); g.set_titles(size=lfs); g.tick_params(labelsize=lfs-1); plt.grid(True,ls=':',lw=0.5); g.tight_layout(pad=0.5); st.pyplot(g.fig)
                else: st.caption("No Sex&Age data.")
            else: st.caption("Sex&Age data unavailable.")

    with row2_c2: # Age Group Overall
        st.markdown("<h6>📊 Age Group Rate (Overall)</h6>", unsafe_allow_html=True)
        if 'age' in df_filtered.columns and not df_filtered['age'].empty:
            data=df_filtered.groupby('age').agg(s_total=('suicides_no','sum'),p_total=('population','sum')).reset_index(); data['rate']=np.where(data['p_total']>0,(data['s_total']/data['p_total'])*100000,0)
            data.dropna(subset=['rate'],inplace=True)
            if not data.empty:
                order=sorted(data['age'].astype(str).dropna().unique(),key=age_sort_key); data['age_cat']=pd.Categorical(data['age'],categories=order,ordered=True); data=data.sort_values('age_cat')
                fig,ax=plt.subplots(figsize=(5.5,2.7)); sns.barplot(data=data,x='age_cat',y='rate',ax=ax,palette="coolwarm_r"); ax.set_xlabel('Age Group',fontsize=lfs); ax.set_ylabel('Rate',fontsize=lfs); plt.xticks(rotation=45,ha="right"); ax.tick_params(labelsize=lfs-1); ax.grid(True,axis='y',ls=':',lw=0.5); plt.tight_layout(pad=0.5); st.pyplot(fig)
            else: st.caption("No Age Dist. data.")
        else: st.caption("Age data unavailable.")

# --- Main App Execution Flow ---
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

# Load data ONCE and store it, or retrieve from session state if already loaded
# This is to avoid reloading data on every rerun if login fails multiple times
if "df_main_data" not in st.session_state:
    loaded_df_or_error = load_data()
    if isinstance(loaded_df_or_error, pd.DataFrame) and not loaded_df_or_error.empty:
        st.session_state["df_main_data"] = loaded_df_or_error
        st.session_state["data_load_error"] = None
    elif isinstance(loaded_df_or_error, str): # Error message string
        st.session_state["df_main_data"] = None
        st.session_state["data_load_error"] = loaded_df_or_error
    else: # None or empty DataFrame
        st.session_state["df_main_data"] = None
        st.session_state["data_load_error"] = "Data could not be loaded or is empty."

# Check login status
if not st.session_state["password_correct"]:
    # Clear sidebar before showing login page for a cleaner look
    # This is a bit of a trick: render an empty sidebar then the login page
    with st.sidebar:
        st.empty() # Clears previous sidebar content if any
    show_login_interface()
else:
    # Password is correct, proceed to dashboard
    if st.session_state["data_load_error"]:
        st.error(st.session_state["data_load_error"])
        if st.button("Logout from Error Screen"): # Allow logout even if data failed
            st.session_state["password_correct"] = False
            if "password_input_login" in st.session_state:
                del st.session_state["password_input_login"]
            st.rerun()
        st.stop()
    elif st.session_state["df_main_data"] is not None:
        display_main_dashboard(st.session_state["df_main_data"])
    else: # Should not be reached if logic is correct, but as a fallback
        st.error("An unexpected error occurred with data handling after login.")
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            if "password_input_login" in st.session_state:
                del st.session_state["password_input_login"]
            st.rerun()
        st.stop()
