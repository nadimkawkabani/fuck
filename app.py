import streamlit as st # MUST BE AT THE TOP, AND 'st' NOT REASSIGNED
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
        # Use st.error directly here. If 'st' is the problem, this will also fail.
        st.error(f"FATAL ERROR (load_data): Data file '{file_path}' not found.")
        return pd.DataFrame() 
    except Exception as e:
        st.error(f"Error loading data (load_data): {e}")
        return pd.DataFrame()

    if not df_loaded.empty:
        try:
            # Check if columns exist before processing
            if 'suicides_no' in df_loaded.columns: 
                df_loaded['suicides_no'] = pd.to_numeric(df_loaded['suicides_no'], errors='coerce')
            else:
                st.warning("Column 'suicides_no' missing in CSV.")
            
            if 'population' in df_loaded.columns: 
                df_loaded['population'] = pd.to_numeric(df_loaded['population'], errors='coerce')
            else:
                st.warning("Column 'population' missing in CSV.")
            
            if 'population' in df_loaded.columns and df_loaded['population'].hasnans is False and \
               'suicides_no' in df_loaded.columns and df_loaded['suicides_no'].hasnans is False:
                df_loaded['suicide_rate'] = np.where(
                    (df_loaded['population'] > 0) & (df_loaded['suicides_no'].notna()), # Double check notna
                    (df_loaded['suicides_no'] / df_loaded['population']) * 100000, 0 
                )
            elif 'population' in df_loaded.columns and 'suicides_no' in df_loaded.columns: # If columns exist but had NaNs
                st.warning("NaNs found in 'population' or 'suicides_no', setting 'suicide_rate' to 0 for affected rows.")
                df_loaded['suicide_rate'] = 0 # Or handle NaNs as you see fit
            else: # If essential columns for rate are missing
                df_loaded['suicide_rate'] = 0 
            
            df_loaded.replace([np.inf, -np.inf], np.nan, inplace=True)
            if 'age' in df_loaded.columns: 
                df_loaded['age'] = df_loaded['age'].astype(str)
            else:
                st.warning("Column 'age' missing in CSV.")
        except Exception as e:
            st.error(f"Error during data cleaning (load_data): {e}")
            return pd.DataFrame()
    return df_loaded

# --- Main Application Logic ---
# Ensure 'st' is THE streamlit module here.
# DO NOT REASSIGN 'st' to anything else in the lines between import and here.
df_original = load_data() # This call to load_data might use 'st.error'

if df_original.empty:
    # If st.error in load_data already showed a message, this might be redundant
    # but it's a fallback.
    st.error("App cannot start: data loading or initial processing failed. Check messages above.")
    st.stop()

# --- Sidebar Filters ---
# Critical section where the error occurs
st.sidebar.markdown("### Filters") # This uses 'st'

# Year Filter - More robust initialization
min_year_global = 1900 # Safe default
max_year_global = 2024 # Safe default
can_create_year_slider = False

# Check if 'st' is still the streamlit module BEFORE line 81
# You could add: st.write(type(st)) to debug locally if it were possible.

if 'year' in df_original.columns and not df_original['year'].empty:
    min_val_series = df_original['year'].min()
    max_val_series = df_original['year'].max()
    if pd.notna(min_val_series) and pd.notna(max_val_series):
        min_year_global = int(min_val_series)
        max_year_global = int(max_val_series)
        if min_year_global <= max_year_global:
            can_create_year_slider = True
        else:
            # THIS IS AROUND LINE 81
            st.sidebar.error("Min year is greater than max year in data. Check data integrity.") # Line A
    else:
        # THIS IS ALSO AROUND LINE 81
        st.sidebar.error("Year column contains NaNs affecting min/max. Cannot create year filter.") # Line B
else:
    # AND THIS IS ALSO AROUND LINE 81 - THIS IS THE ONE FROM YOUR TRACEBACK
    st.sidebar.error("Year column missing or empty. Cannot create year filter.") # Line C - YOUR ERROR
    # If this block is reached, can_create_year_slider remains False

if can_create_year_slider:
    selected_year_range_global = st.sidebar.slider(
        "Year Range:",
        min_year_global, 
        max_year_global, 
        (max(min_year_global, max_year_global - 5), max_year_global),
        key="global_year_slider_sidebar"
    )
else:
    st.sidebar.warning("Year filter cannot be displayed. Using placeholder range.")
    selected_year_range_global = (1900, 2000) 
    # Consider st.stop() if a valid year range is absolutely critical for the rest of the app
    # For now, we let it continue with a placeholder.

# ... (Rest of the code for Sex Filter, Age Filter, applying filters, and plotting) ...
# (Keep the rest of the code for plotting identical to the last working version 
#  you had, as the error is localized to the filter setup)

# --- [PASTE THE REST OF YOUR WORKING PLOTTING CODE HERE] ---
# Starting from:
# Sex Filter
options_sex = []
default_sex = []
# ... and so on ...
# To the end of the file.
# --- [END OF PASTED CODE] ---
