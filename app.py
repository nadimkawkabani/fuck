# app.py - Main application file
import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main application with options to run different parts"""
    st.set_page_config(
        page_title="EST Assessment System",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 EST Assessment System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Data Processing")
        st.markdown("""
        Process the Excel data and create:
        - Excel file with formulas
        - Cleaned CSV file
        """)
        if st.button("Run Data Processing", key="process"):
            import subprocess
            with st.spinner("Processing data..."):
                result = subprocess.run([sys.executable, "data_processor.py"], 
                                      capture_output=True, text=True)
                st.text_area("Processing Output", result.stdout, height=200)
    
    with col2:
        st.markdown("### 📈 Interactive Dashboard")
        st.markdown("""
        Launch the interactive dashboard:
        - Visual analytics
        - School comparisons
        - Student details
        - Data quality checks
        """)
        if st.button("Launch Dashboard", key="dashboard"):
            st.success("Dashboard will open in a new tab")
            st.info("Run: streamlit run dashboard.py")
    
    with col3:
        st.markdown("### 📋 System Information")
        st.markdown("""
        **Files to run:**
        1. `data_processor.py` - Process data
        2. `dashboard.py` - Interactive dashboard
        
        **Requirements:**
        - pandas
        - openpyxl
        - streamlit
        - plotly
        """)
    
    st.markdown("---")
    
    # Quick instructions
    with st.expander("📖 Quick Instructions", expanded=True):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Run Data Processing First:**
           ```bash
           python data_processor.py
           ```
           This creates:
           - `EST_Final_With_Formulas.xlsx` (Excel with formulas)
           - `final_data_cleaned.csv` (Cleaned data)
        
        2. **Launch Dashboard:**
           ```bash
           streamlit run dashboard.py
           ```
           Then open http://localhost:8501 in your browser
        
        3. **Use the Dashboard:**
           - Filter by school, grade, or score
           - View performance analytics
           - Check data quality issues
           - Export reports
        
        ### File Structure:
        ```
        est_assessment/
        ├── data_processor.py    # Data processing script
        ├── dashboard.py         # Streamlit dashboard
        ├── app.py              # This main app
        ├── requirements.txt     # Dependencies
        ├── Employment Test - Dataset - TASK A.xlsx  # Input data
        ├── EST_Final_With_Formulas.xlsx  # Generated Excel
        └── final_data_cleaned.csv        # Generated CSV
        ```
        """)
    
    # File download section
    st.markdown("### 📁 Generated Files")
    if os.path.exists("EST_Final_With_Formulas.xlsx"):
        with open("EST_Final_With_Formulas.xlsx", "rb") as f:
            st.download_button(
                label="📥 Download Excel with Formulas",
                data=f,
                file_name="EST_Final_With_Formulas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    if os.path.exists("final_data_cleaned.csv"):
        with open("final_data_cleaned.csv", "rb") as f:
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=f,
                file_name="final_data_cleaned.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
