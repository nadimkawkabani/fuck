# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="EST Assessment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .highlight {
        background-color: #FFFBEB;
        padding: 0.5rem;
        border-radius: 0.25rem;
        border: 1px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

class ESTDashboard:
    def __init__(self, data):
        self.data = data
    
    def run(self):
        # Sidebar
        with st.sidebar:
            st.title("📊 EST Dashboard")
            st.markdown("---")
            
            # Filters
            st.subheader("🔍 Filters")
            
            # School filter
            schools = ['All Schools'] + sorted(self.data['School Name'].dropna().unique().tolist())
            selected_school = st.selectbox("Select School", schools)
            
            # Grade filter
            grades = ['All Grades'] + sorted(self.data['Grade'].dropna().unique().tolist())
            selected_grade = st.selectbox("Select Grade", grades)
            
            # Exam filter
            exam_options = ['All', 'EST I Only', 'EST II Only', 'Both Exams']
            selected_exam = st.selectbox("Exam Type", exam_options)
            
            # Score range filter
            min_score = int(self.data['EST I Total'].min()) if not self.data['EST I Total'].isna().all() else 0
            max_score = int(self.data['EST I Total'].max()) if not self.data['EST I Total'].isna().all() else 1600
            score_range = st.slider(
                "EST I Score Range",
                min_value=min_score,
                max_value=max_score,
                value=(min_score, max_score)
            )
            
            st.markdown("---")
            
            # Export options
            st.subheader("📤 Export")
            if st.button("Download CSV"):
                self.download_csv()
            if st.button("Generate Report"):
                self.generate_report()
        
        # Apply filters
        filtered_data = self.apply_filters(selected_school, selected_grade, selected_exam, score_range)
        
        # Main content
        st.markdown('<h1 class="main-header">🎯 EST Assessment Dashboard</h1>', unsafe_allow_html=True)
        
        # Key Metrics
        self.display_metrics(filtered_data)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🏫 Schools", 
            "👨‍🎓 Students", 
            "⚠️ Issues", 
            "📊 Analytics"
        ])
        
        with tab1:
            self.overview_tab(filtered_data)
        
        with tab2:
            self.schools_tab(filtered_data)
        
        with tab3:
            self.students_tab(filtered_data)
        
        with tab4:
            self.issues_tab(filtered_data)
        
        with tab5:
            self.analytics_tab(filtered_data)
    
    def apply_filters(self, school, grade, exam, score_range):
        """Apply filters to data"""
        filtered = self.data.copy()
        
        # School filter
        if school != 'All Schools':
            filtered = filtered[filtered['School Name'] == school]
        
        # Grade filter
        if grade != 'All Grades':
            filtered = filtered[filtered['Grade'] == grade]
        
        # Exam filter
        if exam == 'EST I Only':
            filtered = filtered[filtered['Took EST I?'] == 'Yes']
        elif exam == 'EST II Only':
            filtered = filtered[filtered['Took EST II?'] == 'Yes']
        elif exam == 'Both Exams':
            filtered = filtered[(filtered['Took EST I?'] == 'Yes') & (filtered['Took EST II?'] == 'Yes')]
        
        # Score filter
        filtered = filtered[
            (filtered['EST I Total'] >= score_range[0]) & 
            (filtered['EST I Total'] <= score_range[1])
        ]
        
        return filtered
    
    def display_metrics(self, data):
        """Display key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_students = len(data)
            st.metric("Total Students", total_students)
        
        with col2:
            est1_takers = data[data['Took EST I?'] == 'Yes'].shape[0]
            st.metric("EST I Takers", est1_takers)
        
        with col3:
            est2_takers = data[data['Took EST II?'] == 'Yes'].shape[0]
            st.metric("EST II Takers", est2_takers)
        
        with col4:
            avg_score = data['EST I Total'].mean()
            st.metric("Avg EST I Score", 
                     f"{avg_score:.0f}" if not pd.isna(avg_score) else "N/A",
                     delta=f"{(avg_score - 1000):.0f}" if not pd.isna(avg_score) else None)
        
        with col5:
            cheating_cases = data[data['Cheating Flag'] == 'Yes'].shape[0]
            st.metric("Cheating Cases", cheating_cases, delta_color="inverse")
        
        st.markdown("---")
    
    def overview_tab(self, data):
        """Overview tab content"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            st.subheader("📊 EST I Score Distribution")
            if not data['EST I Total'].isna().all():
                fig = px.histogram(
                    data, 
                    x='EST I Total',
                    nbins=30,
                    title="Distribution of EST I Scores",
                    labels={'EST I Total': 'Score'},
                    color_discrete_sequence=['#3B82F6']
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No EST I score data available")
        
        with col2:
            # Attendance vs Registration
            st.subheader("📋 Registration vs Attendance")
            reg_att_data = pd.DataFrame({
                'Status': ['Registered', 'Attended'],
                'EST I': [
                    data[data['Registered EST I?'] == 'Yes'].shape[0],
                    data[data['Took EST I?'] == 'Yes'].shape[0]
                ],
                'EST II': [
                    data[data['Registered EST II?'] == 'Yes'].shape[0],
                    data[data['Took EST II?'] == 'Yes'].shape[0]
                ]
            })
            
            fig = go.Figure(data=[
                go.Bar(name='EST I', x=reg_att_data['Status'], y=reg_att_data['EST I'], marker_color='#3B82F6'),
                go.Bar(name='EST II', x=reg_att_data['Status'], y=reg_att_data['EST II'], marker_color='#10B981')
            ])
            fig.update_layout(barmode
