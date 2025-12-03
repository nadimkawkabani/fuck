# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="EST Assessment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class ESTDashboard:
    def __init__(self, data_path="final_data_cleaned.csv"):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        """Load the processed data"""
        try:
            self.df = pd.read_csv(self.data_path)
            self.original_df = self.df.copy()
            print(f"✅ Loaded {len(self.df)} records from {self.data_path}")
        except:
            # Create sample data if file doesn't exist
            st.warning("Using sample data. Run data_processor.py first for full data.")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_students = 50
        
        self.df = pd.DataFrame({
            'email': [f'est{i:04d}@gmail.com' for i in range(1000, 1000+n_students)],
            'First Name': ['Student'] * n_students,
            'Last Name': [f'{i}' for i in range(1, n_students+1)],
            'gender': np.random.choice(['Male', 'Female'], n_students),
            'Age on 4 Feb 2022': [f'{np.random.randint(15, 19)}y {np.random.randint(0, 12)}m {np.random.randint(0, 30)}d' for _ in range(n_students)],
            'Registered EST I?': np.random.choice(['Yes', 'No'], n_students, p=[0.9, 0.1]),
            'Registered EST II?': np.random.choice(['Yes', 'No'], n_students, p=[0.7, 0.3]),
            'Took EST I?': np.random.choice(['Yes', 'No'], n_students, p=[0.85, 0.15]),
            'Took EST II?': np.random.choice(['Yes', 'No'], n_students, p=[0.6, 0.4]),
            'Total # of Notes': np.random.randint(0, 5, n_students),
            'EST I Attendance Issues?': [''] * n_students,
            'EST I Literacy': np.random.normal(600, 100, n_students).clip(400, 800),
            'EST I Math': np.random.normal(650, 120, n_students).clip(400, 800),
            'EST I Total': np.random.normal(1250, 200, n_students).clip(800, 1600),
            'Grade': np.random.choice(['Grade 10', 'Grade 11', 'Grade 12'], n_students),
            'School Name': np.random.choice(['EST School 1', 'EST School 2', 'EST School 3', 
                                            'EST School 4', 'EST School 5'], n_students),
            'Cheating Flag': np.random.choice(['Yes', ''], n_students, p=[0.05, 0.95])
        })
        self.df['EST I Total'] = self.df['EST I Literacy'] + self.df['EST I Math']
        self.original_df = self.df.copy()
    
    def run(self):
        """Run the dashboard"""
        # Sidebar
        with st.sidebar:
            st.title("📊 EST Dashboard")
            st.markdown("---")
            
            # Filters
            st.subheader("🔍 Filters")
            
            # School filter
            schools = ['All Schools'] + sorted(self.df['School Name'].dropna().unique().tolist())
            selected_school = st.selectbox("Select School", schools)
            
            # Grade filter
            grades = ['All Grades'] + sorted(self.df['Grade'].dropna().unique().tolist())
            selected_grade = st.selectbox("Select Grade", grades)
            
            # Exam status filter
            exam_filter = st.multiselect(
                "Exam Status",
                options=['EST I Registered', 'EST I Attended', 'EST II Registered', 'EST II Attended'],
                default=[]
            )
            
            # Score range filter
            if not self.df['EST I Total'].isna().all():
                min_score = int(self.df['EST I Total'].min())
                max_score = int(self.df['EST I Total'].max())
                score_range = st.slider(
                    "EST I Score Range",
                    min_value=min_score,
                    max_value=max_score,
                    value=(min_score, max_score)
                )
            
            # Cheating cases only
            show_cheating_only = st.checkbox("Show Cheating Cases Only")
            
            st.markdown("---")
            
            # Export buttons
            st.subheader("📤 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 CSV"):
                    self.download_csv()
            with col2:
                if st.button("📊 Excel"):
                    self.download_excel()
            
            st.markdown("---")
            st.info(f"Showing {len(self.apply_filters(selected_school, selected_grade, exam_filter, score_range if 'score_range' in locals() else None, show_cheating_only))} students")
        
        # Apply filters
        filtered_df = self.apply_filters(
            selected_school, 
            selected_grade, 
            exam_filter,
            score_range if 'score_range' in locals() else None,
            show_cheating_only
        )
        
        # Main content
        st.markdown('<h1 class="main-header">🎯 EST Assessment Dashboard</h1>', unsafe_allow_html=True)
        
        # Key Metrics
        self.display_metrics(filtered_df)
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Overview", 
            "🏫 Schools", 
            "📊 Performance", 
            "⚠️ Issues", 
            "👨‍🎓 Students"
        ])
        
        with tab1:
            self.overview_tab(filtered_df)
        
        with tab2:
            self.schools_tab(filtered_df)
        
        with tab3:
            self.performance_tab(filtered_df)
        
        with tab4:
            self.issues_tab(filtered_df)
        
        with tab5:
            self.students_tab(filtered_df)
    
    def apply_filters(self, school, grade, exam_filter, score_range, cheating_only):
        """Apply filters to data"""
        filtered = self.df.copy()
        
        # School filter
        if school != 'All Schools':
            filtered = filtered[filtered['School Name'] == school]
        
        # Grade filter
        if grade != 'All Grades':
            filtered = filtered[filtered['Grade'] == grade]
        
        # Exam filters
        for exam in exam_filter:
            if exam == 'EST I Registered':
                filtered = filtered[filtered['Registered EST I?'] == 'Yes']
            elif exam == 'EST I Attended':
                filtered = filtered[filtered['Took EST I?'] == 'Yes']
            elif exam == 'EST II Registered':
                filtered = filtered[filtered['Registered EST II?'] == 'Yes']
            elif exam == 'EST II Attended':
                filtered = filtered[filtered['Took EST II?'] == 'Yes']
        
        # Score range filter
        if score_range is not None:
            filtered = filtered[
                (filtered['EST I Total'] >= score_range[0]) & 
                (filtered['EST I Total'] <= score_range[1])
            ]
        
        # Cheating filter
        if cheating_only:
            filtered = filtered[filtered['Cheating Flag'] == 'Yes']
        
        return filtered
    
    def display_metrics(self, data):
        """Display key metrics"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_students = len(data)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_students}</div>
                <div class="metric-label">Total Students</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            est1_takers = data[data['Took EST I?'] == 'Yes'].shape[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{est1_takers}</div>
                <div class="metric-label">EST I Takers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            est2_takers = data[data['Took EST II?'] == 'Yes'].shape[0]
                        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{est2_takers}</div>
                <div class="metric-label">EST II Takers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if not data['EST I Total'].isna().all():
                avg_score = data['EST I Total'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{avg_score:.0f}</div>
                    <div class="metric-label">Avg EST I Score</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">N/A</div>
                    <div class="metric-label">Avg EST I Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col5:
            cheating_cases = data[data['Cheating Flag'] == 'Yes'].shape[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cheating_cases}</div>
                <div class="metric-label">Cheating Cases</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def overview_tab(self, data):
        """Overview tab content"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 class="sub-header">📊 EST I Score Distribution</h3>', unsafe_allow_html=True)
            if not data['EST I Total'].isna().all():
                fig = px.histogram(
                    data, 
                    x='EST I Total',
                    nbins=20,
                    title="",
                    labels={'EST I Total': 'Score'},
                    color_discrete_sequence=['#3B82F6'],
                    opacity=0.8
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No EST I score data available")
        
        with col2:
            st.markdown('<h3 class="sub-header">📋 Registration vs Attendance</h3>', unsafe_allow_html=True)
            
            # Calculate registration vs attendance
            reg_att_data = pd.DataFrame({
                'Category': ['EST I Registered', 'EST I Attended', 'EST II Registered', 'EST II Attended'],
                'Count': [
                    data[data['Registered EST I?'] == 'Yes'].shape[0],
                    data[data['Took EST I?'] == 'Yes'].shape[0],
                    data[data['Registered EST II?'] == 'Yes'].shape[0],
                    data[data['Took EST II?'] == 'Yes'].shape[0]
                ]
            })
            
            fig = px.bar(
                reg_att_data, 
                x='Category', 
                y='Count',
                title="",
                color='Category',
                color_discrete_sequence=['#3B82F6', '#10B981', '#6366F1', '#8B5CF6']
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        st.markdown('<h3 class="sub-header">📈 Performance Overview</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not data['EST I Literacy'].isna().all():
                avg_literacy = data['EST I Literacy'].mean()
                st.metric("Avg Literacy Score", f"{avg_literacy:.0f}")
        
        with col2:
            if not data['EST I Math'].isna().all():
                avg_math = data['EST I Math'].mean()
                st.metric("Avg Math Score", f"{avg_math:.0f}")
        
        with col3:
            attendance_rate = (data[data['Took EST I?'] == 'Yes'].shape[0] / len(data) * 100) if len(data) > 0 else 0
            st.metric("EST I Attendance Rate", f"{attendance_rate:.1f}%")
        
        # Data preview
        st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(data.head(10), use_container_width=True)
    
    def schools_tab(self, data):
        """Schools analysis tab"""
        if data['School Name'].nunique() > 1:
            # School comparison
            st.markdown('<h3 class="sub-header">🏫 School Performance Comparison</h3>', unsafe_allow_html=True)
            
            # Calculate school statistics
            school_stats = data.groupby('School Name').agg({
                'email': 'count',
                'EST I Total': 'mean',
                'EST I Literacy': 'mean',
                'EST I Math': 'mean',
                'Took EST I?': lambda x: (x == 'Yes').sum(),
                'Took EST II?': lambda x: (x == 'Yes').sum(),
                'Cheating Flag': lambda x: (x == 'Yes').sum()
            }).round(2)
            
            school_stats.columns = [
                'Total Students', 'Avg Total Score', 'Avg Literacy', 'Avg Math',
                'EST I Takers', 'EST II Takers', 'Cheating Cases'
            ]
            
            # Display school statistics
            st.dataframe(school_stats, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h4>📊 Average Scores by School</h4>', unsafe_allow_html=True)
                fig = px.bar(
                    school_stats,
                    x=school_stats.index,
                    y='Avg Total Score',
                    title="",
                    color=school_stats.index,
                    labels={'Avg Total Score': 'Average EST I Score'}
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="School",
                    yaxis_title="Average Score",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown('<h4>📈 Attendance vs Performance</h4>', unsafe_allow_html=True)
                fig = px.scatter(
                    school_stats,
                    x='EST I Takers',
                    y='Avg Total Score',
                    size='Total Students',
                    color=school_stats.index,
                    hover_name=school_stats.index,
                    title="",
                    labels={
                        'EST I Takers': 'Number of EST I Takers',
                        'Avg Total Score': 'Average EST I Score'
                    }
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # School details
            st.markdown('<h3 class="sub-header">🏫 School Details</h3>', unsafe_allow_html=True)
            selected_school = st.selectbox(
                "Select School for Details",
                sorted(data['School Name'].dropna().unique())
            )
            
            if selected_school:
                school_data = data[data['School Name'] == selected_school]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Students", len(school_data))
                with col2:
                    st.metric("Avg EST I Score", f"{school_data['EST I Total'].mean():.0f}")
                with col3:
                    st.metric("EST I Attendance", 
                             f"{(school_data['Took EST I?'] == 'Yes').sum() / len(school_data) * 100:.1f}%")
                
                # Grade distribution
                st.markdown('<h4>📊 Grade Distribution</h4>', unsafe_allow_html=True)
                grade_dist = school_data['Grade'].value_counts()
                fig = px.pie(
                    values=grade_dist.values,
                    names=grade_dist.index,
                    title=f"Grade Distribution - {selected_school}"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Showing data for {data['School Name'].iloc[0] if not data.empty else 'selected school'}")
    
    def performance_tab(self, data):
        """Performance analysis tab"""
        st.markdown('<h3 class="sub-header">📊 Performance Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Literacy vs Math correlation
            if not data['EST I Literacy'].isna().all() and not data['EST I Math'].isna().all():
                st.markdown('<h4>📈 Literacy vs Math Scores</h4>', unsafe_allow_html=True)
                fig = px.scatter(
                    data,
                    x='EST I Literacy',
                    y='EST I Math',
                    color='Grade',
                    title="",
                    hover_data=['First Name', 'Last Name', 'School Name'],
                    trendline="ols"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution by grade
            if not data['EST I Total'].isna().all():
                st.markdown('<h4>📊 Score Distribution by Grade</h4>', unsafe_allow_html=True)
                fig = px.box(
                    data,
                    x='Grade',
                    y='EST I Total',
                    color='Grade',
                    title="",
                    points="all"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.markdown('<h3 class="sub-header">🏆 Top Performers</h3>', unsafe_allow_html=True)
        if not data['EST I Total'].isna().all():
            top_performers = data.nlargest(10, 'EST I Total')[['First Name', 'Last Name', 'Grade', 'School Name', 'EST I Total']]
            st.dataframe(top_performers, use_container_width=True)
        
        # Performance metrics
        st.markdown('<h3 class="sub-header">📈 Performance Metrics</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not data['EST I Total'].isna().all():
                st.metric("Highest Score", f"{data['EST I Total'].max():.0f}")
        
        with col2:
            if not data['EST I Total'].isna().all():
                st.metric("Lowest Score", f"{data['EST I Total'].min():.0f}")
        
        with col3:
            if not data['EST I Total'].isna().all():
                st.metric("Median Score", f"{data['EST I Total'].median():.0f}")
        
        with col4:
            if not data['EST I Total'].isna().all():
                std_dev = data['EST I Total'].std()
                st.metric("Std Deviation", f"{std_dev:.0f}")
    
    def issues_tab(self, data):
        """Issues and quality control tab"""
        st.markdown('<h3 class="sub-header">⚠️ Data Quality Issues</h3>', unsafe_allow_html=True)
        
        # Calculate issues
        total_students = len(data)
        
        # Attendance issues
        attendance_issues = data[data['EST I Attendance Issues?'] != ''].shape[0]
        
        # Missing scores
        missing_scores = data['EST I Total'].isna().sum()
        
        # Cheating cases
        cheating_cases = data[data['Cheating Flag'] == 'Yes'].shape[0]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Attendance Issues", attendance_issues)
        
        with col2:
            st.metric("Missing Scores", missing_scores)
        
        with col3:
            st.metric("Cheating Cases", cheating_cases)
        
        with col4:
            completeness = ((total_students - missing_scores) / total_students * 100) if total_students > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Detailed issues
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4>🚨 Students with Attendance Issues</h4>', unsafe_allow_html=True)
            issues_df = data[data['EST I Attendance Issues?'] != ''][['First Name', 'Last Name', 'email', 'EST I Attendance Issues?']]
            if not issues_df.empty:
                st.dataframe(issues_df, use_container_width=True)
            else:
                st.success("No attendance issues found!")
        
        with col2:
            st.markdown('<h4>🚫 Cheating Cases</h4>', unsafe_allow_html=True)
            cheating_df = data[data['Cheating Flag'] == 'Yes'][['First Name', 'Last Name', 'email', 'School Name', 'Grade']]
            if not cheating_df.empty:
                st.dataframe(cheating_df, use_container_width=True)
            else:
                st.success("No cheating cases found!")
        
        # Missing data analysis
        st.markdown('<h3 class="sub-header">📊 Missing Data Analysis</h3>', unsafe_allow_html=True)
        
        # Check for missing values in key columns
        missing_data = pd.DataFrame({
            'Column': ['EST I Total', 'EST I Literacy', 'EST I Math', 'School Name', 'Grade'],
            'Missing Count': [
                data['EST I Total'].isna().sum(),
                data['EST I Literacy'].isna().sum(),
                data['EST I Math'].isna().sum(),
                data['School Name'].isna().sum(),
                data['Grade'].isna().sum()
            ]
        })
        missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(data) * 100).round(1)
        
        fig = px.bar(
            missing_data,
            x='Column',
            y='Missing Percentage',
            title="Missing Data by Column",
            color='Missing Percentage',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def students_tab(self, data):
        """Students details tab"""
        st.markdown('<h3 class="sub-header">👨‍🎓 Student Details</h3>', unsafe_allow_html=True)
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("🔍 Search by Name or Email")
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                ['EST I Total (High to Low)', 'EST I Total (Low to High)', 'Name (A-Z)', 'Name (Z-A)']
            )
        
        # Filter data based on search
        filtered_data = data.copy()
        if search_term:
            filtered_data = filtered_data[
                filtered_data['email'].str.contains(search_term, case=False, na=False) |
                filtered_data['First Name'].str.contains(search_term, case=False, na=False) |
                filtered_data['Last Name'].str.contains(search_term, case=False, na=False)
            ]
        
        # Sort data
        if sort_by == 'EST I Total (High to Low)':
            filtered_data = filtered_data.sort_values('EST I Total', ascending=False)
        elif sort_by == 'EST I Total (Low to High)':
            filtered_data = filtered_data.sort_values('EST I Total', ascending=True)
        elif sort_by == 'Name (A-Z)':
            filtered_data = filtered_data.sort_values(['Last Name', 'First Name'])
        elif sort_by == 'Name (Z-A)':
            filtered_data = filtered_data.sort_values(['Last Name', 'First Name'], ascending=False)
        
        # Display student table
        st.dataframe(
            filtered_data[[
                'First Name', 'Last Name', 'email', 'Grade', 'School Name',
                'EST I Total', 'EST I Literacy', 'EST I Math',
                'Registered EST I?', 'Took EST I?', 'Cheating Flag'
            ]],
            use_container_width=True,
            height=400
        )
        
        # Student detail view
        if not filtered_data.empty:
            st.markdown('<h3 class="sub-header">👤 Student Profile</h3>', unsafe_allow_html=True)
            selected_student = st.selectbox(
                "Select Student",
                filtered_data.apply(lambda x: f"{x['First Name']} {x['Last Name']} ({x['email']})", axis=1)
            )
            
            if selected_student:
                # Extract email from selection
                email = selected_student.split('(')[-1].strip(')')
                student_data = filtered_data[filtered_data['email'] == email].iloc[0]
                
                # Display student details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Personal Information**")
                    st.write(f"**Name:** {student_data['First Name']} {student_data['Last Name']}")
                    st.write(f"**Email:** {student_data['email']}")
                    st.write(f"**Gender:** {student_data['gender']}")
                    st.write(f"**Age:** {student_data['Age on 4 Feb 2022']}")
                    st.write(f"**Grade:** {student_data['Grade']}")
                
                with col2:
                    st.markdown("**Exam Information**")
                    st.write(f"**School:** {student_data['School Name']}")
                    st.write(f"**EST I Registered:** {student_data['Registered EST I?']}")
                    st.write(f"**EST I Attended:** {student_data['Took EST I?']}")
                    st.write(f"**EST II Registered:** {student_data['Registered EST II?']}")
                    st.write(f"**EST II Attended:** {student_data['Took EST II?']}")
                    st.write(f"**Total Notes:** {student_data['Total # of Notes']}")
                
                with col3:
                    st.markdown("**Performance**")
                    if pd.notna(student_data['EST I Total']):
                        st.write(f"**EST I Total:** {student_data['EST I Total']:.0f}")
                        st.write(f"**EST I Literacy:** {student_data['EST I Literacy']:.0f}")
                        st.write(f"**EST I Math:** {student_data['EST I Math']:.0f}")
                    
                    st.write(f"**Attendance Issues:** {student_data['EST I Attendance Issues?']}")
                    st.write(f"**Cheating Flag:** {student_data['Cheating Flag']}")
    
    def download_csv(self):
        """Download CSV file"""
        csv = self.df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="est_data_export.csv">📥 Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    def download_excel(self):
        """Download Excel file"""
        try:
            # Create Excel writer
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                self.df.to_excel(writer, sheet_name='EST_Data', index=False)
            
            # Get the Excel file data
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            
            # Create download link
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="est_data_export.xlsx">📊 Download Excel File</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error creating Excel file: {e}")

def main():
    """Main function to run the dashboard"""
    st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = ESTDashboard()
    
    # Run dashboard
    dashboard.run()

if __name__ == "__main__":
    main()
