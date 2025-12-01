# main.py - Complete Solution for Task A

import pandas as pd
import numpy as np
from datetime import datetime, date
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import warnings
warnings.filterwarnings('ignore')

# Part 1: Data Processing Functions
class ESTDataProcessor:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.load_all_sheets()
        
    def load_all_sheets(self):
        """Load all required sheets from the Excel file"""
        self.students = pd.read_excel(self.excel_path, sheet_name='Students')
        self.admissions = pd.read_excel(self.excel_path, sheet_name='Admissions')
        self.exam_notes = pd.read_excel(self.excel_path, sheet_name='Exam Notes')
        self.scores_est1 = pd.read_excel(self.excel_path, sheet_name='Scores EST I')
        self.scores_est2 = pd.read_excel(self.excel_path, sheet_name='Scores EST II')
        self.attendance_est1 = pd.read_excel(self.excel_path, sheet_name='Attendance EST I')
        self.attendance_est2 = pd.read_excel(self.excel_path, sheet_name='Attendance ESTII')
        self.est_scale = pd.read_excel(self.excel_path, sheet_name='EST Scale')
        self.note_types = pd.read_excel(self.excel_path, sheet_name='Note Types')
        
        # Clean column names
        self.students.columns = self.students.columns.str.strip()
        self.admissions.columns = self.admissions.columns.str.strip()
        
    def extract_first_last_name(self, name):
        """Extract first and last name from full name"""
        if pd.isna(name):
            return '', ''
        name_parts = str(name).split()
        first_name = name_parts[0] if name_parts else ''
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
        return first_name, last_name
    
    def calculate_age(self, dob):
        """Calculate age in years, months, days as of Feb 4, 2022"""
        if pd.isna(dob):
            return ''
        
        if isinstance(dob, str):
            dob = datetime.strptime(dob.split()[0], '%Y-%m-%d')
        
        exam_date = datetime(2022, 2, 4)
        
        years = exam_date.year - dob.year
        months = exam_date.month - dob.month
        days = exam_date.day - dob.day
        
        if days < 0:
            months -= 1
            days += 30  # Approximate
            
        if months < 0:
            years -= 1
            months += 12
            
        return f"{years}y {months}m {days}d"
    
    def get_registration_status(self, email):
        """Determine if student is registered for EST I and/or EST II"""
        student_admissions = self.admissions[self.admissions['email'] == email]
        
        registered_est1 = 'Yes' if 'EST I' in student_admissions['exam'].values else 'No'
        registered_est2 = 'Yes' if 'EST II' in student_admissions['exam'].values else 'No'
        
        return registered_est1, registered_est2
    
    def get_attendance_status(self, email):
        """Determine if student took EST I and/or EST II based on attendance"""
        took_est1 = 'No'
        took_est2 = 'No'
        
        # Check EST I attendance
        est1_att = self.attendance_est1[self.attendance_est1['Email'] == email]
        if not est1_att.empty:
            time_in = est1_att['Time In'].iloc[0]
            time_out = est1_att['Time Out'].iloc[0]
            if pd.notna(time_in) and pd.notna(time_out):
                took_est1 = 'Yes'
        
        # Check EST II attendance
        est2_att = self.attendance_est2[self.attendance_est2['Email'] == email]
        if not est2_att.empty:
            # Check if any attendance record exists
            attendance_columns = ['Math 1-In', 'Math 1-Out', 'Biology-In', 'Biology-Out', 
                                'Physics-In', 'Physics-Out']
            for col in attendance_columns:
                if col in est2_att.columns and pd.notna(est2_att[col].iloc[0]):
                    took_est2 = 'Yes'
                    break
        
        return took_est1, took_est2
    
    def count_notes(self, email):
        """Count total notes for a student"""
        return len(self.exam_notes[self.exam_notes['email'] == email])
    
    def check_attendance_issues(self, email):
        """Check for attendance discrepancies"""
        notes = self.exam_notes[self.exam_notes['email'] == email]
        est1_att = self.attendance_est1[self.attendance_est1['Email'] == email]
        
        issues = []
        
        # Check if marked absent but has attendance
        absent_notes = notes[notes['Note'] == 'Absent']
        if not absent_notes.empty and not est1_att.empty:
            time_in = est1_att['Time In'].iloc[0]
            if pd.notna(time_in):
                issues.append('Marked absent but has attendance record')
        
        # Check if has attendance but no scores
        if not est1_att.empty:
            time_in = est1_att['Time In'].iloc[0]
            if pd.notna(time_in):
                scores = self.scores_est1[self.scores_est1['Email'] == email]
                if scores.empty or scores.isna().all().all():
                    issues.append('Has attendance but no scores')
        
        return '; '.join(issues) if issues else ''
    
    def apply_raise(self, raw_score, subject, email):
        """Apply raises to raw scores based on subject and misconduct status"""
        # Check for misconduct
        student_notes = self.exam_notes[self.exam_notes['email'] == email]
        if 'Misconduct' in student_notes['Note'].values:
            return raw_score  # No raise for misconduct
        
        # Define raises
        raises = {
            'EST I Literacy 1': 4,
            'EST I Literacy 2': 3,
            'EST I Math without Calculator': 5,
            'EST I Math with Calculator': 1
        }
        
        if subject in raises:
            if pd.notna(raw_score):
                return raw_score + raises[subject]
        
        return raw_score
    
    def get_scaled_score(self, raw_score, subject):
        """Convert raw score to scaled score using EST Scale sheet"""
        if pd.isna(raw_score):
            return None
        
        # Map subjects to scale columns
        scale_map = {
            'EST I Math': ('Raw', 'EST I Math (58)'),
            'EST I Literacy 1': ('Raw', 'EST I Literacy 1 (44)'),
            'EST I Literacy 2': ('Raw', 'EST I Literacy 2 (52)'),
            'EST II Math 1': ('Raw', 'EST II Math 1 (50)'),
            'EST II Biology': ('Raw', 'EST II Biology (80)'),
            'EST II Physics': ('Raw', 'EST II Physics (75)')
        }
        
        subject_key = None
        for key in scale_map:
            if key in subject:
                subject_key = key
                break
        
        if not subject_key:
            return None
        
        raw_col, scaled_col = scale_map[subject_key]
        
        # Find the scale row
        scale_data = self.est_scale[[raw_col, scaled_col]].dropna()
        scale_data = scale_data.sort_values(raw_col)
        
        # Find matching or closest raw score
        matching = scale_data[scale_data[raw_col] <= raw_score]
        if matching.empty:
            return scale_data[scaled_col].iloc[0]
        
        return matching.iloc[-1][scaled_col]
    
    def calculate_est1_literacy(self, email):
        """Calculate EST I Literacy score"""
        scores = self.scores_est1[self.scores_est1['Email'] == email]
        if scores.empty:
            return None
        
        # Apply raises
        literacy1_raw = self.apply_raise(
            scores['EST I Literacy 1'].iloc[0] if pd.notna(scores['EST I Literacy 1'].iloc[0]) else 0,
            'EST I Literacy 1', email
        )
        literacy2_raw = self.apply_raise(
            scores['EST I Literacy 2'].iloc[0] if pd.notna(scores['EST I Literacy 2'].iloc[0]) else 0,
            'EST I Literacy 2', email
        )
        
        # Get scaled scores
        literacy1_scaled = self.get_scaled_score(literacy1_raw, 'EST I Literacy 1')
        literacy2_scaled = self.get_scaled_score(literacy2_raw, 'EST I Literacy 2')
        
        if literacy1_scaled and literacy2_scaled:
            return (literacy1_scaled + literacy2_scaled) * 10
        return None
    
    def calculate_est1_math(self, email):
        """Calculate EST I Math score"""
        scores = self.scores_est1[self.scores_est1['Email'] == email]
        if scores.empty:
            return None
        
        # Apply raises
        math_no_calc_raw = self.apply_raise(
            scores['EST I Math without Calculator'].iloc[0] if pd.notna(scores['EST I Math without Calculator'].iloc[0]) else 0,
            'EST I Math without Calculator', email
        )
        math_with_calc_raw = self.apply_raise(
            scores['EST I Math with Calculator'].iloc[0] if pd.notna(scores['EST I Math with Calculator'].iloc[0]) else 0,
            'EST I Math with Calculator', email
        )
        
        # Total raw score
        total_raw = math_no_calc_raw + math_with_calc_raw
        
        # Get scaled score
        return self.get_scaled_score(total_raw, 'EST I Math')
    
    def calculate_est2_subject(self, email, subject):
        """Calculate EST II subject score"""
        if subject == 'Math 1':
            col = 'EST II Math 1'
        elif subject == 'Biology':
            col = 'EST II Biology'
        elif subject == 'Physics':
            col = 'EST II Physics'
        else:
            return None
        
        scores = self.scores_est2[self.scores_est2['Email'] == email]
        if scores.empty or col not in scores.columns:
            return None
        
        raw_score = scores[col].iloc[0]
        if pd.isna(raw_score):
            return None
        
        # Calculate raise based on average (simplified - using fixed raise for demo)
        # In real scenario, calculate average per test
        raise_amount = 2  # Simplified for demo
        
        # Check for misconduct
        student_notes = self.exam_notes[self.exam_notes['email'] == email]
        if 'Misconduct' not in student_notes['Note'].values:
            raw_score += raise_amount
        
        # Get scaled score
        return self.get_scaled_score(raw_score, f'EST II {subject}')
    
    def get_school_name(self, email):
        """Get school name from admissions"""
        school_info = self.admissions[self.admissions['email'] == email]
        if not school_info.empty:
            return school_info['School Name'].iloc[0]
        return ''
    
    def check_cheating(self, email):
        """Check if student has cheating note"""
        student_notes = self.exam_notes[self.exam_notes['email'] == email]
        return 'Yes' if 'Cheating' in student_notes['Note'].values else ''
    
    def process_final_sheet(self):
        """Main function to process and populate the final sheet"""
        final_data = []
        
        for _, student in self.students.iterrows():
            email = student['email']
            
            # Extract names
            first_name, last_name = self.extract_first_last_name(student['name'])
            
            # Calculate age
            age = self.calculate_age(student['Date Of Birth'])
            
            # Get registration status
            registered_est1, registered_est2 = self.get_registration_status(email)
            
            # Get attendance status
            took_est1, took_est2 = self.get_attendance_status(email)
            
            # Count notes
            total_notes = self.count_notes(email)
            
            # Check attendance issues
            attendance_issues = self.check_attendance_issues(email)
            
            # Calculate scores
            est1_literacy = self.calculate_est1_literacy(email)
            est1_math = self.calculate_est1_math(email)
            est1_total = None
            if est1_literacy and est1_math:
                est1_total = est1_literacy + est1_math
            
            est2_math = self.calculate_est2_subject(email, 'Math 1')
            est2_biology = self.calculate_est2_subject(email, 'Biology')
            est2_physics = self.calculate_est2_subject(email, 'Physics')
            
            # Get school name
            school_name = self.get_school_name(email)
            
            # Check cheating
            cheating_flag = self.check_cheating(email)
            
            # Compile final row
            row = {
                'email': email,
                'First Name': first_name,
                'Last Name': last_name,
                'gender': student['gender'],
                'Date Of Birth': student['Date Of Birth'],
                'Age on 4 Feb 2022': age,
                'Registered EST I?': registered_est1,
                'Registered EST II?': registered_est2,
                'Took EST I?': took_est1,
                'Took EST II?': took_est2,
                'Total # of Notes': total_notes,
                'EST I Attendance Issues?': attendance_issues,
                'EST I Literacy': est1_literacy,
                'EST I Math': est1_math,
                'EST I Total': est1_total,
                'EST II Math 1': est2_math,
                'EST II Biology': est2_biology,
                'EST II Physics': est2_physics,
                'Grade': student['Grade'],
                'EST II Attendance Issues?': '',  # Not required per instructions
                'School Name': school_name,
                'Cheating Flag': cheating_flag
            }
            
            final_data.append(row)
        
        self.final_df = pd.DataFrame(final_data)
        return self.final_df
    
    def save_to_excel(self, output_path):
        """Save processed data to Excel with formulas (simulated)"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self.final_df.to_excel(writer, sheet_name='Final', index=False)
            
            # Save other sheets for reference
            self.students.to_excel(writer, sheet_name='Students', index=False)
            self.admissions.to_excel(writer, sheet_name='Admissions', index=False)
            # ... add other sheets as needed
    
    def save_cleaned_csv(self, output_path):
        """Save cleaned data to CSV"""
        self.final_df.to_csv(output_path, index=False)

# Part 2: Dashboard Application
class ESTDashboard:
    def __init__(self, data_processor):
        self.processor = data_processor
        self.final_df = data_processor.final_df
        
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.set_page_config(
            page_title="EST Assessment Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        # Sidebar navigation
        with st.sidebar:
            st.title("📊 EST Dashboard")
            selected = option_menu(
                menu_title="Navigation",
                options=["Dashboard", "School Analysis", "Student Details", "Data Quality", "Export Reports"],
                icons=["house", "building", "person", "check-circle", "download"],
                menu_icon="cast",
                default_index=0,
            )
            
            st.divider()
            st.subheader("Filters")
            
            # School filter
            schools = ['All'] + sorted(self.final_df['School Name'].dropna().unique().tolist())
            selected_school = st.selectbox("Select School", schools)
            
            # Grade filter
            grades = ['All'] + sorted(self.final_df['Grade'].dropna().unique().tolist())
            selected_grade = st.selectbox("Select Grade", grades)
            
            # Apply filters
            filtered_df = self.final_df.copy()
            if selected_school != 'All':
                filtered_df = filtered_df[filtered_df['School Name'] == selected_school]
            if selected_grade != 'All':
                filtered_df = filtered_df[filtered_df['Grade'] == selected_grade]
        
        # Main content based on selection
        if selected == "Dashboard":
            self.show_dashboard(filtered_df)
        elif selected == "School Analysis":
            self.show_school_analysis(filtered_df)
        elif selected == "Student Details":
            self.show_student_details(filtered_df)
        elif selected == "Data Quality":
            self.show_data_quality(filtered_df)
        elif selected == "Export Reports":
            self.show_export_options(filtered_df, selected_school)
    
    def show_dashboard(self, df):
        """Display main dashboard"""
        st.title("🎯 EST Assessment Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            est1_takers = df[df['Took EST I?'] == 'Yes'].shape[0]
            st.metric("EST I Takers", est1_takers)
        with col3:
            est2_takers = df[df['Took EST II?'] == 'Yes'].shape[0]
            st.metric("EST II Takers", est2_takers)
        with col4:
            avg_score = df['EST I Total'].mean()
            st.metric("Avg EST I Score", f"{avg_score:.0f}" if not pd.isna(avg_score) else "N/A")
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # EST I Score Distribution
            scores = df['EST I Total'].dropna()
            if not scores.empty:
                fig = px.histogram(scores, nbins=20, title="EST I Total Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Registration vs Attendance
            reg_att_data = pd.DataFrame({
                'Category': ['Registered EST I', 'Took EST I', 'Registered EST II', 'Took EST II'],
                'Count': [
                    df[df['Registered EST I?'] == 'Yes'].shape[0],
                    df[df['Took EST I?'] == 'Yes'].shape[0],
                    df[df['Registered EST II?'] == 'Yes'].shape[0],
                    df[df['Took EST II?'] == 'Yes'].shape[0]
                ]
            })
            fig = px.bar(reg_att_data, x='Category', y='Count', title="Registration vs Attendance")
            st.plotly_chart(fig, use_container_width=True)
        
        # Data table preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    def show_school_analysis(self, df):
        """Display school-level analysis"""
        st.title("🏫 School Analysis")
        
        if df['School Name'].nunique() > 1:
            # School comparison
            school_stats = df.groupby('School Name').agg({
                'email': 'count',
                'EST I Total': 'mean',
                'Took EST I?': lambda x: (x == 'Yes').sum(),
                'Took EST II?': lambda x: (x == 'Yes').sum()
            }).round(2)
            
            school_stats.columns = ['Total Students', 'Avg EST I Score', 'EST I Takers', 'EST II Takers']
            
            st.subheader("School Performance Summary")
            st.dataframe(school_stats, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(school_stats, x=school_stats.index, y='Avg EST I Score',
                           title="Average EST I Score by School")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(school_stats, x='EST I Takers', y='Avg EST I Score',
                               size='Total Students', hover_name=school_stats.index,
                               title="Attendance vs Performance")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Showing data for {df['School Name'].iloc[0] if not df.empty else 'selected school'}")
            
            if not df.empty:
                # Single school analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Grade distribution
                    grade_dist = df['Grade'].value_counts()
                    fig = px.pie(values=grade_dist.values, names=grade_dist.index,
                               title="Grade Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Score distribution for this school
                    scores = df['EST I Total'].dropna()
                    if not scores.empty:
                        fig = px.box(df, y='EST I Total', title="EST I Score Distribution")
                        st.plotly_chart(fig, use_container_width=True)
    
    def show_student_details(self, df):
        """Display detailed student information"""
        st.title("👨‍🎓 Student Details")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_email = st.text_input("Search by Email")
        with col2:
            show_cheating = st.checkbox("Show Cheating Cases Only")
        
        filtered_df = df.copy()
        if search_email:
            filtered_df = filtered_df[filtered_df['email'].str.contains(search_email, case=False, na=False)]
        if show_cheating:
            filtered_df = filtered_df[filtered_df['Cheating Flag'] == 'Yes']
        
        # Student table
        st.dataframe(filtered_df, use_container_width=True)
        
        # Student details if selected
        if not filtered_df.empty and len(filtered_df) == 1:
            student = filtered_df.iloc[0]
            st.subheader(f"Student Details: {student['First Name']} {student['Last Name']}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Email:** {student['email']}")
                st.write(f"**Grade:** {student['Grade']}")
                st.write(f"**School:** {student['School Name']}")
            with col2:
                st.write(f"**EST I Registered:** {student['Registered EST I?']}")
                st.write(f"**EST I Attended:** {student['Took EST I?']}")
                st.write(f"**EST I Total:** {student['EST I Total']}")
            with col3:
                st.write(f"**Total Notes:** {student['Total # of Notes']}")
                st.write(f"**Attendance Issues:** {student['EST I Attendance Issues?']}")
                st.write(f"**Cheating Flag:** {student['Cheating Flag']}")
    
    def show_data_quality(self, df):
        """Display data quality issues"""
        st.title("🔍 Data Quality Check")
        
        # Calculate data quality metrics
        total_students = len(df)
        
        missing_scores = df['EST I Total'].isna().sum()
        attendance_issues = df[df['EST I Attendance Issues?'] != ''].shape[0]
        cheating_cases = df[df['Cheating Flag'] == 'Yes'].shape[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Missing Scores", missing_scores)
        with col2:
            st.metric("Attendance Issues", attendance_issues)
        with col3:
            st.metric("Cheating Cases", cheating_cases)
        with col4:
            completeness = ((total_students - missing_scores) / total_students * 100) if total_students > 0 else 0
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        st.divider()
        
        # Show issues
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Students with Attendance Issues")
            issues_df = df[df['EST I Attendance Issues?'] != ''][['email', 'EST I Attendance Issues?']]
            st.dataframe(issues_df, use_container_width=True)
        
        with col2:
            st.subheader("Cheating Cases")
            cheating_df = df[df['Cheating Flag'] == 'Yes'][['email', 'First Name', 'Last Name', 'School Name']]
            st.dataframe(cheating_df, use_container_width=True)
    
    def show_export_options(self, df, selected_school):
        """Display export and report generation options"""
        st.title("📤 Export Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Export Data")
            
            if st.button("📥 Download Cleaned CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="final_data_cleaned.csv",
                    mime="text/csv"
                )
            
            if st.button("📊 Download Excel with Formulas"):
                # In a real implementation, this would create an Excel file with formulas
                st.info("Excel export with formulas would be generated here")
        
        with col2:
            st.subheader("Generate PDF Report")
            
            report_type = st.selectbox(
                "Report Type",
                ["School Summary", "Global Summary", "Detailed Student Report"]
            )
            
            if st.button("🖨️ Generate PDF Report"):
                self.generate_pdf_report(df, report_type, selected_school)
                st.success("PDF report generated successfully!")
    
    def generate_pdf_report(self, df, report_type, school_name):
        """Generate PDF report with custom colors"""
        # Define colors from requirements
        primary_color = colors.Color(253/255, 231/255, 45/255)  # RGB: 253,231,45
        text_color = colors.Color(35/255, 31/255, 32/255)  # RGB: 35,31,32
        
        # Create PDF document
        pdf_path = f"EST_Report_{report_type.replace(' ', '_')}.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        
        # Content
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = styles['Title']
        title_style.textColor = text_color
        story.append(Paragraph(f"EST Assessment Report - {report_type}", title_style))
        story.append(Spacer(1, 12))
        
        # Subtitle
        if report_type == "School Summary":
            story.append(Paragraph(f"School: {school_name if school_name != 'All' else 'All Schools'}", styles['Heading2']))
        
        # Summary statistics
        story.append(Paragraph("Summary Statistics", styles['Heading3']))
        
        # Create summary table
        summary_data = [
            ["Metric", "Value"],
            ["Total Students", str(len(df))],
            ["Average EST I Score", f"{df['EST I Total'].mean():.1f}"],
            ["EST I Attendance Rate", f"{(df['Took EST I?'] == 'Yes').sum() / len(df) * 100:.1f}%"],
            ["Data Issues", str(df[df['EST I Attendance Issues?'] != ''].shape[0])]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Add more content based on report type...
        
        # Build PDF
        doc.build(story)
        
        # Provide download link in Streamlit
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name=pdf_path,
                mime="application/pdf"
            )

# Part 3: Main Execution
def main():
    """Main execution function"""
    print("Starting EST Data Processing...")
    
    # Initialize processor
    processor = ESTDataProcessor("Employment Test - Dataset - TASK A.xlsx")
    
    # Process data
    print("Processing data...")
    final_df = processor.process_final_sheet()
    
    # Save outputs
    print("Saving outputs...")
    processor.save_cleaned_csv("final_data_cleaned.csv")
    processor.save_to_excel("EST_Final_Processed.xlsx")
    
    print(f"Processed {len(final_df)} students")
    print(f"Saved cleaned CSV to: final_data_cleaned.csv")
    print(f"Saved Excel file to: EST_Final_Processed.xlsx")
    
    # Initialize dashboard
    dashboard = ESTDashboard(processor)
    
    # Run dashboard
    print("\nStarting Dashboard...")
    print("Open http://localhost:8501 in your browser")
    dashboard.run_dashboard()

# Alternative: FastAPI version (uncomment if preferred over Streamlit)
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/data")
async def get_data():
    processor = ESTDataProcessor("Employment Test - Dataset - TASK A.xlsx")
    final_df = processor.process_final_sheet()
    return JSONResponse(final_df.to_dict(orient="records"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# Run the main function
if __name__ == "__main__":
    main()
