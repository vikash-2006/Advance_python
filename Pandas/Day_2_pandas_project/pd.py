# student_id, student_name, course_name, course_duration, total_fees
# insert 10 rows into dataframe
# convert this into csv and excel
# then create a repo ('Data_students') and upload datasets on this github




import pandas as pd

# Student data - 10 rows
student = {
    'Student_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Student_name': ['Abhishek', 'Rishi', 'Vikash', 'Akanksha', 'Sandeep', 'Divyansh', 'Sourbh', 'Tushar', 'Prachi', 'Ayushi'],
    'Course_name': ['Data Science', 'Data Analyst', 'Machine Learning', 'AI', 'Python', 'Power BI', 'SQL', 'Deep Learning', 'Tableau', 'Data Engineering'],
    'Course_duration': ['6 months', '4 months', '6 months', '8 months', '3 months', '2 months', '2 months', '7 months', '2 months', '6 months'],
    'Total_fees': [50000, 40000, 60000, 70000, 20000, 15000, 12000, 65000, 18000, 55000]
}

# Create DataFrame
df = pd.DataFrame(student)
print(df)

# Export to CSV and Excel
df.to_csv('Student_data.csv', index=False)
df.to_excel('Student_data.xlsx', index=False)

# Read back CSV
df_csv = pd.read_csv('Student_data.csv')
print(df_csv)