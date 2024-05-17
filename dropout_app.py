import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing
from prediction import prediction
import sklearn

col0,col1,col3 = st.columns([1,1,5])
with col1:
   #st.image("https://github.com/dicodingacademy/assets/raw/main/logo.png", width=110)
   st.image("topi.png", width=90)
with col3:
   st.title('Jaya Jaya Institut')

col0,col1,col3 = st.columns([1,1,5])


col10,col11,col13 = st.columns([1,3,1])
with col11:  
   st.header('Student Performance')

data = pd.DataFrame()

col1, col2, col3 = st.columns(3)
with col1:
   st.subheader("Demografi")
   container = st.container(border=True)

   Gender = container.selectbox(label='Gender', options=[0, 1], index=1)
   data["Gender"] = [Gender]

   International = container.selectbox(label='International', options=[0, 1], index=1)
   data["International"] = [International]

   Displaced = container.selectbox(label='Displaced', options=[0, 1], index=1)
   data["Displaced"] = [Displaced]

   Marital_status = int(container.number_input(label='Marital status', value=2))
   data["Marital_status"] = Marital_status

   Age_at_enrollment = int(container.number_input(label='Age', value=18))
   data["Age_at_enrollment"] = Age_at_enrollment

   Nacionality = int(container.number_input(label='Nacionality', value=62))
   data["Nacionality"] = Nacionality
   
with col2:
   st.subheader("Sosial Ekonomi")
   container = st.container(border=True)

   Educational_special_needs = container.selectbox(label='Educational special needs', options=[0, 1], index=1)
   data["Educational_special_needs"] = [Educational_special_needs]

   Debtor = container.selectbox(label='Debtor', options=[0, 1], index=1)
   data["Debtor"] = [Debtor]

   Tuition_fees_up_to_date = container.selectbox(label='Tuition fees uptodate', options=[1, 0], index=1)
   data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]

   Scholarship_holder = container.selectbox(label='Scholarship holder', options=[1, 0], index=1)
   data["Scholarship_holder"] = [Scholarship_holder]

   Mothers_qualification = int(container.number_input(label='Mothers qualification', value=39))
   data["Mothers_qualification"] = Mothers_qualification

   Fathers_qualification = int(container.number_input(label='Fathers qualification', value=39))
   data["Fathers_qualification"] = Fathers_qualification

   Mothers_occupation = int(container.number_input(label='Mothers occupation', value=7))
   data["Mothers_occupation"] = Mothers_occupation

   Fathers_occupation = int(container.number_input(label='Fathers occupation', value=7))
   data["Fathers_occupation"] = Fathers_occupation
  
with col3:
   st.subheader("Ekonomi Makro")
   container = st.container(border=True)

   Daytime_evening_attendance = container.selectbox(label='Attendance', options=[0, 1], index=1)
   data["Daytime_evening_attendance"] = [Daytime_evening_attendance]

   Application_mode = int(container.number_input(label='Application mode', value=17))
   data["Application_mode"] = Application_mode

   Application_order = int(container.number_input(label='Application order', value=5))
   data["Application_order"] = Application_order

   Course = int(container.number_input(label='Course', value=9254))
   data["Course"] = Course

   Unemployment_rate = float(container.number_input(label='Unemployment rate', value=10.8))
   data["Unemployment_rate"] = Unemployment_rate

   Inflation_rate = float(container.number_input(label='Inflation rate', value=1.4))
   data["Inflation_rate"] = Inflation_rate

   GDP = float(container.number_input(label='GDP', value=1.74))
   data["GDP"] = GDP

   Previous_qualification = int(container.number_input(label='Previous qualification', value=19))
   data["Previous_qualification"] = Previous_qualification


col1, col2, col3 = st.columns(3)
with col1:
   st.subheader("Akademik 1")
   container = st.container(border=True)

   Admission_grade = float(container.number_input(label='Admission grade', value=128.4))
   data["Admission_grade"] = Admission_grade

   Curricular_units_1st_sem_credited = int(container.number_input(label='Curricular 1st sem credited', value=2))
   data["Curricular_units_1st_sem_credited"] = Curricular_units_1st_sem_credited

   Curricular_units_1st_sem_enrolled = int(container.number_input(label='Curricular 1st sem enrolled', value=6))
   data["Curricular_units_1st_sem_enrolled"] = Curricular_units_1st_sem_enrolled

   Curricular_units_1st_sem_evaluations = int(container.number_input(label='Curricular 1st sem evaluations', value=8))
   data["Curricular_units_1st_sem_evaluations"] = Curricular_units_1st_sem_evaluations

   Curricular_units_1st_sem_approved = int(container.number_input(label='Curricular 1st sem approved', value=5))
   data["Curricular_units_1st_sem_approved"] = Curricular_units_1st_sem_approved

   Curricular_units_1st_sem_grade = float(container.number_input(label='Curricular 1st sem_grade', value=10.5))
   data["Curricular_units_1st_sem_grade"] = Curricular_units_1st_sem_grade

   Curricular_units_1st_sem_without_evaluations = int(container.number_input(label='Curricular 1st sem without evaluations', value=12))
   data["Curricular_units_1st_sem_without_evaluations"] = Curricular_units_1st_sem_without_evaluations

with col2:
   st.subheader("Akademik 2")
   container = st.container(border=True)

   Previous_qualification_grade = float(container.number_input(label='Previous qualification grade', value=154.4))
   data["Previous_qualification_grade"] = Previous_qualification_grade
   
   Curricular_units_2nd_sem_credited = int(container.number_input(label='Curricular 2nd sem credited', value=2))
   data["Curricular_units_2nd_sem_credited"] = Curricular_units_2nd_sem_credited

   Curricular_units_2nd_sem_enrolled = int(container.number_input(label='Curricular 2nd sem enrolled', value=6))
   data["Curricular_units_2nd_sem_enrolled"] = Curricular_units_2nd_sem_enrolled

   Curricular_units_2nd_sem_evaluations = int(container.number_input(label='Curricular 2nd sem evaluations', value=8))
   data["Curricular_units_2nd_sem_evaluations"] = Curricular_units_2nd_sem_evaluations

   Curricular_units_2nd_sem_approved = int(container.number_input(label='Curricular 2nd sem approved', value=5))
   data["Curricular_units_2nd_sem_approved"] = Curricular_units_2nd_sem_approved

   Curricular_units_2nd_sem_grade = float(container.number_input(label='Curricular 2nd sem grade', value=10.5))
   data["Curricular_units_2nd_sem_grade"] = Curricular_units_2nd_sem_grade

   Curricular_units_2nd_sem_without_evaluations = int(container.number_input(label='Curricular 2nd sem without evaluations', value=12))
   data["Curricular_units_2nd_sem_without_evaluations"] = Curricular_units_2nd_sem_without_evaluations

with col3:
   st.subheader("Prediksi")
   container = st.container(border=True)
   with container.expander("View the Raw Data"):
      container.dataframe(data=data, width=800, height=10)

   if container.button('Predict'):
    new_data = data_preprocessing(data=data)
    with container.expander("View the Preprocessed Data"):
       container.dataframe(data=new_data, width=800, height=10)
    container.subheader("Prediction : ")
    container.title(":blue[{}]".format(prediction(new_data)))
