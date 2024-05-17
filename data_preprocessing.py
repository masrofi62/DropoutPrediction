import joblib
import numpy as np
import pandas as pd

pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")
pca_3 = joblib.load("model/pca_3.joblib")

scaler_Marital_status = joblib.load("model/scaler_Marital_status.joblib")
scaler_Application_mode = joblib.load("model/scaler_Application_mode.joblib")
scaler_Course = joblib.load("model/scaler_Course.joblib")
scaler_Daytime_evening_attendance = joblib.load("model/scaler_Daytime_evening_attendance.joblib")
scaler_Previous_qualification = joblib.load("model/scaler_Previous_qualification.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Nacionality = joblib.load("model/scaler_Nacionality.joblib")
scaler_Mothers_qualification = joblib.load("model/scaler_Mothers_qualification.joblib")
scaler_Fathers_qualification = joblib.load("model/scaler_Fathers_qualification.joblib")
scaler_Mothers_occupation = joblib.load("model/scaler_Mothers_occupation.joblib")
scaler_Fathers_occupation = joblib.load("model/scaler_Fathers_occupation.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Displaced = joblib.load("model/scaler_Displaced.joblib")
scaler_Educational_special_needs = joblib.load("model/scaler_Educational_special_needs.joblib")
scaler_Debtor = joblib.load("model/scaler_Debtor.joblib")
scaler_Tuition_fees_up_to_date = joblib.load("model/scaler_Tuition_fees_up_to_date.joblib")
scaler_Gender = joblib.load("model/scaler_Gender.joblib")
scaler_Scholarship_holder = joblib.load("model/scaler_Scholarship_holder.joblib")
scaler_International = joblib.load("model/scaler_International.joblib")
scaler_Unemployment_rate = joblib.load("model/scaler_Unemployment_rate.joblib")
scaler_Inflation_rate = joblib.load("model/scaler_Inflation_rate.joblib")
scaler_GDP = joblib.load("model/scaler_GDP.joblib")
scaler_Application_order = joblib.load("model/scaler_Application_order.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_credited = joblib.load("model/scaler_Curricular_units_1st_sem_credited.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_1st_sem_without_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_without_evaluations.joblib")
scaler_Curricular_units_2nd_sem_credited = joblib.load("model/scaler_Curricular_units_2nd_sem_credited.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_without_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_without_evaluations.joblib")

pca_columns_1 = ['Marital_status',
                 'Nacionality',
                 'Displaced',
                 'Gender',
                 'Age_at_enrollment',
                 'International'
]

pca_columns_2 = ['Mothers_qualification',
                'Fathers_qualification',
                'Mothers_occupation',
                'Fathers_occupation',
                'Educational_special_needs',
                'Debtor',
                'Tuition_fees_up_to_date',
                'Scholarship_holder',
                'Unemployment_rate',
                'Inflation_rate',
                'GDP',
                'Application_mode',
                'Application_order',
                'Course',
                'Daytime_evening_attendance',
                'Previous_qualification'
]

pca_columns_3 = ['Curricular_units_1st_sem_credited',
                 'Curricular_units_1st_sem_enrolled',
                 'Curricular_units_1st_sem_evaluations',
                 'Curricular_units_1st_sem_approved',
                 'Curricular_units_1st_sem_grade',
                 'Curricular_units_1st_sem_without_evaluations',
                 'Curricular_units_2nd_sem_credited',
                 'Curricular_units_2nd_sem_enrolled',
                 'Curricular_units_2nd_sem_evaluations',
                 'Curricular_units_2nd_sem_approved',
                 'Curricular_units_2nd_sem_grade',
                 'Curricular_units_2nd_sem_without_evaluations',
                 'Previous_qualification_grade',
                 'Admission_grade'
]

def data_preprocessing(data):

    data = data.copy()
    df = pd.DataFrame()

    #PCA 1
    data["Marital_status"] = scaler_Marital_status.transform(np.asarray(data["Marital_status"]).reshape(-1,1))[0]
    data['Nacionality'] = scaler_Nacionality.transform(np.asarray(data["Nacionality"]).reshape(-1,1))[0]
    data['Displaced'] = scaler_Displaced.transform(np.asarray(data["Displaced"]).reshape(-1,1))[0]
    data['Gender'] = scaler_Gender.transform(np.asarray(data["Gender"]).reshape(-1,1))[0]
    data['Age_at_enrollment'] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))[0]
    data['International'] = scaler_International.transform(np.asarray(data["International"]).reshape(-1,1))[0]

    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4"]] = pca_1.transform(data[pca_columns_1])

    #PCA 2
    data['Mothers_qualification'] = scaler_Mothers_qualification.transform(np.asarray(data["Mothers_qualification"]).reshape(-1,1))[0]
    data['Fathers_qualification'] = scaler_Fathers_qualification.transform(np.asarray(data["Fathers_qualification"]).reshape(-1,1))[0]
    data['Mothers_occupation'] = scaler_Mothers_occupation.transform(np.asarray(data["Mothers_occupation"]).reshape(-1,1))[0]
    data['Fathers_occupation'] = scaler_Fathers_occupation.transform(np.asarray(data["Fathers_occupation"]).reshape(-1,1))[0]
    data['Educational_special_needs'] = scaler_Educational_special_needs.transform(np.asarray(data["Educational_special_needs"]).reshape(-1,1))[0]
    data['Debtor'] = scaler_Debtor.transform(np.asarray(data["Debtor"]).reshape(-1,1))[0]
    data['Tuition_fees_up_to_date'] = scaler_Tuition_fees_up_to_date.transform(np.asarray(data["Tuition_fees_up_to_date"]).reshape(-1,1))[0]
    data['Scholarship_holder'] = scaler_Scholarship_holder.transform(np.asarray(data["Scholarship_holder"]).reshape(-1,1))[0]
    data['Unemployment_rate'] = scaler_Unemployment_rate.transform(np.asarray(data["Unemployment_rate"]).reshape(-1,1))[0]
    data['Inflation_rate'] = scaler_Inflation_rate.transform(np.asarray(data["Inflation_rate"]).reshape(-1,1))[0]
    data['GDP'] = scaler_GDP.transform(np.asarray(data["GDP"]).reshape(-1,1))[0]
    data['Application_mode'] = scaler_Application_mode.transform(np.asarray(data["Application_mode"]).reshape(-1,1))[0]
    data['Application_order'] = scaler_Application_order.transform(np.asarray(data["Application_order"]).reshape(-1,1))[0]
    data['Course'] = scaler_Course.transform(np.asarray(data["Course"]).reshape(-1,1))[0]
    data['Daytime_evening_attendance'] = scaler_Daytime_evening_attendance.transform(np.asarray(data["Daytime_evening_attendance"]).reshape(-1,1))[0]
    data['Previous_qualification'] = scaler_Previous_qualification.transform(np.asarray(data["Previous_qualification"]).reshape(-1,1))[0]

    df[["pc2_1", "pc2_2", "pc2_3", "pc2_4", "pc2_5", "pc2_6", "pc2_7", "pc2_8", "pc2_9", "pc2_10"]] = pca_2.transform(data[pca_columns_2])

    #PCA 3
    data['Curricular_units_1st_sem_credited'] = scaler_Curricular_units_1st_sem_credited.transform(np.asarray(data["Curricular_units_1st_sem_credited"]).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_enrolled'] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_evaluations'] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_approved'] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_grade'] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))[0]
    data['Curricular_units_1st_sem_without_evaluations'] = scaler_Curricular_units_1st_sem_without_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_without_evaluations"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_credited'] = scaler_Curricular_units_2nd_sem_credited.transform(np.asarray(data["Curricular_units_2nd_sem_credited"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_enrolled'] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_evaluations'] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_approved'] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_grade'] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))[0]
    data['Curricular_units_2nd_sem_without_evaluations'] = scaler_Curricular_units_2nd_sem_without_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_without_evaluations"]).reshape(-1,1))[0]
    data['Previous_qualification_grade'] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1,1))[0]
    data['Admisson_grade'] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1,1))[0]

    df[["pc3_1", "pc3_2", "pc3_3", "pc3_4", "pc3_5", "pc3_6", "pc3_7"]] = pca_3.transform(data[pca_columns_3])
    
    return df