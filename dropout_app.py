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

options_Nacionality=['Portuguese',
                     'German',
                     'Spanish',
                     'Italian',
                     'Dutch',
                     'English',
                     'Lithuanian',
                     'Angolan',
                     'Cape Verdean',
                     'Guinean',
                     'Mozambican',
                     'Santomean',
                     'Turkish',
                     'Brazilian',
                     'Romanian',
                     'Moldova',
                     'Mexican',
                     'Ukrainian',
                     'Russian',
                     'Cuban',
                     'Colombian']

options_Previous_qualification=['1:Secondary education',
                     "2:Higher education - bachelor's degree",
                     '3:Higher education - degree',
                     "4:Higher education - master's",
                     '5:Higher education - doctorate',
                     '6:Frequency of higher education',
                     '9:12th year of schooling - not completed',
                     '10:11th year of schooling - not completed',
                     '12:Other - 11th year of schooling',
                     '14:10th year of schooling',
                     '15:10th year of schooling - not completed',
                     '19:Basic education 3rd cycle (9th/10th/11th year) or equiv',
                     '38:Basic education 2nd cycle (6th/7th/8th year) or equiv',
                     '39:Technological specialization course',
                     '40:Higher education - degree (1st cycle)',
                     '42:Professional higher technical course',
                     '43:Higher education - master (2nd cycle)']

options_Application_mode=['1:1st phase - general contingent',
                     '2:Ordinance No. 612/93',
                     '7:1st phase - special contingent (Azores Island)',
                     '10:Ordinance No. 854-B/99',
                     '15:International student (bachelor)',
                     '16:1st phase - special contingent (Madeira Island)',
                     '17:2nd phase - general contingent',
                     '18:3rd phase - general contingent',
                     '26:Ordinance No. 533-A/99, item b2) (Different Plan)',
                     '27:Ordinance No. 533-A/99, item b3 (Other Institution)',
                     '39:Over 23 years old',
                     '42:Transfer',
                     '43:Change of course',
                     '44:Technological specialization diploma holders',
                     '51:Change of institution/course',
                     '53:Short cycle diploma holders',
                     '57:Change of institution/course (International)']

options_Mothers_qualification=['1:Secondary education - 12th Year of Schooling or Eq',
                     "2:Higher education - bachelor's degree",
                     '3:Higher education - degree',
                     "4:Higher education - master's",
                     '5:Higher education - doctorate',
                     '6:Frequency of higher education',
                     '9:12th year of schooling - not completed',
                     '10:11th year of schooling - not completed',
                     '11:7th Year (Old)',
                     '12:Other - 11th year of schooling',
                     '14:10th year of schooling',
                     '18:General commerce course',
                     '19:Basic education 3rd cycle (9th/10th/11th year) or equiv',
                     '22:Technical-professional course',
                     '26:7th year of schooling',
                     '27:2nd cycle of the general high school course',
                     '29:9th Year of Schooling - Not Completed',
                     '30:8th year of schooling',
                     '34:Unknown',
                     "35:Can't read or write",
                     '36:Can read without having a 4th year of schooling',
                     '37:Basic education 1st cycle (4th/5th year) or equiv',
                     '38:Basic education 2nd cycle (6th/7th/8th year) or equiv',
                     '39:Technological specialization course',
                     '40:Higher education - degree (1st cycle)',
                     '41:Specialized higher studies course',
                     '42:Professional higher technical course',
                     '43:Higher education - master (2nd cycle)',
                     '44:Higher Education - Doctorate (3rd cycle)']

options_Fathers_qualification=['1:Secondary education - 12th Year of Schooling or Eq',
                     "2:Higher education - bachelor's degree",
                     '3:Higher education - degree',
                     "4:Higher education - master's",
                     '5:Higher education - doctorate',
                     '6:Frequency of higher education',
                     '9:12th year of schooling - not completed',
                     '10:11th year of schooling - not completed',
                     '11:7th Year (Old)',
                     '12:Other - 11th year of schooling',
                     '14:10th year of schooling',
                     '18:General commerce course',
                     '19:Basic education 3rd cycle (9th/10th/11th year) or equiv',
                     '22:Technical-professional course',
                     '26:7th year of schooling',
                     '27:2nd cycle of the general high school course',
                     '29:9th Year of Schooling - Not Completed',
                     '30:8th year of schooling',
                     '34:Unknown',
                     "35:Can't read or write",
                     '36:Can read without having a 4th year of schooling',
                     '37:Basic education 1st cycle (4th/5th year) or equiv',
                     '38:Basic education 2nd cycle (6th/7th/8th year) or equiv',
                     '39:Technological specialization course',
                     '40:Higher education - degree (1st cycle)',
                     '41:Specialized higher studies course',
                     '42:Professional higher technical course',
                     '43:Higher education - master (2nd cycle)',
                     '44:Higher Education - Doctorate (3rd cycle)']

options_Mothers_occupation=['0:Student',
                     '1:Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
                     '2:Specialists in Intellectual and Scientific Activities',
                     '3:Intermediate Level Technicians and Professions',
                     '4:Administrative staff',
                     '5:Personal Services, Security and Safety Workers and Sellers',
                     '6:Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
                     '7:Skilled Workers in Industry, Construction and Craftsmen ',
                     '8:Installation and Machine Operators and Assembly Workers',
                     '9:Unskilled Workers',
                     '10:Armed Forces Professions',
                     '90:Other Situation',
                     '99:(blank)',
                     '122:Health professionals',
                     '123:teachers',
                     '125:Specialists in information and communication technologies (ICT)',
                     '131:Intermediate level science and engineering technicians and professions',
                     '132:Technicians and professionals, of intermediate level of health',
                     '134:Intermediate level technicians from legal, social, sports, cultural and similar services',
                     '141:Office workers, secretaries in general and data processing operators',
                     '143:Data, accounting, statistical, financial services and registry-related operators',
                     '144:Other administrative support staff',
                     '151:personal service workers',
                     '152:sellers',
                     '153:Personal care workers and the like',
                     '171:Skilled construction workers and the like, except electricians',
                     '173:Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
                     '175:Workers in food processing, woodworking, clothing and other industries and crafts',
                     '191:cleaning workers',
                     '192:Unskilled workers in agriculture, animal production, fisheries and forestry',
                     '193:Unskilled workers in extractive industry, construction, manufacturing and transport',
                     '194:Meal preparation assistants']

options_Fathers_occupation=['0:Student',
                     '1:Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers',
                     '2:Specialists in Intellectual and Scientific Activities',
                     '3:Intermediate Level Technicians and Professions',
                     '4:Administrative staff',
                     '5:Personal Services, Security and Safety Workers and Sellers',
                     '6:Farmers and Skilled Workers in Agriculture, Fisheries and Forestry',
                     '7:Skilled Workers in Industry, Construction and Craftsmen ',
                     '8:Installation and Machine Operators and Assembly Workers',
                     '9:Unskilled Workers',
                     '10:Armed Forces Professions',
                     '90:Other Situation',
                     '99:(blank)',
                     '122:Health professionals',
                     '123:teachers',
                     '125:Specialists in information and communication technologies (ICT)',
                     '131:Intermediate level science and engineering technicians and professions',
                     '132:Technicians and professionals, of intermediate level of health',
                     '134:Intermediate level technicians from legal, social, sports, cultural and similar services',
                     '141:Office workers, secretaries in general and data processing operators',
                     '143:Data, accounting, statistical, financial services and registry-related operators',
                     '144:Other administrative support staff',
                     '151:personal service workers',
                     '152:sellers',
                     '153:Personal care workers and the like',
                     '171:Skilled construction workers and the like, except electricians',
                     '173:Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like',
                     '175:Workers in food processing, woodworking, clothing and other industries and crafts',
                     '191:cleaning workers',
                     '192:Unskilled workers in agriculture, animal production, fisheries and forestry',
                     '193:Unskilled workers in extractive industry, construction, manufacturing and transport',
                     '194:Meal preparation assistants']

options_Course=['33:Biofuel Production Technologies',
                     '171:Animation and Multimedia Design',
                     '8014:Social Service (evening attendance)',
                     '9003:Agronomy',
                     '9070:Communication Design',
                     '9085:Veterinary Nursing',
                     '9119:Informatics Engineering',
                     '9130:Equinculture',
                     '9147:Management',
                     '9238:Social Service',
                     '9254:Tourism',
                     '9500:Nursing',
                     '9556:Oral Hygiene',
                     '9670:Advertising and Marketing Management',
                     '9773:Journalism and Communication',
                     '9853:Basic Education',
                     '9991:Management (evening attendance)']

col1, col2, col3 = st.columns(3)
with col1:
   st.subheader("Demografi")
   container = st.container(border=True)

   Gender = container.selectbox(label='Gender', options=['male','female'], index=1)
   if Gender == 'male':
       data["Gender"] = [1]
   else :
       data["Gender"] = [0]

   International = container.selectbox(label='International', options=['yes','no'], index=1)
   if International == 'yes':
       data["International"] = [1]
   else :
       data["International"] = [0]

   Nacionality = container.selectbox(label='Nacionality', options = options_Nacionality, index=1)
   if Nacionality == 'Portuguese':
       data["Nacionality"] = [1]
   elif Nacionality == 'German':
       data["Nacionality"] = [2]
   elif Nacionality == 'Spanish':
       data["Nacionality"] = [6]
   elif Nacionality == 'Italian':
       data["Nacionality"] = [11]
   elif Nacionality == 'Dutch':
       data["Nacionality"] = [13]
   elif Nacionality == 'English':
       data["Nacionality"] = [14]
   elif Nacionality == 'Lithuanian':
       data["Nacionality"] = [17]
   elif Nacionality == 'Angolan':
       data["Nacionality"] = [21]  
   elif Nacionality == 'Cape Verdean':
       data["Nacionality"] = [22]
   elif Nacionality == 'Guinean':
       data["Nacionality"] = [24]
   elif Nacionality == 'Mozambican':
       data["Nacionality"] = [25]
   elif Nacionality == 'Santomean':
       data["Nacionality"] = [26]
   elif Nacionality == 'Turkish':
       data["Nacionality"] = [32]
   elif Nacionality == 'Brazilian':
       data["Nacionality"] = [41]
   elif Nacionality == 'Romanian':
       data["Nacionality"] = [62]       
   elif Nacionality == 'Moldova':
       data["Nacionality"] = [100]
   elif Nacionality == 'Mexican':
       data["Nacionality"] = [101]
   elif Nacionality == 'Ukrainian':
       data["Nacionality"] = [103]
   elif Nacionality == 'Russian':
       data["Nacionality"] = [105]
   elif Nacionality == 'Cuban':
       data["Nacionality"] = [108]
   else :
       data["Nacionality"] = [109]

   Age_at_enrollment = int(container.number_input(label='Age', value=18))
   data["Age_at_enrollment"] = Age_at_enrollment

   Displaced = container.selectbox(label='Displaced', options=['yes','no'], index=1)
   if Displaced == 'yes':
       data["Displaced"] = [1]
   else :
       data["Displaced"] = [0]

   Marital_status = container.selectbox(label='Marital status', options = ['single','widower','Graduate','divorced','facto union','legally separated',], index=1)
   if Marital_status == 'Portuguese':
       data["Marital_status"] = [1]
   elif Marital_status == 'German':
       data["Marital_status"] = [2]
   elif Marital_status == 'Spanish':
       data["Marital_status"] = [3]
   elif Marital_status == 'Italian':
       data["Marital_status"] = [4]
   elif Marital_status == 'Dutch':
       data["Marital_status"] = [5]
   else:
       data["Marital_status"] = [6]


   
with col2:
   st.subheader("Sosial Ekonomi")
   container = st.container(border=True)

   Educational_special_needs = container.selectbox(label='Educational special needs', options=['yes','no'], index=1)
   if Educational_special_needs == 'yes':
       data["Educational_special_needs"] = [1]
   else :
       data["Educational_special_needs"] = [0]

   Debtor = container.selectbox(label='Debtor', options=['yes','no'], index=1)
   if Debtor == 'yes':
       data["Debtor"] = [1]
   else :
       data["Debtor"] = [0]

   Tuition_fees_up_to_date = container.selectbox(label='Tuition fees uptodate', options=['yes','no'], index=1)
   if Tuition_fees_up_to_date == 'yes':
       data["Tuition_fees_up_to_date"] = [1]
   else :
       data["Tuition_fees_up_to_date"] = [0]

   Scholarship_holder = container.selectbox(label='Scholarship holder', options=['yes','no'], index=1)
   if Scholarship_holder == 'yes':
       data["Scholarship_holder"] = [1]
   else :
       data["Scholarship_holder"] = [0]

   Mothers_qualification = container.selectbox(label='Mothers qualification', options=options_Mothers_qualification, index=1)
   if Mothers_qualification == 'Secondary education - 12th Year of Schooling or Eq':
       data["Mothers_qualification"] = [1]
   elif Mothers_qualification == "2:Higher education - bachelor's degree":
       data["Mothers_qualification"] = [2]
   elif Mothers_qualification == '3:Higher education - degree':
       data["Mothers_qualification"] = [3]
   elif Mothers_qualification == "4:Higher education - master's":
       data["Mothers_qualification"] = [4]
   elif Mothers_qualification == '5:Higher education - doctorate':
       data["Mothers_qualification"] = [5]
   elif Mothers_qualification == '6:Frequency of higher education':
       data["Mothers_qualification"] = [6]
   elif Mothers_qualification == '9:12th year of schooling - not completed':
       data["Mothers_qualification"] = [9]
   elif Mothers_qualification == '10:11th year of schooling - not completed':
       data["Mothers_qualification"] = [10]  
   elif Mothers_qualification == '11:7th Year (Old)':
       data["Mothers_qualification"] = [11]
   elif Mothers_qualification == '12:Other - 11th year of schooling':
       data["Mothers_qualification"] = [12]
   elif Mothers_qualification == '14:10th year of schooling':
       data["Mothers_qualification"] = [14]
   elif Mothers_qualification == '18:General commerce course':
       data["Mothers_qualification"] = [18]
   elif Mothers_qualification == '19:Basic education 3rd cycle (9th/10th/11th year) or equiv':
       data["Mothers_qualification"] = [19]
   elif Mothers_qualification == '22:Technical-professional course':
       data["Mothers_qualification"] = [22]
   elif Mothers_qualification == '26:7th year of schooling':
       data["Mothers_qualification"] = [26]       
   elif Mothers_qualification == '27:2nd cycle of the general high school course':
       data["Mothers_qualification"] = [27]
   elif Mothers_qualification == '29:9th Year of Schooling - Not Completed':
       data["Mothers_qualification"] = [29]
   elif Mothers_qualification == '30:8th year of schooling':
       data["Mothers_qualification"] = [30]
   elif Mothers_qualification == '34:Unknown':
       data["Mothers_qualification"] = [34]
   elif Mothers_qualification == "35:Can't read or write":
       data["Mothers_qualification"] = [35]
   elif Mothers_qualification == '36:Can read without having a 4th year of schooling':
       data["Mothers_qualification"] = [36]
   elif Mothers_qualification == '37:Basic education 1st cycle (4th/5th year) or equiv':
       data["Mothers_qualification"] = [37]
   elif Mothers_qualification == '38:Basic education 2nd cycle (6th/7th/8th year) or equiv':
       data["Mothers_qualification"] = [38]       
   elif Mothers_qualification == '39:Technological specialization course':
       data["Mothers_qualification"] = [39]
   elif Mothers_qualification == '40:Higher education - degree (1st cycle)':
       data["Mothers_qualification"] = [40]
   elif Mothers_qualification == '41:Specialized higher studies course':
       data["Mothers_qualification"] = [41]
   elif Mothers_qualification == '42:Professional higher technical course':
       data["Mothers_qualification"] = [42]
   elif Mothers_qualification == "43:Higher education - master (2nd cycle)":
       data["Mothers_qualification"] = [43]
   else :
       data["Mothers_qualification"] = [44]
   
   Fathers_qualification = container.selectbox(label='Fathers qualification', options=options_Fathers_qualification, index=1)
   if Fathers_qualification == 'Secondary education - 12th Year of Schooling or Eq':
       data["Fathers_qualification"] = [1]
   elif Fathers_qualification == "2:Higher education - bachelor's degree":
       data["Fathers_qualification"] = [2]
   elif Fathers_qualification == '3:Higher education - degree':
       data["Fathers_qualification"] = [3]
   elif Fathers_qualification == "4:Higher education - master's":
       data["Fathers_qualification"] = [4]
   elif Fathers_qualification == '5:Higher education - doctorate':
       data["Fathers_qualification"] = [5]
   elif Fathers_qualification == '6:Frequency of higher education':
       data["Fathers_qualification"] = [6]
   elif Fathers_qualification == '9:12th year of schooling - not completed':
       data["Fathers_qualification"] = [9]
   elif Fathers_qualification == '10:11th year of schooling - not completed':
       data["Fathers_qualification"] = [10]  
   elif Fathers_qualification == '11:7th Year (Old)':
       data["Fathers_qualification"] = [11]
   elif Fathers_qualification == '12:Other - 11th year of schooling':
       data["Fathers_qualification"] = [12]
   elif Fathers_qualification == '14:10th year of schooling':
       data["Fathers_qualification"] = [14]
   elif Fathers_qualification == '18:General commerce course':
       data["Fathers_qualification"] = [18]
   elif Fathers_qualification == '19:Basic education 3rd cycle (9th/10th/11th year) or equiv':
       data["Fathers_qualification"] = [19]
   elif Fathers_qualification == '22:Technical-professional course':
       data["Fathers_qualification"] = [22]
   elif Fathers_qualification == '26:7th year of schooling':
       data["Fathers_qualification"] = [26]       
   elif Fathers_qualification == '27:2nd cycle of the general high school course':
       data["Fathers_qualification"] = [27]
   elif Fathers_qualification == '29:9th Year of Schooling - Not Completed':
       data["Fathers_qualification"] = [29]
   elif Fathers_qualification == '30:8th year of schooling':
       data["Fathers_qualification"] = [30]
   elif Fathers_qualification == '34:Unknown':
       data["Fathers_qualification"] = [34]
   elif Fathers_qualification == "35:Can't read or write":
       data["Fathers_qualification"] = [35]
   elif Fathers_qualification == '36:Can read without having a 4th year of schooling':
       data["Fathers_qualification"] = [36]
   elif Fathers_qualification == '37:Basic education 1st cycle (4th/5th year) or equiv':
       data["Fathers_qualification"] = [37]
   elif Fathers_qualification == '38:Basic education 2nd cycle (6th/7th/8th year) or equiv':
       data["Fathers_qualification"] = [38]       
   elif Fathers_qualification == '39:Technological specialization course':
       data["Fathers_qualification"] = [39]
   elif Fathers_qualification == '40:Higher education - degree (1st cycle)':
       data["Fathers_qualification"] = [40]
   elif Fathers_qualification == '41:Specialized higher studies course':
       data["Fathers_qualification"] = [41]
   elif Fathers_qualification == '42:Professional higher technical course':
       data["Fathers_qualification"] = [42]
   elif Fathers_qualification == "43:Higher education - master (2nd cycle)":
       data["Fathers_qualification"] = [43]
   else :
       data["Fathers_qualification"] = [44]

   Mothers_occupation = container.selectbox(label='Mothers occupation', options=options_Mothers_occupation, index=1)
   if Mothers_occupation == '0:Student':
       data["Mothers_occupation"] = [0]
   elif Mothers_occupation == "1:Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers":
       data["Mothers_occupation"] = [1]
   elif Mothers_occupation == "2:Specialists in Intellectual and Scientific Activities":
       data["Mothers_occupation"] = [2]
   elif Mothers_occupation == '3:Intermediate Level Technicians and Professions':
       data["Mothers_occupation"] = [3]
   elif Mothers_occupation == "4:Administrative staff":
       data["Mothers_occupation"] = [4]
   elif Mothers_occupation == '5:Personal Services, Security and Safety Workers and Sellers':
       data["Mothers_occupation"] = [5]
   elif Mothers_occupation == '6:Farmers and Skilled Workers in Agriculture, Fisheries and Forestry':
       data["Mothers_occupation"] = [6]
   elif Mothers_occupation == '7:Skilled Workers in Industry, Construction and Craftsmen':
       data["Mothers_occupation"] = [7]
   elif Mothers_occupation == '8:Installation and Machine Operators and Assembly Workers':
       data["Mothers_occupation"] = [8]  
   elif Mothers_occupation == '9:Unskilled Workers':
       data["Mothers_occupation"] = [9]
   elif Mothers_occupation == '10:Armed Forces Professions':
       data["Mothers_occupation"] = [10]
   elif Mothers_occupation == '90:Other Situation':
       data["Mothers_occupation"] = [90]
   elif Mothers_occupation == '99:(blank)':
       data["Mothers_occupation"] = [99]
   elif Mothers_occupation == '122:Health professionals':
       data["Mothers_occupation"] = [122]
   elif Mothers_occupation == '123:teachers':
       data["Mothers_occupation"] = [123]
   elif Mothers_occupation == '125:Specialists in information and communication technologies (ICT)':
       data["Mothers_occupation"] = [125]       
   elif Mothers_occupation == '131:Intermediate level science and engineering technicians and professions':
       data["Mothers_occupation"] = [131]
   elif Mothers_occupation == '132:Technicians and professionals, of intermediate level of health':
       data["Mothers_occupation"] = [132]
   elif Mothers_occupation == '134:Intermediate level technicians from legal, social, sports, cultural and similar services':
       data["Mothers_occupation"] = [134]
   elif Mothers_occupation == '141:Office workers, secretaries in general and data processing operators':
       data["Mothers_occupation"] = [141]
   elif Mothers_occupation == "143:Data, accounting, statistical, financial services and registry-related operators":
       data["Mothers_occupation"] = [143]
   elif Mothers_occupation == '144:Other administrative support staff':
       data["Mothers_occupation"] = [144]
   elif Mothers_occupation == '151:personal service workers':
       data["Mothers_occupation"] = [151]
   elif Mothers_occupation == '152:sellers':
       data["Mothers_occupation"] = [152]       
   elif Mothers_occupation == '153:Personal care workers and the like':
       data["Mothers_occupation"] = [153]
   elif Mothers_occupation == '171:Skilled construction workers and the like, except electricians':
       data["Mothers_occupation"] = [171]
   elif Mothers_occupation == '173:Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like':
       data["Mothers_occupation"] = [173]
   elif Mothers_occupation == '175:Workers in food processing, woodworking, clothing and other industries and crafts':
       data["Mothers_occupation"] = [175]
   elif Mothers_occupation == "191:cleaning workers":
       data["Mothers_occupation"] = [191]
   elif Mothers_occupation == "192:Unskilled workers in agriculture, animal production, fisheries and forestry":
       data["Mothers_occupation"] = [192]
   elif Mothers_occupation == "193:Unskilled workers in extractive industry, construction, manufacturing and transport":
       data["Mothers_occupation"] = [193]
   else :
       data["Mothers_occupation"] = [194]

   Fathers_occupation = container.selectbox(label='Fathers occupation', options=options_Fathers_occupation, index=1)
   if Fathers_occupation == '0:Student':
       data["Fathers_occupation"] = [0]
   elif Fathers_occupation == "1:Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers":
       data["Fathers_occupation"] = [1]
   elif Fathers_occupation == "2:Specialists in Intellectual and Scientific Activities":
       data["Fathers_occupation"] = [2]
   elif Fathers_occupation == '3:Intermediate Level Technicians and Professions':
       data["Fathers_occupation"] = [3]
   elif Fathers_occupation == "4:Administrative staff":
       data["Fathers_occupation"] = [4]
   elif Fathers_occupation == '5:Personal Services, Security and Safety Workers and Sellers':
       data["Fathers_occupation"] = [5]
   elif Fathers_occupation == '6:Farmers and Skilled Workers in Agriculture, Fisheries and Forestry':
       data["Fathers_occupation"] = [6]
   elif Fathers_occupation == '7:Skilled Workers in Industry, Construction and Craftsmen':
       data["Fathers_occupation"] = [7]
   elif Fathers_occupation == '8:Installation and Machine Operators and Assembly Workers':
       data["Fathers_occupation"] = [8]  
   elif Fathers_occupation == '9:Unskilled Workers':
       data["Fathers_occupation"] = [9]
   elif Fathers_occupation == '10:Armed Forces Professions':
       data["Fathers_occupation"] = [10]
   elif Fathers_occupation == '90:Other Situation':
       data["Fathers_occupation"] = [90]
   elif Fathers_occupation == '99:(blank)':
       data["Fathers_occupation"] = [99]
   elif Fathers_occupation == '122:Health professionals':
       data["Fathers_occupation"] = [122]
   elif Fathers_occupation == '123:teachers':
       data["Fathers_occupation"] = [123]
   elif Fathers_occupation == '125:Specialists in information and communication technologies (ICT)':
       data["Fathers_occupation"] = [125]       
   elif Fathers_occupation == '131:Intermediate level science and engineering technicians and professions':
       data["Fathers_occupation"] = [131]
   elif Fathers_occupation == '132:Technicians and professionals, of intermediate level of health':
       data["Fathers_occupation"] = [132]
   elif Fathers_occupation == '134:Intermediate level technicians from legal, social, sports, cultural and similar services':
       data["Fathers_occupation"] = [134]
   elif Fathers_occupation == '141:Office workers, secretaries in general and data processing operators':
       data["Fathers_occupation"] = [141]
   elif Fathers_occupation == "143:Data, accounting, statistical, financial services and registry-related operators":
       data["Fathers_occupation"] = [143]
   elif Fathers_occupation == '144:Other administrative support staff':
       data["Fathers_occupation"] = [144]
   elif Fathers_occupation == '151:personal service workers':
       data["Fathers_occupation"] = [151]
   elif Fathers_occupation == '152:sellers':
       data["Fathers_occupation"] = [152]       
   elif Fathers_occupation == '153:Personal care workers and the like':
       data["Fathers_occupation"] = [153]
   elif Fathers_occupation == '171:Skilled construction workers and the like, except electricians':
       data["Fathers_occupation"] = [171]
   elif Fathers_occupation == '173:Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like':
       data["Fathers_occupation"] = [173]
   elif Fathers_occupation == '175:Workers in food processing, woodworking, clothing and other industries and crafts':
       data["Fathers_occupation"] = [175]
   elif Fathers_occupation == "191:cleaning workers":
       data["Fathers_occupation"] = [191]
   elif Fathers_occupation == "192:Unskilled workers in agriculture, animal production, fisheries and forestry":
       data["Fathers_occupation"] = [192]
   elif Fathers_occupation == "193:Unskilled workers in extractive industry, construction, manufacturing and transport":
       data["Fathers_occupation"] = [193]
   else :
       data["Fathers_occupation"] = [194]

  
with col3:
   st.subheader("Ekonomi Makro")
   container = st.container(border=True)

   Daytime_evening_attendance = container.selectbox(label='Attendance', options=['daytime','evening'], index=1)
   if Daytime_evening_attendance == 'daytime':
       data["Daytime_evening_attendance"] = [1]
   else :
       data["Daytime_evening_attendance"] = [0]

   Application_mode = int(container.number_input(label='Application mode', value=17))
   data["Application_mode"] = Application_mode
   Application_mode = container.selectbox(label='Application mode', options=options_Application_mode, index=1)
   if Application_mode == '1:Secondary education':
       data["Application_mode"] = [1]
   elif Application_mode == "2:Higher education - bachelor's degree":
       data["Application_mode"] = [2]
   elif Application_mode == '3:Higher education - degree':
       data["Application_mode"] = [3]
   elif Application_mode == "4:Higher education - master's":
       data["Application_mode"] = [4]
   elif Application_mode == '5:Higher education - doctorate':
       data["Application_mode"] = [5]
   elif Application_mode == '6:Frequency of higher education':
       data["Application_mode"] = [6]
   elif Application_mode == '9:12th year of schooling - not completed':
       data["Application_mode"] = [9]
   elif Application_mode == '10:11th year of schooling - not completed':
       data["Application_mode"] = [10]  
   elif Application_mode == '12:Other - 11th year of schooling':
       data["Application_mode"] = [12]
   elif Application_mode == '14:10th year of schooling':
       data["Application_mode"] = [14]
   elif Application_mode == '15:10th year of schooling - not completed':
       data["Application_mode"] = [15]
   elif Application_mode == '19:Basic education 3rd cycle (9th/10th/11th year) or equiv':
       data["Application_mode"] = [19]
   elif Application_mode == '38:Basic education 2nd cycle (6th/7th/8th year) or equiv':
       data["Application_mode"] = [38]
   elif Application_mode == '39:Technological specialization course':
       data["Application_mode"] = [39]
   elif Application_mode == '40:Higher education - degree (1st cycle)':
       data["Application_mode"] = [40]       
   elif Application_mode == '42:Professional higher technical course':
       data["Application_mode"] = [42]
   else :
       data["Application_mode"] = [43]
   

   Application_order = int(container.number_input(label='Application order', value=5))
   data["Application_order"] = Application_order

   Course = container.selectbox(label='Course', options=options_Course, index=1)
   if Course == '33:Biofuel Production Technologies':
       data["Course"] = [33]
   elif Course == '171:Animation and Multimedia Design':
       data["Course"] = [171]
   elif Course == '8014:Social Service (evening attendance)':
       data["Course"] = [8014]
   elif Course == '9003:Agronomy':
       data["Course"] = [9003]
   elif Course == '9070:Communication Design':
       data["Course"] = [9070]
   elif Course == '9085:Veterinary Nursing':
       data["Course"] = [9085]
   elif Course == '9119:Informatics Engineering':
       data["Course"] = [9119]
   elif Course == '9130:Equinculture':
       data["Course"] = [9130 ]  
   elif Course == '9147:Management':
       data["Course"] = [9147]
   elif Course == '9238:Social Service':
       data["Course"] = [9238]
   elif Course == '9254:Tourism':
       data["Course"] = [9254]
   elif Course == '9500:Nursing':
       data["Course"] = [9500]
   elif Course == '9556:Oral Hygiene':
       data["Course"] = [9556]
   elif Course == '9670:Advertising and Marketing Management':
       data["Course"] = [9670]
   elif Course == '9773:Journalism and Communication':
       data["Course"] = [9773]       
   elif Course == '9853:Basic Education':
       data["Course"] = [9853]
   else :
       data["Course"] = [9991]


   Unemployment_rate = float(container.number_input(label='Unemployment rate', value=10.8))
   data["Unemployment_rate"] = Unemployment_rate

   Inflation_rate = float(container.number_input(label='Inflation rate', value=1.4))
   data["Inflation_rate"] = Inflation_rate

   GDP = float(container.number_input(label='GDP', value=1.74))
   data["GDP"] = GDP


   Previous_qualification = container.selectbox(label='Previous qualification', options=options_Previous_qualification, index=1)
   if Previous_qualification == '1:Secondary education':
       data["Previous_qualification"] = [1]
   elif Previous_qualification == "2:Higher education - bachelor's degree":
       data["Previous_qualification"] = [2]
   elif Previous_qualification == '3:Higher education - degree':
       data["Previous_qualification"] = [3]
   elif ApplicatiPrevious_qualificationon_mode == "4:Higher education - master's":
       data["Previous_qualification"] = [4]
   elif Previous_qualification == '5:Higher education - doctorate':
       data["Previous_qualification"] = [5]
   elif Previous_qualification == '6:Frequency of higher education':
       data["Previous_qualification"] = [6]
   elif Previous_qualification == '9:12th year of schooling - not completed':
       data["Previous_qualification"] = [9]
   elif Previous_qualification == '10:11th year of schooling - not completed':
       data["Previous_qualification"] = [10]  
   elif Previous_qualification == '12:Other - 11th year of schooling':
       data["Previous_qualification"] = [12]
   elif Previous_qualification == '14:10th year of schooling':
       data["Previous_qualification"] = [14]
   elif Previous_qualification == '15:10th year of schooling - not completed':
       data["Previous_qualification"] = [15]
   elif Previous_qualification == '19:Basic education 3rd cycle (9th/10th/11th year) or equiv':
       data["Previous_qualification"] = [19]
   elif Previous_qualification == '38:Basic education 2nd cycle (6th/7th/8th year) or equiv':
       data["Previous_qualification"] = [38]
   elif Previous_qualification == '39:Technological specialization course':
       data["Previous_qualification"] = [39]
   elif Previous_qualification == '40:Higher education - degree (1st cycle)':
       data["Previous_qualification"] = [40]       
   elif Previous_qualification == '42:Professional higher technical course':
       data["Previous_qualification"] = [42]
   else :
       data["Previous_qualification"] = [43]


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
