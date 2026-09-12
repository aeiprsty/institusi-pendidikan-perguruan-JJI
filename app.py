import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("model/dropout_prediction_model.pkl")

preprocessor = model.named_steps["preprocessor"]
encoder = preprocessor.named_transformers_["cat"]


# =========================================================
# FEATURE LIST
# =========================================================

categorical_features = [
    "Marital_status",
    "Application_mode",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Nacionality",
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
    "Displaced",
    "Educational_special_needs",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "International"
]


# =========================================================
# CATEGORY MAPPINGS
# =========================================================

marital_status_labels = {
    1: "Single",
    2: "Married",
    3: "Widower",
    4: "Divorced",
    5: "Facto Union",
    6: "Legally Separated"
}

application_mode_labels = {
    1: "1st Phase - General Contingent",
    2: "Ordinance No. 612/93",
    5: "1st Phase - Special Contingent (Azores Island)",
    7: "Holders of Other Higher Courses",
    10: "Ordinance No. 854-B/99",
    15: "International Student (Bachelor)",
    16: "1st Phase - Special Contingent (Madeira Island)",
    17: "2nd Phase - General Contingent",
    18: "3rd Phase - General Contingent",
    26: "Ordinance No. 533-A/99, Item b2",
    27: "Ordinance No. 533-A/99, Item b3",
    39: "Over 23 Years Old",
    42: "Transfer",
    43: "Change of Course",
    44: "Technological Specialization Diploma Holders",
    51: "Change of Institution/Course",
    53: "Short Cycle Diploma Holders",
    57: "Change of Institution/Course (International)"
}

course_labels = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (Evening Attendance)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (Evening Attendance)"
}

previous_qualification_labels = {
    1: "Secondary Education",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    12: "Other - 11th Year of Schooling",
    14: "10th Year of Schooling",
    15: "10th Year of Schooling - Not Completed",
    19: "Basic Education 3rd Cycle",
    38: "Basic Education 2nd Cycle",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)"
}

nationality_labels = {
    1: "Portuguese",
    2: "German",
    6: "Spanish",
    11: "Italian",
    13: "Dutch",
    14: "English",
    17: "Lithuanian",
    21: "Angolan",
    22: "Cape Verdean",
    24: "Guinean",
    25: "Mozambican",
    26: "Santomean",
    32: "Turkish",
    41: "Brazilian",
    62: "Romanian",
    100: "Moldova",
    101: "Mexican",
    103: "Ukrainian",
    105: "Russian",
    108: "Cuban",
    109: "Colombian"
}

parent_qualification_labels = {
    1: "Secondary Education - 12th Year or Equivalent",
    2: "Higher Education - Bachelor's Degree",
    3: "Higher Education - Degree",
    4: "Higher Education - Master's",
    5: "Higher Education - Doctorate",
    6: "Frequency of Higher Education",
    9: "12th Year of Schooling - Not Completed",
    10: "11th Year of Schooling - Not Completed",
    11: "7th Year (Old)",
    12: "Other - 11th Year of Schooling",
    13: "2nd Year Complementary High School Course",
    14: "10th Year of Schooling",
    18: "General Commerce Course",
    19: "Basic Education 3rd Cycle",
    20: "Complementary High School Course",
    22: "Technical-Professional Course",
    25: "Complementary High School Course - Not Concluded",
    26: "7th Year of Schooling",
    27: "2nd Cycle of General High School Course",
    29: "9th Year of Schooling - Not Completed",
    30: "8th Year of Schooling",
    31: "General Course of Administration and Commerce",
    33: "Supplementary Accounting and Administration",
    34: "Unknown",
    35: "Can't Read or Write",
    36: "Can Read Without Having a 4th Year of Schooling",
    37: "Basic Education 1st Cycle",
    38: "Basic Education 2nd Cycle",
    39: "Technological Specialization Course",
    40: "Higher Education - Degree (1st Cycle)",
    41: "Specialized Higher Studies Course",
    42: "Professional Higher Technical Course",
    43: "Higher Education - Master (2nd Cycle)",
    44: "Higher Education - Doctorate (3rd Cycle)"
}

occupation_labels = {
    0: "Student",
    1: "Legislative/Executive Representatives, Directors and Managers",
    2: "Intellectual and Scientific Specialists",
    3: "Intermediate Technicians and Professionals",
    4: "Administrative Staff",
    5: "Personal Services, Security and Sales Workers",
    6: "Agriculture, Forestry and Fishery Workers",
    7: "Skilled Industry, Construction and Crafts Workers",
    8: "Machine and Assembly Operators",
    9: "Unskilled Workers",
    10: "Armed Forces Professions",
    90: "Other Situation",
    99: "Blank",
    101: "Armed Forces Officers",
    102: "Armed Forces Sergeants",
    103: "Other Armed Forces Personnel",
    112: "Administrative and Commercial Directors",
    114: "Directors of Administrative and Commercial Services",
    121: "Specialists in Physical, Mathematical and Engineering Sciences",
    122: "Health Professionals",
    123: "Teachers",
    124: "Legal, Social and Cultural Specialists",
    125: "ICT Specialists",
    131: "Technicians and Associate Professionals",
    132: "Health Associate Professionals",
    134: "Legal, Social and Cultural Associate Professionals",
    141: "Office Workers",
    143: "Customer Service Clerks",
    144: "Other Administrative Support Workers",
    151: "Personal Service Workers",
    152: "Sales Workers",
    153: "Personal Care Workers",
    171: "Skilled Agricultural Workers",
    172: "Skilled Building and Related Trades Workers",
    173: "Skilled Metal, Machinery and Related Trades Workers",
    174: "Skilled Electrical and Electronic Trades Workers",
    175: "Food Processing, Woodworking and Garment Trades Workers",
    181: "Stationary Plant and Machine Operators",
    182: "Drivers and Mobile Plant Operators",
    183: "Other Machine Operators",
    191: "Cleaners and Helpers",
    192: "Agricultural, Forestry and Fishery Labourers",
    193: "Labourers in Mining, Construction, Manufacturing and Transport",
    194: "Food Preparation Assistants and Related Workers"
}

binary_labels = {
    "Daytime_evening_attendance": {
        0: "Evening",
        1: "Daytime"
    },
    "Displaced": {
        0: "No",
        1: "Yes"
    },
    "Educational_special_needs": {
        0: "No",
        1: "Yes"
    },
    "Debtor": {
        0: "No",
        1: "Yes"
    },
    "Tuition_fees_up_to_date": {
        0: "No",
        1: "Yes"
    },
    "Gender": {
        0: "Female",
        1: "Male"
    },
    "Scholarship_holder": {
        0: "No",
        1: "Yes"
    },
    "International": {
        0: "No",
        1: "Yes"
    }
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_categories(feature_name):
    index = categorical_features.index(feature_name)
    return list(encoder.categories_[index])


def get_label(feature_name, value):
    mappings = {
        "Marital_status": marital_status_labels,
        "Application_mode": application_mode_labels,
        "Course": course_labels,
        "Previous_qualification": previous_qualification_labels,
        "Nacionality": nationality_labels,
        "Mothers_qualification": parent_qualification_labels,
        "Fathers_qualification": parent_qualification_labels,
        "Mothers_occupation": occupation_labels,
        "Fathers_occupation": occupation_labels,
        **binary_labels
    }

    mapping = mappings.get(feature_name, {})

    if value in mapping:
        return f"{value} — {mapping[value]}"

    return str(value)


def labeled_selectbox(label, feature_name, key=None):
    categories = get_categories(feature_name)

    return st.selectbox(
        label,
        options=categories,
        format_func=lambda x: get_label(feature_name, x),
        key=key
    )


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("Student Dropout Risk Prediction")

st.write(
    "Prototype machine learning untuk membantu mengidentifikasi "
    "mahasiswa yang berisiko mengalami dropout berdasarkan "
    "data akademik dan karakteristik mahasiswa"
)

st.divider()


# =========================================================
# STUDENT CHARACTERISTICS
# =========================================================

st.subheader("Karakteristik Mahasiswa")

col1, col2 = st.columns(2)

with col1:

    marital_status = labeled_selectbox(
        "Marital Status",
        "Marital_status"
    )

    application_mode = labeled_selectbox(
        "Application Mode",
        "Application_mode"
    )

    application_order = st.number_input(
        "Application Order",
        min_value=0,
        max_value=9,
        value=1
    )

    course = labeled_selectbox(
        "Course",
        "Course"
    )

    daytime_evening_attendance = labeled_selectbox(
        "Daytime / Evening Attendance",
        "Daytime_evening_attendance"
    )

    previous_qualification = labeled_selectbox(
        "Previous Qualification",
        "Previous_qualification"
    )

    previous_qualification_grade = st.number_input(
        "Previous Qualification Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

    nationality = labeled_selectbox(
        "Nationality",
        "Nacionality"
    )

    admission_grade = st.number_input(
        "Admission Grade",
        min_value=0.0,
        max_value=200.0,
        value=120.0
    )

    age_at_enrollment = st.number_input(
        "Age at Enrollment",
        min_value=15,
        max_value=100,
        value=20
    )


with col2:

    mothers_qualification = labeled_selectbox(
        "Mother's Qualification",
        "Mothers_qualification"
    )

    fathers_qualification = labeled_selectbox(
        "Father's Qualification",
        "Fathers_qualification"
    )

    mothers_occupation = labeled_selectbox(
        "Mother's Occupation",
        "Mothers_occupation"
    )

    fathers_occupation = labeled_selectbox(
        "Father's Occupation",
        "Fathers_occupation"
    )

    displaced = labeled_selectbox(
        "Displaced",
        "Displaced"
    )

    educational_special_needs = labeled_selectbox(
        "Educational Special Needs",
        "Educational_special_needs"
    )

    debtor = labeled_selectbox(
        "Debtor",
        "Debtor"
    )

    tuition_fees_up_to_date = labeled_selectbox(
        "Tuition Fees Up to Date",
        "Tuition_fees_up_to_date"
    )

    gender = labeled_selectbox(
        "Gender",
        "Gender"
    )

    scholarship_holder = labeled_selectbox(
        "Scholarship Holder",
        "Scholarship_holder"
    )

    international = labeled_selectbox(
        "International",
        "International"
    )


# =========================================================
# ACADEMIC PERFORMANCE
# =========================================================

st.divider()

st.subheader("Performa Akademik")

st.write(
    "Masukkan informasi performa akademik mahasiswa "
    "pada semester pertama dan kedua"
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### Semester 1")

    sem1_credited = st.number_input(
        "Curricular Units Credited",
        min_value=0,
        value=0,
        key="sem1_credited"
    )

    sem1_enrolled = st.number_input(
        "Curricular Units Enrolled",
        min_value=0,
        value=6,
        key="sem1_enrolled"
    )

    sem1_evaluations = st.number_input(
        "Curricular Units Evaluations",
        min_value=0,
        value=6,
        key="sem1_evaluations"
    )

    sem1_approved = st.number_input(
        "Curricular Units Approved",
        min_value=0,
        value=5,
        key="sem1_approved"
    )

    sem1_grade = st.number_input(
        "Curricular Units Grade",
        min_value=0.0,
        value=10.0,
        key="sem1_grade"
    )

    sem1_without_evaluations = st.number_input(
        "Curricular Units Without Evaluations",
        min_value=0,
        value=0,
        key="sem1_without_evaluations"
    )


with col2:

    st.markdown("### Semester 2")

    sem2_credited = st.number_input(
        "Curricular Units Credited",
        min_value=0,
        value=0,
        key="sem2_credited"
    )

    sem2_enrolled = st.number_input(
        "Curricular Units Enrolled",
        min_value=0,
        value=6,
        key="sem2_enrolled"
    )

    sem2_evaluations = st.number_input(
        "Curricular Units Evaluations",
        min_value=0,
        value=6,
        key="sem2_evaluations"
    )

    sem2_approved = st.number_input(
        "Curricular Units Approved",
        min_value=0,
        value=5,
        key="sem2_approved"
    )

    sem2_grade = st.number_input(
        "Curricular Units Grade",
        min_value=0.0,
        value=10.0,
        key="sem2_grade"
    )

    sem2_without_evaluations = st.number_input(
        "Curricular Units Without Evaluations",
        min_value=0,
        value=0,
        key="sem2_without_evaluations"
    )


# =========================================================
# ECONOMIC INFORMATION
# =========================================================

st.divider()

st.subheader("Informasi Ekonomi")

col1, col2, col3 = st.columns(3)

with col1:
    unemployment_rate = st.number_input(
        "Unemployment Rate",
        value=10.0
    )

with col2:
    inflation_rate = st.number_input(
        "Inflation Rate",
        value=2.0
    )

with col3:
    gdp = st.number_input(
        "GDP",
        value=1.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Prediksi Risiko Dropout",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    input_data = pd.DataFrame([{

        "Marital_status": marital_status,
        "Application_mode": application_mode,
        "Application_order": application_order,
        "Course": course,
        "Daytime_evening_attendance": daytime_evening_attendance,
        "Previous_qualification": previous_qualification,
        "Previous_qualification_grade": previous_qualification_grade,
        "Nacionality": nationality,
        "Mothers_qualification": mothers_qualification,
        "Fathers_qualification": fathers_qualification,
        "Mothers_occupation": mothers_occupation,
        "Fathers_occupation": fathers_occupation,
        "Admission_grade": admission_grade,
        "Displaced": displaced,
        "Educational_special_needs": educational_special_needs,
        "Debtor": debtor,
        "Tuition_fees_up_to_date": tuition_fees_up_to_date,
        "Gender": gender,
        "Scholarship_holder": scholarship_holder,
        "Age_at_enrollment": age_at_enrollment,
        "International": international,

        "Curricular_units_1st_sem_credited": sem1_credited,
        "Curricular_units_1st_sem_enrolled": sem1_enrolled,
        "Curricular_units_1st_sem_evaluations": sem1_evaluations,
        "Curricular_units_1st_sem_approved": sem1_approved,
        "Curricular_units_1st_sem_grade": sem1_grade,
        "Curricular_units_1st_sem_without_evaluations":
            sem1_without_evaluations,

        "Curricular_units_2nd_sem_credited": sem2_credited,
        "Curricular_units_2nd_sem_enrolled": sem2_enrolled,
        "Curricular_units_2nd_sem_evaluations": sem2_evaluations,
        "Curricular_units_2nd_sem_approved": sem2_approved,
        "Curricular_units_2nd_sem_grade": sem2_grade,
        "Curricular_units_2nd_sem_without_evaluations":
            sem2_without_evaluations,

        "Unemployment_rate": unemployment_rate,
        "Inflation_rate": inflation_rate,
        "GDP": gdp
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.divider()

    st.subheader("Hasil Prediksi")

    if prediction == 1:

        st.error("Berisiko Dropout")

        st.metric(
            "Probabilitas Risiko Dropout",
            f"{probability * 100:.2f}%"
        )

        st.warning(
            "Mahasiswa terindikasi memiliki risiko dropout "
            "berdasarkan data yang diberikan. Institusi dapat "
            "mempertimbangkan pemberian pendampingan atau "
            "intervensi lebih lanjut"
        )

    else:

        st.success("Tidak Terindikasi Risiko Dropout")

        st.metric(
            "Probabilitas Risiko Dropout",
            f"{probability * 100:.2f}%"
        )

        st.info(
            "Berdasarkan data yang diberikan, mahasiswa tidak "
            "terindikasi memiliki risiko dropout yang tinggi"
        )