import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Alzheimer's Risk Predictor", page_icon="🧠", layout="wide")

# ---------- Load saved files ----------
@st.cache_resource
def load_files():
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    features = joblib.load("models/features.pkl")
    return model, scaler, features

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned.csv")

model, scaler, features = load_files()
df = load_data()

# ---------- Text choices -> numbers used by the model ----------
YES_NO = {"No": 0, "Yes": 1}

CATEGORICAL = {
    "Gender": ("Gender", {"Male": 0, "Female": 1}),
    "Ethnicity": ("Ethnicity", {"Caucasian": 0, "African American": 1, "Asian": 2, "Other": 3}),
    "EducationLevel": ("Education level", {"None": 0, "High School": 1, "Bachelor's": 2, "Higher (Master's/PhD)": 3}),
}

YES_NO_LABELS = {
    "Smoking": "Does the patient smoke?",
    "FamilyHistoryAlzheimers": "Family history of Alzheimer's?",
    "CardiovascularDisease": "Cardiovascular (heart) disease?",
    "Diabetes": "Diabetes?",
    "Depression": "Depression?",
    "HeadInjury": "Previous head injury?",
    "Hypertension": "High blood pressure (hypertension)?",
    "MemoryComplaints": "Memory complaints?",
    "BehavioralProblems": "Behavioral problems?",
    "Confusion": "Episodes of confusion?",
    "Disorientation": "Disorientation (loses track of place/time)?",
    "PersonalityChanges": "Personality changes?",
    "DifficultyCompletingTasks": "Difficulty completing tasks?",
    "Forgetfulness": "Forgetfulness?",
}

# column: (label, min, max, default, step)
NUMERIC = {
    "Age": ("Age (years)", 60.0, 90.0, 75.0, 1.0),
    "BMI": ("BMI (Body Mass Index)", 15.0, 40.0, 27.0, 0.1),
    "AlcoholConsumption": ("Alcohol (units per week)", 0.0, 20.0, 5.0, 0.5),
    "PhysicalActivity": ("Physical activity (hours per week)", 0.0, 10.0, 5.0, 0.5),
    "DietQuality": ("Diet quality (0 = poor, 10 = excellent)", 0.0, 10.0, 5.0, 0.5),
    "SleepQuality": ("Sleep quality (4 = poor, 10 = excellent)", 4.0, 10.0, 7.0, 0.5),
    "SystolicBP": ("Systolic blood pressure (mmHg)", 90.0, 180.0, 120.0, 1.0),
    "DiastolicBP": ("Diastolic blood pressure (mmHg)", 60.0, 120.0, 80.0, 1.0),
    "CholesterolTotal": ("Total cholesterol (mg/dL)", 150.0, 300.0, 200.0, 1.0),
    "CholesterolLDL": ("LDL cholesterol (mg/dL)", 50.0, 200.0, 100.0, 1.0),
    "CholesterolHDL": ("HDL cholesterol (mg/dL)", 20.0, 100.0, 50.0, 1.0),
    "CholesterolTriglycerides": ("Triglycerides (mg/dL)", 50.0, 400.0, 150.0, 1.0),
    "MMSE": ("MMSE memory test score (0 = severe, 30 = normal)", 0.0, 30.0, 24.0, 0.5),
    "FunctionalAssessment": ("Functional assessment (0 = poor, 10 = good)", 0.0, 10.0, 5.0, 0.5),
    "ADL": ("Daily living ability (0 = needs help, 10 = independent)", 0.0, 10.0, 5.0, 0.5),
}

# Sections shown on the page
GROUPS = {
    "👤 Personal Information": ["Age", "Gender", "Ethnicity", "EducationLevel"],
    "🏃 Lifestyle": ["BMI", "Smoking", "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality"],
    "🩺 Medical History": ["FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes",
                           "Depression", "HeadInjury", "Hypertension"],
    "🧪 Clinical Measurements": ["SystolicBP", "DiastolicBP", "CholesterolTotal", "CholesterolLDL",
                                 "CholesterolHDL", "CholesterolTriglycerides"],
    "🧠 Cognitive and Symptoms": ["MMSE", "FunctionalAssessment", "ADL", "MemoryComplaints",
                                  "BehavioralProblems", "Confusion", "Disorientation",
                                  "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness"],
}

def make_input(col):
    """Show a friendly widget and return the NUMBER the model needs."""
    if col in CATEGORICAL:
        label, options = CATEGORICAL[col]
        choice = st.selectbox(label, list(options.keys()), key=col)
        return options[choice]
    if col in YES_NO_LABELS:
        choice = st.radio(YES_NO_LABELS[col], list(YES_NO.keys()), horizontal=True, key=col)
        return YES_NO[choice]
    if col in NUMERIC:
        label, lo, hi, default, step = NUMERIC[col]
        return st.slider(label, lo, hi, default, step, key=col)
    # fallback for any column not listed above
    s = df[col]
    return st.slider(col, float(s.min()), float(s.max()), float(s.median()), key=col)

# ---------- Title ----------
st.title("🧠 Alzheimer's Disease Risk Predictor")
st.caption("Educational project only. This is NOT a medical diagnosis tool.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Overview", "📈 Model Comparison"])

# ---------- Tab 1: Prediction ----------
with tab1:
    inputs = {}
    shown = set()

    for title, cols_in_group in GROUPS.items():
        group_cols = [c for c in cols_in_group if c in features]
        if not group_cols:
            continue
        st.subheader(title)
        columns = st.columns(3)
        for i, col_name in enumerate(group_cols):
            with columns[i % 3]:
                inputs[col_name] = make_input(col_name)
            shown.add(col_name)

    # any feature not placed in a section
    leftover = [c for c in features if c not in shown]
    if leftover:
        st.subheader("Other")
        columns = st.columns(3)
        for i, col_name in enumerate(leftover):
            with columns[i % 3]:
                inputs[col_name] = make_input(col_name)

    st.markdown("---")
    if st.button("Predict", type="primary"):
        X_new = pd.DataFrame([inputs])[features]
        X_scaled = scaler.transform(X_new)
        prob = model.predict_proba(X_scaled)[0][1]

        st.metric("Predicted risk of Alzheimer's", f"{prob:.1%}")
        st.progress(float(prob))

        if prob >= 0.5:
            st.error("Higher risk. The model predicts Alzheimer's.")
        else:
            st.success("Lower risk. The model predicts no Alzheimer's.")

# ---------- Tab 2: Data Overview ----------
with tab2:
    st.subheader("Dataset preview")
    st.write("Rows:", df.shape[0], "| Columns:", df.shape[1])
    st.dataframe(df.head(20))

    st.subheader("Diagnosis distribution")
    counts = df["Diagnosis"].map({0: "No Alzheimer's", 1: "Alzheimer's"}).value_counts()
    st.bar_chart(counts)

    st.subheader("Summary statistics")
    st.dataframe(df.describe())

# ---------- Tab 3: Model Comparison ----------
with tab3:
    st.subheader("Performance of all models")
    comparison = pd.read_csv("models/model_comparison.csv")
    st.dataframe(comparison)
    st.bar_chart(comparison.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])