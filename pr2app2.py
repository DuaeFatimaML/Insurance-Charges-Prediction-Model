import streamlit as st
import pandas as pd
import numpy as np
import joblib
 
# =======================================================
#   PAGE CONFIG
# =======================================================
st.set_page_config(
    page_title="Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)
 
# =======================================================
#   FEATURE ENGINEERING (must match training code exactly)
# =======================================================
def engineer_features(df):
    df = df.copy()
    df['age_squared']           = df['age'] ** 2
    df['bmi_squared']           = df['bmi'] ** 2
    df['age_bmi']               = df['age'] * df['bmi']
    df['bmi_obese']             = (df['bmi'] >= 30).astype(int)
    df['bmi_severely_obese']    = (df['bmi'] >= 35).astype(int)
    df['bmi_obese_interaction'] = df['bmi'] * df['bmi_obese']
    df['age_group']             = pd.cut(df['age'], bins=[17, 30, 45, 60, 100],
                                         labels=[0, 1, 2, 3]).astype(int)
    df['age_children']          = df['age'] * df['children']
    smoker_flag                 = (df['smoker'].str.lower() == 'yes').astype(int)
    df['smoker_bmi']            = smoker_flag * df['bmi']
    df['smoker_age']            = smoker_flag * df['age']
    df['smoker_bmi_obese']      = smoker_flag * df['bmi_obese']
    return df
 
categorical_features = ['sex', 'smoker', 'region']
numerical_features   = [
    'age', 'bmi', 'children',
    'age_squared', 'bmi_squared', 'age_bmi',
    'bmi_obese', 'bmi_severely_obese', 'bmi_obese_interaction',
    'age_group', 'age_children',
    'smoker_bmi', 'smoker_age', 'smoker_bmi_obese'
]
 
# =======================================================
#   LOAD MODELS
# =======================================================
@st.cache_resource
def load_models():
    smoker_model    = joblib.load('model_smoker.joblib')
    nonsmoker_model = joblib.load('model_nonsmoker.joblib')
    smoker_prep     = joblib.load('preprocessor_smoker.joblib')
    nonsmoker_prep  = joblib.load('preprocessor_nonsmoker.joblib')
    return smoker_model, nonsmoker_model, smoker_prep, nonsmoker_prep
 
smoker_model, nonsmoker_model, smoker_prep, nonsmoker_prep = load_models()
 
# =======================================================
#   PREDICTION FUNCTION
# =======================================================
def predict_charges(age, sex, bmi, children, smoker, region):
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi,
        'children': children, 'smoker': smoker, 'region': region
    }])
    input_df = engineer_features(input_df)
    input_df = input_df[numerical_features + categorical_features]
 
    if smoker.lower() == 'yes':
        input_prep = smoker_prep.transform(input_df)
        prediction = smoker_model.predict(input_prep)[0]
    else:
        input_prep = nonsmoker_prep.transform(input_df)
        prediction = nonsmoker_model.predict(input_prep)[0]
 
    return round(prediction, 2)
 
# =======================================================
#   HEADER
# =======================================================
st.title("🏥 Insurance Charges Predictor")
st.markdown("##### Predict your annual medical insurance charges instantly using Machine Learning.")
st.markdown("---")
 
# =======================================================
#   ABOUT SECTION (above the model)
# =======================================================
with st.expander("📌 About This Project", expanded=True):
    st.markdown("""
    Most solutions for this dataset on Kaggle use **Polynomial Regression** — a single model
    trained on all records together. This project takes a different approach.
 
    After analysing the data, I identified that **smoking status creates a structural break** —
    smokers and non-smokers follow fundamentally different cost patterns. Forcing one model to
    learn both groups simultaneously weakens predictions for both.
 
    **My solution:** A **Segmented Dual Random Forest** — one dedicated model per group.
    Each model learns its population fully without interference from the other.
 
    > This is a domain-aware, business-driven design decision — not just algorithm selection.
    """)
 
with st.expander("📊 Model Performance"):
    st.markdown("Both models evaluated on the same **268 held-out test rows** — never seen during training.")
    perf_df = pd.DataFrame({
        'Segment':          ['Smoker', 'Non-Smoker', 'Combined'],
        'Polynomial R²':    [0.7810,   0.4444,       0.8678],
        'Segmented RF R²':  [0.8290,   0.4612,       0.8783],
        'Improvement':      ['+6.14%', '+3.77%',     '+1.22%']
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    st.caption(
        "⚠️ Non-smoker R² of 0.46 is a data ceiling — this dataset lacks health history "
        "and pre-existing conditions which are the primary drivers of non-smoker cost variation. "
        "Every algorithm tested converged at the same score, confirming the limitation is in the data."
    )
 
with st.expander("🧠 Why Segmented RF Over Polynomial?"):
    st.markdown("""
    | | Polynomial Regression | Segmented RF |
    |---|---|---|
    | Fits one curve to all data | ✅ | ❌ |
    | Handles structural data breaks | ❌ | ✅ |
    | Captures non-linear interactions | Partially | ✅ |
    | Dedicated model per risk group | ❌ | ✅ |
    | Combined R² on this dataset | 0.8678 | **0.8783** |
 
    Polynomial regression assumes the entire population follows one mathematical relationship.
    That assumption fails here because smokers and non-smokers are two different populations
    sitting in the same dataset. Random Forest makes no such assumption — and with dedicated
    models per segment, each learns its group's patterns fully.
    """)
 
st.markdown("---")
 
# =======================================================
#   INPUT FORM
# =======================================================
st.subheader("🔢 Enter Your Details")
 
col1, col2 = st.columns(2)
 
with col1:
    age      = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi      = st.slider("BMI", min_value=10.0, max_value=55.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children", options=[0, 1, 2, 3, 4, 5], index=0)
 
with col2:
    sex    = st.selectbox("Sex", options=["male", "female"])
    smoker = st.selectbox("Smoker", options=["no", "yes"])
    region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])
 
# BMI category indicator
if bmi < 18.5:
    bmi_label, bmi_color = "Underweight", "🔵"
elif bmi < 25:
    bmi_label, bmi_color = "Normal", "🟢"
elif bmi < 30:
    bmi_label, bmi_color = "Overweight", "🟡"
elif bmi < 35:
    bmi_label, bmi_color = "Obese", "🟠"
else:
    bmi_label, bmi_color = "Severely Obese", "🔴"
 
st.caption(f"BMI Category: {bmi_color} {bmi_label}")
st.markdown("---")
 
# =======================================================
#   PREDICT BUTTON + RESULT
# =======================================================
if st.button("💰 Predict My Insurance Charges", use_container_width=True):
    with st.spinner("Calculating..."):
        prediction = predict_charges(age, sex, bmi, children, smoker, region)
 
    st.markdown("---")
    st.subheader("📈 Prediction Result")
 
    st.metric(
        label="Estimated Annual Insurance Charges",
        value=f"${prediction:,.2f}"
    )
 
    if smoker == 'yes':
        st.info("🌲 **Smoker RF Model** used — dedicated model for smoker predictions (R² = 0.83)")
    else:
        st.info("🌲 **Non-Smoker RF Model** used — dedicated model for non-smoker predictions (R² = 0.46)")
 
    # Risk profile
    st.markdown("---")
    st.subheader("⚠️ Your Risk Profile")
 
    risk_col1, risk_col2, risk_col3 = st.columns(3)
    with risk_col1:
        age_risk = "High" if age > 45 else "Medium" if age > 30 else "Low"
        st.metric("Age Risk", age_risk)
    with risk_col2:
        st.metric("BMI Category", bmi_label)
    with risk_col3:
        st.metric("Smoker Risk", "High ⚠️" if smoker == 'yes' else "Low ✅")
 
    st.markdown("---")
    st.caption(
        "💡 Predictions are based on 1,338 insurance records. "
        "Actual charges may vary based on your specific policy, provider, and health history."
    )