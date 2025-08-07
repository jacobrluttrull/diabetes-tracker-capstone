import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join("models", "diabetes_pipeline.pkl")
model = joblib.load(MODEL_PATH)

threshold = 0.301

st.title ("Diabetes Prediction App")
st.write("Below enter a patients details to predict if they have diabetes or not.")

#Input fields for patient details

age = st.number_input("Age:", min_value=1, max_value=120, value=40)
hypertension = st.selectbox("Hypertension:", [0,1], format_func=lambda x : "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease:", [0,1], format_func=lambda x : "No" if x == 0 else "Yes")
bmi = st.number_input("BMI:", min_value=10.0, max_value=60.0, value = 25.0, step=0.1)
hba1c = st.number_input("HbA1c Level:", min_value=4.0, max_value=16.5, value=6.0, step=0.1)
glucose = st.number_input("Blood Sugar: (mg/dL)", min_value=50, max_value = 800, value=90, step=1)
gender = st.selectbox("Biological Gender:", ["Male", "Female"])
smoking = st.selectbox(
    "Smoking History",
    ["never", "current", "former", "ever", "not current"]
)
# On prediction button click
if st.button("Predict"):
    # Wrap inputs into a DataFrame
    input_df = pd.DataFrame([{
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "gender": gender,
        "smoking_history": smoking
    }])

    try:
        prob = model.predict_proba(input_df)[0][1]
        result = "🔴 Diabetic" if prob >= threshold else "🟢 Not Diabetic"
        st.markdown(f"### Probability: `{prob:.2f}`")
        st.markdown(f"### Prediction: **{result}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")