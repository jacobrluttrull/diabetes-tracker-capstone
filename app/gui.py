import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Load model
MODEL_PATH = os.path.join("models", "diabetes_pipeline.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(MODEL_PATH)

# Sidebar for threshold setting
st.sidebar.header("Settings")
threshold = st.sidebar.slider(
    "Classification Threshold", min_value=0.0, max_value=1.0, value=0.301, step=0.01
)

# App header
st.title("Type 2 Diabetes Prediction App")
st.write("This tool estimates the likelihood of **Type 2 Diabetes** based on key health indicators.")
st.write("It is intended for educational purposes and should not be used as a substitute for professional medical advice.")

# Input fields
age = st.number_input("Age:", min_value=1, max_value=120, value=40)
hypertension = st.selectbox("Hypertension:", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
heart_disease = st.selectbox("Heart Disease:", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
bmi = st.number_input("BMI:", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
hba1c = st.number_input("HbA1c Level:", min_value=4.0, max_value=16.5, value=6.0, step=0.1)
glucose = st.number_input("Blood Sugar (mg/dL):", min_value=50, max_value=800, value=90, step=1)
gender = st.selectbox("Biological Gender:", ["Male", "Female"])
smoking = st.selectbox("Smoking History", ["never", "current", "former", "ever", "not current"])

# Predict button
if st.button("Predict"):
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
        result = "🔴 Likely Type 2 Diabetes" if prob >= threshold else "🟢 Unlikely Type 2 Diabetes"
        st.markdown(f"### Probability: `{prob:.2f}`")
        st.markdown(f"### Prediction: **{result}**")

        # Risk tier
        tier = "Low" if prob < 0.2 else "Medium" if prob < 0.5 else "High"
        st.metric("Predicted Probability", f"{prob:.2f}", help="Probability of Type 2 Diabetes.")
        st.caption(f"Risk Tier: **{tier}**  •  Decision threshold = {threshold:.3f}")
        st.caption("This tool is educational and not clinical advice.")

        # -------------------------
        # Phase 3 — What-if analysis
        # -------------------------
        with st.expander("What-if analysis"):
            st.caption("Adjust a value to see how the probability would change (Type 2 Diabetes only).")

            new_bmi = st.slider("Adjust BMI", 10.0, 60.0, float(bmi), 0.1)
            new_hba1c = st.slider("Adjust HbA1c", 4.0, 12.0, float(hba1c), 0.1)
            new_glucose = st.slider("Adjust Blood Sugar (mg/dL)", 50, 300, int(glucose), 1)

            alt_df = input_df.copy()
            alt_df["bmi"] = new_bmi
            alt_df["HbA1c_level"] = new_hba1c
            alt_df["blood_glucose_level"] = new_glucose

            alt_prob = model.predict_proba(alt_df)[0, 1]
            st.write(
                f"If BMI = **{new_bmi:.1f}**, HbA1c = **{new_hba1c:.1f}**, Blood Sugar = **{new_glucose}**, "
                f"→ Probability of **Type 2 Diabetes** would be **{alt_prob:.2f}** (threshold {threshold:.3f})."
            )

        # --- SHAP explainability ---
        st.subheader("📊 Why this prediction?")

        try:
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            # Transform the single row
            X_row = preprocessor.transform(input_df)

            # Compute SHAP
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(X_row)  # shape: (1, n_features)
            base_value = explainer.expected_value
            feat_names = preprocessor.get_feature_names_out()

            NUM_MAP = {
                "age": "Age",
                "bmi": "BMI",
                "HbA1c_level": "HbA1c",
                "blood_glucose_level": "Blood Sugar"
            }

            def friendly_label(raw_feature_name: str) -> str:
                if raw_feature_name.startswith("num__"):
                    col = raw_feature_name.split("__", 1)[1]
                    return NUM_MAP.get(col, col.replace("_", " ").title())
                if raw_feature_name.startswith("cat__"):
                    rest = raw_feature_name.split("__", 1)[1]
                    if rest.startswith("gender_"):
                        val = rest.split("gender_", 1)[1]
                        return f"Gender = {val}"
                    if rest.startswith("smoking_history_"):
                        val = rest.split("smoking_history_", 1)[1]
                        return f"Smoking: {val}"
                    return rest.replace("_", " ").title()
                return raw_feature_name

            friendly_labels = [friendly_label(f) for f in feat_names]

            contrib = pd.DataFrame({
                "Feature": friendly_labels,
                "Raw Feature": feat_names,
                "SHAP": shap_vals[0]
            })
            contrib["|SHAP|"] = contrib["SHAP"].abs()
            contrib_sorted = contrib.sort_values("|SHAP|", ascending=False)

            def user_value_for_engineered(raw_feature_name: str) -> str:
                if raw_feature_name.startswith("num__"):
                    col = raw_feature_name.split("__", 1)[1]
                    return str(input_df[col].iat[0])
                if raw_feature_name.startswith("cat__"):
                    rest = raw_feature_name.split("__", 1)[1]
                    if rest.startswith("gender_"):
                        return input_df['gender'].iat[0]
                    if rest.startswith("smoking_history_"):
                        return input_df['smoking_history'].iat[0]
                return "—"

            # Phase 3 wording: consistent “Type 2 Diabetes” + natural phrasing
            def effect_label_with_value(feature_name, shap_value, value):
                thresholds = {
                    "Blood Sugar": 140,
                    "HbA1c": 6.5,
                    "BMI": 30,
                    "Age": 45
                }
                direction = "increases" if shap_value >= 0 else "lowers"

                if feature_name in thresholds:
                    threshold_val = thresholds[feature_name]
                    comp = "above" if float(value) >= threshold_val else "below"
                    if direction == "increases":
                        return (
                            f"Since your {feature_name.lower()} was {value} and {comp} the {threshold_val} threshold, "
                            f"your risk of **Type 2 Diabetes is higher."
                        )
                    else:
                        return (
                            f"Since your {feature_name.lower()} was {value} and {comp} the {threshold_val} threshold, "
                            f"your risk of **Type 2 Diabetes is lower."
                        )
                else:
                    if direction == "increases":
                        return (
                            f"Your value for {feature_name.lower()} was {value}, which increases your risk of Type 2 Diabetes**."
                        )
                    else:
                        return (
                            f"Your value for {feature_name.lower()} was {value}, which lowers your risk of Type 2 Diabetes**."
                        )

            topk = contrib_sorted.head(6).copy()
            topk["Your value"] = topk["Raw Feature"].map(user_value_for_engineered)
            topk["Effect"] = [
                effect_label_with_value(f, s, v) for f, s, v in zip(topk["Feature"], topk["SHAP"], topk["Your value"])
            ]

            # Plain-English bullets
            st.markdown("**Top factors for this prediction:**")
            for _, r in topk.head(3).iterrows():
                st.write(f"- {r['Effect']}")

            # Friendly table
            st.dataframe(
                topk[["Feature", "Your value", "Effect"]],
                use_container_width=True
            )

            # Full waterfall
            with st.expander("See full SHAP waterfall (advanced)"):
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots._waterfall.waterfall_legacy(
                    base_value, shap_vals[0], feature_names=friendly_labels,
                    max_display=14, show=False
                )
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"⚠️ SHAP explainability failed: {e}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# -------------------------
# Phase 5 — Mini Model Card
# -------------------------
with st.expander("About this model"):
    st.markdown(f"""
**Task:** Type 2 Diabetes risk prediction (binary classification)  
**Algorithm:** XGBoost inside a scikit-learn Pipeline (preprocessing + model)  
**Decision threshold:** {threshold:.3f} (adjustable in sidebar)  
**Reported test metrics:** Accuracy 0.965, Precision 0.83, Recall 0.74, F1 0.78  
**Data:** `diabetes_prediction_dataset.csv` (structured health indicators)  
**Privacy & ethics:** No PHI stored; all processing is local; outputs are educational and not medical advice.
""")
