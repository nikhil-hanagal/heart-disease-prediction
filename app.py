import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

MODEL_PATH = "models/model.pkl"
DATA_PATH = "data/heart.csv"

# Must match the categories in your training data (UCI heart disease dataset)
CP_OPTIONS = ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
RESTECG_OPTIONS = ["normal", "st-t abnormality", "lv hypertrophy"]
SLOPE_OPTIONS = ["upsloping", "flat", "downsloping"]
THAL_OPTIONS = ["normal", "fixed defect", "reversable defect"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalch", "exang", "oldpeak", "slope", "ca", "thal"]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None


def gauge(prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "Heart disease risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#d62728" if prob >= 0.5 else "#2ca02c"},
            "steps": [
                {"range": [0, 35], "color": "#e8f5e9"},
                {"range": [35, 65], "color": "#fff8e1"},
                {"range": [65, 100], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=10, l=20, r=20))
    return fig


model = load_model()

st.sidebar.title("❤️ Heart Disease App")
page = st.sidebar.radio("Navigate", ["Home", "Single Prediction", "Batch Prediction", "Model Insights"])
st.sidebar.caption("Educational project. Not a medical device or a substitute for a doctor.")

if model is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Run your training script first.")
    st.stop()

# ---------------------------------------------------------------- Home
if page == "Home":
    st.title("Heart Disease Prediction")
    st.write("An XGBoost model (tuned with GridSearchCV) that estimates the likelihood "
             "of heart disease from clinical measurements.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost")
    c2.metric("Input features", len(FEATURES))
    df = load_data()
    c3.metric("Training records", f"{len(df):,}" if df is not None else "n/a")
    st.info("Use **Single Prediction** for one patient, or **Batch Prediction** to upload a CSV.")

# ---------------------------------------------------------------- Single
elif page == "Single Prediction":
    st.title("Single Patient Prediction")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", 1, 120, 50)
        sex = c1.selectbox("Sex", ["Male", "Female"])
        cp = c1.selectbox("Chest pain type", CP_OPTIONS)
        trestbps = c1.number_input("Resting blood pressure (mm Hg)", 50, 250, 120)
        chol = c2.number_input("Cholesterol (mg/dl)", 0, 700, 200)
        fbs = c2.selectbox("Fasting blood sugar > 120 mg/dl", [False, True])
        restecg = c2.selectbox("Resting ECG", RESTECG_OPTIONS)
        thalch = c2.number_input("Max heart rate achieved", 40, 250, 150)
        exang = c3.selectbox("Exercise-induced angina", [False, True])
        oldpeak = c3.number_input("ST depression (oldpeak)", -3.0, 10.0, 1.0, step=0.1)
        slope = c3.selectbox("Slope of peak exercise ST", SLOPE_OPTIONS)
        ca = c3.selectbox("Major vessels colored (0-3)", [0, 1, 2, 3])
        thal = c3.selectbox("Thalassemia", THAL_OPTIONS)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": restecg, "thalch": thalch, "exang": exang,
            "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        }])[FEATURES]
        prob = float(model.predict_proba(row)[0, 1])
        left, right = st.columns([1, 1])
        left.plotly_chart(gauge(prob), use_container_width=True)
        with right:
            if prob >= 0.5:
                st.error("⚠️ High risk of heart disease detected.")
            else:
                st.success("✅ Low risk of heart disease.")
            st.write(f"Predicted probability: **{prob:.1%}**")
            st.caption("Please consult a medical professional for any real health concern.")

# ---------------------------------------------------------------- Batch
elif page == "Batch Prediction":
    st.title("Batch Prediction")
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{f}`" for f in FEATURES))
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = model.predict_proba(data[FEATURES])[:, 1]
            out = data.copy()
            out["risk_probability"] = probs.round(4)
            out["prediction"] = (probs >= 0.5).astype(int)
            c1, c2 = st.columns(2)
            c1.metric("Patients", len(out))
            c2.metric("Flagged high risk", int(out["prediction"].sum()))
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- Insights
else:
    st.title("Model Insights")
    pre = model.named_steps["preprocessor"]
    clf = model.named_steps["classifier"]
    imp = pd.Series(clf.feature_importances_, index=pre.get_feature_names_out())
    imp.index = [i.split("__", 1)[-1] for i in imp.index]
    top = imp.sort_values().tail(15)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
    fig.update_layout(title="Top 15 feature importances", height=500)
    st.plotly_chart(fig, use_container_width=True)

    df = load_data()
    if df is not None:
        st.subheader("Dataset explorer")
        df = df.assign(has_disease=(df["num"] > 0).astype(int))
        col = st.selectbox("Numeric feature", ["age", "trestbps", "chol", "thalch", "oldpeak"])
        hist = go.Figure()
        for label, name in [(0, "No disease"), (1, "Disease")]:
            hist.add_trace(go.Histogram(x=df.loc[df.has_disease == label, col], name=name, opacity=0.65))
        hist.update_layout(barmode="overlay", height=400)
        st.plotly_chart(hist, use_container_width=True)
        st.dataframe(df.head(50), use_container_width=True)