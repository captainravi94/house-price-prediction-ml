import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime
import requests
from streamlit_lottie import st_lottie

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
st.set_page_config(page_title="EstateAI", layout="wide")

# ------------------------------------------------
# THEME TOGGLE
# ------------------------------------------------
theme = st.sidebar.selectbox("Theme", ["Dark", "Light"])

if theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    h1,h2,h3,h4 { color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f8fafc, #e2e8f0); }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("house_price_pipeline.pkl")

try:
    feature_defaults = joblib.load("feature_defaults.pkl")
except:
    feature_defaults = {}

# ------------------------------------------------
# LOTTIE
# ------------------------------------------------
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json")

# ------------------------------------------------
# HEADER
# ------------------------------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.title("🏢 EstateAI - Smart Property Valuation")
with col2:
    st_lottie(lottie, height=120)

st.markdown("---")

# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
page = st.sidebar.radio("Navigation", ["Single Prediction", "Batch Prediction", "Admin Dashboard"])

# ------------------------------------------------
# SINGLE PREDICTION
# ------------------------------------------------
if page == "Single Prediction":

    st.sidebar.header("Property Details")

    overall_qual = st.sidebar.slider("Overall Quality", 1, 10, 5)
    overall_cond = st.sidebar.slider("Overall Condition", 1, 10, 5)
    living_area = st.sidebar.slider("Living Area (sq ft)", 300, 6000, 1500)
    lot_area = st.sidebar.slider("Lot Area", 1000, 20000, 8000)
    garage = st.sidebar.slider("Garage Cars", 0, 4, 2)

    current_year = datetime.now().year
    building_age = st.sidebar.slider("Building Age", 0, 100, 10)
    year_built = current_year - building_age

    input_data = {
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "GrLivArea": living_area,
        "LotArea": lot_area,
        "GarageCars": garage,
        "YearBuilt": year_built
    }

    expected = model.named_steps["preprocess"].feature_names_in_

    for col in expected:
        if col not in input_data:
            input_data[col] = feature_defaults.get(col, 0)

    input_df = pd.DataFrame([input_data])

    if st.button("Predict Price"):
        pred = model.predict(input_df)[0]
        st.metric("Estimated Property Value", f"₹ {int(pred):,}")

# ------------------------------------------------
# BATCH PREDICTION
# ------------------------------------------------
elif page == "Batch Prediction":

    st.subheader("Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)

        preds = model.predict(df)
        df["PredictedPrice"] = preds

        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "batch_predictions.csv", "text/csv")

# ------------------------------------------------
# ADMIN DASHBOARD
# ------------------------------------------------
elif page == "Admin Dashboard":

    st.subheader("Admin Analytics Dashboard")

    # Fake analytics example
    sample_data = pd.DataFrame({
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "AvgPrice": [200000, 210000, 190000, 230000, 250000]
    })

    fig = px.line(sample_data, x="Month", y="AvgPrice", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance")

    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        importances = model.named_steps["regressor"].feature_importances_
        df_imp = pd.DataFrame({"Feature": names, "Importance": importances})
        df_imp = df_imp.sort_values("Importance", ascending=False).head(10)

        fig2 = px.bar(df_imp, x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    except:
        st.info("Feature importance not available")

st.markdown("---")
st.markdown("EstateAI © 2026 | Powered by Machine Learning")
