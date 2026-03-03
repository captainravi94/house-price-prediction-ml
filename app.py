import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(
    page_title="House Price AI",
    page_icon="🏠",
    layout="wide"
)

# ------------------------------------------------
# Load Model (Pipeline)
# ------------------------------------------------
model = joblib.load("house_price_pipeline.pkl")

# ------------------------------------------------
# Load Feature Defaults (Mean values from training)
# ------------------------------------------------
try:
    feature_defaults = joblib.load("feature_defaults.pkl")
except:
    feature_defaults = {}

# ------------------------------------------------
# Header
# ------------------------------------------------
st.title("🏠 House Price Prediction App")
st.markdown("### Smart Property Valuation Using Machine Learning")
st.markdown("---")

col1, col2 = st.columns([1, 2])

# ------------------------------------------------
# Input Section
# ------------------------------------------------
with col1:
    st.subheader("📋 Property Details")

    # Core Important Features
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    overall_cond = st.slider("Overall Condition (1–10)", 1, 10, 5)
    living_area = st.slider("Living Area (sq ft)", 300, 6000, 1500)
    basement_area = st.slider("Basement Area (sq ft)", 0, 3000, 800)
    lot_area = st.slider("Lot Area (sq ft)", 1000, 20000, 8000)
    garage_cars = st.slider("Garage Capacity (Cars)", 0, 4, 2)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    bedrooms = st.slider("Bedrooms Above Ground", 0, 6, 3)

    # Building Age (Converted to YearBuilt)
    current_year = datetime.now().year
    building_age = st.slider("Building Age (Years)", 0, 100, 10)
    year_built = current_year - building_age

    # Create Input Dictionary
    input_data = {
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "GrLivArea": living_area,
        "TotalBsmtSF": basement_area,
        "LotArea": lot_area,
        "GarageCars": garage_cars,
        "FullBath": full_bath,
        "BedroomAbvGr": bedrooms,
        "YearBuilt": year_built
    }

    # ------------------------------------------------
    # Ensure All Model Features Exist
    # ------------------------------------------------
    expected_features = model.named_steps["preprocess"].feature_names_in_

    for col in expected_features:
        if col not in input_data:
            input_data[col] = feature_defaults.get(col, 0)

    input_df = pd.DataFrame([input_data])

    predict_button = st.button("🚀 Predict Price")

# ------------------------------------------------
# Prediction Section
# ------------------------------------------------
with col2:
    st.subheader("📊 Prediction Result")

    if predict_button:
        prediction = model.predict(input_df)

        st.markdown("## 💰 Estimated Property Value")
        st.success(f"₹ {int(prediction[0]):,}")

        st.markdown("---")
        st.subheader("📈 Top Influential Features")

        try:
            feature_names = model.named_steps["preprocess"].get_feature_names_out()
            importances = model.named_steps["regressor"].feature_importances_

            indices = np.argsort(importances)[-10:]

            fig, ax = plt.subplots()
            ax.barh(np.array(feature_names)[indices], importances[indices])
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)

        except:
            st.info("Feature importance not available.")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Scikit-Learn, XGBoost & Streamlit")