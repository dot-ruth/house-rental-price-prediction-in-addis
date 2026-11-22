
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Addis Ababa Rental Price Predictor", page_icon="")

MODEL_PATH = Path(__file__).parent / "models" / "gradient_boosting_rental_price_model.joblib"
DATA_PATH = Path(__file__).parent / "real-estate_dataset_addis-ababa_v1.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_options():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["listing_type"].astype("string").str.contains("rent", case=False, na=False)]
    return {
        col: sorted(df[col].dropna().astype(str).str.strip().str.lower().unique())
        for col in ["property_type", "subcity"]
    }

model = load_model()
options = load_options()

categorical_features = ["property_type", "subcity", "condition"]
numerical_features = ["num_bedrooms", "num_bathrooms"]

def prepare_input(data):
    data[categorical_features] = data[categorical_features].apply(
        lambda col: col.astype("string").str.strip().str.lower()
    ).astype(object).where(data[categorical_features].notna(), np.nan)
    data[numerical_features] = data[numerical_features].apply(pd.to_numeric, errors="coerce")
    return data

st.title("Addis Ababa Rental Price Predictor")
st.write("Estimate residential rental prices in Addis Ababa using machine learning.")

with st.form("prediction_form"):
    property_type = st.selectbox("Property Type", options["property_type"])
    subcity = st.selectbox("Subcity", options["subcity"])
    condition = st.selectbox("Property Condition", ["Newly-Built", "Fairly Used", "Renovated", "Old"])

    col1, col2 = st.columns(2)
    bedrooms = col1.number_input("Bedrooms", 0, 30, 2)
    bathrooms = col2.number_input("Bathrooms", 0, 30, 1)

    submitted = st.form_submit_button("Predict Rental Price", use_container_width=True)

if submitted:
    data = pd.DataFrame([{
        "property_type": property_type,
        "subcity": subcity,
        "condition": condition,
        "num_bedrooms": bedrooms,
        "num_bathrooms": bathrooms
    }])

    prediction = model.predict(prepare_input(data))[0]

    st.divider()
    st.subheader("Estimated Rental Price")
    st.metric("Predicted Price", f"{prediction:,.2f}")
    st.caption("This is a machine learning estimate, not a guaranteed market price.")

st.divider()
st.caption("Machine Learning Assignment | Residential Rental Price Prediction in Addis Ababa")
