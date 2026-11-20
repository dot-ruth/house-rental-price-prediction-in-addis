
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Addis Ababa Rental Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).parent
    / "models"
    / "gradient_boosting_rental_price_model.joblib"
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# --------------------------------------------------
# INPUT PREPARATION
# Must match the training-time preparation logic
# --------------------------------------------------

def normalize_property_type(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    if "studio" in value:
        return "Studio Apartment"

    if "apartment" in value and "house" in value:
        return "Mixed residential"

    if "villa" in value:
        return "Villa"

    if "penthouse" in value:
        return "Penthouse"

    if "townhouse" in value or "terrace" in value:
        return "Townhouse"

    if "duplex" in value:
        return "Duplex"

    if "mansion" in value:
        return "Mansion"

    if "condo" in value:
        return "Condo"

    if "apartment" in value:
        return "Apartment"

    if "house" in value:
        return "House"

    if "building" in value:
        return "Building"

    if "residential" in value:
        return "Residential"

    return "Other"


def prepare_input(data):

    data["property_type"] = (
        data["property_type"]
        .apply(normalize_property_type)
        .astype(object)
    )

    for col in ["location", "condition"]:
        data[col] = (
            data[col]
            .astype(object)
            .where(data[col].notna(), np.nan)
        )

        data[col] = data[col].apply(
            lambda value: (
                str(value).strip().lower()
                if pd.notna(value)
                else np.nan
            )
        )

    for col in ["bedrooms", "bathrooms"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    return data


# --------------------------------------------------
# APPLICATION INTERFACE
# --------------------------------------------------

st.title("🏠 Addis Ababa Rental Price Predictor")

st.write(
    """
    Estimate the monthly rental price of a residential
    property in Addis Ababa using a machine learning model.
    """
)

st.divider()

st.subheader("Property Information")

with st.form("prediction_form"):

    property_type = st.text_input(
        "Property Type",
        placeholder="e.g. Apartment, House, Villa"
    )

    location = st.text_input(
        "Location",
        placeholder="e.g. Bole, Addis Ababa"
    )

    condition = st.selectbox(
        "Property Condition",
        [
            "Newly-Built",
            "Fairly Used",
            "Renovated",
            "Old"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=30,
            value=2,
            step=1
        )

    with col2:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0,
            max_value=30,
            value=1,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict Monthly Rent",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if not property_type.strip() or not location.strip():
        st.warning(
            "Please enter both property type and location."
        )

    else:

        input_data = pd.DataFrame([{
            "property_type": property_type,
            "location": location,
            "condition": condition,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms
        }])

        prepared_data = prepare_input(input_data)

        prediction = model.predict(prepared_data)[0]

        st.divider()

        st.subheader("Estimated Monthly Rent")

        st.metric(
            label="Predicted Rental Price",
            value=f"ETB {prediction:,.2f}"
        )

        st.caption(
            "This is a machine learning estimate, not an "
            "official property valuation or guaranteed market price."
        )

        st.info(
            "Prediction quality depends on the available training "
            "data and the information provided."
        )


st.divider()

st.caption(
    "Machine Learning Assignment | Residential Rental Price "
    "Prediction in Addis Ababa"
)
