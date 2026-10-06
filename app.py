import streamlit as st
import pandas as pd
import joblib

# Load the trained model
with open("Model/model.pkl", "rb") as f:
    model = joblib.load(f)

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Titanic Survival Prediction")

st.write(
    "This app predicts whether a passenger would survive "
    "the Titanic disaster based on their features."
)

# User input for features
Pclass = st.selectbox(
    "Passenger Class (Pclass)",
    [1, 2, 3]
)

Sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

Age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=25
)

SibSp = st.number_input(
    "Number of Siblings/Spouses Aboard",
    min_value=0,
    max_value=10,
    value=0
)

Parch = st.number_input(
    "Number of Parents/Children Aboard",
    min_value=0,
    max_value=10,
    value=0
)

Fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=32.2
)

Embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Survival"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "Pclass": [Pclass],
        "Sex": [Sex],
        "Age": [Age],
        "SibSp": [SibSp],
        "Parch": [Parch],
        "Fare": [Fare],
        "Embarked": [Embarked]
    })

    # Make prediction
    prediction = model.predict(input_data)

    # Display probability if supported
    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_data)[0][1]

        st.write(
            f"### Survival Probability: {probability:.2%}"
        )

    # Display prediction
    if prediction[0] == 1:
        st.success("🚢 Passenger would likely SURVIVE.")
    else:
        st.error("❌ Passenger would likely NOT SURVIVE.")
