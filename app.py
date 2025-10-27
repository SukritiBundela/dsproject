import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("Student Performance Predictor (Math Score)")

# Sidebar for input
st.sidebar.header("Enter Student Details")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race_ethnicity = st.sidebar.selectbox("Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_education = st.sidebar.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.sidebar.number_input("Reading Score", min_value=0, max_value=100, value=50)
writing_score = st.sidebar.number_input("Writing Score", min_value=0, max_value=100, value=50)

# Button to predict
if st.button("Predict Math Score"):
    # Create input data
    input_data = CustomData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_education,
        lunch=lunch,
        test_preparation_course=test_prep,
        reading_score=reading_score,
        writing_score=writing_score
    )
    input_df = input_data.get_data_as_data_frame()

    # Predict
    pipeline = PredictPipeline()
    prediction = pipeline.predict(input_df)

    st.success(f"Predicted Math Score: {prediction[0]:.2f}")
