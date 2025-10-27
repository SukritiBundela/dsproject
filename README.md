Student Performance Predictor
A Streamlit web app that predicts student's Math Scores using a trained Machine Learning model.
Users enter details like gender, ethnicity, parental education, and reading/writing scores — the app predicts the likely math score.

How to Run
git clone https://github.com/SukritiBundela/dsproject.git
cd dsproject
python -m venv venv
venv\Scripts\activate       
pip install -r requirements.txt
streamlit run app.py
Then open 👉 http://localhost:8501

Files
app.py → Streamlit UI
src/pipeline/predict_pipeline.py → Model & preprocessing
linear_model.pkl → Trained model
notebook/ 

Features
Instant math score prediction
Clean UI with sidebar inputs
Uses preprocessor + regression model