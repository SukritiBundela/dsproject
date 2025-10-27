import os
import numpy as np
import pandas as pd
import joblib

class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education,
                 lunch, test_preparation_course, reading_score, writing_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        data_dict = {
            'gender': [self.gender],
            'race_ethnicity': [self.race_ethnicity],
            'parental_level_of_education': [self.parental_level_of_education],
            'lunch': [self.lunch],
            'test_preparation_course': [self.test_preparation_course],
            'reading_score': [self.reading_score],
            'writing_score': [self.writing_score]
        }
        return pd.DataFrame(data_dict)


class PredictPipeline:
    def __init__(self):
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'linear_model.pkl'))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        saved_obj = joblib.load(model_path)
        self.model = saved_obj['model']
        self.preprocessor = saved_obj['preprocessor']

    def predict(self, dataframe):
        """Preprocess and predict math_score from the input dataframe"""
        processed_data = self.preprocessor.transform(dataframe)
        preds = self.model.predict(processed_data)
        preds = np.clip(preds, 0, 100)
        return preds
