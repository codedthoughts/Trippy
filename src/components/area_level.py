import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model

warnings.filterwarnings('ignore')

with open('../../artifacts/idx_to_area.pkl', 'rb') as file:
    idx_to_area = pickle.load(file)

model = load_model('../../artifacts/area_recommendation_model.h5')

with open('../../artifacts/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

categories = ['Accommodation and Wellness', 'Adventure and Recreation', 'Beach and Water',
              'Educational and Research', 'Entertainment and Leisure', 'Heritage and Culture',
              'Nature and Wildlife', 'Religious', 'Shopping and Urban', 'Transportation and Services']

def preprocess_user_input(lat, lon, category, default_rating=4.0, default_score=5.0):
    average_rating = default_rating
    final_score = default_score

    scaled_input = scaler.transform([[lat, lon, average_rating, final_score]])

    category_vector = [1 if cat == category else 0 for cat in categories]

    user_input = np.concatenate([scaled_input[0], category_vector])

    return user_input.reshape(1, -1)  

def predict_top_n_areas(lat, lon, category, top_n=10):
    processed_input = preprocess_user_input(lat, lon, category)

    predictions = model.predict(processed_input)

    top_n_indices = np.argsort(predictions[0])[-top_n:][::-1]  
    top_n_areas = [idx_to_area[idx] for idx in top_n_indices]

    return top_n_areas

latitude = 18.5204
longitude = 73.8567 #Pune
category = 'Religious'  

recommended_areas = predict_top_n_areas(latitude, longitude, category, top_n=10)

print(f"Top 10 Recommended Areas: {recommended_areas}")
