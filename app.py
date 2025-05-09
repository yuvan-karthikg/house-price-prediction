# app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

st.title('California House Price Prediction (ANN)')
st.write('Enter the details below to predict the median house value:')

# Input fields
longitude = st.number_input('Longitude', value=-118.0)
latitude = st.number_input('Latitude', value=34.0)
housing_median_age = st.number_input('Housing Median Age', value=20)
total_rooms = st.number_input('Total Rooms', value=2000)
total_bedrooms = st.number_input('Total Bedrooms', value=400)
population = st.number_input('Population', value=1000)
households = st.number_input('Households', value=400)
median_income = st.number_input('Median Income', value=3.0)
ocean_proximity = st.selectbox(
    'Ocean Proximity',
    options=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
)
# Map ocean proximity to code (same as during training)
ocean_dict = {'<1H OCEAN': 0, 'INLAND': 1, 'ISLAND': 2, 'NEAR BAY': 3, 'NEAR OCEAN': 4}
ocean_code = ocean_dict[ocean_proximity]

if st.button('Predict'):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms,
                            total_bedrooms, population, households, median_income, ocean_code]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f'Estimated Median House Value: ${prediction[0][0]:,.2f}')
