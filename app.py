import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import urllib.request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'housing.csv'
DATA_URL = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.info("Downloading California housing dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes
    df = df.dropna()
    return df

def train_and_save_model(df):
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    st.write(f"Model trained with R² score: {r2:.3f}")

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def main():
    st.title("California House Price Prediction with ANN")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.info("Training model, please wait...")
        df = load_data()
        model, scaler = train_and_save_model(df)
    else:
        model, scaler = load_model_and_scaler()

    # Input form
    longitude = st.number_input('Longitude', value=-118.0)
    latitude = st.number_input('Latitude', value=34.0)
    housing_median_age = st.number_input('Housing Median Age', value=20)
    total_rooms = st.number_input('Total Rooms', value=2000)
    total_bedrooms = st.number_input('Total Bedrooms', value=400)
    population = st.number_input('Population', value=1000)
    households = st.number_input('Households', value=400)
    median_income = st.number_input('Median Income', value=3.0)
    ocean_proximity = st.selectbox('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])
    ocean_dict = {'<1H OCEAN': 0, 'INLAND': 1, 'ISLAND': 2, 'NEAR BAY': 3, 'NEAR OCEAN': 4}
    ocean_code = ocean_dict[ocean_proximity]

    if st.button('Predict'):
        input_data = np.array([[longitude, latitude, housing_median_age, total_rooms,
                                total_bedrooms, population, households, median_income, ocean_code]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Estimated Median House Value: ${prediction[0][0]:,.2f}")

if __name__ == '__main__':
    main()
