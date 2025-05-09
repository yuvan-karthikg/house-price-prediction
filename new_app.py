import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'new_housing.csv'

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset 'new_housing.csv' not found. Please upload it to your repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    st.write("Columns in your dataset:", df.columns.tolist())  # Debugging aid

    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'price']
    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"Missing columns in your data: {missing}")
        st.stop()
    df = df.dropna(subset=features)
    return df

def train_and_save_model(df):
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
    X = df[features]
    y = df['price']

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

    # Evaluate
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metrics = {
        'R² Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Test Size': len(y_test)
    }

    return model, scaler, metrics

def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def main():
    st.title("House Price Prediction (ANN) - Custom Dataset")

    # Train or load model and scaler, and get metrics
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        st.info("Training model, please wait...")
        df = load_data()
        model, scaler, metrics = train_and_save_model(df)
    else:
        df = load_data()
        model, scaler = load_model_and_scaler()
        # Evaluate metrics on load
        features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition']
        X = df[features]
        y = df['price']
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test).flatten()
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        metrics = {
            'R² Score': r2,
            'MAE': mae,
            'RMSE': rmse,
            'Test Size': len(y_test)
        }

    st.header("Model Performance on Test Data")
    st.write(f"**R² Score:** {metrics['R² Score']:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** ${metrics['MAE']:,.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** ${metrics['RMSE']:,.2f}")
    st.write(f"**Test Samples:** {metrics['Test Size']}")

    st.header("Enter house details to predict price:")
    bedrooms = st.number_input('Bedrooms', min_value=0, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=0.0, value=2.0, format="%.1f")
    sqft_living = st.number_input('Sqft Living', min_value=0, value=1500)
    sqft_lot = st.number_input('Sqft Lot', min_value=0, value=5000)
    floors = st.number_input('Floors', min_value=0.0, value=1.0, format="%.1f")
    waterfront = st.selectbox('Waterfront', options=[0, 1])
    view = st.number_input('View', min_value=0, max_value=4, value=0)
    condition = st.number_input('Condition', min_value=1, max_value=5, value=3)

    if st.button('Predict'):
        input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

if __name__ == '__main__':
    main()
