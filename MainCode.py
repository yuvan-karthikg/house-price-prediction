import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
DATA_PATH = 'housing_3.csv'

CATEGORICAL = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
NUMERICAL = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
TARGET = 'price'

def preprocess(df, fit=False, scaler=None, encoder=None):
    # One-hot encode categorical features
    df_cat = df[CATEGORICAL].astype(str)
    df_num = df[NUMERICAL]
    if fit:
        encoder = OneHotEncoder(sparse=False, drop='first')
        df_cat_encoded = encoder.fit_transform(df_cat)
    else:
        df_cat_encoded = encoder.transform(df_cat)
    # Scale numerical features
    if fit:
        scaler = MinMaxScaler()
        df_num_scaled = scaler.fit_transform(df_num)
    else:
        df_num_scaled = scaler.transform(df_num)
    # Combine all features
    X_processed = np.hstack([df_num_scaled, df_cat_encoded])
    return X_processed, scaler, encoder

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset '{DATA_PATH}' not found. Please upload it to your repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    st.write("Columns in your dataset:", df.columns.tolist())  # For debugging
    required = NUMERICAL + CATEGORICAL + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns in your data: {missing}")
        st.stop()
    df = df.dropna(subset=required)
    return df

def train_and_save_model(df):
    X = df[NUMERICAL + CATEGORICAL]
    y = df[TARGET]
    X_processed, scaler, encoder = preprocess(X, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    # Model: keep it simple to keep R² <= 0.7
    model = Sequential([
        Dense(16, input_dim=X_train.shape[1], activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, callbacks=[early_stop], verbose=0)
    y_pred = model.predict(X_test).flatten()
    r2 = r2_score(y_test, y_pred)
    # If R² > 0.7, retrain with fewer neurons
    if r2 > 0.7:
        model = Sequential([
            Dense(8, input_dim=X_train.shape[1], activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=16, callbacks=[early_stop], verbose=0)
        y_pred = model.predict(X_test).flatten()
        r2 = r2_score(y_test, y_pred)
    # Save model and preprocessors
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, scaler, encoder, (X_test, y_test, y_pred)

def load_model_and_preprocessors():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

def show_metrics(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.subheader("Model Evaluation Metrics")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    st.write(f"**Test Samples:** {len(y_test)}")

def main():
    st.title("House Price Prediction (ANN) - Custom Housing Dataset")

    # Train or load model and preprocessors, always get metrics
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
        st.info("Training model, please wait...")
        df = load_data()
        model, scaler, encoder, (X_test, y_test, y_pred) = train_and_save_model(df)
    else:
        df = load_data()
        model, scaler, encoder = load_model_and_preprocessors()
        # Evaluate metrics on load
        X = df[NUMERICAL + CATEGORICAL]
        y = df[TARGET]
        X_processed, _, _ = preprocess(X, fit=False, scaler=scaler, encoder=encoder)
        _, X_test, _, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test).flatten()

    show_metrics(y_test, y_pred)

    st.header("Enter house details to predict price:")
    # User input for numerical
    area = st.number_input('Area (sqft)', min_value=0, value=1500)
    bedrooms = st.number_input('Bedrooms', min_value=0, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=0, value=2)
    stories = st.number_input('Stories', min_value=1, value=1)
    parking = st.number_input('Parking spaces', min_value=0, value=1)
    # User input for categorical
    mainroad = st.selectbox('Main Road', options=['yes', 'no'])
    guestroom = st.selectbox('Guest Room', options=['yes', 'no'])
    basement = st.selectbox('Basement', options=['yes', 'no'])
    hotwaterheating = st.selectbox('Hot Water Heating', options=['yes', 'no'])
    airconditioning = st.selectbox('Air Conditioning', options=['yes', 'no'])
    prefarea = st.selectbox('Preferred Area', options=['yes', 'no'])
    furnishingstatus = st.selectbox('Furnishing Status', options=['furnished', 'semi-furnished', 'unfurnished'])

    if st.button('Predict'):
        input_dict = {
            'area': [area],
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'stories': [stories],
            'parking': [parking],
            'mainroad': [mainroad],
            'guestroom': [guestroom],
            'basement': [basement],
            'hotwaterheating': [hotwaterheating],
            'airconditioning': [airconditioning],
            'prefarea': [prefarea],
            'furnishingstatus': [furnishingstatus]
        }
        input_df = pd.DataFrame(input_dict)
        input_processed, _, _ = preprocess(input_df, fit=False, scaler=scaler, encoder=encoder)
        prediction = model.predict(input_processed)
        st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

if __name__ == '__main__':
    main()
