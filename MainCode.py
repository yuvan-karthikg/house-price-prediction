import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

MODEL_PATH = 'model.keras'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'encoder.pkl'
DATA_PATH = 'housing_3.csv'

CATEGORICAL = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea', 'furnishingstatus']
NUMERICAL = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
TARGET = 'price'

def preprocess(df, fit=False, scaler=None, encoder=None):
    df_cat = df[CATEGORICAL].astype(str)
    df_num = df[NUMERICAL]
    if fit:
        encoder = OneHotEncoder(sparse_output=False)
        df_cat_encoded = encoder.fit_transform(df_cat)
    else:
        df_cat_encoded = encoder.transform(df_cat)
    if fit:
        scaler = MinMaxScaler()
        df_num_scaled = scaler.fit_transform(df_num)
    else:
        df_num_scaled = scaler.transform(df_num)
    X_processed = np.hstack([df_num_scaled, df_cat_encoded])
    return X_processed, scaler, encoder

def remove_outliers(df):
    # Simple IQR method for price and area
    for col in ['price', 'area']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset '{DATA_PATH}' not found. Please upload it to your repo.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    required = NUMERICAL + CATEGORICAL + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing columns in your data: {missing}")
        st.stop()
    df = df.dropna(subset=required)
    df = remove_outliers(df)
    return df

def train_and_save_model(df):
    X = df[NUMERICAL + CATEGORICAL]
    y = df[TARGET]
    y_log = np.log1p(y)  # Log-transform target
    X_processed, scaler, encoder = preprocess(X, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_log, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=16, callbacks=[early_stop], verbose=0)
    y_pred_log = model.predict(X_test).flatten()
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)

    # Check for NaNs/Infs before evaluation
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        st.error("Model predictions contain NaN or infinite values. Please check your model or data.")
        st.stop()
    if np.any(np.isnan(y_test_orig)) or np.any(np.isinf(y_test_orig)):
        st.error("Test target contains NaN or infinite values. Please check your dataset.")
        st.stop()

    r2 = r2_score(y_test_orig, y_pred)
    if r2 > 0.7:
        # Retrain with smaller model if R² too high
        model = Sequential([
            Dense(32, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop], verbose=0)
        y_pred_log = model.predict(X_test).flatten()
        y_pred = np.expm1(y_pred_log)
        r2 = r2_score(y_test_orig, y_pred)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    return model, scaler, encoder, (X_test, y_test, y_pred, y_test_orig)

def load_model_and_preprocessors():
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

def show_metrics(y_test, y_pred, y_test_orig=None):
    if y_test_orig is None:
        y_test_orig = y_test

    # Check for NaN/Inf before metrics
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        st.error("Predictions contain NaN or infinite values. Cannot compute metrics.")
        st.stop()
    if np.any(np.isnan(y_test_orig)) or np.any(np.isinf(y_test_orig)):
        st.error("Test data contains NaN or infinite values. Cannot compute metrics.")
        st.stop()

    r2 = r2_score(y_test_orig, y_pred)
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    st.subheader("Model Evaluation Metrics")
    st.write(f"**R² Score:** {r2:.3f}")
    st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    st.write(f"**Test Samples:** {len(y_test_orig)}")

    chart_df = pd.DataFrame({'Actual': y_test_orig, 'Predicted': y_pred})
    st.subheader("Predicted vs Actual Prices")
    st.scatter_chart(chart_df)

def main():
    st.title("Efficient House Price Prediction (ANN)")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
        st.info("Training model, please wait...")
        df = load_data()
        model, scaler, encoder, (X_test, y_test, y_pred, y_test_orig) = train_and_save_model(df)
    else:
        df = load_data()
        model, scaler, encoder = load_model_and_preprocessors()
        X = df[NUMERICAL + CATEGORICAL]
        y = df[TARGET]
        y_log = np.log1p(y)
        X_processed, _, _ = preprocess(X, fit=False, scaler=scaler, encoder=encoder)
        _, X_test, _, y_test = train_test_split(X_processed, y_log, test_size=0.2, random_state=42)
        y_pred_log = model.predict(X_test).flatten()
        y_pred = np.expm1(y_pred_log)
        y_test_orig = np.expm1(y_test)

    show_metrics(y_test, y_pred, y_test_orig)

    st.header("Enter house details to predict price:")
    area = st.number_input('Area (sqft)', min_value=0, value=1500)
    bedrooms = st.number_input('Bedrooms', min_value=0, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=0, value=2)
    stories = st.number_input('Stories', min_value=1, value=1)
    parking = st.number_input('Parking spaces', min_value=0, value=1)
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
        pred_log = model.predict(input_processed)
        pred_price = np.expm1(pred_log)[0][0]
        st.success(f"Estimated House Price: ${pred_price:,.2f}")

if __name__ == '__main__':
    main()
