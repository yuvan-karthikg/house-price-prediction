import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

st.title("House Price Prediction using Neural Networks")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    # Read the uploaded file directly (no filename needed)
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Drop 'date' if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Fill missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Check for 'price' column
    if 'price' not in df.columns:
        st.error("The uploaded file must contain a 'price' column as the target variable.")
        st.stop()

    # Feature/Target Split
    X = df.drop('price', axis=1)
    y = df['price']

    # Train/Test Split and Scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and Train Neural Network
    @st.cache_resource
    def train_model(X_train_scaled, y_train):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
        return model

    model = train_model(X_train_scaled, y_train)

    # Streamlit UI for Prediction
    st.header("Predict the Price of a House")

    def user_input_features():
        bedrooms = st.number_input('Bedrooms', min_value=0, max_value=20, value=3)
        bathrooms = st.number_input('Bathrooms', min_value=0.0, max_value=10.0, step=0.25, value=2.0)
        sqft_living = st.number_input('Sqft Living', min_value=100, max_value=20000, value=2000)
        sqft_lot = st.number_input('Sqft Lot', min_value=100, max_value=1000000, value=5000)
        floors = st.number_input('Floors', min_value=1.0, max_value=4.0, step=0.5, value=1.0)
        waterfront = st.selectbox('Waterfront', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        view = st.slider('View', 0, 4, 0)
        condition = st.slider('Condition', 1, 5, 3)
        data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': waterfront,
            'view': view,
            'condition': condition
        }
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()
    input_scaled = scaler.transform(input_df)

    if st.button('Predict Price'):
        prediction = model.predict(input_scaled)
        st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

    # Model Performance
    if st.checkbox('Show Model Performance on Test Data'):
        test_preds = model.predict(X_test_scaled)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        st.write(f"Test MAE: ${mae:,.2f}")
        st.write(f"Test R² Score: {r2:.3f}")

else:
    st.info("Please upload a CSV file to use the app.")

