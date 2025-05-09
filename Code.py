# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# -------------------------------
# 1. Load and Preprocess Data
# -------------------------------
@st.cache_data
def load_data():
    # Replace with your actual file path or upload widget
    df = pd.read_csv('housing.csv')
    # Drop 'date' column (not useful for prediction)
    df = df.drop(columns=['date'])
    # Handle missing values (fill with median)
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

# -------------------------------
# 2. Feature Selection
# -------------------------------
X = df.drop('price', axis=1)
y = df['price']

# -------------------------------
# 3. Train/Test Split & Scaling
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Build and Train Neural Network
# -------------------------------
@st.cache_resource
def train_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
    return model

model = train_model()

# -------------------------------
# 5. Streamlit UI
# -------------------------------
st.title("House Price Prediction using Neural Networks")

st.write("Enter the details of the house to predict its price:")

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

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button('Predict Price'):
    prediction = model.predict(input_scaled)
    st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

# -------------------------------
# 6. Model Evaluation (Optional)
# -------------------------------
if st.checkbox('Show Model Performance on Test Data'):
    test_preds = model.predict(X_test_scaled)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, test_preds)
    r2 = r2_score(y_test, test_preds)
    st.write(f"Test MAE: ${mae:,.2f}")
    st.write(f"Test R² Score: {r2:.3f}")
