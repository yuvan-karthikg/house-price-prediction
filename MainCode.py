import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
@st.cache_data

def load_data():
    df = pd.read_csv('housing_3.csv')
    return df

df = load_data()

st.title("House Price Prediction Debug App")

# --- Data Preprocessing ---
st.header("Data Preview")
st.write(df.head())

categorical = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
numerical = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Encode categorical features
df_encoded = df.copy()
for col in categorical:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Log-transform the price
st.subheader("Log Transforming Price")
df_encoded['price'] = np.log1p(df_encoded['price'])
st.write("Sample log(price):", df_encoded['price'].head())

# Split data
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Model ---
st.header("Neural Network Model Training")
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X_train_scaled.shape[1])
history = model.fit(X_train_scaled, y_train, epochs=10, validation_split=0.1, verbose=1)
st.success("Neural Network Trained")

# --- Predictions ---
y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_log)
actual = np.expm1(y_test)

mse = mean_squared_error(actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(actual, y_pred)

st.header("Model Evaluation")
st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:,.2f}")
st.write(f"**R-squared (R²):** {r2:.2f}")

# --- Plot Predictions ---
fig1, ax1 = plt.subplots()
sns.scatterplot(x=actual, y=y_pred, ax=ax1)
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted Prices")
st.pyplot(fig1)

# --- Sanity Check with Linear Regression ---
st.header("Sanity Check: Linear Regression")
lr = LinearRegression()
lr.fit(X_train_scaled, np.expm1(y_train))
y_pred_lr = lr.predict(X_test_scaled)
st.write(f"Linear Regression R²: {r2_score(np.expm1(y_test), y_pred_lr):.2f}")

# --- Outlier Check ---
st.header("Target Distribution (Exponential Scale)")
fig2, ax2 = plt.subplots()
sns.histplot(np.expm1(y_train), bins=30, kde=True, ax=ax2)
ax2.set_title("Price Distribution")
st.pyplot(fig2)

# --- Sample Predictions ---
st.header("Sample Predictions vs Actual")
sample_df = pd.DataFrame({
    'Actual Price': actual[:5].values,
    'Predicted Price (NN)': y_pred[:5],
    'Predicted Price (LR)': y_pred_lr[:5]
})
st.write(sample_df)  # Display comparison
