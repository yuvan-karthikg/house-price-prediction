import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('housing_3.csv')
    return df

df = load_data()

st.title("House Price Prediction with Neural Networks (Improved)")

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

# Log-transform the target
df_encoded['log_price'] = np.log1p(df_encoded['price'])

X = df_encoded.drop(['price', 'log_price'], axis=1)
y = df_encoded['log_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Neural Network Model ---
st.header("Model Training")

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # No activation for regression
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_model(X_train_scaled.shape[1])

# Early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    validation_split=0.1,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)

st.success("Model trained successfully!")

# --- Evaluation Metrics ---
st.header("Model Evaluation")

y_pred_log = model.predict(X_test_scaled).flatten()
y_pred = np.expm1(y_pred_log)  # Inverse log1p
y_test_real = np.expm1(y_test)

mse = mean_squared_error(y_test_real, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred)

st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
st.write(f"**R-squared (R2):** {r2:.2f}")

fig, ax = plt.subplots()
sns.scatterplot(x=y_test_real, y=y_pred, ax=ax)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Actual vs Predicted Prices")
st.pyplot(fig)

# --- Prediction Interface ---
st.header("Predict House Price")

def user_input_features():
    area = st.number_input('Area (sq ft)', min_value=500, max_value=10000, value=2000)
    bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Bathrooms', min_value=1, max_value=10, value=2)
    stories = st.number_input('Stories', min_value=1, max_value=4, value=2)
    mainroad = st.selectbox('Main Road', ['yes', 'no'])
    guestroom = st.selectbox('Guest Room', ['yes', 'no'])
    basement = st.selectbox('Basement', ['yes', 'no'])
    hotwaterheating = st.selectbox('Hot Water Heating', ['yes', 'no'])
    airconditioning = st.selectbox('Air Conditioning', ['yes', 'no'])
    parking = st.number_input('Parking Spots', min_value=0, max_value=5, value=1)
    prefarea = st.selectbox('Preferred Area', ['yes', 'no'])
    furnishingstatus = st.selectbox('Furnishing Status', ['furnished', 'semi-furnished', 'unfurnished'])

    data = {
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

for col in categorical:
    le = LabelEncoder()
    le.fit(df[col])
    input_df[col] = le.transform(input_df[col])

input_scaled = scaler.transform(input_df)

if st.button('Predict'):
    log_pred = model.predict(input_scaled).flatten()[0]
    prediction = np.expm1(log_pred)
    st.subheader(f"Estimated House Price: {prediction:,.2f}")

