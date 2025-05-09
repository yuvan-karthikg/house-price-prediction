import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Streamlit UI
st.title("🏠 House Price Prediction using ANN")
st.write("Upload your dataset to begin:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.head())

    st.subheader("Data Description")
    st.write(df.describe())

    # Preprocessing
    st.subheader("Preprocessing Data")
    df = df.dropna()
    X = df.drop(columns=["price"], errors='ignore')
    y = df["price"] if "price" in df.columns else df.iloc[:, -1]  # fallback

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Model
    st.subheader("Training ANN Model")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        callbacks=[early_stop],
        verbose=0
    )

    st.success("Model trained successfully!")

    # Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

    if st.button("Show Prediction Samples"):
        results_df = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
        st.write(results_df)
else:
    st.info("Awaiting CSV file upload.")
