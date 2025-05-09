import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("House Price Prediction with ANN")

uploaded_file = st.file_uploader("Upload your house price CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.drop(['date', 'street', 'country'], axis=1)

    X = df.drop('price', axis=1)
    y = df['price']

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    input_dim = X_train_prep.shape[1]
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

    early_stopping = EarlyStopping(
        monitor='val_mse',
        patience=40,
        restore_best_weights=True,
        verbose=1
    )

    with st.spinner("Training model..."):
        history = model.fit(
            X_train_prep, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

    y_pred = model.predict(X_test_prep)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance on Test Set")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    st.subheader("Predict Price for a New House")

    # User inputs for all features
    user_input = {}
    for col in num_cols:
        user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    for col in cat_cols:
        user_input[col] = st.text_input(f"{col}", value=str(X[col].iloc[0]))

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        input_prep = preprocessor.transform(input_df)
        pred_price = model.predict(input_prep)
        st.success(f"Predicted price: ${pred_price[0][0]:,.2f}")

else:
    st.warning("Please upload a CSV file to proceed.")
