import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras

st.title("House Price Prediction using Neural Networks")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Drop 'date' if present
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    # Fill missing values: median for numeric, mode for categorical
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Check for 'price' column
    if 'price' not in df.columns:
        st.error("The uploaded file must contain a 'price' column as the target variable.")
        st.stop()

    # Separate features and target
    X_raw = df.drop('price', axis=1)
    y = df['price']

    # Identify categorical and numeric columns
    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if categorical_cols:
        X_cat = pd.DataFrame(
            encoder.fit_transform(X_raw[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        X = pd.concat([X_raw[numeric_cols].reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X = X_raw[numeric_cols]

    # Fill again in case encoding introduces NaNs
    X = X.fillna(0)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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

    st.header("Predict the Price of a House")

    # User input for all features
    user_input = {}
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.number_input(
            f"{col}", value=mean_val, min_value=min_val, max_value=max_val
        )

    for col in categorical_cols:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 10:
            user_input[col] = st.selectbox(f"{col}", unique_vals)
        else:
            user_input[col] = st.text_input(f"{col}", str(unique_vals[0]))

    # Build input DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode input
    if categorical_cols:
        input_cat = pd.DataFrame(
            encoder.transform(input_df[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols)
        )
        input_num = input_df[numeric_cols].reset_index(drop=True)
        input_X = pd.concat([input_num, input_cat], axis=1)
    else:
        input_X = input_df[numeric_cols]

    # Align columns with training data (add missing columns if any)
    for col in X.columns:
        if col not in input_X.columns:
            input_X[col] = 0
    input_X = input_X[X.columns.tolist()]

    # Scale
    input_scaled = scaler.transform(input_X)

    if st.button('Predict Price'):
        prediction = model.predict(input_scaled)
        st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

    if st.checkbox('Show Model Performance on Test Data'):
        test_preds = model.predict(X_test_scaled)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        st.write(f"Test MAE: ${mae:,.2f}")
        st.write(f"Test R² Score: {r2:.3f}")

else:
    st.info("Please upload a CSV file to use the app.")
