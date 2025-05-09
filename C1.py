import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

st.title("🏠 House Price Prediction using Neural Network (ANN)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    st.subheader("📄 Data Preview")
    st.write(df.head())

    # Drop unnecessary columns
    drop_cols = ['id', 'date', 'zipcode']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Feature Engineering
    if 'yr_built' in df.columns:
        df['house_age'] = 2025 - df['yr_built']
        df.drop('yr_built', axis=1, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Drop outliers
    df = df[df['price'] < df['price'].quantile(0.98)]

    if 'price' not in df.columns:
        st.error("The uploaded file must contain a 'price' column.")
        st.stop()

    # Define features and target
    X_raw = df.drop('price', axis=1)
    y_raw = df['price']
    y = np.log1p(y_raw)  # Log-transform target

    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Build and train ANN
    def build_ann(input_dim):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=input_dim))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
        return model

    @st.cache_resource
    def train_ann_model(X_train, y_train, X_test, y_test):
        model = build_ann(X_train.shape[1])
        early_stop = EarlyStopping(patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=100, batch_size=32, callbacks=[early_stop], verbose=0)
        return model

    model = train_ann_model(X_train, y_train, X_test, y_test)

    st.header("🧮 Predict a New House Price")

    # User Inputs
    user_input = {}
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.number_input(col, value=mean_val, min_value=min_val, max_value=max_val)

    for col in categorical_cols:
        options = df[col].dropna().unique()
        user_input[col] = st.selectbox(col, options)

    # Prediction
    input_df = pd.DataFrame([user_input])
    input_processed = preprocessor.transform(input_df)

    if st.button("💰 Predict Price"):
        pred_log = model.predict(input_processed)[0][0]
        pred_actual = np.expm1(pred_log)
        st.success(f"Estimated House Price: ${pred_actual:,.2f}")

    if st.checkbox("📊 Show Model Performance"):
        test_preds_log = model.predict(X_test).flatten()
        test_preds = np.expm1(test_preds_log)
        y_test_actual = np.expm1(y_test)

        mae = mean_absolute_error(y_test_actual, test_preds)
        r2 = r2_score(y_test_actual, test_preds)

        st.write(f"Test MAE: ${mae:,.2f}")
        st.write(f"Test R² Score: {r2:.3f}")

else:
    st.info("Upload a CSV file to get started.")
