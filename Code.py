import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow import keras

st.title("House Price Prediction using Neural Networks")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Drop 'date' and 'id' columns if present (commonly non-predictive)
    for col in ['date', 'id']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Check for 'price' column
    if 'price' not in df.columns:
        st.error("The uploaded file must contain a 'price' column as the target variable.")
        st.stop()

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Fill missing values
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # Preprocessing: OneHotEncode categoricals, scale numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # Model definition and training
    @st.cache_resource
    def train_model(X_train, y_train):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        return model

    model = train_model(X_train, y_train)

    st.header("Predict the Price of a House")

    # Generate input fields for all features
    def user_input_features():
        input_data = {}
        for col in numeric_cols:
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            default = float(X[col].median())
            input_data[col] = st.number_input(
                f"{col}", min_value=min_val, max_value=max_val, value=default
            )
        for col in categorical_cols:
            options = sorted(X[col].unique())
            default = options[0]
            input_data[col] = st.selectbox(f"{col}", options, index=options.index(default))
        return pd.DataFrame([input_data])

    input_df = user_input_features()

    # Preprocess user input using the same pipeline as training
    input_processed = preprocessor.transform(input_df)

    if st.button('Predict Price'):
        prediction = model.predict(input_processed)
        st.success(f"Estimated House Price: ${prediction[0][0]:,.2f}")

    if st.checkbox('Show Model Performance on Test Data'):
        test_preds = model.predict(X_test)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, test_preds)
        r2 = r2_score(y_test, test_preds)
        st.write(f"Test MAE: ${mae:,.2f}")
        st.write(f"Test R² Score: {r2:.3f}")

else:
    st.info("Please upload a CSV file to use the app.")
