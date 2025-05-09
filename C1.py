import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

st.title("🏡 House Price Prediction using XGBoost (Optimized)")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📄 Preview of Uploaded Data")
    st.write(df.head())

    # Drop irrelevant columns
    drop_cols = ['id', 'date', 'zipcode']  # drop more if needed
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Feature Engineering: Age of house
    if 'yr_built' in df.columns:
        df['house_age'] = 2025 - df['yr_built']
        df.drop('yr_built', axis=1, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Remove outliers
    df = df[df['price'] < df['price'].quantile(0.98)]

    if 'price' not in df.columns:
        st.error("The uploaded file must contain a 'price' column as the target variable.")
        st.stop()

    X_raw = df.drop('price', axis=1)
    y_raw = df['price']

    # Log-transform the target
    y = np.log1p(y_raw)

    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    X_processed = preprocessor.fit_transform(X_raw)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    @st.cache_resource
    def train_model(X_train, y_train, X_test, y_test):
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        return model

    model = train_model(X_train, y_train, X_test, y_test)

    st.header("🔍 Predict the Price of a House")

    # User input
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

    input_df = pd.DataFrame([user_input])
    input_processed = preprocessor.transform(input_df)

    if st.button("💰 Predict Price"):
        pred_log = model.predict(input_processed)
        pred_actual = np.expm1(pred_log)  # inverse of log1p
        st.success(f"Estimated House Price: ${pred_actual[0]:,.2f}")

    if st.checkbox("📊 Show Model Performance on Test Data"):
        test_preds_log = model.predict(X_test)
        test_preds = np.expm1(test_preds_log)
        y_test_actual = np.expm1(y_test)

        mae = mean_absolute_error(y_test_actual, test_preds)
        r2 = r2_score(y_test_actual, test_preds)

        st.write(f"Test MAE: ${mae:,.2f}")
        st.write(f"Test R² Score: {r2:.3f}")

else:
    st.info("Please upload a CSV file to use the app.")


