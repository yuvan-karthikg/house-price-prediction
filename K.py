import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Use the DataFrame already loaded in your notebook (replace 'df' if your variable is different)
# df = ... # Already loaded in your notebook

# 2. Drop unnecessary columns
df = df.drop(['date', 'street', 'country'], axis=1)

# 3. Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# 4. Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# 5. Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Fit and transform
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 8. Build ANN Model
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

# 9. Early stopping
early_stopping = EarlyStopping(
    monitor='val_mse',
    patience=40,
    restore_best_weights=True,
    verbose=1
)

# 10. Train the model
history = model.fit(
    X_train_prep, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 11. Evaluate
y_pred = model.predict(X_test_prep)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")

# 12. Predict on new data example
new_data = pd.DataFrame([{
    'bedrooms': 3,
    'bathrooms': 2,
    'sqft_living': 1800,
    'sqft_lot': 5000,
    'floors': 1,
    'waterfront': 0,
    'view': 0,
    'condition': 3,
    'sqft_above': 1800,
    'sqft_basement': 0,
    'yr_built': 1990,
    'yr_renovated': 0,
    'city': 'Seattle',
    'statezip': 'WA 98103'
}])
new_data_prep = preprocessor.transform(new_data)
predicted_price = model.predict(new_data_prep)
print(f"Predicted price: {predicted_price[0][0]:.2f}")


