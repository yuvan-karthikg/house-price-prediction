import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
df = pd.read_csv('your_file.csv')

# 2. Drop uninformative columns
df = df.drop(['date', 'street', 'country'], axis=1)

# 3. Handle missing values
df = df.dropna()  # or use imputation

# 4. Feature engineering
df['age'] = 2025 - df['yr_built']
df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)
df = df.drop(['yr_built', 'yr_renovated'], axis=1)

# 5. Remove outliers (example: remove top/bottom 1% prices)
df = df[(df['price'] < df['price'].quantile(0.99)) & (df['price'] > df['price'].quantile(0.01))]

# 6. Encode categorical features
X = df.drop('price', axis=1)
y = df['price']

num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])

X_prep = preprocessor.fit_transform(X)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=0.2, random_state=42)

# 8. Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=1)

# 9. Evaluation
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# 10. Plot actual vs predicted
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()
