# train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv('housing.csv')

# Encode 'ocean_proximity' as categorical codes
df['ocean_proximity'] = df['ocean_proximity'].astype('category').cat.codes

# Drop missing values
df = df.dropna()

# Features and target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build ANN model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred = model.predict(X_test).flatten()
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2:.3f}')  # Should be between 0 and 0.7

# Save model and scaler
model.save('model.h5')
joblib.dump(scaler, 'scaler.pkl')
