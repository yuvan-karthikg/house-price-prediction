HOUSE PRICE PREDICTION
House Price Prediction using Neural Networks

OVERVIEW:

A deep learning-based solution for predicting residential property prices using a feedforward neural network (TensorFlow/Keras), deployed as a Streamlit web app.

STREAMLIT APP:
https://house-price-prediction-796xgfuzsxagekygtpaggb.streamlit.app/

KEY FEATURES:

ANN with three hidden layers (128, 64, 32 neurons)

Standard scaling and one-hot encoding

Streamlit app for interactive predictions

Metrics: MSE, RMSE, R²

DATASET:

Features: Area, bedrooms, bathrooms, stories, parking, main road, guest room, basement, hot water heating, air conditioning, preferred area, furnishing status

PROJECT STRUCTURE:

data/: Datasets

notebooks/: EDA and prototyping

src/: Preprocessing, training, evaluation scripts

app/: Streamlit app code

README.md: Documentation

HOW IT WORKS:

Data preprocessing: scaling and encoding

Model training with TensorFlow/Keras

Evaluation on test data

Deployment via Streamlit

RESULTS:

MSE: ~1.81 × 10¹²

RMSE: ~1,343,619

R²: 0.64

USAGE:

bash
pip install -r requirements.txt
streamlit run app/house_price_app.py
Or use the hosted app:
https://house-price-prediction-796xgfuzsxagekygtpaggb.streamlit.app/

LIMITATIONS & FUTURE WORKS:

Accuracy can be improved with additional features and data

Future: Advanced architectures, cloud deployment, enhanced UI

AUTHORS:

Yuvan Karthik G

Aditya Sajith

Nihal Muhammedali

Mohammed Farhaan
