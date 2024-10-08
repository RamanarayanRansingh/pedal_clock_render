import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Bike Demand Prediction", page_icon="🚲", layout="wide")

# Load the model, scaler, and columns
@st.cache_resource
def load_model():
    model_path = 'resources/xgboost_grid_r2_0_94_v3.pkl'
    return pickle.load(open(model_path, "rb"))

@st.cache_resource
def load_scaler():
    sc_dump_path = 'resources/sc.pkl'
    return pickle.load(open(sc_dump_path, "rb"))

@st.cache_resource
def load_columns():
    column_path = 'resources/columns.pkl'
    return pickle.load(open(column_path, "rb"))

model = load_model()
scaler = load_scaler()
columns = load_columns()

def preprocess_input(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Convert "Date" to "year", "month", and "day"
    input_df['Date'] = pd.to_datetime(input_df['Date'], format="%Y-%m-%d")
    input_df['year'] = input_df['Date'].dt.year
    input_df['month'] = input_df['Date'].dt.month
    input_df['day'] = input_df['Date'].dt.day_name()

    # Create the weekdays_weekend column
    input_df['weekdays_weekend'] = input_df['day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Drop the 'Date', 'day', and 'year' columns
    input_df = input_df.drop(['Date', 'day', 'year'], axis=1)

    # One-hot encode the categorical features
    categorical_features = ['Hour', 'Seasons', 'Holiday', 'Functioning_Day', 'month', 'weekdays_weekend']
    for col in categorical_features:
        dummies = pd.get_dummies(input_df[col], prefix=col, drop_first=True)
        input_df = pd.concat([input_df, dummies], axis=1)
        input_df = input_df.drop([col], axis=1)

    # Ensure all expected columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the order used during training
    input_df = input_df[columns]

    return input_df

def make_prediction(input_data):
    processed_input = preprocess_input(input_data)
    
    # Scale the input features
    scaled_input = scaler.transform(processed_input)
    
    # Make the prediction (this will be the square root of the bike count)
    predicted_sqrt_value = model.predict(scaled_input)[0]
    
    # Square the prediction to get the actual bike count
    predicted_value = predicted_sqrt_value ** 2
    
    return round(predicted_value)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Get top 10 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(10), importances[indices])
    ax.set_yticks(range(10))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_title("Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    
    return fig

# Streamlit UI
st.title("🚲 Bike Demand Prediction")
st.markdown("Enter the details below to predict the number of rented bikes. This model estimates bike rental demand based on weather conditions and time of the day.")

col1, col2 = st.columns(2)

with col1:
    date = st.date_input("Date", datetime.today())
    hour = st.slider("Hour", 0, 23, 12)
    temperature = st.number_input("Temperature (°C)", -20.0, 40.0, 20.0)
    humidity = st.number_input("Humidity (%)", 0, 100, 50)
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 2.0)

with col2:
    visibility = st.number_input("Visibility (meters)", 0, 2000, 1000)
    solar_radiation = st.number_input("Solar Radiation", 0.0, 5.0, 1.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 0.0)
    snowfall = st.number_input("Snowfall (mm)", 0.0, 100.0, 0.0)
    seasons = st.selectbox("Season", ['Spring', 'Summer', 'Autumn', 'Winter'])
    holiday = st.selectbox("Holiday", ['No Holiday', 'Holiday'])
    functioning_day = st.selectbox("Functioning Day", ['Yes', 'No'])

if st.button("Predict Bike Demand", key="predict"):
    input_data = {
        'Hour': hour,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_speed': wind_speed,
        'Visibility': visibility,
        'Solar_Radiation': solar_radiation,
        'Rainfall': rainfall,
        'Snowfall': snowfall,
        'Seasons': seasons,
        'Holiday': holiday,
        'Functioning_Day': functioning_day,
        'Date': date.strftime("%Y-%m-%d")
    }
    
    prediction = make_prediction(input_data)
    
    st.success(f"Predicted number of rented bikes: {prediction}")

    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Bike Demand"},
            gauge = {
                'axis': {'range': [None, 3000]},
                'bar': {'color': "darkblue"},
                'steps' : [
                    {'range': [0, 1000], 'color': "lightgray"},
                    {'range': [1000, 2000], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 2500}}))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.pyplot(plot_feature_importance(model, columns))

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")