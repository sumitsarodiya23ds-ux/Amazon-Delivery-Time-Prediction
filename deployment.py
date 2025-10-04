import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Amazon Delivery Time Predictor 🚚", layout="centered")

st.title("Amazon Delivery Time Predictor 🚚")

#Load model 
MODEL_PATH = "Best_Model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found. Please check the path.")
else:
    model = joblib.load(MODEL_PATH, mmap_mode='r')

    model_columns = [
        'Agent_Age', 'Agent_Rating', 'Weather', 'Traffic', 'Vehicle', 
        'Area', 'Category', 'order_hour', 'order_dayofweek', 'pickup_delay_mins'
    ]

    st.sidebar.header("Input Features")

    agent_age = st.sidebar.number_input("Agent Age", 18, 60, 25)
    agent_rating = st.sidebar.number_input("Agent Rating (1-5)", 1, 5, 4)
    weather = st.sidebar.selectbox("Weather", ["Clear", "Rainy", "Snowy", "Cloudy"])
    traffic = st.sidebar.selectbox("Traffic", ["Low", "Medium", "High"])
    vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Bike", "Car", "Truck"])
    area = st.sidebar.number_input("Area code", 0, 100, 1)
    category = st.sidebar.number_input("Category code", 0, 20, 1)
    order_hour = st.sidebar.slider("Order Hour (0-23)", 0, 23, 12)
    order_dayofweek = st.sidebar.selectbox("Day of Week (0=Monday)", list(range(7)))
    pickup_delay_mins = st.sidebar.number_input("Pickup Delay (minutes)", 0, 500, 10)

    
    #Encode categorical variables
    weather_map = {"Clear":0, "Rainy":1, "Snowy":2, "Cloudy":3}
    traffic_map = {"Low":0, "Medium":1, "High":2}
    vehicle_map = {"Bike":0, "Car":1, "Truck":2}

    input_data = pd.DataFrame({
        "Agent_Age": [agent_age],
        "Agent_Rating": [agent_rating],
        "Weather": [weather_map[weather]],
        "Traffic": [traffic_map[traffic]],
        "Vehicle": [vehicle_map[vehicle_type]],
        "Area": [area],
        "Category": [category],
        "order_hour": [order_hour],
        "order_dayofweek": [order_dayofweek],
        "pickup_delay_mins": [pickup_delay_mins]
    })

    #Align columns with training
    try:
        input_data = input_data[model_columns]
    except KeyError as e:
        st.error(f"Column mismatch: {e}")

    #Predict button
    if st.button("Predict Delivery Time"):
     try:
        prediction = model.predict(input_data)
        total_minutes = prediction[0]
        hours = int(total_minutes // 60)  # Convert minutes to hours
        st.success(f"Estimated Delivery Time: {hours} hours")
     except Exception as e:
        st.error(f"Prediction error: {e}")
