import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

# Set up a clean webpage header
st.set_page_config(page_title="Car Price Predictor", layout="centered")
st.title("🚗 Car Price Prediction Web App")
st.markdown("Enter the car details below to estimate its market price instantly.")

# 1. Load your saved machine learning files safely
@st.cache_resource
def load_ml_components():
    model = pk.load(open('model.pkl', 'rb'))
    mmscaler = pk.load(open('mmscaler.pkl', 'rb'))
    mmscaler_y = pk.load(open('mmscaler_y.pkl', 'rb'))
    le_fuel = pk.load(open('label_encoder_fuel.pkl', 'rb'))
    return model, mmscaler, mmscaler_y, le_fuel

try:
    model, mmscaler, mmscaler_y, le_fuel = load_ml_components()
except FileNotFoundError:
    st.error("⚠️ Error: One or more pickle (.pkl) files are missing from your folder! Make sure model.pkl, mmscaler.pkl, mmscaler_y.pkl, and label_encoder_fuel.pkl are sitting next to app.py.")
    st.stop()

# 2. Build the User Input Fields (Matching your notebook structure)
st.subheader("Enter Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacture Year (e.g., 2014)", min_value=1980, max_value=2026, value=2015, step=1)
    kms_driven = st.number_input("Total Kilometers Driven", min_value=0, max_value=1000000, value=35000, step=500)

with col2:
    # Fuel types mapped directly to how your LabelEncoder was trained
    fuel_options = ["Diesel", "LPG", "Petrol"] 
    selected_fuel = st.selectbox("Fuel Type", fuel_options)

# 3. Handle the Prediction Process
if st.button("🔮 Predict Car Price", type="primary"):
    # Convert fuel type back into its exact trained number (0, 1, or 2)
    fuel_encoded = le_fuel.transform([selected_fuel])[0]
    
    # Pack inputs into an array matching your exact notebook columns: [year, kms_driven, fuel_type]
    raw_features = np.array([[year, kms_driven, fuel_encoded]])
    
    # Scale your X features using your saved MinMaxScaler
    scaled_features = mmscaler.transform(raw_features)
    
    # Predict the scaled price
    scaled_prediction = model.predict(scaled_features)
    
    # Inverse transform back from scaled numbers to raw Indian Rupees
    final_price_array = mmscaler_y.inverse_transform(scaled_prediction.reshape(-1, 1))
    predicted_price = float(final_price_array[0][0])
    
    # Display the result beautifully on screen
    st.success(f"### 🎉 Estimated Market Value: ₹ {predicted_price:,.2f}")
