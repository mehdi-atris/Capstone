import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

# Define expected features based on the training process
expected_features = [
    'event_duration',  # Numeric feature
    'event_type_Business & Finance', 
    'event_type_Construction, Infrastructure & Manufacturing',
    'event_type_Consumer Goods & Retail',
    'event_type_Entertainment, Media & Design',
    'event_type_Healthcare, Life Sciences & Biotechnology',
    'event_type_Technology & Data',
    'event_type_Tourism & Hospitality',
    'continent_Asia', 
    'continent_Europe', 
    'continent_North America', 
    'continent_South America'
]

# App title and description
st.title("Event Booking Prediction App")
st.write("Predict the expected number of bookings per visitor based on event details.")

# Sidebar inputs
st.sidebar.header("Input Event Details")

# Function to capture user inputs
def get_user_input():
    # Slider for event duration
    event_duration = st.sidebar.slider("Event Duration (days)", min_value=2, max_value=5, value=3)
    
    # Dropdown for event type
    event_type = st.sidebar.selectbox(
        "Event Type", 
        [
            "Business & Finance", 
            "Construction, Infrastructure & Manufacturing",
            "Consumer Goods & Retail",
            "Entertainment, Media & Design",
            "Healthcare, Life Sciences & Biotechnology",
            "Technology & Data",
            "Tourism & Hospitality"
        ]
    )
    
    # Dropdown for continent
    continent = st.sidebar.selectbox(
        "Continent", 
        ["Asia", "Europe", "North America", "South America"]
    )
    
    # Initialize input dictionary with zeros
    input_features = {feature: 0 for feature in expected_features}
    
    # Populate features
    input_features['event_duration'] = event_duration
    if f"event_type_{event_type}" in input_features:
        input_features[f"event_type_{event_type}"] = 1
    if f"continent_{continent}" in input_features:
        input_features[f"continent_{continent}"] = 1
    
    # Convert to array in correct order
    return np.array([input_features[feature] for feature in expected_features]).reshape(1, -1)

# Capture user inputs
user_input = get_user_input()

# Make prediction when user clicks the button
if st.button("Predict Booking/Visitor"):
    prediction = model.predict(user_input)[0]
    st.subheader(f"Predicted Booking/Visitor: {prediction:.2f}")

# Additional notes for the user
st.write("""
### Instructions:
- Use the sliders and dropdown menus in the sidebar to input event details.
- Click the **Predict Booking/Visitor** button to see the prediction.
""")
