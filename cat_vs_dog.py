import streamlit as st
import joblib
import numpy as np

model = joblib.load('cat_vs_dog_model.pkl')

st.title("Cat vs Dog Classification")

height = st.number_input("Height (0.3 to 0.9)", min_value=0.3, max_value=0.9, value=0.5, step=0.01)
weight = st.number_input("Weight (2 to 10)", min_value=2.0, max_value=10.0, value=5.0, step=0.1)
fur_intensity = st.number_input("Fur Intensity (0.1 to 1.0)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
tail_length = st.number_input("Tail Length (0.1 to 1.0)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
energy_level = st.number_input("Energy Level (0.1 to 1.0)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
ear_size = st.number_input("Ear Size (0.1 to 0.9)", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
paw_size = st.number_input("Paw Size (0.1 to 0.9)", min_value=0.1, max_value=0.9, value=0.5, step=0.01)
interaction_term = st.number_input("Interaction Term (0.1 to 1)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)

if st.button("Predict"):
    try:
        input_data = np.array([[height, weight, fur_intensity, tail_length, energy_level, ear_size, paw_size, interaction_term]])
        st.write(f"Input data shape: {input_data.shape}")
            
        prediction = model.predict(input_data)
            
            
        result = "Dog" if prediction[0] == 1 else "Cat"
        st.write(f"The model predicts: **{result}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")