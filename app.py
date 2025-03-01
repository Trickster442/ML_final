import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Hair-Fall Prediction API")

# User input (only selected features)
total_protein = st.number_input("Total Protein", min_value=0)
total_keratine = st.number_input("Total Keratine", min_value=0)
vitamin = st.number_input("Vitamin Level", min_value=0)
manganese = st.number_input("Manganese Level", min_value=0)
iron = st.number_input("Iron Level", min_value=0)
calcium = st.number_input("Calcium Level", min_value=0)
liver_data = st.number_input("Liver Data", min_value=0)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[total_protein, total_keratine, vitamin, 
                            manganese, iron, calcium, liver_data]])  # Only selected features
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Hair-Fall Level: {prediction[0]}")
