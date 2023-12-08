# app.py
import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load the pre-trained model
model = joblib.load('gradient_boosting_model.pkl')

def main():
    st.title("Crime Prediction App")

    # Add user inputs, e.g., a form to input data
    st.sidebar.header("User Inputs")

    # Example: Input for OCC_MONTH
    occ_month = st.sidebar.slider("Select OCC_MONTH", 1, 12, 6)

    # You can add more input fields based on your model's features

    # Create a DataFrame from user inputs
    user_data = pd.DataFrame({
        'OCC_MONTH': [occ_month],
        # Add other features here
    })

    # Make predictions
    prediction = model.predict(user_data)

    # Display the prediction
    st.subheader("Prediction:")
    st.write(prediction)

if __name__ == '__main__':
    main()
