import numpy as np
import pandas as pd
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta

# Register the custom loss function
import keras.losses
keras.losses.root_mean_squared_error = keras.losses.mean_squared_error

# Load the pre-trained Keras model
with keras.utils.custom_object_scope({'root_mean_squared_error': keras.losses.root_mean_squared_error}):
    model = load_model('Model.keras')

# Set run_eagerly=True
model.compile(run_eagerly=True)

st.header('Load Forecasting')

data = st.file_uploader("Upload CSV file", type=["csv"])

if data is not None:
    df = pd.read_csv(data)  # Read the CSV file without specifying index_col
    
    st.subheader('Electricity Data')
    st.write("Uploaded DataFrame:", df.head())
    
    # Data Preprocessing
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    data_scaled = scaler.fit_transform(df[['nat_demand']])
    
    # Reshape input data to match the model's expected input shape
    x = np.zeros((len(data_scaled) - 100, 24, 75))  # Initialize an array to hold reshaped data
    
    # Populate the input array
    for i in range(len(data_scaled) - 100):
        seq_len = min(24*75, len(data_scaled) - i)  # Ensure sequence length does not exceed available data
        x[i, :seq_len] = data_scaled[i:i+seq_len, 0].reshape(-1, 75)

    # Model Prediction
    predict = model.predict(x)

    # Inverse transform the predicted data to original scale
    predict = scaler.inverse_transform(predict)  # Inverse transform predictions

    # Get the next date and time for prediction
    last_date = datetime.strptime(df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S')  # Adjust the datetime format
    next_date = last_date + timedelta(hours=2)

    # Output predicted value for the next 2 hours
    st.subheader('Predicted Value for Next 2 Hours')
    st.write(f"Date: {next_date.strftime('%Y-%m-%d')}, Time: {next_date.strftime('%H:%M')}, Predicted Value: {predict[-1][0]}")
