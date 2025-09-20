from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and the scaler once when starting the server
model = load_model('my_model.h5')
scaler = joblib.load('scaler.joblib')  # Load the fitted scaler
# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('candles')  # Get the data from the POST request
    if not data or len(data) != 10:
        return jsonify({'error': 'You need to provide 10 candles'}), 400
    
    try:
        # Ensure data is a 2D NumPy array
        data_array = np.array(data, dtype=float)
        
        # Preprocess the input data
        input_data = preprocess_data(data_array)
        
        # Predict using the pre-loaded model
        prediction = model.predict(input_data)
        
        # Return prediction as JSON response
        return jsonify({'prediction': float(prediction[0, 0])})
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400

# Preprocess data function
def preprocess_data(data):
    # Assuming 'data' is a 2D NumPy array with shape (10, 7)
    # +91 7202883415
    scaled_data = scaler.transform(data)
    return np.reshape(scaled_data, (1, len(data), 7))  # Shape as required for the model


@app.route('/hello', methods=['GET'])
def hello():
    return "hello"


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
