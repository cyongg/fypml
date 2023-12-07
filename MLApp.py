pip install flask
from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the Ridge Regression model and scaler
with open('ridge_model.pkl', 'rb') as model_file:
    ridge_model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the request
        features = request.json['features']

        # Standardize the input features
        features_scaled = scaler.transform([features])

        # Make predictions
        prediction = ridge_model.predict(features_scaled)

        # Send the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
