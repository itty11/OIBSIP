from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained fraud detection model
model = joblib.load("fraud_model.pkl")  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        return jsonify({'prediction': int(prediction), 'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
