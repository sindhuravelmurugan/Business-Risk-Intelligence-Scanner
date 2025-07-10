from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = tf.keras.models.load_model('business_risk_model.h5')

# Paste from scaler_stats.js manually
scaler_mean = [<paste mean values here>]
scaler_std = [<paste std values here>]

scaler = StandardScaler()
scaler.mean_ = np.array(scaler_mean)
scaler.scale_ = np.array(scaler_std)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_features = np.array(data['features']).reshape(1, -1)
        scaled_input = (input_features - scaler.mean_) / scaler.scale_
        prediction = model.predict(scaled_input)[0][0]
        return jsonify({'risk_score': round(float(prediction), 4)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
