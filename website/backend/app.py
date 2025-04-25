import os
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)

# Directory configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))       # .../website/backend
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)                    # .../website
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')          # .../website/frontend
MODEL_PATH   = os.path.join(PROJECT_ROOT, 'ML-Model', 'model', 'trained_model.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model on startup
def load_model():
    if not os.path.exists(MODEL_PATH):
        logging.error("Model file not found at %s", MODEL_PATH)
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    logging.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
    return model

model = load_model()

# Define features
FEATURES = [
    'Irregular / Missed periods', 'Cramping', 'Menstrual clots', 'Infertility',
    'Pain / Chronic pain', 'Diarrhea', 'Long menstruation', 'Vomiting / constant vomiting',
    'Migraines', 'Extreme Bloating', 'Leg pain', 'Depression', 'Fertility Issues',
    'Ovarian cysts', 'Painful urination', 'Pain after Intercourse',
    'Digestive / GI problems', 'Anaemia / Iron deficiency', 'Hip pain',
    'Vaginal Pain/Pressure', 'Cysts (unspecified)', 'Abnormal uterine bleeding',
    'Hormonal problems', 'Feeling sick', 'Abdominal Cramps during Intercourse',
    'Insomnia / Sleeplessness', 'Loss of appetite'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = [int(data.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]

        return jsonify({
            'prediction': int(prediction),
            'confidence': float(proba),
            'diagnosis': 'Endometriosis' if prediction == 1 else 'No Endometriosis'
        })
    except Exception as e:
        logging.error("Prediction error: %s", e)
        return jsonify({'error': str(e)}), 400

@app.route('/')
def serve_home():
    # Debug: log directory contents
    logging.info("Frontend dir contents: %s", os.listdir(FRONTEND_DIR))
    return send_from_directory(FRONTEND_DIR, 'home.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
