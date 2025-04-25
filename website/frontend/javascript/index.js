from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging
import xgboost as xgb

app = Flask(__name__)
CORS(app)  # This will allow all origins; you can specify origins if needed

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model on startup
def load_model():
    try:
        model_path = "ML-Model/model/trained_model.pkl"  # Ensure this path is correct

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Model loading error: {e}")
        raise

model = load_model()  # Load the model when the app starts

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
        data = request.get_json()
        input_data = [int(data.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for prediction

        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]

        return jsonify({
            'prediction': int(prediction),
            'confidence': float(proba),
            'diagnosis': 'Endometriosis' if prediction == 1 else 'No Endometriosis'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Serve homepage
@app.route('/')
def serve_home():
    try:
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')

        if os.path.exists(frontend_dir):
            logging.info(f"Contents of frontend directory: {os.listdir(frontend_dir)}")
        else:
            logging.error(f"Frontend directory does not exist: {frontend_dir}")
            return "Error: Frontend directory does not exist.", 500
        
        return send_from_directory(frontend_dir, 'home.html')
    except Exception as e:
        return f"Error loading homepage: {e}", 500

# Serve static files (CSS, JS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    try:
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')
        
        return send_from_directory(frontend_dir, filename)
    except Exception as e:
        return f"Error loading file: {e}", 500

# Start the Flask application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # PORT environment variable used by Render
    app.run(host='0.0.0.0', port=port)
