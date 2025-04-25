from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)

# Configure CORS for all possible frontend URLs
frontend_urls = [
    "http://localhost:5500",                # Base URL
    "http://127.0.0.1:5500",                # Base URL (alternative)
    "http://localhost:5500/website/frontend",  # Specific path
    "http://127.0.0.1:5500/website/frontend", # Specific path (alternative)
    "http://localhost:5500/website/frontend/",
    "http://127.0.0.1:5500/website/frontend/"
]

CORS(app, resources={
    r"/api/*": {
        "origins": frontend_urls,
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

# Robust model path resolution
def load_model():
    try:
        # Get the directory where app.py is located
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate up two levels to project root, then to ML-Model
        project_root = os.path.dirname(os.path.dirname(backend_dir))
        model_path = os.path.join(project_root, 'ML-Model', 'model', 'trained_model.pkl')
        
        print(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Load model at startup
model = load_model()

# Define features in exact order expected by model
FEATURES = [
    'Irregular / Missed periods', 'Cramping', 'Menstrual clots',
    'Infertility', 'Pain / Chronic pain', 'Diarrhea',
    'Long menstruation', 'Vomiting / constant vomiting', 'Migraines',
    'Extreme Bloating', 'Leg pain', 'Depression',
    'Fertility Issues', 'Ovarian cysts', 'Painful urination',
    'Pain after Intercourse', 'Digestive / GI problems', 'Anaemia / Iron deficiency',
    'Hip pain', 'Vaginal Pain/Pressure', 'Cysts (unspecified)',
    'Abnormal uterine bleeding', 'Hormonal problems', 'Feeling sick',
    'Abdominal Cramps during Intercourse', 'Insomnia / Sleeplessness', 'Loss of appetite'
]

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        print("\n Received data:", data)
        
        # Create input array with correct feature order
        input_data = [int(data.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)
        
        print(" Input array:", input_array)
        
        # Make prediction
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]
        
        response = {
            'prediction': int(prediction),
            'confidence': float(proba),
            'diagnosis': 'Endometriosis' if prediction == 1 else 'No Endometriosis',
            'features_used': FEATURES  # For debugging
        }
        
        print("Sending response:", response)
        return _corsify_actual_response(jsonify(response))
    
    except Exception as e:
        print("Prediction error:", str(e))
        return _corsify_actual_response(jsonify({'error': str(e)}), 400)

# CORS helper functions
def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/')
def home():
    return """
    <h1>Endometriosis Prediction API</h1>
    <p>Endpoints:</p>
    <ul>
        <li>POST /api/predict - Make predictions</li>
    </ul>
    <p>Allowed frontend origins:</p>
    <ul>
        <li>http://localhost:5500</li>
        <li>http://127.0.0.1:5500</li>
        <li>http://localhost:5500/website/frontend</li>
        <li>http://127.0.0.1:5500/website/frontend</li>
    </ul>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)