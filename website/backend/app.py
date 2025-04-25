
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load model on startup
def load_model():
    try:
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        model_path = os.path.join(project_root, 'ML-Model', 'model', 'trained_model.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found.")
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Model loading error: {e}")

model = load_model()

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
        input_array = np.array(input_data).reshape(1, -1)

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
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        frontend_dir = os.path.join(project_root, 'frontend')
        return send_from_directory(frontend_dir, 'website-home.html')
    except Exception as e:
        return f"Error loading homepage: {e}"

# Serve static files (CSS, JS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    frontend_dir = os.path.join(project_root, 'frontend')
    return send_from_directory(frontend_dir, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # PORT is used by Render
    app.run(host='0.0.0.0', port=port)
