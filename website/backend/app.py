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
        # Get the directory paths
        backend_dir = os.path.dirname(os.path.abspath(__file__))  # Backend folder
        project_root = os.path.dirname(backend_dir)  # Root of the project
        model_path = os.path.join(project_root, 'ML-Model', 'model', 'trained_model.pkl')

        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found.")
        
        # Load the model
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Model loading error: {e}")

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
        # Get the input data from the request
        data = request.get_json()
        input_data = [int(data.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)  # Reshape for prediction

        # Get the prediction and confidence
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]

        # Return the result as JSON
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
        # Get the directory paths for frontend
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        frontend_dir = os.path.join(project_root, 'frontend')
        
        # Return the homepage HTML file
        return send_from_directory(frontend_dir, 'website-home.html')
    except Exception as e:
        return f"Error loading homepage: {e}"

# Serve static files (CSS, JS, images)
@app.route('/<path:filename>')
def serve_static(filename):
    try:
        # Get the path for static files
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(backend_dir)
        frontend_dir = os.path.join(project_root, 'frontend')
        
        # Return the requested static file
        return send_from_directory(frontend_dir, filename)
    except Exception as e:
        return f"Error loading file: {e}"

# Start the Flask application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # PORT environment variable used by Render
    app.run(host='0.0.0.0', port=port)  # Run the app on all interfaces (useful for deployment)
