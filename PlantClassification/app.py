from flask import Flask, request, jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'plant_classification_model.h5'
CLASS_NAMES = []  # Will be loaded from file
IMG_SIZE = (128, 128)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and class names
def load_model_and_classes():
    global model, CLASS_NAMES
    try:
        model = load_model(MODEL_PATH)
        # Load class names from file (assuming it's saved as a JSON file)
        import json
        with open('class_names.json', 'r') as f:
            CLASS_NAMES = json.load(f)
        print(f"Model loaded successfully with {len(CLASS_NAMES)} classes")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model_and_classes()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            img = preprocess_image(filepath)
            
            # Make prediction
            predictions = model.predict(img)
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Prepare response
            response = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'all_predictions': {}
            }
            
            for i, class_name in enumerate(CLASS_NAMES):
                response['all_predictions'][class_name] = float(predictions[0][i])
            
            # Clean up
            os.remove(filepath)
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({'status': 'healthy', 'classes_loaded': len(CLASS_NAMES)})
    else:
        return jsonify({'status': 'unhealthy'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)