import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

class PlantClassifier:
    def __init__(self, model_path, class_names, img_size=(128, 128)):
        
        self.model = load_model(model_path)
        self.class_names = class_names
        self.img_size = img_size
    
    def preprocess_image(self, image_path):
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize and normalize
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path):
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img)
        
        # Get top prediction
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = self.class_names[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        
        # Get all probabilities
        results = {}
        for i, class_name in enumerate(self.class_names):
            results[class_name] = float(predictions[0][i])
        
        return {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': results
        }

def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Predict plant species from image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--class_names', type=str, required=True, 
                        help='JSON file with class names or comma-separated list')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image to classify')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128], 
                        help='Image size (height width)')
    
    args = parser.parse_args()
    
    # Load class names
    if args.class_names.endswith('.json'):
        with open(args.class_names, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = args.class_names.split(',')
    
    # Initialize classifier
    classifier = PlantClassifier(args.model_path, class_names, tuple(args.img_size))
    
    # Make prediction
    try:
        result = classifier.predict(args.image_path)
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll predictions:")
        for class_name, prob in result['all_predictions'].items():
            print(f"  {class_name}: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()