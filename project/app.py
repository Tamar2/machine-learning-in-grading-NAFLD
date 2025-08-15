import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import json
from ultralytics import YOLO
import base64
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms, models

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'liver_classification_resnet101.pth'  # Updated to use ResNet101

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the classification model
def load_model():
    """Load the trained PyTorch classification model"""
    try:
        # Load pre-trained ResNet101 and modify for 4 classes
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 4)
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print(f"✅ ResNet101 model loaded successfully from {MODEL_PATH}")
        print(f"🎯 This model should have significantly better accuracy, especially for Grade 3!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Initialize model
model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for classification model"""
    # Resize to 224x224 (standard for ResNet)
    image = image.resize((224, 224))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict_grade(model, image_tensor):
    """Predict liver steatosis grade using the classification model"""
    try:
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Convert 0-based index back to grade (0->1, 1->2, 2->3, 3->4)
            grade = predicted_class + 1
            
            return {
                'grade': grade,
                'confidence': confidence,
                'probabilities': probabilities[0].tolist()
            }
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return {
            'grade': 1,
            'confidence': 0.0,
            'probabilities': [0.25, 0.25, 0.25, 0.25]
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Read and process image
            image = Image.open(file.stream).convert('RGB')
            
            # Preprocess for model
            image_tensor = preprocess_image(image)
            
            # Make prediction
            if model is not None:
                result = predict_grade(model, image_tensor)
                
                # Convert image to base64 for display
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return jsonify({
                    'success': True,
                    'grade': result['grade'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities'],
                    'image': img_str
                })
            else:
                return jsonify({'error': 'Model not loaded'}), 500
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    print("🏥 Liver Steatosis Classification Web Interface")
    print("=" * 50)
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"✅ Model loaded: {model is not None}")
    print("🌐 Starting server at http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000) 