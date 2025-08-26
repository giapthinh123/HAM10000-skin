from flask import Blueprint, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import base64
import io
import json
import numpy as np
import torch.nn as nn
from ultralytics import YOLO
import os


object_detection_bp = Blueprint('object_detection', __name__)

yolo_model = None
classification_model = None
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

DISEASE_NAMES = {
    'akiec': 'Actinic keratosis (solar keratosis)',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions'
}

class CustomCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = self.create_conv_block(3, 16)
        self.conv2 = self.create_conv_block(16, 32)
        self.conv3 = self.create_conv_block(32, 64)
        self.conv4 = self.create_conv_block(64, 64)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 13 * 13, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(

            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes),
        )
        
        self.global_avg_pool = nn.AvgPool2d(3, stride=2)

    def create_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)

        x = self.global_avg_pool(x)
        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.fc2(x) 
        x = self.fc3(x)
        return x

def load_model():
    global yolo_model, classification_model, class_names
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(script_dir, 'object_detection.pt')
        if not os.path.exists(yolo_path):
            print(f"YOLO model file not found: {yolo_path}")
            return False
        yolo_model = YOLO(yolo_path)

        checkpoint_path = os.path.join(script_dir, 'classification_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        classification_model = CustomCNN(num_classes=7)
        classification_model.load_state_dict(checkpoint['model_state_dict'])
        classification_model.eval()
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def preprocess_image_for_classification(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    processed_image = transform(pil_image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image

def detect_objects(image, threshold=0.6):
    global yolo_model, classification_model, class_names
    if yolo_model is None or classification_model is None:
        if not load_model():
            return {"error": "Models not loaded"}

    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = yolo_model(image_rgb)
        if results is None:
            return {"error": "YOLO model returned None results"}
        
        all_boxes = []
        if not isinstance(results, list):
            results = [results]
        
        for result in results:
            if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
                continue
                
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                if conf > threshold:
                    all_boxes.append({
                        'box': box,
                        'coords': [x1, y1, x2, y2],
                        'confidence': conf
                    })
        
        if not all_boxes:
            return {"objects": [], "status": "success", "message": "No objects detected above threshold"}
        
        best_box_info = max(all_boxes, key=lambda x: x['confidence'])
        box = best_box_info['box']
        x1, y1, x2, y2 = best_box_info['coords']
        conf = best_box_info['confidence']
        
        bbox = [int(x1), int(y1), int(x2), int(y2)]
        cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if cropped_image.size == 0:
            return {"objects": [], "status": "success", "message": "Invalid cropped image"}
        
        processed_crop = preprocess_image_for_classification(cropped_image)
        with torch.no_grad():
            classification_output = classification_model(processed_crop)
            probabilities = torch.softmax(classification_output, dim=1)
            
            disease_probabilities = {}
            for i, class_name in enumerate(class_names):
                prob_percent = probabilities[0][i].item() * 100
                disease_probabilities[class_name] = {
                    "percentage": round(prob_percent, 2),
                    "description": DISEASE_NAMES[class_name]
                }
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print(f"Detected: {class_names[predicted_class]} with confidence: {conf:.2f}")
        
        detection = {
            "bbox": bbox,
            "confidence": float(conf),
            "class": class_names[predicted_class],
            "predicted_class": class_names[predicted_class],
            "detection_confidence": float(conf),
            "classification_confidence": probabilities[0][predicted_class].item(),
            "disease_probabilities": disease_probabilities,
            "description": DISEASE_NAMES[class_names[predicted_class]]
        }
        
        detections = [detection]

        return {"objects": detections, "status": "success"}
    except Exception as e:
        return {"error": f"Detection failed: {str(e)}"}

@object_detection_bp.route('/detect', methods=['POST'])
def detect_objects_endpoint():
    try:
        
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
            
        if 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
            
        image_data = request.json['image']
        
        
        if not image_data:
            return jsonify({"error": "Empty image data"}), 400
            
       
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
            
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {str(e)}"}), 400
            
        
        threshold = 0.65
            
        result = detect_objects(image, threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@object_detection_bp.route('/detect/file', methods=['POST'])
def detect_objects_file():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        
        threshold = 0.65
            
        result = detect_objects(image, threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@object_detection_bp.route('/health', methods=['GET'])
def health_check():
    global yolo_model, classification_model
    return jsonify({
        "status": "healthy",
        "yolo_model_loaded": yolo_model is not None,
        "classification_model_loaded": classification_model is not None,
        "class_names": class_names,
        "disease_names": DISEASE_NAMES
    })

load_model()

