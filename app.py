from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
import os
from PIL import Image

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde cualquier origen

# Configuración
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Clase del sistema híbrido
class HybridStrokePredictor:
    """"Sistema hídrido que combina predicciones clínicas e imágenes"""
    def __init__(self, clinical_model, cnm_model, scaler, label_encoders):
        self.clinical_model = clinical_model
        self.cnn_model = cnm_model
        self.scaler = scaler
        self.label_encoders = label_encoders

    def predict_clinical(self, clinical_data):
        """"Predicción basada en datos clínicos"""
        clinical_scaled = self.sacler.transform(clinical_data)
        return self.clinical_model.predict_proba(clinical_scaled)[:, 1]

    def predict_image(sel, image_path):
        """"Predicción basada en imagen"""
        img = keras_image.load_img(image_path, target_size=IMG_SIZE, color_mode='grayscale')
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return self.cnn_model.predict(img_array, verbose=0)[0][0]
    
    def predict_hydrid(self, clinical_data, image_path, weights=(0.6,0.4)):
        """"Predicción hídrida combinando ambos modelos"""
        clinical_prob = self.predict_clinical(clinical_data)
        image_prob = self.predict_image(image_path)

        #combinación ponderada
        hydrid_prob = weights[0] * clinical_prob + weights [1] * image_prob

        return {
            'clinical_probability': float(clinical_prob[0]),
            'image_probability': float(image_prob),
            'hydrid_probabilty': float(hydrid_prob[0]),
            'prediction': 'Alto riesgo ACV' if hydrid_prob[0] > 0.5 else 'Bajo Riesgo ACV',
            'confidence': float(max(hydrid_prob[0], 1 - hydrid_prob[0])) 
        }
    