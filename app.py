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
        """"Predicció basada en datos clínicos"""
        clinical_scaled = self.sacler.transform(clinical_data)
        return self.clinical_model.predict_proba(clinical_scaled)[:, 1]