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
    """Sistema híbrido que combina predicciones clínicas e imágenes"""
    
    def __init__(self, clinical_model, cnn_model, scaler, label_encoders):
        self.clinical_model = clinical_model
        self.cnn_model = cnn_model
        self.scaler = scaler
        self.label_encoders = label_encoders
    
    def predict_clinical(self, clinical_data):
        """Predicción basada en datos clínicos"""
        clinical_scaled = self.scaler.transform(clinical_data)
        return self.clinical_model.predict_proba(clinical_scaled)[:, 1]
    
    def predict_image(self, image_path):
        """Predicción basada en imagen"""
        img = keras_image.load_img(image_path, target_size=IMG_SIZE, color_mode='grayscale')
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return self.cnn_model.predict(img_array, verbose=0)[0][0]
    
    def predict_hybrid(self, clinical_data, image_path, weights=(0.6, 0.4)):
        """Predicción híbrida combinando ambos modelos"""
        clinical_prob = self.predict_clinical(clinical_data)
        image_prob = self.predict_image(image_path)
        
        # Combinación ponderada
        hybrid_prob = weights[0] * clinical_prob + weights[1] * image_prob
        
        return {
            'clinical_probability': float(clinical_prob[0]),
            'image_probability': float(image_prob),
            'hybrid_probability': float(hybrid_prob[0]),
            'prediction': 'Alto Riesgo ACV' if hybrid_prob[0] > 0.5 else 'Bajo Riesgo ACV',
            'confidence': float(max(hybrid_prob[0], 1 - hybrid_prob[0]))
        }

# Cargar modelos al iniciar
print("Cargando modelos...")
try:
    clinical_model = joblib.load('modelo/clinical_stroke_model.pkl')
    scaler = joblib.load('modelo/clinical_scaler.pkl')
    label_encoders = joblib.load('modelo/label_encoders.pkl')
    cnn_model = load_model('modelo/cnn_stroke_model.h5')
    
    with open('modelo/hybrid_config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    hybrid_system = HybridStrokePredictor(
        clinical_model=clinical_model,
        cnn_model=cnn_model,
        scaler=scaler,
        label_encoders=label_encoders
    )
    
    print("Modelos cargados exitosamente")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    hybrid_system = None

# Funciones auxiliares
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_clinical_data(data):
    """Valida que todos los campos clínicos estén presentes"""
    required_fields = [
        'gender', 'age', 'hypertension', 'heart_disease',
        'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    
    missing = [field for field in required_fields if field not in data]
    if missing:
        return False, f"Campos faltantes: {', '.join(missing)}"
    
    return True, None

# ENDPOINTS

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado de la API"""
    return jsonify({
        'status': 'ok',
        'models_loaded': hybrid_system is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint principal para hacer predicciones
    
    Recibe:
    - Datos clínicos como form-data o JSON
    - Imagen como archivo
    
    Retorna:
    - Probabilidades de cada modelo
    - Predicción final
    - Nivel de confianza
    """
    
    if hybrid_system is None:
        return jsonify({'error': 'Modelos no cargados correctamente'}), 500
    
    try:
        # 1. Verificar que hay imagen
        if 'image' not in request.files:
            return jsonify({'error': 'No se encontró imagen en la petición'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato de imagen no permitido. Use PNG, JPG o JPEG'}), 400
        
        # 2. Obtener datos clínicos
        clinical_data = {}
        
        # Intentar obtener desde form-data
        if request.form:
            clinical_data = request.form.to_dict()
        # O desde JSON
        elif request.get_json():
            clinical_data = request.get_json()
        else:
            return jsonify({'error': 'No se encontraron datos clínicos'}), 400
        
        # 3. Validar datos clínicos
        is_valid, error_msg = validate_clinical_data(clinical_data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        
        # 4. Guardar imagen temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 5. Procesar datos clínicos
        # Convertir tipos de datos
        clinical_data['age'] = float(clinical_data['age'])
        clinical_data['hypertension'] = int(clinical_data['hypertension'])
        clinical_data['heart_disease'] = int(clinical_data['heart_disease'])
        clinical_data['avg_glucose_level'] = float(clinical_data['avg_glucose_level'])
        clinical_data['bmi'] = float(clinical_data['bmi'])
        
        # Normalizar strings
        clinical_data['gender'] = clinical_data['gender'].lower()
        clinical_data['ever_married'] = clinical_data['ever_married'].lower()
        clinical_data['work_type'] = clinical_data['work_type'].lower()
        clinical_data['Residence_type'] = clinical_data['Residence_type'].lower()
        clinical_data['smoking_status'] = clinical_data['smoking_status'].lower()
        
        # Crear DataFrame
        clinical_df = pd.DataFrame([clinical_data])
        
        # Aplicar label encoders
        for col, le in hybrid_system.label_encoders.items():
            if col in clinical_df.columns:
                try:
                    clinical_df[col] = le.transform(clinical_df[col])
                except ValueError as e:
                    os.remove(filepath)  # Limpiar archivo temporal
                    return jsonify({
                        'error': f'Valor inválido en campo {col}: {str(e)}'
                    }), 400
        
        # 6. Hacer predicción
        result = hybrid_system.predict_hybrid(clinical_df.values, filepath)
        
        # 7. Limpiar archivo temporal
        os.remove(filepath)
        
        # 8. Retornar resultado
        return jsonify({
            'success': True,
            'result': result,
            'input_data': {
                'clinical': clinical_data,
                'image': filename
            }
        })
        
    except Exception as e:
        # Limpiar archivo temporal si existe
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'error': f'Error al procesar predicción: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Retorna información sobre los modelos cargados"""
    if hybrid_system is None:
        return jsonify({'error': 'Modelos no cargados'}), 500
    
    return jsonify({
        'model_version': config.get('model_version', 'unknown'),
        'weights': config.get('weights', 'unknown'),
        'img_size': config.get('img_size', 'unknown'),
        'clinical_features': list(label_encoders.keys())
    })

# Manejo de errores
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'Archivo demasiado grande. Máximo 16MB'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Servidor Flask iniciado")
    print("API disponible en: http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
