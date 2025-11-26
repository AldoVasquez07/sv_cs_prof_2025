import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
import logging

# ===================================================
# CONFIGURACIÓN DE LOGGING
# ===================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================================================
# CONFIG GLOBAL
# ===================================================
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===================================================
# CLASE DEL MODELO HÍBRIDO
# ===================================================
class HybridStrokePredictor:
    def __init__(self, clinical_model, cnn_model, scaler, label_encoders):
        self.clinical_model = clinical_model
        self.cnn_model = cnn_model
        self.scaler = scaler
        self.label_encoders = label_encoders

    def predict_clinical(self, clinical_data):
        # Convertir a DataFrame con nombres de columnas para evitar warning
        feature_names = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        if not isinstance(clinical_data, pd.DataFrame):
            clinical_df = pd.DataFrame(clinical_data, columns=feature_names)
        else:
            clinical_df = clinical_data
            
        clinical_scaled = self.scaler.transform(clinical_df)
        return self.clinical_model.predict_proba(clinical_scaled)[:, 1]

    def predict_image(self, image_path):
        img = keras_image.load_img(image_path, target_size=IMG_SIZE, color_mode='grayscale')
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return self.cnn_model.predict(img_array, verbose=0)[0][0]

    def predict_hybrid(self, clinical_data, image_path, weights=(0.6, 0.4)):
        clinical_prob = self.predict_clinical(clinical_data)
        image_prob = self.predict_image(image_path)
        hybrid_prob = weights[0] * clinical_prob + weights[1] * image_prob

        return {
            'clinical_probability': float(clinical_prob[0]),
            'image_probability': float(image_prob),
            'hybrid_probability': float(hybrid_prob[0]),
            'prediction': 'Alto Riesgo ACV' if hybrid_prob[0] > 0.5 else 'Bajo Riesgo ACV',
            'confidence': float(max(hybrid_prob[0], 1 - hybrid_prob[0]))
        }

# ===================================================
# CARGA DE MODELOS AL INICIO (FUERA DE LA FUNCIÓN)
# ===================================================
logger.info("🔄 Iniciando carga de modelos...")

try:
    clinical_model = joblib.load('modelo/clinical_stroke_model.pkl')
    logger.info("✅ Modelo clínico cargado")
    
    scaler = joblib.load('modelo/clinical_scaler.pkl')
    logger.info("✅ Scaler cargado")
    
    label_encoders = joblib.load('modelo/label_encoders.pkl')
    logger.info("✅ Label encoders cargados")
    
    cnn_model = load_model('modelo/cnn_stroke_model.h5')
    logger.info("✅ Modelo CNN cargado")

    with open('modelo/hybrid_config.pkl', 'rb') as f:
        config = pickle.load(f)
    logger.info("✅ Configuración cargada")

    hybrid_system = HybridStrokePredictor(
        clinical_model=clinical_model,
        cnn_model=cnn_model,
        scaler=scaler,
        label_encoders=label_encoders
    )
    
    logger.info("✅✅✅ TODOS LOS MODELOS CARGADOS CORRECTAMENTE")
    
except Exception as e:
    logger.error(f"❌ ERROR CRÍTICO cargando modelos: {e}")
    hybrid_system = None
    config = None
    raise

# ===================================================
# CREACIÓN DE LA APP
# ===================================================
app = Flask(__name__)

# CORS
CORS(app, resources={
    r"/*": {
        "origins": ["https://mn-cs-prof-2025.vercel.app", "http://localhost:*"],
        "supports_credentials": True
    }
})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ===================================================
# ENDPOINTS
# ===================================================

@app.route("/")
def home():
    return jsonify({
        "status": "API de Predicción ACV funcionando 👍",
        "models_loaded": hybrid_system is not None
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": hybrid_system is not None,
        "version": config.get("model_version") if config else "unknown"
    })

@app.route('/predict', methods=['POST'])
def predict():
    if hybrid_system is None:
        logger.error("Intento de predicción con modelos no cargados")
        return jsonify({"error": "Modelos no disponibles"}), 503

    filepath = None
    
    try:
        # Validar imagen
        if 'image' not in request.files:
            return jsonify({'error': 'Imagen requerida'}), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Nombre de archivo vacío'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Formato no permitido. Use: png, jpg, jpeg'}), 400

        # Guardar imagen temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{os.getpid()}_{filename}")
        file.save(filepath)
        logger.info(f"Imagen guardada: {filepath}")

        # Obtener datos clínicos
        data = {}
        if request.form:
            data = request.form.to_dict()
        elif request.json:
            data = request.json
        else:
            return jsonify({'error': 'Datos clínicos requeridos'}), 400

        # Validar campos requeridos
        required = [
            'gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]

        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f"Faltan campos: {', '.join(missing)}"}), 400

        # Conversión de tipos con validación
        try:
            data['age'] = float(data['age'])
            data['hypertension'] = int(data['hypertension'])
            data['heart_disease'] = int(data['heart_disease'])
            data['avg_glucose_level'] = float(data['avg_glucose_level'])
            data['bmi'] = float(data['bmi'])
        except ValueError as ve:
            return jsonify({'error': f'Error en formato de datos: {ve}'}), 400

        # Crear DataFrame
        df = pd.DataFrame([data])

        # Aplicar label encoders
        for col, le in hybrid_system.label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError as ve:
                    return jsonify({'error': f'Valor no válido en {col}: {ve}'}), 400

        # Realizar predicción
        logger.info("Iniciando predicción híbrida...")
        result = hybrid_system.predict_hybrid(df, filepath)
        logger.info(f"Predicción completada: {result['prediction']}")

        return jsonify({"success": True, "result": result})

    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500
    
    finally:
        # Limpiar archivo temporal
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Archivo temporal eliminado: {filepath}")
            except Exception as e:
                logger.warning(f"No se pudo eliminar archivo temporal: {e}")

@app.route('/model-info')
def model_info():
    if hybrid_system is None:
        return jsonify({"error": "Modelos no cargados"}), 503

    return jsonify({
        "model_version": config.get("model_version") if config else "unknown",
        "img_size": IMG_SIZE,
        "clinical_features": list(hybrid_system.label_encoders.keys()),
        "status": "operational"
    })

# ===================================================
# PUNTO DE ENTRADA
# ===================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)