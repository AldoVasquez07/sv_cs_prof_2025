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

# ===================================================
# CONFIG GLOBAL PARA RENDER
# ===================================================

UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ===================================================
# CLASE PRINCIPAL DEL MODELO HÍBRIDO
# ===================================================

class HybridStrokePredictor:
    """Sistema híbrido que combina predicciones clínicas e imágenes"""

    def __init__(self, clinical_model, cnn_model, scaler, label_encoders):
        self.clinical_model = clinical_model
        self.cnn_model = cnn_model
        self.scaler = scaler
        self.label_encoders = label_encoders

    def predict_clinical(self, clinical_data):
        clinical_scaled = self.scaler.transform(clinical_data)
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
# CARGA DE MODELOS (RENDER)
# ===================================================

def load_models():
    print("Cargando modelos...")

    try:
        clinical_model = joblib.load('modelo_n/clinical_stroke_model.pkl')
        scaler = joblib.load('modelo_n/clinical_scaler.pkl')
        label_encoders = joblib.load('modelo_n/label_encoders.pkl')
        cnn_model = load_model('modelo_n/cnn_stroke_model.h5')

        with open('modelo_n/hybrid_config.pkl', 'rb') as f:
            config = pickle.load(f)

        hybrid_system = HybridStrokePredictor(
            clinical_model=clinical_model,
            cnn_model=cnn_model,
            scaler=scaler,
            label_encoders=label_encoders
        )

        print("Modelos cargados correctamente")
        return hybrid_system, config

    except Exception as e:
        print(f"Error cargando modelos: {e}")
        return None, None


# ===================================================
# CREATE_APP PARA RENDER
# ===================================================

def create_app():
    app = Flask(__name__)
    
    # CORS seguro para frontend
    CORS(app, origins=["https://mn-cs-prof-2025.vercel.app"])

    hybrid_system, config = load_models()

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

    # Helpers
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    # ===================================================
    # ENDPOINTS
    # ===================================================

    @app.route("/")
    def home():
        return jsonify({"status": "API de Predicción ACV funcionando 👍"})


    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({
            "status": "ok",
            "models_loaded": hybrid_system is not None
        })


    @app.route('/predict', methods=['POST'])
    def predict():
        if hybrid_system is None:
            return jsonify({"error": "Modelos no cargados"}), 500

        try:
            # Imagen obligatoria
            if 'image' not in request.files:
                return jsonify({'error': 'Imagen requerida'}), 400

            file = request.files['image']

            if not allowed_file(file.filename):
                return jsonify({'error': 'Formato no permitido'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Datos clínicos
            data = request.form.to_dict() if request.form else request.get_json()

            required = [
                'gender', 'age', 'hypertension', 'heart_disease',
                'ever_married', 'work_type', 'Residence_type',
                'avg_glucose_level', 'bmi', 'smoking_status'
            ]

            missing = [f for f in required if f not in data]
            if missing:
                return jsonify({'error': f"Faltan campos: {', '.join(missing)}"}), 400

            # Convertir tipos
            data['age'] = float(data['age'])
            data['hypertension'] = int(data['hypertension'])
            data['heart_disease'] = int(data['heart_disease'])
            data['avg_glucose_level'] = float(data['avg_glucose_level'])
            data['bmi'] = float(data['bmi'])

            df = pd.DataFrame([data])

            # Label encoders
            for col, le in hybrid_system.label_encoders.items():
                df[col] = le.transform(df[col])

            result = hybrid_system.predict_hybrid(df.values, filepath)

            os.remove(filepath)

            return jsonify({"success": True, "result": result})

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route('/model-info')
    def model_info():
        if hybrid_system is None:
            return jsonify({"error": "Modelos no cargados"}), 500

        return jsonify({
            "model_version": config.get("model_version"),
            "img_size": IMG_SIZE,
            "clinical_features": list(hybrid_system.label_encoders.keys())
        })

    return app


# ===================================================
# RUN LOCAL
# ===================================================

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
