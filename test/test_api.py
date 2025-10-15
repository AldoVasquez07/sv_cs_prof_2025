import requests
import json

# URL base de la API
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Prueba el endpoint de health check"""
    print("\n" + "="*50)
    print("🔍 Probando Health Check")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Prueba el endpoint de información del modelo"""
    print("\n" + "="*50)
    print("📊 Probando Model Info")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction(image_path):
    """Prueba el endpoint de predicción"""
    print("\n" + "="*50)
    print("🎯 Probando Predicción")
    print("="*50)
    
    # Datos clínicos de ejemplo
    clinical_data = {
        'gender': 'male',
        'age': '67.0',
        'hypertension': '1',
        'heart_disease': '1',
        'ever_married': 'yes',
        'work_type': 'private',
        'Residence_type': 'urban',
        'avg_glucose_level': '228.69',
        'bmi': '36.6',
        'smoking_status': 'formerly smoked'
    }
    
    # Preparar archivos
    files = {
        'image': open(image_path, 'rb')
    }
    
    # Hacer petición
    response = requests.post(
        f"{BASE_URL}/predict",
        data=clinical_data,
        files=files
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Predicción exitosa:")
        print(f"   - Probabilidad Clínica: {result['result']['clinical_probability']:.3f}")
        print(f"   - Probabilidad Imagen: {result['result']['image_probability']:.3f}")
        print(f"   - Probabilidad Híbrida: {result['result']['hybrid_probability']:.3f}")
        print(f"   - Predicción: {result['result']['prediction']}")
        print(f"   - Confianza: {result['result']['confidence']:.3f}")
    else:
        print(f"\n❌ Error: {response.json()}")

def test_prediction_json(image_path):
    """Prueba el endpoint usando JSON para datos clínicos"""
    print("\n" + "="*50)
    print("🎯 Probando Predicción (con JSON)")
    print("="*50)
    
    # Datos clínicos como JSON
    clinical_data = {
        'gender': 'female',
        'age': 45.0,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'yes',
        'work_type': 'private',
        'Residence_type': 'rural',
        'avg_glucose_level': 150.0,
        'bmi': 28.5,
        'smoking_status': 'never smoked'
    }
    
    # Preparar archivos
    files = {
        'image': open(image_path, 'rb')
    }
    
    # Convertir clinical_data a strings para form-data
    clinical_data_str = {k: str(v) for k, v in clinical_data.items()}
    
    # Hacer petición
    response = requests.post(
        f"{BASE_URL}/predict",
        data=clinical_data_str,
        files=files
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ Predicción exitosa:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ Error: {response.json()}")

def test_error_cases():
    """Prueba casos de error"""
    print("\n" + "="*50)
    print("⚠️ Probando Casos de Error")
    print("="*50)
    
    # 1. Sin imagen
    print("\n1. Sin imagen:")
    response = requests.post(f"{BASE_URL}/predict", data={'age': '50'})
    print(f"   Status: {response.status_code}")
    print(f"   Error: {response.json()['error']}")
    
    # 2. Datos clínicos incompletos
    print("\n2. Datos clínicos incompletos:")
    files = {'image': open('test_image.jpg', 'rb')}
    response = requests.post(
        f"{BASE_URL}/predict",
        data={'age': '50', 'gender': 'male'},
        files=files
    )
    print(f"   Status: {response.status_code}")
    print(f"   Error: {response.json()['error']}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 INICIANDO PRUEBAS DE LA API")
    print("="*60)
    
    # Cambia esta ruta por la ubicación de tu imagen de prueba
    IMAGE_PATH = "test_image.jpg"
    
    try:
        # 1. Health check
        test_health_check()
        
        # 2. Model info
        test_model_info()
        
        # 3. Predicción normal
        test_prediction(IMAGE_PATH)
        
        # 4. Predicción con JSON
        test_prediction_json(IMAGE_PATH)
        
        # 5. Casos de error
        test_error_cases()
        
        print("\n" + "="*60)
        print("✅ PRUEBAS COMPLETADAS")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error durante las pruebas: {e}")
        print("\nAsegúrate de que:")
        print("1. El servidor Flask está corriendo")
        print("2. Tienes una imagen de prueba en la ruta especificada")
        print("3. Los modelos están cargados correctamente")