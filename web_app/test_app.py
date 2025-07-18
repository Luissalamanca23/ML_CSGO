#!/usr/bin/env python3
"""
Script de pruebas para la aplicación web CS:GO ML Predictor
"""

import os
import sys
import subprocess
import time
import requests
import json

def check_dependencies():
    """Verificar que todas las dependencias están instaladas"""
    print("Verificando dependencias...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'pickle', 'json', 'os'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'pickle':
                import pickle
            elif package == 'json':
                import json
            elif package == 'os':
                import os
            else:
                __import__(package)
            print(f"  [OK] {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  [MISSING] {package}")
    
    if missing_packages:
        print(f"\nPaquetes faltantes: {', '.join(missing_packages)}")
        print("Instalar con: pip install -r requirements.txt")
        return False
    
    print("[SUCCESS] Todas las dependencias están instaladas")
    return True

def check_model_files():
    """Verificar que los archivos de modelo existen"""
    print("\n🔍 Verificando archivos de modelo...")
    
    model_files = {
        'Classification Model': '../models/classification/best_classification_model.pkl',
        'Classification Metadata': '../models/classification/classification_metadata.json',
        'Classification Label Encoder': '../models/classification/label_encoder.pkl',
        'Regression Model': '../models/regression/best_regression_model.pkl',
        'Regression Metadata': '../models/regression/regression_metadata.json'
    }
    
    missing_files = []
    for name, path in model_files.items():
        if os.path.exists(path):
            print(f"  ✅ {name}")
        else:
            missing_files.append((name, path))
            print(f"  ❌ {name} - {path}")
    
    if missing_files:
        print(f"\n⚠️  Archivos faltantes:")
        for name, path in missing_files:
            print(f"    - {name}: {path}")
        print("\nEjecuta los notebooks para generar los modelos:")
        print("  - effectiveness_modeling.ipynb")
        print("  - regression_modeling.ipynb")
        return False
    
    print("✅ Todos los archivos de modelo están presentes")
    return True

def test_app_startup():
    """Probar que la aplicación puede iniciarse"""
    print("\n🚀 Probando inicio de aplicación...")
    
    try:
        # Importar la aplicación
        from app import app, predictor
        
        print("  ✅ Aplicación importada correctamente")
        
        # Verificar que los modelos se cargaron
        if predictor.classification_model is not None:
            print("  ✅ Modelo de clasificación cargado")
        else:
            print("  ⚠️  Modelo de clasificación no cargado")
        
        if predictor.regression_model is not None:
            print("  ✅ Modelo de regresión cargado")
        else:
            print("  ⚠️  Modelo de regresión no cargado")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error al importar aplicación: {str(e)}")
        return False

def test_api_endpoints():
    """Probar los endpoints de API (requiere app corriendo)"""
    print("\n🧪 Probando endpoints de API...")
    
    base_url = "http://localhost:5000"
    
    # Test data
    classification_data = {
        "round_headshots": 2,
        "grenade_effectiveness": 3.5
    }
    
    regression_data = {
        "round_kills": 1.2,
        "match_headshots": 10,
        "team_equipment_value": 22000,
        "match_assists": 6
    }
    
    tests = [
        {
            "name": "Model Info",
            "url": f"{base_url}/api/model_info",
            "method": "GET",
            "data": None
        },
        {
            "name": "Classification Prediction",
            "url": f"{base_url}/api/predict/effectiveness",
            "method": "POST",
            "data": classification_data
        },
        {
            "name": "Regression Prediction",
            "url": f"{base_url}/api/predict/match_kills",
            "method": "POST",
            "data": regression_data
        }
    ]
    
    for test in tests:
        try:
            if test["method"] == "GET":
                response = requests.get(test["url"], timeout=5)
            else:
                response = requests.post(
                    test["url"], 
                    json=test["data"], 
                    headers={"Content-Type": "application/json"},
                    timeout=5
                )
            
            if response.status_code == 200:
                print(f"  ✅ {test['name']}")
                
                # Mostrar respuesta para endpoints de predicción
                if "predict" in test["url"]:
                    result = response.json()
                    if result.get("success"):
                        print(f"    📊 Predicción exitosa")
                    else:
                        print(f"    ⚠️  Predicción con errores: {result.get('error', 'Unknown')}")
            else:
                print(f"  ❌ {test['name']} - Status: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️  {test['name']} - App no está corriendo")
        except Exception as e:
            print(f"  ❌ {test['name']} - Error: {str(e)}")

def create_sample_test_script():
    """Crear script de ejemplo para probar manualmente"""
    test_script = '''#!/usr/bin/env python3
"""
Script de prueba manual para CS:GO ML Predictor
"""

import requests
import json

# Configuración
BASE_URL = "http://localhost:5000"

def test_classification():
    """Probar predicción de efectividad"""
    print("Testing Classification Prediction...")
    
    data = {
        "round_headshots": 3,
        "grenade_effectiveness": 5.0
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict/effectiveness",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        if result['success']:
            pred = result['result']
            print(f"Prediction: {pred['prediction']}")
            print(f"Confidence: {pred['confidence']:.3f}")
            print(f"Probabilities: {pred['probabilities']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_regression():
    """Probar predicción de kills"""
    print("\\nTesting Regression Prediction...")
    
    data = {
        "round_kills": 1.5,
        "match_headshots": 12,
        "team_equipment_value": 24000,
        "match_assists": 8
    }
    
    response = requests.post(
        f"{BASE_URL}/api/predict/match_kills",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        if result['success']:
            pred = result['result']
            print(f"Predicted kills: {pred['prediction']:.2f}")
            print(f"Target range: {pred['target_range']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("CS:GO ML Predictor - Manual Test Script")
    print("="*50)
    print("Make sure the Flask app is running: python app.py")
    print("="*50)
    
    try:
        test_classification()
        test_regression()
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Flask app. Is it running?")
    except Exception as e:
        print(f"Error: {str(e)}")
'''
    
    with open("manual_test.py", "w") as f:
        f.write(test_script)
    
    print("\n📝 Script de prueba manual creado: manual_test.py")
    print("   Para usar:")
    print("   1. python app.py  (en una terminal)")
    print("   2. python manual_test.py  (en otra terminal)")

def main():
    """Función principal de pruebas"""
    print("🧪 CS:GO ML Predictor - Test Suite")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        print("\n❌ Pruebas fallidas: Dependencias faltantes")
        return False
    
    # Verificar archivos de modelo
    if not check_model_files():
        print("\n❌ Pruebas fallidas: Archivos de modelo faltantes")
        print("\n💡 Para generar los modelos:")
        print("   1. cd ../notebooks")
        print("   2. Ejecutar effectiveness_modeling.ipynb")
        print("   3. Ejecutar regression_modeling.ipynb")
        return False
    
    # Probar inicio de aplicación
    if not test_app_startup():
        print("\n❌ Pruebas fallidas: Error al iniciar aplicación")
        return False
    
    print("\n✅ Pruebas básicas completadas exitosamente!")
    
    # Crear script de prueba manual
    create_sample_test_script()
    
    print("\n🚀 Para probar la aplicación completa:")
    print("   1. python app.py")
    print("   2. Abrir http://localhost:5000 en el navegador")
    print("   3. O usar: python manual_test.py")
    
    print("\n📊 Endpoints disponibles:")
    print("   - GET  /                           (Página principal)")
    print("   - GET  /classification             (Clasificación)")
    print("   - GET  /regression                 (Regresión)")
    print("   - GET  /api/model_info             (Info de modelos)")
    print("   - POST /api/predict/effectiveness  (API clasificación)")
    print("   - POST /api/predict/match_kills    (API regresión)")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)