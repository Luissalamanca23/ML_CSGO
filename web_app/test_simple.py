#!/usr/bin/env python3
"""
Script de pruebas simple para la aplicación web CS:GO ML Predictor
"""

import os
import sys

def check_dependencies():
    """Verificar que todas las dependencias están instaladas"""
    print("Verificando dependencias...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'sklearn', 
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
    print("\nVerificando archivos de modelo...")
    
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
            print(f"  [OK] {name}")
        else:
            missing_files.append((name, path))
            print(f"  [MISSING] {name} - {path}")
    
    if missing_files:
        print(f"\nArchivos faltantes:")
        for name, path in missing_files:
            print(f"    - {name}: {path}")
        print("\nEjecuta los notebooks para generar los modelos:")
        print("  - effectiveness_modeling.ipynb")
        print("  - regression_modeling.ipynb")
        return False
    
    print("[SUCCESS] Todos los archivos de modelo están presentes")
    return True

def test_app_startup():
    """Probar que la aplicación puede iniciarse"""
    print("\nProbando inicio de aplicación...")
    
    try:
        # Importar la aplicación
        from app import app, predictor
        
        print("  [OK] Aplicación importada correctamente")
        
        # Verificar que los modelos se cargaron
        if predictor.classification_model is not None:
            print("  [OK] Modelo de clasificación cargado")
        else:
            print("  [WARNING] Modelo de clasificación no cargado")
        
        if predictor.regression_model is not None:
            print("  [OK] Modelo de regresión cargado")
        else:
            print("  [WARNING] Modelo de regresión no cargado")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error al importar aplicación: {str(e)}")
        return False

def main():
    """Función principal de pruebas"""
    print("CS:GO ML Predictor - Test Suite")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        print("\n[FAILED] Pruebas fallidas: Dependencias faltantes")
        return False
    
    # Verificar archivos de modelo
    if not check_model_files():
        print("\n[FAILED] Pruebas fallidas: Archivos de modelo faltantes")
        print("\nPara generar los modelos:")
        print("   1. cd ../notebooks")
        print("   2. Ejecutar effectiveness_modeling.ipynb")
        print("   3. Ejecutar regression_modeling.ipynb")
        return False
    
    # Probar inicio de aplicación
    if not test_app_startup():
        print("\n[FAILED] Pruebas fallidas: Error al iniciar aplicación")
        return False
    
    print("\n[SUCCESS] Pruebas básicas completadas exitosamente!")
    
    print("\nPara probar la aplicación completa:")
    print("   1. python app.py")
    print("   2. Abrir http://localhost:5000 en el navegador")
    
    print("\nEndpoints disponibles:")
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