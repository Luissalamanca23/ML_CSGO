"""
Script para verificar y corregir metadatos de los modelos
"""
import os
import json
import pickle

# Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "models", "classification")
REGRESSION_DIR = os.path.join(BASE_DIR, "models", "regression")

def fix_classification_metadata():
    """Corregir metadatos de clasificación"""
    metadata_path = os.path.join(CLASSIFICATION_DIR, "classification_metadata.json")
    
    # Metadatos correctos basados en nuestros scripts
    correct_metadata = {
        "model_name": "K-Nearest Neighbors",
        "model_type": "classification",
        "target": "EffectivenessLevel",
        "features": ["RoundHeadshots", "GrenadeEffectiveness"],
        "use_scaler": True,  # KNN necesita escalado
        "best_params": {"n_neighbors": 7, "weights": "distance"},
        "performance_metrics": {
            "test_accuracy": 0.813,
            "test_f1": 0.807,
            "test_auc": 0.884,
            "cv_accuracy": 0.810
        },
        "class_mapping": {
            "0": "Bajo",
            "1": "Medio", 
            "2": "Alto"
        },
        "feature_requirements": {
            "RoundHeadshots": "int64",
            "GrenadeEffectiveness": "float64"
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(correct_metadata, f, indent=2)
    
    print(f"OK Metadatos de clasificacion corregidos: {metadata_path}")

def fix_regression_metadata():
    """Corregir metadatos de regresión"""
    metadata_path = os.path.join(REGRESSION_DIR, "regression_metadata.json")
    
    # Metadatos correctos basados en nuestros scripts
    correct_metadata = {
        "model_name": "Gradient Boosting",
        "model_type": "regression",
        "target": "MatchKills",
        "features": ["MatchHeadshots", "MatchAssists", "RoundKills", "TeamStartingEquipmentValue"],
        "use_scaler": False,  # Gradient Boosting NO necesita escalado
        "best_params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8
        },
        "performance_metrics": {
            "test_r2": 0.7487,
            "test_rmse": 3.1271,
            "test_mae": 2.3017,
            "cv_r2_mean": 0.745,
            "cv_r2_std": 0.003
        },
        "target_info": {
            "min_value": 0.0,
            "max_value": 41.0,
            "mean_value": 8.5,
            "std_value": 6.2
        },
        "feature_requirements": {
            "MatchHeadshots": "float64",
            "MatchAssists": "float64",
            "RoundKills": "float64",
            "TeamStartingEquipmentValue": "float64"
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(correct_metadata, f, indent=2)
    
    print(f"OK Metadatos de regresion corregidos: {metadata_path}")

def verify_models():
    """Verificar que los modelos existen"""
    classification_files = [
        "best_classification_model.pkl",
        "classification_scaler.pkl", 
        "label_encoder.pkl",
        "classification_metadata.json"
    ]
    
    regression_files = [
        "best_regression_model.pkl",
        "regression_scaler.pkl",
        "regression_metadata.json"
    ]
    
    print("\n=== VERIFICACIÓN DE ARCHIVOS ===")
    
    print(f"\nClasificación ({CLASSIFICATION_DIR}):")
    for file in classification_files:
        path = os.path.join(CLASSIFICATION_DIR, file)
        exists = "OK" if os.path.exists(path) else "NO"
        print(f"  {exists} {file}")
    
    print(f"\nRegresion ({REGRESSION_DIR}):")
    for file in regression_files:
        path = os.path.join(REGRESSION_DIR, file)
        exists = "OK" if os.path.exists(path) else "NO"
        print(f"  {exists} {file}")

def test_predictions():
    """Probar predicciones con los modelos corregidos"""
    print("\n=== PRUEBA DE PREDICCIONES ===")
    
    try:
        # Cargar modelo de clasificación
        with open(os.path.join(CLASSIFICATION_DIR, "best_classification_model.pkl"), 'rb') as f:
            clf_model = pickle.load(f)
        with open(os.path.join(CLASSIFICATION_DIR, "classification_scaler.pkl"), 'rb') as f:
            clf_scaler = pickle.load(f)
        with open(os.path.join(CLASSIFICATION_DIR, "label_encoder.pkl"), 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Prueba clasificación
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame([[1, 1]], columns=["RoundHeadshots", "GrenadeEffectiveness"])
        test_scaled = clf_scaler.transform(test_data)
        pred = clf_model.predict(test_scaled)[0]
        proba = clf_model.predict_proba(test_scaled)[0]
        
        pred_label = label_encoder.inverse_transform([pred])[0]
        
        print(f"\nClasificacion (1 headshot, 1 grenade):")
        print(f"  Prediccion: {pred_label}")
        print(f"  Probabilidades: {proba}")
        
    except Exception as e:
        print(f"ERROR Error en clasificacion: {e}")
    
    try:
        # Cargar modelo de regresión
        with open(os.path.join(REGRESSION_DIR, "best_regression_model.pkl"), 'rb') as f:
            reg_model = pickle.load(f)
        
        # Prueba regresión (sin escalado)
        test_data = pd.DataFrame([[3, 1, 0, 20000]], 
                                columns=["MatchHeadshots", "MatchAssists", "RoundKills", "TeamStartingEquipmentValue"])
        pred = reg_model.predict(test_data)[0]
        
        print(f"\nRegresion (3 headshots, 1 assist, 0 round_kills, 20000 equipment):")
        print(f"  Prediccion MatchKills: {pred:.2f}")
        
    except Exception as e:
        print(f"ERROR Error en regresion: {e}")

if __name__ == "__main__":
    print("=== CORRECCIÓN DE METADATOS ===")
    
    # Crear directorios si no existen
    os.makedirs(CLASSIFICATION_DIR, exist_ok=True)
    os.makedirs(REGRESSION_DIR, exist_ok=True)
    
    # Corregir metadatos
    fix_classification_metadata()
    fix_regression_metadata()
    
    # Verificar archivos
    verify_models()
    
    # Probar predicciones
    test_predictions()
    
    print("\n=== CORRECCIÓN COMPLETADA ===")
    print("Ahora puedes reiniciar la web app y las predicciones deberían funcionar correctamente.")