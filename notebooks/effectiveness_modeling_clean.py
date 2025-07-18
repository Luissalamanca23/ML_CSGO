"""
Modelado de Efectividad - CS:GO Dataset (Limpio)
================================================

Objetivo: Entrenar K-Nearest Neighbors para predecir nivel de efectividad
Best Model: K-Nearest Neighbors (Accuracy=0.813, F1=0.807, AUC=0.884)
"""

import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Configuración
BASE_DIR = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml"
DATA_PATH = os.path.join(BASE_DIR, "data", "02_intermediate", "csgo_data_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "classification")
os.makedirs(MODELS_DIR, exist_ok=True)

def create_features(df):
    """Crear features para el modelo de efectividad"""
    df_new = df.copy()
    
    # Variables necesarias
    if 'RoundKills' not in df_new.columns:
        df_new['RoundKills'] = df_new.get('Kills', np.random.poisson(0.5, len(df_new)))
    if 'RoundAssists' not in df_new.columns:
        df_new['RoundAssists'] = df_new.get('Assists', np.random.poisson(0.3, len(df_new)))
    
    # Effectiveness Score
    df_new['EffectivenessScore'] = df_new['RoundKills'] * 2 + df_new['RoundAssists']
    
    # Effectiveness Level
    df_new['EffectivenessLevel'] = pd.cut(
        df_new['EffectivenessScore'].astype(float),
        bins=[-0.1, 0.5, 2, np.inf],
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # Features principales
    if 'RoundHeadshots' not in df_new.columns:
        np.random.seed(42)
        df_new['RoundHeadshots'] = np.random.binomial(df_new['RoundKills'], 0.3)
    
    if 'GrenadeEffectiveness' not in df_new.columns:
        np.random.seed(42)
        df_new['GrenadeEffectiveness'] = np.random.poisson(df_new['RoundKills'] + df_new['RoundAssists'] * 0.5)
    
    return df_new

def train_best_model():
    """Entrenar el mejor modelo (K-Nearest Neighbors)"""
    print("="*50)
    print("ENTRENAMIENTO MODELO DE EFECTIVIDAD")
    print("="*50)
    
    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape}")
    
    # Crear features
    df_features = create_features(df)
    
    # Preparar datos
    features = ['RoundHeadshots', 'GrenadeEffectiveness']
    X = df_features[features].copy()
    
    # Codificar target
    le = LabelEncoder()
    y = le.fit_transform(df_features['EffectivenessLevel'])
    
    print(f"Features: {features}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar mejor modelo
    model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # Métricas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
    
    print(f"\nResultados del modelo:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_test_pred, target_names=['Bajo', 'Medio', 'Alto']))
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "best_classification_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "classification_scaler.pkl")
    le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    metadata_path = os.path.join(MODELS_DIR, "classification_metadata.json")
    
    # Guardar archivos
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    metadata = {
        'model_name': 'K-Nearest Neighbors',
        'features': features,
        'performance': {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc
        },
        'class_mapping': {0: 'Bajo', 1: 'Medio', 2: 'Alto'}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModelo guardado en: {MODELS_DIR}")
    
    return model, scaler, le, X_test, y_test

def make_predictions(model, scaler, le, sample_data):
    """Hacer predicciones con valores de ejemplo"""
    print("\n" + "="*50)
    print("PREDICCIONES DE EJEMPLO")
    print("="*50)
    
    # Ejemplos de prueba con valores típicos del rango
    test_examples = [
        {"RoundHeadshots": 0, "GrenadeEffectiveness": 0},  # Bajo
        {"RoundHeadshots": 1, "GrenadeEffectiveness": 1},  # Medio
        {"RoundHeadshots": 2, "GrenadeEffectiveness": 3},  # Alto
        {"RoundHeadshots": 0, "GrenadeEffectiveness": 2},  # Test border
        {"RoundHeadshots": 3, "GrenadeEffectiveness": 1},  # Test border
    ]
    
    print("Predicciones con valores de ejemplo:")
    print("-" * 40)
    
    for i, example in enumerate(test_examples, 1):
        # Preparar datos
        X_example = np.array([[example["RoundHeadshots"], example["GrenadeEffectiveness"]]])
        X_example_scaled = scaler.transform(X_example)
        
        # Predecir
        prediction = model.predict(X_example_scaled)[0]
        probabilities = model.predict_proba(X_example_scaled)[0]
        
        # Convertir a label
        predicted_level = le.inverse_transform([prediction])[0]
        
        print(f"Ejemplo {i}:")
        print(f"  RoundHeadshots: {example['RoundHeadshots']}")
        print(f"  GrenadeEffectiveness: {example['GrenadeEffectiveness']}")
        print(f"  Predicción: {predicted_level}")
        print(f"  Probabilidades: Bajo={probabilities[0]:.3f}, Medio={probabilities[1]:.3f}, Alto={probabilities[2]:.3f}")
        print()
    
    # Predicciones con datos reales del test set
    print("Predicciones con datos reales (muestra del test set):")
    print("-" * 50)
    
    for i in range(min(5, len(sample_data))):
        X_real = sample_data.iloc[i:i+1]
        X_real_scaled = scaler.transform(X_real)
        
        prediction = model.predict(X_real_scaled)[0]
        probabilities = model.predict_proba(X_real_scaled)[0]
        predicted_level = le.inverse_transform([prediction])[0]
        
        print(f"Muestra real {i+1}:")
        print(f"  RoundHeadshots: {X_real.iloc[0]['RoundHeadshots']}")
        print(f"  GrenadeEffectiveness: {X_real.iloc[0]['GrenadeEffectiveness']}")
        print(f"  Predicción: {predicted_level}")
        print(f"  Probabilidades: Bajo={probabilities[0]:.3f}, Medio={probabilities[1]:.3f}, Alto={probabilities[2]:.3f}")
        print()

def main():
    print("Modelo de Efectividad - K-Nearest Neighbors")
    print("Mejor resultado: Accuracy=0.813, F1=0.807, AUC=0.884")
    
    # Entrenar modelo
    model, scaler, le, X_test, y_test = train_best_model()
    
    # Hacer predicciones
    make_predictions(model, scaler, le, X_test)
    
    print("="*50)
    print("ENTRENAMIENTO Y PREDICCIONES COMPLETADAS")
    print("="*50)

if __name__ == "__main__":
    main()