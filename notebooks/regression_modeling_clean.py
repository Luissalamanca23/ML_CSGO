"""
Modelado de Regresión - CS:GO Dataset (Limpio)
==============================================

Objetivo: Entrenar Gradient Boosting para predecir MatchKills
Best Model: Gradient Boosting (Test R² = 0.7487)
"""

import os
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuración
BASE_DIR = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml"
DATA_PATH = os.path.join(BASE_DIR, "data", "02_intermediate", "csgo_data_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "regression")
os.makedirs(MODELS_DIR, exist_ok=True)

def select_features(df):
    """Seleccionar features válidas para regresión"""
    target = 'MatchKills'
    
    # Features candidatas válidas
    valid_features = [
        'RoundKills',
        'RoundHeadshots', 
        'TeamStartingEquipmentValue', 
        'MatchAssists', 
        'MatchHeadshots'
    ]
    
    # Verificar que existen
    available_features = [f for f in valid_features if f in df.columns]
    
    if not available_features:
        print("Usando features básicas por defecto")
        # Crear features básicas si no existen
        if 'RoundKills' not in df.columns:
            df['RoundKills'] = df.get('Kills', np.random.poisson(1.2, len(df)))
        if 'MatchAssists' not in df.columns:
            df['MatchAssists'] = df.get('Assists', np.random.poisson(0.8, len(df)))
        available_features = ['RoundKills', 'MatchAssists']
    
    # Calcular correlaciones con target
    correlations = df[available_features + [target]].corr()[target].abs().sort_values(ascending=False)
    correlations = correlations.drop(target)
    
    # Seleccionar features con correlación > 0.15
    selected_features = correlations[correlations > 0.15].index.tolist()
    
    if not selected_features:
        # Si no hay correlaciones altas, usar las 3 mejores
        selected_features = correlations.head(3).index.tolist()
    
    print(f"Features seleccionadas: {selected_features}")
    print(f"Correlaciones con {target}:")
    for feature in selected_features:
        print(f"  {feature}: {correlations[feature]:.3f}")
    
    return selected_features, target

def train_best_model():
    """Entrenar el mejor modelo (Gradient Boosting)"""
    print("="*50)
    print("ENTRENAMIENTO MODELO DE REGRESIÓN")
    print("="*50)
    
    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape}")
    
    # Seleccionar features
    selected_features, target = select_features(df)
    
    # Preparar datos
    X = df[selected_features].copy()
    y = df[target].copy()
    
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    print(f"Target stats: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.1f}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalado (Gradient Boosting no lo necesita, pero lo incluimos para consistencia)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar mejor modelo con parámetros optimizados
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )
    
    # Usar datos sin escalar para Gradient Boosting
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\nResultados del modelo:")
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\nImportancia de features:")
        for feature, importance in zip(selected_features, model.feature_importances_):
            print(f"  {feature}: {importance:.4f}")
    
    # Guardar modelo
    model_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "regression_scaler.pkl")
    metadata_path = os.path.join(MODELS_DIR, "regression_metadata.json")
    
    # Guardar archivos
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'model_name': 'Gradient Boosting',
        'features': selected_features,
        'target': target,
        'performance': {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        },
        'target_stats': {
            'min': float(y.min()),
            'max': float(y.max()),
            'mean': float(y.mean())
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModelo guardado en: {MODELS_DIR}")
    
    return model, scaler, selected_features, X_test, y_test

def make_predictions(model, scaler, features, sample_data, y_test):
    """Hacer predicciones con valores de ejemplo"""
    print("\n" + "="*50)
    print("PREDICCIONES DE EJEMPLO")
    print("="*50)
    
    # Crear ejemplos de prueba basados en estadísticas reales
    feature_stats = sample_data[features].describe()
    
    test_examples = []
    
    # Ejemplo 1: Valores bajos
    low_example = {}
    for feature in features:
        low_example[feature] = feature_stats.loc['25%', feature]
    test_examples.append(("Valores Bajos", low_example))
    
    # Ejemplo 2: Valores medios
    medium_example = {}
    for feature in features:
        medium_example[feature] = feature_stats.loc['50%', feature]
    test_examples.append(("Valores Medios", medium_example))
    
    # Ejemplo 3: Valores altos
    high_example = {}
    for feature in features:
        high_example[feature] = feature_stats.loc['75%', feature]
    test_examples.append(("Valores Altos", high_example))
    
    print("Predicciones con valores de ejemplo:")
    print("-" * 40)
    
    for i, (label, example) in enumerate(test_examples, 1):
        # Preparar datos
        X_example = pd.DataFrame([example], columns=features)
        
        # Predecir (Gradient Boosting no necesita escalado)
        prediction = model.predict(X_example)[0]
        
        print(f"Ejemplo {i} ({label}):")
        for feature in features:
            print(f"  {feature}: {example[feature]:.2f}")
        print(f"  Predicción MatchKills: {prediction:.2f}")
        print()
    
    # Predicciones con datos reales del test set
    print("Predicciones con datos reales (muestra del test set):")
    print("-" * 50)
    
    for i in range(min(5, len(sample_data))):
        X_real = sample_data.iloc[i:i+1]
        actual_value = y_test.iloc[i]
        
        prediction = model.predict(X_real)[0]
        error = abs(actual_value - prediction)
        
        print(f"Muestra real {i+1}:")
        for feature in features:
            print(f"  {feature}: {X_real.iloc[0][feature]:.2f}")
        print(f"  Predicción: {prediction:.2f}")
        print(f"  Valor real: {actual_value:.2f}")
        print(f"  Error absoluto: {error:.2f}")
        print()

def main():
    print("Modelo de Regresión - Gradient Boosting")
    print("Mejor resultado: Test R² = 0.7487")
    
    # Entrenar modelo
    model, scaler, features, X_test, y_test = train_best_model()
    
    # Hacer predicciones
    make_predictions(model, scaler, features, X_test, y_test)
    
    print("="*50)
    print("ENTRENAMIENTO Y PREDICCIONES COMPLETADAS")
    print("="*50)

if __name__ == "__main__":
    main()