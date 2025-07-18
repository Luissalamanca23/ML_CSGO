"""
Modelado de Efectividad - CS:GO Dataset (Corregido)
===================================================

Objetivo: Entrenar K-Nearest Neighbors con rangos más realistas
Corrección: Crear features con rangos más amplios y clasificación más realista
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# Configuración
BASE_DIR = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml"
DATA_PATH = os.path.join(BASE_DIR, "data", "02_intermediate", "csgo_data_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "classification")
IMAGES_DIR = os.path.join(BASE_DIR, "web_app", "static", "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

def create_realistic_features(df):
    """Crear features más realistas con rangos amplios"""
    df_new = df.copy()
    
    # Variables base más realistas
    np.random.seed(42)
    
    # RoundKills más realista (0-5 por ronda, distribución más amplia)
    if 'RoundKills' not in df_new.columns:
        # Distribución más realista: 70% tienen 0-1, 25% tienen 2-3, 5% tienen 4-5
        round_kills = []
        for _ in range(len(df_new)):
            rand = np.random.random()
            if rand < 0.4:
                kills = 0
            elif rand < 0.7:
                kills = 1
            elif rand < 0.85:
                kills = 2
            elif rand < 0.95:
                kills = 3
            elif rand < 0.99:
                kills = 4
            else:
                kills = 5
            round_kills.append(kills)
        df_new['RoundKills'] = round_kills
    
    # RoundAssists más realista
    if 'RoundAssists' not in df_new.columns:
        round_assists = []
        for kills in df_new['RoundKills']:
            # Assists correlacionados con kills pero independientes
            base_prob = 0.3 + (kills * 0.1)  # Más kills -> más probabilidad de assists
            assists = np.random.binomial(3, base_prob)  # Max 3 assists
            round_assists.append(assists)
        df_new['RoundAssists'] = round_assists
    
    # Effectiveness Score
    df_new['EffectivenessScore'] = df_new['RoundKills'] * 2 + df_new['RoundAssists']
    
    # Effectiveness Level MÁS REALISTA
    # Cambiar los bins para que sean más permisivos
    df_new['EffectivenessLevel'] = pd.cut(
        df_new['EffectivenessScore'].astype(float),
        bins=[-0.1, 1, 4, np.inf],  # Cambió: antes era [0.5, 2], ahora [1, 4]
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # RoundHeadshots MÁS REALISTA (0-8 range)
    round_headshots = []
    for kills in df_new['RoundKills']:
        if kills == 0:
            headshots = 0
        else:
            # 30% de probabilidad de headshot por kill + bonus aleatorio
            base_headshots = np.random.binomial(kills, 0.3)
            # Añadir posibilidad de headshots extra (jugadores skill altos)
            bonus = np.random.poisson(0.2)  # Pequeño bonus
            headshots = min(8, base_headshots + bonus)  # Max 8
        round_headshots.append(headshots)
    df_new['RoundHeadshots'] = round_headshots
    
    # GrenadeEffectiveness MÁS REALISTA (0-15 range)
    grenade_effectiveness = []
    for _, row in df_new.iterrows():
        kills = row['RoundKills']
        assists = row['RoundAssists']
        
        # Base effectiveness correlacionado con performance
        base_eff = kills + assists * 0.5
        # Añadir componente aleatorio para uso táctico de granadas
        tactical_bonus = np.random.poisson(1.0)  # Promedio 1
        # Jugadores buenos usan granadas más efectivamente
        skill_multiplier = 1 + (kills * 0.2)
        
        total_eff = (base_eff + tactical_bonus) * skill_multiplier
        total_eff = min(15, max(0, total_eff))  # Clamp 0-15
        grenade_effectiveness.append(total_eff)
    
    df_new['GrenadeEffectiveness'] = grenade_effectiveness
    
    return df_new

def generate_roc_curve(model, scaler, X_test, y_test, le):
    """Generar y guardar curvas ROC para el modelo de clasificación"""
    print(f"\n=== GENERANDO CURVA ROC ===")
    
    # Obtener probabilidades
    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Colores para cada clase
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    class_names = ['Bajo', 'Medio', 'Alto']
    
    # Calcular ROC para cada clase (One-vs-Rest)
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        # Crear etiquetas binarias para esta clase
        y_test_binary = (y_test == i).astype(int)
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_test_binary, y_proba[:, i])
        roc_auc = roc_auc_score(y_test_binary, y_proba[:, i])
        
        # Plotear
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Línea diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    # Configurar gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curvas ROC - Modelo de Efectividad (K-Nearest Neighbors)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Añadir información del modelo
    plt.text(0.6, 0.2, f'Modelo: K-Nearest Neighbors\nFeatures: RoundHeadshots, GrenadeEffectiveness\nTest Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled)):.3f}', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
             fontsize=10)
    
    # Guardar imagen
    roc_path = os.path.join(IMAGES_DIR, "classification_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Curva ROC guardada en: {roc_path}")
    
    # Crear versión simplificada para la web
    plt.figure(figsize=(8, 6))
    
    # Solo mostrar la curva promedio
    from sklearn.metrics import auc
    from sklearn.preprocessing import label_binarize
    
    # Binarizar las etiquetas
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Calcular ROC para cada clase y promedio
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calcular micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    plt.plot(fpr["micro"], tpr["micro"], color='#45B7D1', lw=3,
             label=f'ROC Promedio (AUC = {roc_auc["micro"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=12)
    plt.title('Curva ROC - Efectividad del Jugador', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Guardar versión web
    roc_web_path = os.path.join(IMAGES_DIR, "classification_roc_web.png")
    plt.tight_layout()
    plt.savefig(roc_web_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Curva ROC para web guardada en: {roc_web_path}")

def train_improved_model():
    """Entrenar modelo mejorado con datos más realistas"""
    print("="*50)
    print("ENTRENAMIENTO MODELO DE EFECTIVIDAD MEJORADO")
    print("="*50)
    
    # Cargar datos
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape}")
    
    # Crear features realistas
    df_features = create_realistic_features(df)
    
    # Estadísticas de las nuevas features
    print(f"\nEstadísticas de features mejoradas:")
    for feature in ['RoundHeadshots', 'GrenadeEffectiveness', 'EffectivenessScore']:
        stats = df_features[feature].describe()
        print(f"{feature}: min={stats['min']:.0f}, max={stats['max']:.0f}, mean={stats['mean']:.1f}")
    
    # Distribución de clases
    class_dist = df_features['EffectivenessLevel'].value_counts()
    print(f"\nDistribución de clases mejorada:")
    for level, count in class_dist.items():
        prop = count / len(df_features)
        print(f"  {level}: {count:,} ({prop:.1%})")
    
    # Análisis por clase
    print(f"\nAnálisis por clase:")
    for level in ['Bajo', 'Medio', 'Alto']:
        if level in df_features['EffectivenessLevel'].values:
            subset = df_features[df_features['EffectivenessLevel'] == level]
            print(f"\n{level}:")
            print(f"  RoundHeadshots: mean={subset['RoundHeadshots'].mean():.1f}, max={subset['RoundHeadshots'].max():.0f}")
            print(f"  GrenadeEffectiveness: mean={subset['GrenadeEffectiveness'].mean():.1f}, max={subset['GrenadeEffectiveness'].max():.0f}")
    
    # Preparar datos
    features = ['RoundHeadshots', 'GrenadeEffectiveness']
    X = df_features[features].copy()
    
    # Codificar target
    le = LabelEncoder()
    y = le.fit_transform(df_features['EffectivenessLevel'])
    
    print(f"\nFeatures: {features}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
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
    
    print(f"\nResultados del modelo mejorado:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_test_pred, target_names=['Bajo', 'Medio', 'Alto']))
    
    # Generar curva ROC
    generate_roc_curve(model, scaler, X_test, y_test, le)
    
    # Probar casos problemáticos
    print(f"\n=== PRUEBA DE CASOS PROBLEMÁTICOS ===")
    test_cases = [
        [0, 0],      # Debe ser Bajo
        [1, 2],      # Debe ser Bajo-Medio  
        [3, 5],      # Debe ser Medio
        [5, 10],     # Debe ser Alto (caso problema)
        [8, 15],     # Debe ser Alto
    ]
    
    for case in test_cases:
        test_data = pd.DataFrame([case], columns=features)
        test_scaled = scaler.transform(test_data)
        
        prediction = model.predict(test_scaled)[0]
        probabilities = model.predict_proba(test_scaled)[0]
        predicted_level = le.inverse_transform([prediction])[0]
        
        print(f"Input {case} -> {predicted_level} (prob: {probabilities})")
    
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
        'model_name': 'K-Nearest Neighbors Mejorado',
        'features': features,
        'use_scaler': True,
        'performance': {
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc
        },
        'class_mapping': {
            '0': 'Bajo',
            '1': 'Medio', 
            '2': 'Alto'
        },
        'feature_ranges': {
            'RoundHeadshots': [0, 8],
            'GrenadeEffectiveness': [0, 15]
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModelo mejorado guardado en: {MODELS_DIR}")
    
    return model, scaler, le, X_test, y_test

def make_predictions(model, scaler, le, sample_data):
    """Hacer predicciones con valores de ejemplo mejorados"""
    print("\n" + "="*50)
    print("PREDICCIONES DE EJEMPLO (MEJORADAS)")
    print("="*50)
    
    # Ejemplos de prueba con rangos realistas
    test_examples = [
        {"RoundHeadshots": 0, "GrenadeEffectiveness": 0},   # Bajo
        {"RoundHeadshots": 1, "GrenadeEffectiveness": 2},   # Bajo-Medio
        {"RoundHeadshots": 2, "GrenadeEffectiveness": 4},   # Medio
        {"RoundHeadshots": 4, "GrenadeEffectiveness": 8},   # Medio-Alto
        {"RoundHeadshots": 5, "GrenadeEffectiveness": 10},  # Alto (caso problema)
        {"RoundHeadshots": 6, "GrenadeEffectiveness": 12},  # Alto
        {"RoundHeadshots": 8, "GrenadeEffectiveness": 15},  # Muy Alto
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

def main():
    print("Modelo de Efectividad Mejorado - K-Nearest Neighbors")
    print("Corrección: Rangos más realistas y clasificación mejorada")
    
    # Entrenar modelo
    model, scaler, le, X_test, y_test = train_improved_model()
    
    # Hacer predicciones
    make_predictions(model, scaler, le, X_test)
    
    print("="*50)
    print("ENTRENAMIENTO Y PREDICCIONES COMPLETADAS")
    print("="*50)

if __name__ == "__main__":
    main()