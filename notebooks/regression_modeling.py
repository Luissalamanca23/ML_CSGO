"""
Modelado de Regresión - CS:GO Dataset
=====================================

Objetivo: Predecir MatchKills (número de kills en partida)
Target: MatchKills 
Features válidas: Equipamiento del equipo, tipo de armas, granadas, contexto del mapa/ronda

Este script ejecuta todo el pipeline de regresión de manera automática,
guarda las imágenes generadas y los mejores modelos.
"""

import os
import time
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Librerías principales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn para ML
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_validate, learning_curve
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from scipy import stats
import signal
from contextlib import contextmanager

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (12, 8)

# Configuración de directorios
BASE_DIR = r"C:\Users\LuisSalamanca\Desktop\Duoc\Machine\csgo-ml"
DATA_PATH = os.path.join(BASE_DIR, "data", "02_intermediate", "csgo_data_clean.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "notebooks", "regression_images")
MODELS_DIR = os.path.join(BASE_DIR, "models", "regression")

# Crear directorios si no existen
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def save_plot(fig, name, img_counter):
    """Guardar plot con número consecutivo"""
    filename = f"{img_counter:02d}_{name}.png"
    filepath = os.path.join(IMAGES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Imagen guardada: {filename}")
    return img_counter + 1

@contextmanager
def timeout(duration):
    """Context manager para timeout de operaciones"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operación excedió {duration} segundos")
    
    # Configurar handler de timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

def main():
    print("="*60)
    print("MODELADO DE REGRESIÓN - CS:GO DATASET")
    print("="*60)
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
    
    img_counter = 1
    
    # ============================================================================
    # 1. CARGA Y EXPLORACIÓN DE DATOS
    # ============================================================================
    print("\n" + "="*50)
    print("1. CARGA Y EXPLORACIÓN DE DATOS")
    print("="*50)
    
    # Cargar dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {df.shape}")
    print(f"- Filas: {df.shape[0]:,}")
    print(f"- Columnas: {df.shape[1]}")
    print(f"\nTipos de datos:")
    print(df.dtypes.value_counts())
    
    # ============================================================================
    # 2. ANÁLISIS DE CORRELACIÓN Y SELECCIÓN DE FEATURES
    # ============================================================================
    print("\n" + "="*50)
    print("2. ANÁLISIS DE CORRELACIÓN Y SELECCIÓN DE FEATURES")
    print("="*50)
    
    # Definir target y features válidas
    target = 'MatchKills'
    valid_features = [
        'RoundKills',
        'RoundHeadshots', 
        'TeamStartingEquipmentValue', 
        'MatchAssists', 
        'MatchHeadshots'
    ]
    
    # Verificar que las features existen
    candidate_features = [f for f in valid_features if f in df.columns]
    print(f"Features candidatas disponibles: {len(candidate_features)}")
    
    # Calcular correlación con el target
    correlations = df[candidate_features + [target]].corr()[target].abs().sort_values(ascending=False)
    correlations = correlations.drop(target)  # Remover autocorrelación
    
    print(f"\nTARGET: {target}")
    print(f"- Valores únicos: {df[target].nunique()}")
    print(f"- Rango: {df[target].min():.0f} - {df[target].max():.0f}")
    print(f"- Media: {df[target].mean():.0f}")
    
    print(f"\nCORRELACIONES CON {target}:")
    for i, (feature, corr) in enumerate(correlations.items()):
        print(f"{i+1:2d}. {feature:35s}: {corr:.3f}")
    
    # Seleccionar features con correlación > 0.15
    high_corr_features = correlations[correlations > 0.15].index.tolist()
    print(f"\nFeatures con correlación > 0.15: {len(high_corr_features)}")
    
    # Análisis de multicolinealidad
    if len(high_corr_features) > 1:
        feature_corr_matrix = df[high_corr_features].corr()
        
        # Visualizar matriz de correlación
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(feature_corr_matrix, dtype=bool))
        sns.heatmap(feature_corr_matrix, 
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0, 
                    mask=mask,
                    square=True, 
                    fmt='.2f',
                    cbar_kws={'shrink': 0.8},
                    ax=ax)
        ax.set_title('Matriz de Correlación - Features Válidas', fontsize=16, fontweight='bold')
        plt.tight_layout()
        img_counter = save_plot(fig, "matriz_correlacion_features", img_counter)
        plt.close()
    
    # Features finales seleccionadas
    selected_features = high_corr_features.copy()
    print(f"\nFEATURES SELECCIONADAS PARA MODELADO: {len(selected_features)}")
    for i, feature in enumerate(selected_features):
        corr_with_target = correlations[feature]
        print(f"{i+1:2d}. {feature:35s}: r = {corr_with_target:.3f}")
    
    # Verificar valores nulos
    null_counts = df[selected_features + [target]].isnull().sum()
    if null_counts.sum() > 0:
        print(f"\nValores nulos encontrados:")
        print(null_counts[null_counts > 0])
    else:
        print(f"\nNo hay valores nulos en las features seleccionadas ✓")
    
    print(f"\nMax correlación alcanzada: {correlations.max():.3f}")
    if correlations.max() > 0.7:
        print("✅ EXCELENTE para regresión - correlaciones altas")
    elif correlations.max() > 0.5:
        print("✅ BUENO para regresión - correlaciones moderadas")
    else:
        print("⚠️ Correlaciones bajas pero aceptables")
    
    # ============================================================================
    # 3. PREPARACIÓN DE DATOS
    # ============================================================================
    print("\n" + "="*50)
    print("3. PREPARACIÓN DE DATOS")
    print("="*50)
    
    # Preparar datos para regresión
    X = df[selected_features].copy()
    y = df[target].copy()
    
    print(f"Shape de X: {X.shape}")
    print(f"Shape de y: {y.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nDivisión train/test:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Escalado de features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir de vuelta a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
    
    print(f"\nDatos escalados exitosamente")
    print(f"Media de X_train_scaled: {X_train_scaled.mean().mean():.6f}")
    print(f"Std de X_train_scaled: {X_train_scaled.std().mean():.6f}")
    
    # Visualizar distribución del target
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribución en train
    y_train.hist(bins=30, ax=axes[0], alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribución Target - Train Set', fontweight='bold')
    axes[0].set_xlabel(f'{target}')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True, alpha=0.3)
    
    # Distribución en test
    y_test.hist(bins=30, ax=axes[1], alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Distribución Target - Test Set', fontweight='bold')
    axes[1].set_xlabel(f'{target}')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_counter = save_plot(fig, "distribucion_target", img_counter)
    plt.close()
    
    # ============================================================================
    # 4. DEFINICIÓN DE MODELOS Y CONFIGURACIÓN
    # ============================================================================
    print("\n" + "="*50)
    print("4. DEFINICIÓN DE MODELOS Y CONFIGURACIÓN")
    print("="*50)
    
    # Definir modelos optimizados
    models = {
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=3),
            'scaled': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'scaled': False
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=3, eval_metric='rmse'),
            'scaled': False
        },
        'Ridge Regression': {
            'model': Ridge(random_state=42),
            'scaled': True
        },
        'Lasso': {
            'model': Lasso(random_state=42, max_iter=2000),
            'scaled': True
        },
        'Linear Regression': {
            'model': LinearRegression(),
            'scaled': True
        },
        'Decision Tree': {
            'model': DecisionTreeRegressor(random_state=42),
            'scaled': False
        },
        'KNN': {
            'model': KNeighborsRegressor(n_jobs=3),
            'scaled': True
        },
        'SVR': {
            'model': SVR(),
            'scaled': True,
            'timeout': 180  # 3 minutos máximo para SVR
        }
    }
    
    print(f"MODELOS CONFIGURADOS: {len(models)}")
    for name, config in models.items():
        print(f"- {name}: Escalado = {config['scaled']}")
    
    # Configuración de hiperparámetros para GridSearchCV
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'subsample': [0.8, 1.0]
        },
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'cholesky']
        },
        'Lasso': {
            'alpha': [0.01, 0.1, 1.0],
            'selection': ['cyclic']
        },
        'Linear Regression': {
            'fit_intercept': [True, False]
        },
        'Decision Tree': {
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 10],
            'min_samples_leaf': [1, 5],
            'criterion': ['squared_error']
        },
        'KNN': {
            'n_neighbors': [5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf']
        }
    }
    
    print(f"\nConfiguración de GridSearchCV completada")
    print(f"Modelos con hiperparámetros definidos: {len(param_grids)}")
    
    # ============================================================================
    # 5. ENTRENAMIENTO CON GRIDSEARCHCV
    # ============================================================================
    print("\n" + "="*50)
    print("5. ENTRENAMIENTO CON GRIDSEARCHCV")
    print("="*50)
    
    results = {}
    best_models = {}
    best_params = {}
    training_times = {}
    
    # Configurar GridSearchCV
    cv_folds = 3
    scoring_metric = 'r2'
    n_jobs = -1
    
    for name, config in models.items():
        print(f"\nEntrenando {name}...")
        
        # Determinar si usar datos escalados
        if config['scaled']:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled
            print(f"   Usando datos escalados")
        else:
            X_train_used = X_train
            X_test_used = X_test
            print(f"   Usando datos sin escalar")
        
        base_model = config['model']
        param_grid = param_grids.get(name, {})
        model_timeout = config.get('timeout', 300)  # Default 5 min
        
        if not param_grid:
            # Entrenamiento simple sin GridSearch
            try:
                if name == 'SVR':
                    print(f"   Entrenando con timeout de {model_timeout}s...")
                    with timeout(model_timeout):
                        best_model = base_model
                        best_model.fit(X_train_used, y_train)
                else:
                    best_model = base_model
                    best_model.fit(X_train_used, y_train)
                
                best_params[name] = "Parámetros por defecto"
                grid_search_time = 0
                print(f"   ✓ Entrenamiento completado")
                
            except TimeoutError:
                print(f"   ⚠️ TIMEOUT: {name} excedió {model_timeout}s - SALTANDO")
                continue
            except Exception as e:
                print(f"   ❌ Error en {name}: {str(e)}")
                continue
        else:
            # GridSearchCV con manejo de timeout para SVR
            try:
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring=scoring_metric,
                    n_jobs=n_jobs,
                    verbose=0
                )
                
                start_time = time.time()
                
                if name == 'SVR':
                    print(f"   GridSearch con timeout de {model_timeout}s...")
                    with timeout(model_timeout):
                        grid_search.fit(X_train_used, y_train)
                else:
                    grid_search.fit(X_train_used, y_train)
                
                grid_search_time = time.time() - start_time
                
                best_model = grid_search.best_estimator_
                best_params[name] = grid_search.best_params_
                
                print(f"   ✓ GridSearch completado en {grid_search_time:.1f}s")
                print(f"   Mejor CV Score: {grid_search.best_score_:.4f}")
                
            except TimeoutError:
                print(f"   ⚠️ TIMEOUT: {name} excedió {model_timeout}s - SALTANDO")
                continue
            except Exception as e:
                print(f"   ❌ Error en GridSearch para {name}: {str(e)}")
                continue
        
        training_times[name] = grid_search_time
        
        # Predicciones y métricas
        y_train_pred = best_model.predict(X_train_used)
        y_test_pred = best_model.predict(X_test_used)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        cv_scores = cross_val_score(best_model, X_train_used, y_train, 
                                    cv=cv_folds, scoring=scoring_metric)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_test_pred': y_test_pred,
            'training_time': grid_search_time,
            'best_params': best_params[name]
        }
        
        best_models[name] = best_model
        
        print(f"   Test R²: {test_r2:.4f}")
        print(f"   CV R²: {cv_mean:.4f} ± {cv_std:.4f}")
        
        if test_r2 > 0.8:
            print(f"   🎯 ¡OBJETIVO CUMPLIDO! R² = {test_r2:.4f} > 0.8")
        else:
            print(f"   📈 R² = {test_r2:.4f} < 0.8")
    
    print(f"\nEntrenamiento completado")
    total_time = sum(training_times.values())
    print(f"Tiempo total: {total_time:.1f}s")
    
    # Visualización del progreso de entrenamiento
    if training_times:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico de tiempos de entrenamiento
        trained_models = list(training_times.keys())
        times = list(training_times.values())
        
        bars = axes[0].bar(range(len(trained_models)), times, alpha=0.7, color='skyblue')
        axes[0].set_xlabel('Modelos')
        axes[0].set_ylabel('Tiempo (segundos)')
        axes[0].set_title('Tiempos de Entrenamiento por Modelo', fontweight='bold')
        axes[0].set_xticks(range(len(trained_models)))
        axes[0].set_xticklabels(trained_models, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir valores a las barras
        for bar, time_val in zip(bars, times):
            if time_val > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                            f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Gráfico de progreso acumulativo
        cumulative_times = np.cumsum([0] + times)
        axes[1].plot(range(len(cumulative_times)), cumulative_times, 'o-', linewidth=2, markersize=8)
        axes[1].fill_between(range(len(cumulative_times)), cumulative_times, alpha=0.3)
        axes[1].set_xlabel('Modelos Completados')
        axes[1].set_ylabel('Tiempo Acumulado (segundos)')
        axes[1].set_title('Progreso Acumulativo de Entrenamiento', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Añadir etiquetas de modelos
        model_labels = ['Inicio'] + trained_models
        axes[1].set_xticks(range(len(model_labels)))
        axes[1].set_xticklabels(model_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        img_counter = save_plot(fig, "progreso_entrenamiento", img_counter)
        plt.close()
    
    # ============================================================================
    # 6. COMPARACIÓN Y VISUALIZACIÓN DE RESULTADOS
    # ============================================================================
    print("\n" + "="*50)
    print("6. COMPARACIÓN Y VISUALIZACIÓN DE RESULTADOS")
    print("="*50)
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV_R2': [results[model]['cv_mean'] for model in results.keys()],
        'CV_Std': [results[model]['cv_std'] for model in results.keys()],
        'Train_R2': [results[model]['train_r2'] for model in results.keys()],
        'Test_R2': [results[model]['test_r2'] for model in results.keys()],
        'Test_RMSE': [results[model]['test_rmse'] for model in results.keys()],
        'Test_MAE': [results[model]['test_mae'] for model in results.keys()],
        'Training_Time': [results[model]['training_time'] for model in results.keys()]
    })
    
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    print("COMPARACIÓN DE RESULTADOS:")
    print(results_df.round(4).to_string(index=False))
    
    # Identificar mejor modelo
    best_model_name = results_df.iloc[0]['Model']
    best_test_r2 = results_df.iloc[0]['Test_R2']
    
    print(f"\nMEJOR MODELO: {best_model_name}")
    print(f"Test R²: {best_test_r2:.4f}")
    
    # Visualización principal
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Comparación R² scores
    x_pos = np.arange(len(results_df))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, results_df['Train_R2'], width, 
               label='Train R²', alpha=0.8, color='skyblue')
    axes[0].bar(x_pos + width/2, results_df['Test_R2'], width, 
               label='Test R²', alpha=0.8, color='orange')
    
    axes[0].axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Objetivo R² = 0.8')
    axes[0].set_xlabel('Modelos')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Comparación R² Score: Train vs Test')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test R² ranking
    colors = ['green' if x > 0.8 else 'orange' if x > 0.7 else 'red' for x in results_df['Test_R2']]
    axes[1].barh(range(len(results_df)), results_df['Test_R2'], color=colors, alpha=0.7)
    
    axes[1].axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Objetivo R² = 0.8')
    axes[1].set_yticks(range(len(results_df)))
    axes[1].set_yticklabels(results_df['Model'])
    axes[1].set_xlabel('Test R² Score')
    axes[1].set_title('Ranking de Modelos por Test R²')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Añadir valores a las barras del ranking
    for i, v in enumerate(results_df['Test_R2']):
        axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    img_counter = save_plot(fig, "comparacion_modelos", img_counter)
    plt.close()
    
    # ============================================================================
    # 7. ANÁLISIS DEL MEJOR MODELO
    # ============================================================================
    print("\n" + "="*50)
    print("7. ANÁLISIS DEL MEJOR MODELO")
    print("="*50)
    
    best_model = best_models[best_model_name]
    best_results = results[best_model_name]
    
    print(f"ANÁLISIS DETALLADO: {best_model_name}")
    print("=" * 50)
    
    print(f"Mejores hiperparámetros:")
    if isinstance(best_results['best_params'], dict):
        for param, value in best_results['best_params'].items():
            print(f"  {param}: {value}")
    else:
        print(f"  {best_results['best_params']}")
    
    print(f"\nMétricas de rendimiento:")
    print(f"  Train R²: {best_results['train_r2']:.4f}")
    print(f"  Test R²: {best_results['test_r2']:.4f}")
    print(f"  CV R²: {best_results['cv_mean']:.4f} ± {best_results['cv_std']:.4f}")
    print(f"  Test RMSE: {best_results['test_rmse']:.4f}")
    print(f"  Test MAE: {best_results['test_mae']:.4f}")
    
    # Feature importance si está disponible
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nImportancia de features:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")
        
        # Visualizar feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis', ax=ax)
        ax.set_title(f'Feature Importances - {best_model_name}')
        ax.set_xlabel('Importance')
        plt.tight_layout()
        img_counter = save_plot(fig, "feature_importance", img_counter)
        plt.close()
    
    elif hasattr(best_model, 'coef_'):
        feature_coef = pd.DataFrame({
            'feature': selected_features,
            'coefficient': abs(best_model.coef_)
        }).sort_values('coefficient', ascending=False)
        
        print(f"\nCoeficientes (valor absoluto):")
        for idx, row in feature_coef.iterrows():
            print(f"  {row['feature']:30s}: {row['coefficient']:.4f}")
    
    # ============================================================================
    # 8. VISUALIZACIÓN DE PREDICCIONES Y RESIDUOS
    # ============================================================================
    print("\n" + "="*50)
    print("8. VISUALIZACIÓN DE PREDICCIONES Y RESIDUOS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: Predicciones vs Valores Reales
    axes[0,0].scatter(y_test, best_results['y_test_pred'], alpha=0.6, color='blue')
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Valores Reales')
    axes[0,0].set_ylabel('Predicciones')
    axes[0,0].set_title(f'Predicciones vs Reales - {best_model_name}')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].text(0.05, 0.95, f'R² = {best_results["test_r2"]:.3f}', 
                   transform=axes[0,0].transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residuos vs Predicciones
    residuals = y_test - best_results['y_test_pred']
    axes[0,1].scatter(best_results['y_test_pred'], residuals, alpha=0.6, color='green')
    axes[0,1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0,1].set_xlabel('Predicciones')
    axes[0,1].set_ylabel('Residuos')
    axes[0,1].set_title('Residuos vs Predicciones')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Distribución de residuos
    axes[1,0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].axvline(residuals.mean(), color='red', linestyle='--', 
                      label=f'Media: {residuals.mean():.3f}')
    axes[1,0].set_xlabel('Residuos')
    axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].set_title('Distribución de Residuos')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot para normalidad de residuos
    stats.probplot(residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot - Normalidad de Residuos')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_counter = save_plot(fig, "analisis_residuos", img_counter)
    plt.close()
    
    print(f"\nESTADÍSTICAS DE RESIDUOS:")
    print(f"Media: {residuals.mean():.4f}")
    print(f"Desviación estándar: {residuals.std():.4f}")
    print(f"Mínimo: {residuals.min():.4f}")
    print(f"Máximo: {residuals.max():.4f}")
    
    # ============================================================================
    # 9. VALIDACIÓN CRUZADA DETALLADA
    # ============================================================================
    print("\n" + "="*50)
    print("9. VALIDACIÓN CRUZADA DETALLADA")
    print("="*50)
    
    # Seleccionar datos apropiados según el modelo
    if best_model_name in ['Ridge Regression', 'Lasso', 'Linear Regression', 'KNN']:
        X_cv = X_train_scaled
    else:
        X_cv = X_train
    
    cv_results = cross_validate(
        best_model, X_cv, y_train, 
        cv=5, 
        scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], 
        return_train_score=True
    )
    
    # Procesar resultados
    metrics_summary = {
        'R²': {
            'train': cv_results['train_r2'],
            'test': cv_results['test_r2']
        },
        'RMSE': {
            'train': np.sqrt(-cv_results['train_neg_mean_squared_error']),
            'test': np.sqrt(-cv_results['test_neg_mean_squared_error'])
        },
        'MAE': {
            'train': -cv_results['train_neg_mean_absolute_error'],
            'test': -cv_results['test_neg_mean_absolute_error']
        }
    }
    
    # Mostrar estadísticas
    for metric_name, metric_data in metrics_summary.items():
        train_scores = metric_data['train']
        test_scores = metric_data['test']
        
        print(f"\n{metric_name}:")
        print(f"  Train: {train_scores.mean():.4f} ± {train_scores.std():.4f}")
        print(f"  Test:  {test_scores.mean():.4f} ± {test_scores.std():.4f}")
    
    # Visualización de validación cruzada
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (metric_name, metric_data) in enumerate(metrics_summary.items()):
        train_scores = metric_data['train']
        test_scores = metric_data['test']
        
        x_pos = [1, 2]
        means = [train_scores.mean(), test_scores.mean()]
        stds = [train_scores.std(), test_scores.std()]
        
        axes[idx].bar(x_pos, means, yerr=stds, capsize=5, 
                      color=['skyblue', 'orange'], alpha=0.7)
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(['Train', 'Test'])
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'{metric_name} - Validación Cruzada')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    img_counter = save_plot(fig, "validacion_cruzada", img_counter)
    plt.close()
    
    # ============================================================================
    # 10. CURVAS DE APRENDIZAJE
    # ============================================================================
    print("\n" + "="*50)
    print("10. CURVAS DE APRENDIZAJE")
    print("="*50)
    
    # Seleccionar datos apropiados
    if best_model_name in ['Ridge Regression', 'Lasso', 'Linear Regression', 'KNN']:
        X_learning = X_train_scaled
    else:
        X_learning = X_train
    
    # Generar curva de aprendizaje
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_learning, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring='r2', n_jobs=-1
    )
    
    # Calcular medias y desviaciones
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Visualizar curva de aprendizaje
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Objetivo R² = 0.8')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R² Score')
    ax.set_title(f'Curva de Aprendizaje - {best_model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    img_counter = save_plot(fig, "curva_aprendizaje", img_counter)
    plt.close()
    
    print(f"Rendimiento final:")
    print(f"Training Score: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"Validation Score: {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"Gap Train-Val: {abs(train_mean[-1] - val_mean[-1]):.4f}")
    
    # ============================================================================
    # 11. GUARDAR MODELOS Y METADATOS
    # ============================================================================
    print("\n" + "="*50)
    print("11. GUARDAR MODELOS Y METADATOS")
    print("="*50)
    
    # Guardar el mejor modelo
    best_model_for_save = best_models[best_model_name]
    best_params_for_save = results[best_model_name]['best_params']
    
    # Determinar si usar datos escalados
    scale_sensitive = ['Ridge Regression', 'Lasso', 'Linear Regression', 'KNN']
    use_scaler = best_model_name in scale_sensitive
    
    # Preparar metadatos del modelo
    model_metadata = {
        'model_name': best_model_name,
        'model_type': 'regression',
        'target': target,
        'features': selected_features,
        'use_scaler': use_scaler,
        'best_params': best_params_for_save,
        'performance_metrics': {
            'test_r2': results[best_model_name]['test_r2'],
            'test_rmse': results[best_model_name]['test_rmse'],
            'test_mae': results[best_model_name]['test_mae'],
            'cv_r2_mean': results[best_model_name]['cv_mean'],
            'cv_r2_std': results[best_model_name]['cv_std']
        },
        'target_info': {
            'min_value': float(y.min()),
            'max_value': float(y.max()),
            'mean_value': float(y.mean()),
            'std_value': float(y.std())
        },
        'feature_requirements': {
            feature: 'float64' for feature in selected_features
        }
    }
    
    # Guardar modelo principal
    model_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model_for_save, f)
    
    # Guardar scaler si se usa
    if use_scaler:
        scaler_path = os.path.join(MODELS_DIR, "regression_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler guardado: regression_scaler.pkl")
    
    # Guardar metadatos
    metadata_path = os.path.join(MODELS_DIR, "regression_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Guardar DataFrame de resultados
    results_path = os.path.join(MODELS_DIR, "regression_results.csv")
    results_df.to_csv(results_path, index=False)
    
    print(f"Modelo guardado: best_regression_model.pkl")
    print(f"Metadatos guardados: regression_metadata.json")
    print(f"Resultados guardados: regression_results.csv")
    
    # Verificar carga del modelo
    print(f"\nVerificando modelo guardado...")
    try:
        # Cargar modelo
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Cargar metadatos
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        
        # Cargar scaler si existe
        if use_scaler:
            with open(scaler_path, 'rb') as f:
                loaded_scaler = pickle.load(f)
        
        print(f"✓ Todos los archivos se cargaron correctamente")
        
        # Probar predicción
        test_sample = X_test.iloc[:1]
        if use_scaler:
            test_sample_scaled = loaded_scaler.transform(test_sample)
            test_pred = loaded_model.predict(test_sample_scaled)
        else:
            test_pred = loaded_model.predict(test_sample)
        
        actual_value = y_test.iloc[0]
        
        print(f"Predicción de prueba:")
        print(f"  - Valor real: {actual_value:.2f}")
        print(f"  - Predicción: {test_pred[0]:.2f}")
        print(f"  - Error absoluto: {abs(actual_value - test_pred[0]):.2f}")
        
    except Exception as e:
        print(f"❌ Error al verificar modelo: {str(e)}")
    
    # ============================================================================
    # 12. RESUMEN FINAL
    # ============================================================================
    print("\n" + "="*60)
    print("12. RESUMEN FINAL")
    print("="*60)
    
    print(f"Target Variable: {target}")
    print(f"Features utilizadas: {len(selected_features)}")
    print(f"Tamaño del dataset: {df.shape[0]:,} registros")
    print(f"Split train/test: {len(X_train)}/{len(X_test)} (80/20)")
    
    print(f"\nMODELOS EVALUADOS:")
    for idx, row in results_df.iterrows():
        print(f"{idx+1}. {row['Model']:25s}: Test R² = {row['Test_R2']:.4f}")
    
    print(f"\nMEJOR MODELO: {best_model_name}")
    print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
    print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.4f}")
    
    # Verificar objetivo
    objective_met = results_df.iloc[0]['Test_R2'] > 0.8
    print(f"\nOBJETIVO R² > 0.8: {'✅ CUMPLIDO' if objective_met else '❌ NO CUMPLIDO'}")
    
    if not objective_met:
        print(f"\nRECOMENDACIONES PARA MEJORAR:")
        print(f"- Incluir más features relevantes")
        print(f"- Probar feature engineering adicional")
        print(f"- Considerar ensemble methods")
        print(f"- Revisar calidad de datos y outliers")
    
    # Análisis de overfitting
    train_test_gap = results_df.iloc[0]['Train_R2'] - results_df.iloc[0]['Test_R2']
    print(f"\nANÁLISIS DE OVERFITTING:")
    print(f"Gap Train-Test: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print(f"⚠️ ADVERTENCIA: Posible overfitting detectado")
    elif train_test_gap < 0.05:
        print(f"✅ EXCELENTE: Buen balance entre bias y variance")
    else:
        print(f"✅ BUENO: Gap aceptable")
    
    print(f"\nARCHIVOS GENERADOS:")
    print(f"📂 Imágenes: {IMAGES_DIR}")
    print(f"📂 Modelos: {MODELS_DIR}")
    print(f"📊 Total imágenes generadas: {img_counter - 1}")
    
    print(f"\n" + "="*60)
    print(f"MODELADO DE REGRESIÓN COMPLETADO EXITOSAMENTE")
    print(f"="*60)
    
    return {
        'best_model_name': best_model_name,
        'best_r2': best_test_r2,
        'objective_met': objective_met,
        'results_df': results_df,
        'images_generated': img_counter - 1
    }

if __name__ == "__main__":
    results = main()