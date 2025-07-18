# CS:GO ML Predictor Web Application

Una aplicación web elegante para realizar predicciones de Machine Learning sobre datos de CS:GO, incluyendo clasificación de efectividad de jugadores y regresión de kills en partidas.

## Características

### 🎯 Predicción de Efectividad (Clasificación)
- Predice el nivel de efectividad del jugador: **Bajo**, **Medio**, **Alto**
- Basado en headshots por ronda y efectividad con granadas
- Modelo: **Gradient Boosting Classifier** con ~82% de accuracy
- Muestra probabilidades por clase y nivel de confianza

### 📈 Predicción de Kills (Regresión)
- Predice el número esperado de kills en una partida
- Basado en kills por ronda, headshots, equipamiento del equipo y asistencias
- Modelo: **Gradient Boosting Regressor** con ~75% R² score
- Incluye análisis comparativo y percentiles

### 🎨 Interfaz Moderna
- Diseño responsive con Bootstrap 5
- Animaciones y transiciones suaves
- Validación en tiempo real de formularios
- Tema oscuro/claro (opcional)
- Iconos de Font Awesome

## Requisitos del Sistema

- Python 3.8+
- Modelos entrenados (generados por los notebooks)
- Dependencias listadas en `requirements.txt`

## Instalación

1. **Clonar el repositorio y navegar al directorio web:**
   ```bash
   cd csgo-ml/web_app
   ```

2. **Crear entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar que existen los modelos entrenados:**
   ```
   ../models/
   ├── classification/
   │   ├── best_classification_model.pkl
   │   ├── classification_metadata.json
   │   ├── classification_scaler.pkl (si es necesario)
   │   └── label_encoder.pkl
   └── regression/
       ├── best_regression_model.pkl
       ├── regression_metadata.json
       └── regression_scaler.pkl (si es necesario)
   ```

5. **Ejecutar los notebooks para generar modelos (si no existen):**
   ```bash
   cd ../notebooks
   jupyter notebook effectiveness_modeling.ipynb
   jupyter notebook regression_modeling.ipynb
   ```

## Uso

### Ejecutar la Aplicación
```bash
python app.py
```

La aplicación estará disponible en: `http://localhost:5000`

### Rutas Disponibles

- **`/`** - Página principal con información general
- **`/classification`** - Interfaz de predicción de efectividad
- **`/regression`** - Interfaz de predicción de kills
- **`/api/model_info`** - Información de los modelos (JSON)
- **`/api/predict/effectiveness`** - API de clasificación (POST)
- **`/api/predict/match_kills`** - API de regresión (POST)

### APIs REST

#### Predicción de Efectividad
```bash
curl -X POST http://localhost:5000/api/predict/effectiveness \
  -H "Content-Type: application/json" \
  -d '{
    "round_headshots": 2,
    "grenade_effectiveness": 3.5
  }'
```

#### Predicción de Kills
```bash
curl -X POST http://localhost:5000/api/predict/match_kills \
  -H "Content-Type: application/json" \
  -d '{
    "round_kills": 1.2,
    "match_headshots": 10,
    "team_equipment_value": 22000,
    "match_assists": 6
  }'
```

## Estructura del Proyecto

```
web_app/
├── app.py                    # Aplicación Flask principal
├── requirements.txt          # Dependencias Python
├── README.md                # Esta documentación
├── static/
│   ├── css/
│   │   └── style.css        # Estilos personalizados
│   └── js/
│       └── main.js          # JavaScript principal
└── templates/
    ├── base.html            # Template base
    ├── index.html           # Página principal
    ├── classification.html   # Página de clasificación
    ├── regression.html      # Página de regresión
    └── error.html           # Página de error
```

## Configuración

### Variables de Entorno (Opcional)
```bash
export FLASK_ENV=development    # Para desarrollo
export FLASK_DEBUG=1           # Para debugging
export FLASK_APP=app.py        # Archivo principal
```

### Configuración de Producción
Para despliegue en producción, considerar:

1. **Usar un servidor WSGI como Gunicorn:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Configurar proxy reverso (Nginx):**
   ```nginx
   server {
       listen 80;
       server_name tu-dominio.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. **Variables de entorno de producción:**
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

## Desarrollo

### Agregar Nuevos Modelos
1. Entrenar modelo en notebook
2. Guardar con el formato esperado:
   ```python
   import pickle
   import json
   
   # Guardar modelo
   with open('nuevo_modelo.pkl', 'wb') as f:
       pickle.dump(modelo, f)
   
   # Guardar metadatos
   metadata = {
       'model_name': 'NuevoModelo',
       'features': ['feature1', 'feature2'],
       'performance_metrics': {...}
   }
   with open('metadata.json', 'w') as f:
       json.dump(metadata, f)
   ```
3. Actualizar `app.py` para cargar el nuevo modelo
4. Crear nueva ruta y template si es necesario

### Personalización del Frontend
- Modificar `static/css/style.css` para estilos
- Actualizar `static/js/main.js` para funcionalidad
- Editar templates en `templates/` para estructura

## Troubleshooting

### Error: "Modelo no encontrado"
- Verificar que los notebooks han generado los archivos de modelo
- Confirmar rutas en `MODELS_DIR` dentro de `app.py`

### Error: "ModuleNotFoundError"
- Instalar dependencias: `pip install -r requirements.txt`
- Verificar versión de Python (3.8+)

### Error: "Permission denied"
- Verificar permisos de archivos
- En Linux/Mac: `chmod +x app.py`

### Problemas de Rendimiento
- Los modelos se cargan en memoria al iniciar
- Para datasets grandes, considerar caching
- Usar múltiples workers con Gunicorn

## Información Técnica

### Dataset
- **79,154 registros** de rondas de CS:GO
- **32 variables** incluyendo kills, headshots, equipamiento, etc.
- **5 mapas diferentes**: de_inferno, de_dust2, de_mirage, de_nuke, otros

### Modelos
- **Clasificación**: Gradient Boosting con accuracy ~82%
- **Regresión**: Gradient Boosting con R² ~75%
- **Features**: Seleccionadas por correlación y relevancia
- **Validación**: Cross-validation 5-fold estratificado

### Tecnologías
- **Backend**: Flask, scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5, Font Awesome, vanilla JavaScript
- **ML**: Gradient Boosting, preprocessing con StandardScaler/RobustScaler

## Licencia

Este proyecto es parte de un análisis académico de datos de CS:GO. Los modelos y datos son para fines educativos únicamente.

## Contribución

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## Contacto

Para consultas sobre el proyecto o problemas técnicos, crear un issue en el repositorio.