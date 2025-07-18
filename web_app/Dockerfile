# Usar Python 3.12 como imagen base
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Copiar los modelos desde el directorio padre
COPY ../models ./models

# Crear los directorios necesarios
RUN mkdir -p templates static/css static/js

# Exponer el puerto 5000
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]