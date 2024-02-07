# Utiliza una imagen base de Python
FROM python:3.8.5-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY requirements.txt .
COPY training.py .
COPY scoring.py .

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando por defecto al iniciar el contenedor
CMD [ "sh", "-c", "python training.py && python scoring.py" ]
