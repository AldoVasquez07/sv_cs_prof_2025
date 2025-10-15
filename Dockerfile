# Usa una imagen base con TensorFlow preinstalado (reduce el build de 10 min a <1 min)
FROM tensorflow/tensorflow:2.20.0

# Evitar buffering de logs en tiempo real
ENV PYTHONUNBUFFERED=1

# Crear y usar directorio de trabajo
WORKDIR /app

# Copiar dependencias primero (para aprovechar la cache de Docker)
COPY requirements.txt .

# Instalar dependencias adicionales
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer el puerto Flask
EXPOSE 5000

# Comando para ejecutar Flask
CMD ["python", "app.py"]
