# Imagen base oficial de Python
FROM python:3.10-slim

# Evitar buffering en logs
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Exponer el puerto Flask
EXPOSE 5000

# Comando para ejecutar Flask
CMD ["python", "app.py"]
