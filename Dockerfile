# Usar una imagen oficial de Python como base
FROM python:3.7.12

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app
# Copiar los archivos necesarios al contenedor
COPY . /app

# Instalar las dependencias necesarias
RUN pip install --no-cache-dir Flask numpy tensorflow keras flask-cors Pillow waitress

EXPOSE 5000

# CMD ["python", "API.py"]
CMD ["waitress-serve", "--port=5000", "API:app"]