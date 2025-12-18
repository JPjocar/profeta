#!/usr/bin/env bash
# exit on error
set -o errexit

chmod a+x build.sh

# 1. Instalar librerías
pip install -r requirements.txt

# 2. Recolectar archivos estáticos (CSS, JS, Imágenes del sistema)
python manage.py collectstatic --no-input

# 3. Crear las tablas de la base de datos (Aunque se borren, Django las necesita para arrancar)
python manage.py migrate