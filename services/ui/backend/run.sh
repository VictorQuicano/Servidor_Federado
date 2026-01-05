#!/bin/bash
# Script para iniciar el servicio de monitoreo
# Puerto 8083

if ! command -v uvicorn &> /dev/null
then
    echo "uvicorn no encontrado. Instalando..."
    pip install uvicorn fastapi sqlalchemy
fi

echo "Iniciando servicio de monitoreo en http://0.0.0.0:8083"
uvicorn main:app --host 0.0.0.0 --port 8083 --reload
