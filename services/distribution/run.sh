#!/bin/bash

# Activar entorno si es necesario o correr directamente con uvicorn
APP="main:app"
HOST="0.0.0.0"
PORT="8081"
LOG_FILE="server.log"
PID_FILE="server.pid"

echo "Iniciando servidor FastAPI en $HOST:$PORT..."

# Ejecutar en segundo plano con nohup
nohup uvicorn $APP --host $HOST --port $PORT > $LOG_FILE 2>&1 &

# Guardar el PID (ID del proceso)
echo $! > $PID_FILE

echo "Servidor iniciado en segundo plano"
echo "PID: $(cat $PID_FILE)"
echo "Logs: $LOG_FILE"
echo "Para detener: ./stop_server.sh"