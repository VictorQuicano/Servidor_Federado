#!/bin/bash

# stop_server.sh - Detener servidor FastAPI

PID_FILE="server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    
    echo "Deteniendo servidor con PID: $PID"
    
    # Matar el proceso
    kill $PID
    
    # Esperar a que se cierre
    sleep 2
    
    # Verificar si aún está corriendo
    if ps -p $PID > /dev/null 2>&1; then
        echo "Forzando cierre..."
        kill -9 $PID
    fi
    
    # Eliminar archivo PID
    rm $PID_FILE
    
    echo "Servidor detenido"
else
    echo "No se encontró archivo PID. Servidor puede no estar corriendo."
    echo "Buscando procesos de uvicorn..."
    
    # Buscar y mostrar procesos relacionados
    ps aux | grep uvicorn | grep -v grep
#!/bin/bash

# stop_server.sh - Detener servidor FastAPI

PID_FILE="server.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    
    echo "Deteniendo servidor con PID: $PID"
    
    # Matar el proceso
    kill $PID
    
    # Esperar a que se cierre
    sleep 2
    
    # Verificar si aún está corriendo
    if ps -p $PID > /dev/null 2>&1; then
        echo "Forzando cierre..."
        kill -9 $PID
    fi
    
    # Eliminar archivo PID
    rm $PID_FILE
    
    echo "Servidor detenido"
else
    echo "No se encontró archivo PID. Servidor puede no estar corriendo."
    echo "Buscando procesos de uvicorn..."
    
    # Buscar y mostrar procesos relacionados
    ps aux | grep uvicorn | grep -v grep
