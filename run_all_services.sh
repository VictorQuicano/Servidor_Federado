#!/bin/bash

# Obtener la ruta absoluta del directorio del script actual
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "🚀 Iniciando todos los servicios del servidor..."

# 1. Activar el entorno virtual (especificado por el usuario)
VENV_PATH="$BASE_DIR/services/search/venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "📦 Activando entorno virtual..."
    source "$VENV_PATH"
else
    echo "⚠️ Advertencia: No se encontró el entorno virtual en $VENV_PATH"
fi

# 2. Ejecutar Servicio de Distribución
echo "🔹 Iniciando Servicio de Distribución..."
cd "$BASE_DIR/services/distribution" && ./run.sh

# 3. Ejecutar Servicio de Búsqueda
echo "🔹 Iniciando Servicio de Búsqueda..."
cd "$BASE_DIR/services/search" && ./run.sh

# 4. Ejecutar Servicio de UI Backend
echo "🔹 Iniciando Servicio de UI Backend..."
cd "$BASE_DIR/services/ui/backend" && ./run.sh

echo "✅ Todos los servicios han sido lanzados."
echo "Puedes verificar los logs en cada directorio de servicio (server.log)."
