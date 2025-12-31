#!/bin/bash

# Configuración
BUCKET_NAME="recommender-system-datasets-tesis-experiment"
DEST_DIR="/mnt/shared-storage/data"

echo "Iniciando descarga de datasets desde gs://${BUCKET_NAME}..."

# Crear el directorio de destino si no existe
sudo mkdir -p ${DEST_DIR}
sudo chown -R $USER:$USER /mnt/shared-storage/

# Descargar las 2 carpetas (music_dataset y user_histories)
# Usamos -m para descarga en paralelo (más rápido) y -r para recursivo
gsutil -m cp -r gs://${BUCKET_NAME}/music_dataset ${DEST_DIR}/
gsutil -m cp -r gs://${BUCKET_NAME}/user_histories ${DEST_DIR}/

echo "Descarga completada en ${DEST_DIR}"
ls -lh ${DEST_DIR}
