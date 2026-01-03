import os
import json
import glob

def generate_summary(directory, output_file):
    # Buscar todos los archivos que coincidan con el patrón user_<id>_processed.csv
    search_pattern = os.path.join(directory, "user_*_processed.csv")
    files = glob.glob(search_pattern)
    
    user_counts = {}
    
    print(f"Procesando {len(files)} archivos en {directory}...")

    for f in files:
        filename = os.path.basename(f)
        # Extraer user_<id> del nombre del archivo
        user_id = filename.replace("_processed.csv", "")
        
        # Contar líneas del CSV de forma eficiente
        try:
            with open(f, 'r') as file:
                # Restamos 1 para no contar el header
                count = sum(1 for line in file) - 1
                if count < 0: count = 0
                user_counts[user_id] = count
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

    if not user_counts:
        print("No se encontraron archivos o los archivos están vacíos.")
        return

    # Cálculos estadísticos
    vals = list(user_counts.values())
    total_users = len(user_counts)
    avg_records = sum(vals) / total_users
    
    min_user_id = min(user_counts, key=user_counts.get)
    max_user_id = max(user_counts, key=user_counts.get)
    
    # Construir el JSON final
    summary = {
        "total_users": total_users,
        "avg_records_per_user": round(avg_records, 2),
        "min_user": {
            "user": min_user_id,
            "count": user_counts[min_user_id]
        },
        "max_user": {
            "user": max_user_id,
            "count": user_counts[max_user_id]
        },
        "difference_max_min": user_counts[max_user_id] - user_counts[min_user_id],
        "counts": user_counts
    }

    # Guardar en el archivo de salida
    with open(output_file, 'w') as out:
        json.dump(summary, out, indent=2)
    
    print(f"Resumen generado exitosamente en: {output_file}")

if __name__ == "__main__":
    DATA_DIR = "/mnt/shared-storage/data/user_histories"
    OUTPUT = "/home/ubuntu/Servidor_Federado/services/distribution/user_summary.json"
    
    generate_summary(DATA_DIR, OUTPUT)