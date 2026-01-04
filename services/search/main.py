from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List as ListType, Dict, Optional, Any
import numpy as np
import uvicorn
from search import *
import pandas as pd
from annoy import AnnoyIndex
import os
from datetime import datetime
import json
import argparse
import sys

from google.cloud import storage

# Setup paths
BUCKET_NAME = "recommender-system-datasets-tesis-experiment"
BUCKET_PREFIX = "music_dataset"

# Load Data
for_humans = "id_information.csv"
duration_item = "id_metadata.csv"
history_count = "userid_trackid_count.tsv.bz2"
embeddings_compress_path = "music_4_all_compress_64.csv"

# Download from GCP if missing
_files_to_check = {
    for_humans: f"{BUCKET_PREFIX}/id_information.csv",
    duration_item: f"{BUCKET_PREFIX}/id_metadata.csv",
    history_count: f"{BUCKET_PREFIX}/userid_trackid_count.tsv.bz2",
    embeddings_compress_path: f"{BUCKET_PREFIX}/music_4_all_compress_64.csv"
}

if any(not os.path.exists(f) for f in _files_to_check):
    _bucket = storage.Client().bucket(BUCKET_NAME)
    for _local, _remote in _files_to_check.items():
        if not os.path.exists(_local):
            _bucket.blob(_remote).download_to_filename(_local)

# Models Pydantic para validación
class SimilarityRequest(BaseModel):
    item_id: str
    n_results: int = 5
    include_embedding: bool = False
    include_distances: bool = True

class VectorSearchRequest(BaseModel):
    vector: List[float]
    n_results: int = 5
    include_embedding: bool = False
    include_distances: bool = True

class BatchInfoRequest(BaseModel):
    item_ids: List[str]
    info_type: str = "all"  # 'human', 'duration', 'embedding', 'all'

class SearchResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    timestamp: str
    message: Optional[str] = None

def sanitize_for_json(data):
    """Convierte objetos no serializables a formatos compatibles con JSON"""
    import numpy as np
    
    if isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_for_json(item) for item in data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):  # Para tipos numpy como np.float32, np.int64, etc.
        return data.item()
    elif hasattr(data, 'tolist'):  # Para memmap y otros tipos similares
        return data.tolist()
    elif hasattr(data, '__dict__'):
        # Para objetos con atributos
        return sanitize_for_json(data.__dict__)
    else:
        # Para tipos básicos de Python
        return data

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Music Similarity Search API",
    description="API para búsqueda de similitud musical usando embeddings optimizados",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable global para almacenar la instancia del buscador
music_searcher = None

@app.on_event("startup")
async def startup_event():
    """Inicializa el buscador al arrancar el servidor"""
    global music_searcher
    
    print("Inicializando Music Similarity Search...")


    
    try:
        # Cargar embeddings
        embeddings_df = pd.read_csv(embeddings_compress_path, index_col=0)
        print(f"Embeddings cargados: {len(embeddings_df)} items")
        
        # Inicializar buscador
        music_searcher = OptimizedMusicSimilaritySearch(
            for_humans_path=for_humans,
            duration_item_path=duration_item,
            embedding_dim=300,
            use_memory_mapping=True
        )
        
        mmap_filename = "embeddings_mmap.dat"
        mmap_path = os.path.join(os.path.dirname(__file__), mmap_filename)

        if os.path.exists(mmap_path):
            print(f"Encontrado archivo mmap en {mmap_path}, cargando índice Annoy desde mmap...")
            try:
                music_searcher.load_annoy_index(mmap_path)
                # Si quieres mantener la misma optimización que se hace luego de construir el índice:
                try:
                    music_searcher.optimize_memory()
                except Exception as e:
                    print(f"Warning: error al optimizar memoria después de cargar mmap: {e}")
                print("Índice cargado correctamente desde mmap. Se omitirá la construcción desde el dataframe.")
                return  # salir del startup para evitar reconstruir el índice
            except Exception as e:
                print(f"Error cargando índice desde mmap: {e}")
                # Si la carga falla, se continúa y se construirá el índice desde el dataframe
        else:
            print("No se encontró archivo mmap; se construirá el índice desde el dataframe.")
            # Construir índice
            music_searcher.build_annoy_index_from_dataframe(
                embeddings_df=embeddings_df,
                n_trees=20,
                build_on_disk=False
            )
        
        # Optimizar memoria
        music_searcher.optimize_memory()
        
        print(f"Buscador inicializado con éxito")
        print(f"Total items en índice: {music_searcher.annoy_index.get_n_items() if music_searcher.annoy_index else 0}")
        
    except Exception as e:
        print(f"Error al inicializar: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar el servidor"""
    global music_searcher
    if music_searcher:
        del music_searcher
    print("Servidor apagado")

@app.get("/")
async def root():
    """Endpoint raíz con información del servicio"""
    total_items = music_searcher.annoy_index.get_n_items() if music_searcher and music_searcher.annoy_index else 0
    return {
        "service": "Music Similarity Search API",
        "status": "running",
        "total_items": total_items,
        "endpoints": {
            "health": "/health",
            "info": "/info/{item_id}",
            "similar": "/similar/{item_id}",
            "search_by_vector": "/search/vector",
            "batch_info": "/batch/info"
        }
    }

@app.get("/health")
async def health_check():
    """Verifica el estado del servicio"""
    if music_searcher is None or music_searcher.annoy_index is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    total_items = music_searcher.annoy_index.get_n_items()
    return {
        "status": "healthy",
        "total_items": total_items,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info/{item_id}", response_model=SearchResponse)
async def get_item_info(
    item_id: str,
    info_type: str = Query("all", description="Tipo de información a obtener: 'human', 'duration', 'embedding' o 'all'")
):
    """
    Obtiene información de un item específico
    
    Args:
        item_id: ID del item musical
        info_type: Tipo de información a obtener:
            - 'all': toda la información (humana, duración y embedding)
            - 'human': solo información humana (artista, título, etc.)
            - 'duration': solo información de duración
            - 'embedding': solo el vector de embedding
    """
    try:
        if not music_searcher:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Validar que el tipo sea válido
        valid_types = {'human', 'duration', 'embedding', 'all'}
        info_type_lower = info_type.lower()
        print(f"🦃🦃🦃 Requested info_type: {info_type_lower}")
        if info_type_lower not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de información inválido: '{info_type}'. Valores permitidos: {', '.join(valid_types)}"
            )
        
        # Obtener información
        info = music_searcher.get_item_info(item_id, info_type_lower)
        
        # Verificar si hay error
        if 'error' in info:
            error_msg = info['error']
            if 'inválido' in error_msg:
                # Esto no debería pasar si validamos arriba, pero por si acaso
                raise HTTPException(status_code=400, detail=error_msg)
            else:
                raise HTTPException(status_code=404, detail=error_msg)
        
        # Sanitizar para JSON (especialmente importante para embeddings)
        sanitized_info = sanitize_for_json(info)
        
        return SearchResponse(
            success=True,
            data=sanitized_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
    
@app.post("/similar/{item_id}", response_model=SearchResponse)
async def get_similar_items(
    item_id: str,
    request: SimilarityRequest
):
    """
    Encuentra items similares a un item dado
    """
    try:
        if not music_searcher:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Usar parámetros del request o valores por defecto
        n = request.n_results
        include_embedding = request.include_embedding
        include_distances = request.include_distances
        
        result = music_searcher.get_similar_items(
            item_id=item_id,
            n=n,
            include_embedding=include_embedding,
            include_distances=include_distances
        )
        
        # Añadir información básica de los items similares
        if 'similar_ids' in result:
            items_info = []
            for similar_id in result['similar_ids']:
                basic_info = music_searcher.get_item_info(similar_id, 'human')
                if 'human_info' in basic_info:
                    items_info.append({
                        'id': similar_id,
                        'info': basic_info['human_info']
                    })
            result['similar_items_info'] = items_info
        
        result = sanitize_for_json(result)
        
        return SearchResponse(
            success=True,
            data=result,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/search/vector", response_model=SearchResponse)
async def search_by_vector(request: VectorSearchRequest):
    """
    Busca items similares usando un vector de embedding directamente
    """
    try:
        if not music_searcher:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Convertir lista a numpy array
        vector = np.array(request.vector, dtype=np.float32)
        
        if len(vector) != music_searcher.embedding_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension mismatch. Expected {music_searcher.embedding_dim}, got {len(vector)}"
            )
        
        result = music_searcher.get_similar_by_vector(
            vector=vector,
            n=request.n_results,
            include_embedding=request.include_embedding,
            include_distances=request.include_distances
        )
        
        # Añadir información básica
        if 'similar_ids' in result:
            items_info = []
            for similar_id in result['similar_ids']:
                basic_info = music_searcher.get_item_info(similar_id, 'human')
                if 'human_info' in basic_info:
                    items_info.append({
                        'id': similar_id,
                        'info': basic_info['human_info']
                    })
            result['similar_items_info'] = items_info
        
        result = sanitize_for_json(result)
        
        return SearchResponse(
            success=True,
            data=result,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/batch/info", response_model=SearchResponse)
async def batch_get_info(request: BatchInfoRequest):
    """
    Obtiene información para múltiples items en un solo request
    """
    try:
        if not music_searcher:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Validar tipo de información
        valid_types = ['human', 'duration', 'embedding', 'all']
        if request.info_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid info_type. Must be one of: {valid_types}"
            )
        
        # Limitar tamaño del batch para evitar sobrecarga
        max_batch_size = 100
        if len(request.item_ids) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum is {max_batch_size}"
            )
        
        results = []
        for item_id in request.item_ids:
            try:
                info = music_searcher.get_item_info(item_id, request.info_type)
                results.append({
                    'item_id': item_id,
                    'info': info
                })
            except Exception as e:
                results.append({
                    'item_id': item_id,
                    'error': str(e)
                })

        result = sanitize_for_json(result)

        return SearchResponse(
            success=True,
            data={'results': results},
            timestamp=datetime.now().isoformat(),
            message=f"Processed {len(results)} items"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/stats")
async def get_statistics():
    """
    Obtiene estadísticas del sistema
    """
    if not music_searcher:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    stats = {
        "total_items": music_searcher.annoy_index.get_n_items() if music_searcher.annoy_index else 0,
        "embedding_dim": music_searcher.embedding_dim,
        "index_size": len(music_searcher.id_to_index),
        "memory_mapping": music_searcher.use_memory_mapping,
        "for_humans_records": len(music_searcher.for_humans) if hasattr(music_searcher, 'for_humans') else 0,
        "duration_records": len(music_searcher.duration_item) if hasattr(music_searcher, 'duration_item') else 0
    }
    
    return SearchResponse(
        success=True,
        data=stats,
        timestamp=datetime.now().isoformat()
    )

def main():
    """Función principal para ejecutar el servidor"""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Servidor FastAPI para Music Similarity Search')
    parser.add_argument('--port', type=int, default=8000,
                       help='Puerto para el servidor (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host para el servidor (default: 0.0.0.0)')
    parser.add_argument('--reload', action='store_true',
                       help='Habilitar recarga automática para desarrollo')
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Configuración del servidor con los argumentos
    host = args.host
    port = args.port
    reload_enabled = args.reload
    
    print(f"Iniciando servidor FastAPI en http://{host}:{port}")
    print(f"Puerto: {port}")
    print(f"Host: {host}")
    print(f"Recarga automática: {'Habilitada' if reload_enabled else 'Deshabilitada'}")
    print("Documentación disponible en:")
    print(f"  - http://{host}:{port}/docs (Swagger UI)")
    print(f"  - http://{host}:{port}/redoc (ReDoc)")
    
    # Ejecutar servidor
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_enabled
    )

if __name__ == "__main__":
    main()