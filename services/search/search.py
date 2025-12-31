import pandas as pd
from annoy import AnnoyIndex
import numpy as np
import os
from typing import Dict, List, Union, Optional, Iterable

class OptimizedMusicSimilaritySearch:
    def __init__(self, for_humans_path: str, duration_item_path: str, 
                 embedding_dim: int, metric: str = 'angular',
                 use_memory_mapping: bool = True):
        """
        Versión optimizada para bajo consumo de RAM
        
        Args:
            for_humans_path: ruta al archivo id_information.csv
            duration_item_path: ruta al archivo id_metadata.csv
            embedding_dim: dimensión de los embeddings
            metric: métrica para Annoy
            use_memory_mapping: usar memory mapping para reducir RAM
        """
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.use_memory_mapping = use_memory_mapping
        
        # Cargar solo columnas necesarias y optimizar tipos de datos
        self.for_humans = self._load_and_optimize_csv(for_humans_path)
        self.duration_item = self._load_and_optimize_csv(duration_item_path)
        
        # Estructuras ligeras
        self.annoy_index = None
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        
        # Memory mapping para embeddings grandes
        self.embeddings_mmap = None
        self.embeddings_file_path = None
    
    def _load_and_optimize_csv(self, filepath: str) -> pd.DataFrame:
        """Carga CSV optimizando tipos de datos para ahorrar RAM"""
        # Primero leemos solo las primeras filas para inferir tipos
        sample = pd.read_csv(filepath,sep='\t', nrows=1000)
        
        type_map = {}
        for col in sample.columns:
            dtype = sample[col].dtype
            
            if dtype == 'object':
                # Para strings, usar category si tiene pocos valores únicos
                unique_ratio = sample[col].nunique() / len(sample)
                if unique_ratio < 0.5:  # Si menos del 50% son únicos
                    type_map[col] = 'category'
                else:
                    type_map[col] = 'string'
            elif dtype in ['int64', 'int32']:
                type_map[col] = 'int32'  # Reducir a 32 bits
            elif dtype in ['float64', 'float32']:
                type_map[col] = 'float32'
        
        # Cargar con tipos optimizados
        return pd.read_csv(filepath, sep='\t', dtype=type_map, usecols=lambda x: x != 'Unnamed: 0')
    
    def build_annoy_index_from_dataframe(self, embeddings_df: pd.DataFrame, 
                                       n_trees: int = 10,  # Reducir árboles
                                       build_on_disk: bool = False) -> None:
        """
        Construye índice Annoy optimizado para memoria
        
        Args:
            embeddings_df: DataFrame con embeddings
            n_trees: número de árboles (reducido para ahorrar memoria)
            build_on_disk: construir en disco para datasets muy grandes
        """
        # Limpiar estructuras previas
        self.id_to_index.clear()
        self.index_to_id.clear()
        
        # Guardar embeddings de forma eficiente
        if self.use_memory_mapping:
            self._save_embeddings_mmap(embeddings_df)
        else:
            self._save_embeddings_light(embeddings_df)
        
        self.embedding_dim = len(embeddings_df.columns)
        # Configurar Annoy
        self.annoy_index = AnnoyIndex(self.embedding_dim, self.metric)
        
        # Construir índice
        print(f"Construyendo índice con {len(embeddings_df)} items y {n_trees} árboles...")
        
        if build_on_disk:
            # Para datasets muy grandes, construir en disco
            temp_index_path = "temp_annoy_index"
            self.annoy_index.on_disk_build(temp_index_path)
        
        # Añadir items
        for i, (idx, row) in enumerate(embeddings_df.iterrows()):
            if self.use_memory_mapping and self.embeddings_mmap is not None:
                vector = self.embeddings_mmap[i]
            else:
                vector = row.values.astype('float32')
            
            self.annoy_index.add_item(i, vector)
            self.id_to_index[str(idx)] = i
            self.index_to_id[i] = str(idx)
        
        # Construir con menos árboles para ahorrar RAM
        self.annoy_index.build(n_trees)
        print("Índice construido exitosamente")
    
    def _save_embeddings_mmap(self, embeddings_df: pd.DataFrame) -> None:
        """Guarda embeddings usando memory mapping"""
        embeddings_array = embeddings_df.values.astype('float32')
        
        # Guardar en archivo temporal para memory mapping
        self.embeddings_file_path = "embeddings_mmap.dat"
        np.memmap(self.embeddings_file_path, dtype='float32', mode='w+', 
                 shape=embeddings_array.shape)[:] = embeddings_array
        
        # Cargar en modo lectura
        self.embeddings_mmap = np.memmap(self.embeddings_file_path, dtype='float32', 
                                        mode='r', shape=embeddings_array.shape)
        
        # Liberar memoria del array original
        del embeddings_array
    
    def _save_embeddings_light(self, embeddings_df: pd.DataFrame) -> None:
        """Guarda embeddings de forma compacta"""
        self.embeddings_array = embeddings_df.values.astype('float32')
        # Forzar garbage collection
        import gc
        gc.collect()
    
    def load_annoy_index(self, filepath: str) -> None:
        """Carga índice Annoy existente"""
        self.annoy_index = AnnoyIndex(self.embedding_dim, self.metric)
        self.annoy_index.load(filepath)
        print(f"Índice cargado: {self.annoy_index.get_n_items()} items")
    
    def get_embedding(self, item_id: str) -> np.ndarray:
        """Obtiene embedding por ID de forma eficiente"""
        if item_id not in self.id_to_index:
            raise ValueError(f"Item ID {item_id} no encontrado")
        
        idx = self.id_to_index[item_id]
        
        if self.use_memory_mapping and self.embeddings_mmap is not None:
            return self.embeddings_mmap[idx]
        elif hasattr(self, 'embeddings_array'):
            return self.embeddings_array[idx]
        else:
            raise ValueError("Embeddings no disponibles")
    
    from typing import Dict

    def get_item_info(
        self,
        item_id: str,
        info_types: str = 'all'
    ) -> Dict:
        """
        Obtiene información del item de forma eficiente.
        info_types debe ser: 'human', 'duration', 'embedding' o 'all'
        """

        allowed_types = {'human', 'duration', 'embedding', 'all'}

        if info_types not in allowed_types:
            return {'error': f'info_types inválido: {info_types}'}

        result = {}

        def _get_from_df(df, key):
            row = df.loc[df['id'] == item_id]
            if not row.empty:
                result[key] = row.iloc[0].to_dict()

        handlers = {
            'human': lambda: _get_from_df(self.for_humans, 'human_info'),
            'duration': lambda: _get_from_df(self.duration_item, 'duration_info'),
            'embedding': lambda: result.update({
                'embedding': self.get_embedding(item_id)
            })
        }

        if info_types == 'all':
            types_to_process = ('human', 'duration', 'embedding')
        else:
            types_to_process = (info_types,)

        for info_type in types_to_process:
            try:
                handlers[info_type]()
            except ValueError:
                pass

        return result or {'error': f'Item {item_id} no encontrado'}

    
    def get_similar_items(self, item_id: str, n: int = 5, 
                         include_embedding: bool = False,
                         include_distances: bool = True) -> Dict:
        """
        Búsqueda eficiente de items similares
        """
        if self.annoy_index is None:
            raise ValueError("Índice Annoy no inicializado")
        
        if item_id not in self.id_to_index:
            raise ValueError(f"Item ID {item_id} no encontrado")
        
        # Obtener embedding de forma eficiente
        query_vector = self.get_embedding(item_id)
        
        return self._get_similar_by_vector(query_vector, n, include_embedding, include_distances)
    
    def get_similar_by_vector(self, vector: np.ndarray, n: int = 5,
                            include_embedding: bool = False,
                            include_distances: bool = True) -> Dict:
        """Búsqueda por vector con gestión eficiente de memoria"""
        return self._get_similar_by_vector(vector, n, include_embedding, include_distances)
    
    def _get_similar_by_vector(self, vector: np.ndarray, n: int,
                             include_embedding: bool, include_distances: bool) -> Dict:
        """Implementación interna eficiente"""
        vector = vector.astype('float32')
        
        # Búsqueda en Annoy
        if include_distances:
            indices, distances = self.annoy_index.get_nns_by_vector(
                vector, n, include_distances=True, search_k=-1  # Busqueda exacta
            )
        else:
            indices = self.annoy_index.get_nns_by_vector(
                vector, n, include_distances=False, search_k=-1
            )
            distances = []
        
        # Mapear a IDs originales
        similar_ids = [self.index_to_id[i] for i in indices]
        
        result = {
            'similar_ids': similar_ids,
            'distances': distances if include_distances else None
        }
        
        # Embeddings solo si se solicitan
        if include_embedding:
            embeddings = []
            for idx in indices:
                if self.use_memory_mapping and self.embeddings_mmap is not None:
                    embeddings.append(self.embeddings_mmap[idx])
                elif hasattr(self, 'embeddings_array'):
                    embeddings.append(self.embeddings_array[idx])
            result['embeddings'] = embeddings
        
        return result
    
    def batch_get_info(self, item_ids: List[str], info_type: str = 'all') -> List[Dict]:
        """Procesamiento por lotes eficiente"""
        return [self.get_item_info(item_id, info_type) for item_id in item_ids]
    
    def optimize_memory(self) -> None:
        """Ejecuta optimizaciones adicionales de memoria"""
        import gc
        
        # Forzar garbage collection
        gc.collect()
        
        # Liberar memoria de DataFrames si es posible
        if hasattr(self, 'for_humans'):
            self.for_humans = self.for_humans.copy()
        if hasattr(self, 'duration_item'):
            self.duration_item = self.duration_item.copy()
        
        gc.collect()
    
    def __del__(self):
        """Cleanup al destruir la instancia"""
        if hasattr(self, 'embeddings_mmap') and self.embeddings_mmap is not None:
            del self.embeddings_mmap
        if (hasattr(self, 'embeddings_file_path') and 
            self.embeddings_file_path and 
            os.path.exists(self.embeddings_file_path)):
            try:
                os.remove(self.embeddings_file_path)
            except:
                pass
            