import json
import random
from typing import List, Dict, Set, Optional
from collections import defaultdict
import os

class UserDistributionManager:
    def __init__(self, 
                 user_summary_path: str,
                 n_grupos: int = 5,
                 n_rondas: int = 1,
                 training_file: Optional[str] = None):
        """
        Clase para distribuir usuarios en grupos (quintiles) y asignarlos de manera cíclica.
        
        Args:
            user_summary_path: Ruta al archivo JSON con el resumen de usuarios
            n_grupos: Número de grupos en los que dividir los usuarios
            n_rondas: Número de rondas completas a realizar
            training_file: Ruta a archivo de entrenamiento previo para continuar
        """
        # Cargar datos iniciales
        with open(user_summary_path, 'r') as f:
            self.user_data = json.load(f)
        
        self.n_grupos = n_grupos
        self.n_rondas = n_rondas
        
        # Extraer y ordenar usuarios por cantidad
        self.users_by_count = list(self.user_data['counts'].items())
        self.users_by_count.sort(key=lambda x: x[1])  # Ordenar por count ascendente
        
        # Dividir en quintiles/grupos
        self.groups = self._create_groups()
        
        # Variables de estado
        self.assigned_users: Set[str] = set()
        self.group_records: Dict[int, int] = defaultdict(int)
        self.round_records: List[Dict[int, int]] = []
        self.current_round = 0
        self.current_group_pointer = 0  # Para seguir qué grupo toca en la ronda actual
        
        # Cargar estado previo si existe
        if training_file and os.path.exists(training_file):
            self._load_training_state(training_file)
        else:
            self.training_file = f"training_{n_grupos}.json"
            self._initialize_training_file()
    
    def _create_groups(self) -> List[List[str]]:
        """Divide los usuarios en n_grupos grupos basados en sus counts."""
        total_users = len(self.users_by_count)
        group_size = total_users // self.n_grupos
        remainder = total_users % self.n_grupos
        
        groups = []
        start = 0
        
        for i in range(self.n_grupos):
            # Ajustar tamaño del grupo para distribuir el resto
            end = start + group_size + (1 if i < remainder else 0)
            group_users = [user_id for user_id, _ in self.users_by_count[start:end]]
            groups.append(group_users)
            start = end
        
        return groups
    
    def _initialize_training_file(self):
        """Inicializa el archivo de entrenamiento con estructura básica."""
        initial_state = {
            "n_grupos": self.n_grupos,
            "n_rondas": self.n_rondas,
            "assigned_users": list(self.assigned_users),
            "group_records": dict(self.group_records),
            "round_records": self.round_records,
            "current_round": self.current_round,
            "current_group_pointer": self.current_group_pointer
        }
        
        with open(self.training_file, 'w') as f:
            json.dump(initial_state, f, indent=2)
    
    def _load_training_state(self, training_file: str):
        """Carga el estado desde un archivo de entrenamiento previo."""
        with open(training_file, 'r') as f:
            state = json.load(f)
        
        self.training_file = training_file
        self.assigned_users = set(state['assigned_users'])
        self.group_records = defaultdict(int, state['group_records'])
        self.round_records = state['round_records']
        self.current_round = state['current_round']
        self.current_group_pointer = state['current_group_pointer']
        
        # Verificar que los parámetros coincidan
        if state['n_grupos'] != self.n_grupos:
            print(f"Advertencia: n_grupos diferente ({state['n_grupos']} vs {self.n_grupos})")
        if state['n_rondas'] != self.n_rondas:
            print(f"Advertencia: n_rondas diferente ({state['n_rondas']} vs {self.n_rondas})")
    
    def _save_training_state(self):
        """Guarda el estado actual en el archivo de entrenamiento."""
        state = {
            "n_grupos": self.n_grupos,
            "n_rondas": self.n_rondas,
            "assigned_users": list(self.assigned_users),
            "group_records": dict(self.group_records),
            "round_records": self.round_records,
            "current_round": self.current_round,
            "current_group_pointer": self.current_group_pointer
        }
        
        with open(self.training_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _get_next_user_from_group(self, group_idx: int) -> Optional[str]:
        """Obtiene el próximo usuario no asignado del grupo especificado."""
        available_users = [
            user for user in self.groups[group_idx] 
            if user not in self.assigned_users
        ]
        
        if not available_users:
            return None
        
        # Seleccionar aleatoriamente de los usuarios disponibles
        return random.choice(available_users)
    
    def get_next_user(self) -> Optional[str]:
        """
        Obtiene el próximo usuario a asignar.
        
        Returns:
            user_id o None si no hay más usuarios disponibles
        """
        # Verificar si hemos completado todas las rondas
        if self.current_round >= self.n_rondas:
            return None
        
        # Intentar asignar usuario del grupo actual
        for attempt in range(self.n_grupos):
            group_idx = (self.current_group_pointer + attempt) % self.n_grupos
            user_id = self._get_next_user_from_group(group_idx)
            
            if user_id:
                # Actualizar estado
                self.assigned_users.add(user_id)
                
                # Actualizar registros del grupo
                user_count = self.user_data['counts'][user_id]
                self.group_records[group_idx] += user_count
                
                # Mover puntero al siguiente grupo para próxima llamada
                self.current_group_pointer = (group_idx + 1) % self.n_grupos
                
                # Guardar estado
                self._save_training_state()
                
                return user_id
        
        # Si llegamos aquí, no hay usuarios disponibles en ningún grupo
        # Comenzar nueva ronda si es necesario
        self._start_new_round()
        
        # Intentar nuevamente después de comenzar nueva ronda
        return self.get_next_user()
    
    def _start_new_round(self):
        """Inicia una nueva ronda de asignación."""
        self.current_round += 1
        
        # Guardar registros de la ronda anterior
        if self.current_round > 0:  # Si ya teníamos una ronda activa
            self.round_records.append(dict(self.group_records))
        
        # Limpiar para nueva ronda
        self.assigned_users.clear()
        self.group_records.clear()
        self.current_group_pointer = 0
        
        # Guardar estado
        self._save_training_state()
    
    def get_distribution_summary(self) -> Dict:
        """Obtiene un resumen de la distribución actual."""
        summary = {
            "total_assigned_users": len(self.assigned_users),
            "current_round": self.current_round,
            "total_rounds": self.n_rondas,
            "group_distribution": {},
            "round_summary": []
        }
        
        # Distribución por grupo actual
        for i in range(self.n_grupos):
            users_in_group = len([u for u in self.groups[i] if u in self.assigned_users])
            total_in_group = len(self.groups[i])
            summary["group_distribution"][f"group_{i}"] = {
                "assigned": users_in_group,
                "total": total_in_group,
                "remaining": total_in_group - users_in_group,
                "total_records": self.group_records.get(i, 0)
            }
        
        # Resumen de rondas completadas
        for idx, round_data in enumerate(self.round_records):
            summary["round_summary"].append({
                "round": idx,
                "group_records": round_data,
                "total_records": sum(round_data.values())
            })
        
        return summary
    
    def reset_distribution(self):
        """Reinicia la distribución completamente."""
        self.assigned_users.clear()
        self.group_records.clear()
        self.round_records.clear()
        self.current_round = 0
        self.current_group_pointer = 0
        self._initialize_training_file()
    
    def get_available_users_count(self) -> int:
        """Obtiene el número de usuarios disponibles para asignar."""
        total_users = sum(len(group) for group in self.groups)
        assigned_in_current_round = len(self.assigned_users)
        users_per_round = total_users // self.n_rondas + (1 if total_users % self.n_rondas else 0)
        
        return min(users_per_round - assigned_in_current_round, 
                  total_users - len(self.assigned_users))
