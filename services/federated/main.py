import flwr as fl
import logging
import os
from server.services.federated.server import get_strategy

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Obtener configuración desde variables de entorno
    SERVER_ADDRESS = os.getenv("FEDERATED_SERVER_ADDRESS", "0.0.0.0:8080")
    NUM_ROUNDS = int(os.getenv("FEDERATED_ROUNDS", 50))
    MIN_CLIENTS = int(os.getenv("FEDERATED_MIN_CLIENTS", 2))
    
    # Configuración de distribución de usuarios
    SUMMARY_PATH = os.getenv("USER_SUMMARY_PATH", "server/services/distribution/user_summary.json")
    N_GRUPOS = int(os.getenv("N_GRUPOS", 5))

    logging.info("Iniciando Servidor Federado...")
    logging.info(f"Dirección: {SERVER_ADDRESS}")
    logging.info(f"Rondas estimadas: {NUM_ROUNDS}")
    logging.info(f"Mínimo de clientes requeridos: {MIN_CLIENTS}")

    # Inicializar el manager de distribución de usuarios
    from server.services.distribution.user_distro_manager import UserDistributionManager
    user_manager = UserDistributionManager(
        user_summary_path=SUMMARY_PATH,
        n_grupos=N_GRUPOS,
        n_rondas=NUM_ROUNDS
    )

    # Definir la estrategia con el manager
    strategy = get_strategy(
        user_manager=user_manager,
        min_fit_clients=MIN_CLIENTS, 
        min_available_clients=MIN_CLIENTS
    )

    # Iniciar el servidor
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
