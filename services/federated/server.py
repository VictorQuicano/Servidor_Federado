import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calcula el promedio ponderado de las métricas recibidas de los clientes."""
    # Multiplicar cada métrica por el número de ejemplos y sumar
    accuracies = [num_examples * m.get("precision@5", 0.0) for num_examples, m in metrics]
    rewards = [num_examples * m.get("avg_reward", 0.0) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Promedio ponderado
    total_examples = sum(examples)
    return {
        "precision@5": sum(accuracies) / total_examples if total_examples > 0 else 0.0,
        "avg_reward": sum(rewards) / total_examples if total_examples > 0 else 0.0,
    }

def get_strategy(min_fit_clients: int = 2, min_available_clients: int = 2) -> fl.server.strategy.Strategy:
    """Configura la estrategia de agregación FedAvg."""
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Proporción de clientes a usar para entrenamiento
        fraction_evaluate=1.0,  # Proporción de clientes a usar para evaluación
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {
            "epochs": 1,
            "epsilon_start": max(0.1, 0.9 * (0.95 ** server_round)) # Decay de exploración global
        }
    )
    return strategy
