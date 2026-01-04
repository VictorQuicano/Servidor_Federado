import flwr as fl
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from .distribution.user_distro_manager import UserDistributionManager
import csv
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

class MetricLogger:
    def __init__(self):
        self.global_csv_path = os.path.join(LOG_DIR, "global_metrics.csv")
        self.client_csv_path = os.path.join(LOG_DIR, "client_metrics.csv")
        
        # Initialize files with headers if they don't exist
        if not os.path.exists(self.global_csv_path):
            with open(self.global_csv_path, "w") as f:
                f.write("timestamp,round,stage,metric_name,value\n")
        
        if not os.path.exists(self.client_csv_path):
            with open(self.client_csv_path, "w") as f:
                f.write("timestamp,round,stage,client_id,metric_name,value\n")

    def log_global(self, server_round: int, stage: str, metrics: Metrics):
        timestamp = datetime.now().isoformat()
        rows = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                rows.append([timestamp, server_round, stage, k, v])
        
        with open(self.global_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def log_clients(self, server_round: int, stage: str, output_metrics: List[Tuple[ClientProxy, Metrics]]):
        """
        output_metrics is list of (ClientProxy, FitRes/EvaluateRes) IN FLOWER.
        But here we might just receive the raw (cid, metrics) list if we process it before aggregation.
        Actually, CustomFedAvg receives (ClientProxy, FitRes) list.
        """
        timestamp = datetime.now().isoformat()
        rows = []
        for client_proxy, res in output_metrics:
            # FitRes and EvaluateRes both have 'metrics' attribute
            # But wait, FitRes.metrics is a dictionary.
            metrics = res.metrics
            cid = client_proxy.cid
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    rows.append([timestamp, server_round, stage, cid, k, v])
        
        with open(self.client_csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(rows)


class MetricManager:
    def __init__(self):
        self.best_metrics = {}

    def aggregate(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        if not metrics:
            return {}

        # Collect all metric keys present across all clients
        all_keys = set()
        for _, m in metrics:
            all_keys.update(m.keys())
        
        aggregated = {}

        for key in all_keys:
            values = []
            weighted_sum = 0.0
            total_weight_for_key = 0
            
            for n, m in metrics:
                if key in m:
                    val = m[key]
                    if isinstance(val, (int, float)):
                        values.append(val)
                        weighted_sum += val * n
                        total_weight_for_key += n
            
            if not values:
                continue

            # Calculate Min, Max, Avg
            metric_min = min(values)
            metric_max = max(values)
            metric_avg = weighted_sum / total_weight_for_key if total_weight_for_key > 0 else 0.0

            aggregated[f"{key}_min"] = metric_min
            aggregated[f"{key}_max"] = metric_max
            aggregated[f"{key}_avg"] = metric_avg
            aggregated[key] = metric_avg # Default/plain key is average

            # Update Best (Track statefully)
            current_best = self.best_metrics.get(key)
            
            # Heuristic: 'loss' is better if lower, others (reward, precision, etc) better if higher
            is_loss = "loss" in key.lower()
            
            if current_best is None:
                self.best_metrics[key] = metric_avg
            else:
                if is_loss:
                    if metric_avg < current_best:
                        self.best_metrics[key] = metric_avg
                else:
                    if metric_avg > current_best:
                        self.best_metrics[key] = metric_avg
            
            aggregated[f"{key}_best"] = self.best_metrics[key]

        return aggregated

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, user_manager: UserDistributionManager, metric_logger: MetricLogger, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_manager = user_manager
        self.client_user_assignments = {} # Mapeo persistente para evaluación si se desea
        self.logger = metric_logger

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        # Log individual client metrics before aggregation
        if results:
            self.logger.log_clients(server_round, "fit", results)

        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        # Log aggregated global metrics
        if metrics:
            self.logger.log_global(server_round, "fit", metrics)
            
        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results using weighted average."""
        
        # Log individual client metrics before aggregation
        if results:
            self.logger.log_clients(server_round, "evaluate", results)

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log aggregated global metrics
        if metrics:
            self.logger.log_global(server_round, "evaluate", metrics)

        return loss, metrics

    def configure_fit(

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Asigna un user_id único a cada cliente para la ronda de entrenamiento."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Obtener clientes disponibles
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Crear instrucciones para cada cliente
        fit_configurations = []
        for client in clients:
            # Obtener el próximo usuario del manager
            target_user_id = self.user_manager.get_next_user()
            
            # Copiar config y añadir el user_id
            client_config = config.copy()
            client_config["target_user_id"] = target_user_id
            
            # Guardar asignación para consistencia en evaluación posterior (opcional)
            self.client_user_assignments[client.cid] = target_user_id
            
            fit_configurations.append((client, FitIns(parameters, client_config)))
            
            print(f"📡 Asignando usuario {target_user_id} al cliente {client.cid} (Ronda {server_round})")

        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configura la evaluación, asegurando que se use el mismo usuario asignado en fit."""
        if self.fraction_evaluate == 0.0:
            return []

        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        evaluate_configurations = []
        for client in clients:
            client_config = config.copy()
            # Usar el usuario que se le asignó en esta ronda (o la última conocida)
            target_user_id = self.client_user_assignments.get(client.cid)
            if target_user_id:
                client_config["target_user_id"] = target_user_id
            
            evaluate_configurations.append((client, EvaluateIns(parameters, client_config)))

        return evaluate_configurations

def get_strategy(user_manager: UserDistributionManager, min_fit_clients: int = 2, min_available_clients: int = 2) -> fl.server.strategy.Strategy:
    """Configura la estrategia de agregación CustomFedAvg."""
    # Instanciar el gestor de métricas para mantener estado (best metrics)
    metric_manager = MetricManager()
    metric_logger = MetricLogger()
    
    # We need to wrap the aggregation functions to also log the global result
    def shared_aggregate_fit(metrics_list):
        aggregated = metric_manager.aggregate(metrics_list)
        # We don't have server_round easily here without currying or changing signature.
        # But wait, FedAvg strategy calls 'fit_metrics_aggregation_fn(metrics)'.
        # It does NOT pass server_round.
        # However, we can log it. But we lack the round number.
        # Alternative: The `aggregate_fit` method in CustomFedAvg returns (parameters, metrics).
        # Depending on recent Flower versions, the strategy's aggregate_fit returns aggregated metrics as the 2nd return value.
        # So we can capture the result of super().aggregate_fit inside CustomFedAvg.aggregate_fit and log it there!
        return aggregated

    def shared_aggregate_eval(metrics_list):
        aggregated = metric_manager.aggregate(metrics_list)
        return aggregated

    start_strategy = CustomFedAvg(
        user_manager=user_manager,
        metric_logger=metric_logger,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        evaluate_metrics_aggregation_fn=shared_aggregate_eval,
        fit_metrics_aggregation_fn=shared_aggregate_fit,
        on_fit_config_fn=lambda server_round: {
            "epochs": 1,
            "epsilon_start": max(0.1, 0.9 * (0.95 ** server_round))
        }
    )
    
    # Monkey patch or modify CustomFedAvg to log global metrics after super calls?
    # Actually, let's just modify the aggregate methods in CustomFedAvg I defined above to log the *returned* metrics from super().
    
    # Redefine CustomFedAvg methods above correctly to capture return values
    
    return start_strategy
