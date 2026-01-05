from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import logging
from fastapi.middleware.cors import CORSMiddleware

from database import SessionLocal, init_db, TrainingSession, Client, ClientStatus, GlobalRoundMetric, ClientRoundMetric

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MonitoringService")

app = FastAPI(title="Federated Monitoring Service", version="1.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencia de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Inicializar DB al arranque
@app.on_event("startup")
def startup_event():
    init_db()

# --- Pydantic Models ---

class SessionCreate(BaseModel):
    total_rounds: int = 10

class ClientHeartbeat(BaseModel):
    user_id: str
    session_id: Optional[int] = None  # Nuevo: Vincular heartbeat a una sesión
    status: str
    current_round: int = 0
    ip_address: Optional[str] = None
    system_info: Optional[Dict] = None

class GlobalMetricLog(BaseModel):
    round_number: int
    metrics: Dict[str, Any]

class ClientMetricLog(BaseModel):
    user_id: str
    round_number: int
    metrics: Dict[str, Any]

# --- Rutas ---

@app.get("/")
def read_root():
    return {"status": "online", "service": "Federated Monitoring"}

# 1. Gestión de Sesiones
@app.post("/training/start")
def start_training_session(session_data: SessionCreate, db: Session = Depends(get_db)):
    """Inicia una nueva sesión de entrenamiento federado"""
    # Finalizar sesiones activas anteriores
    active_sessions = db.query(TrainingSession).filter(TrainingSession.status == "ACTIVE").all()
    for s in active_sessions:
        s.status = "INTERRUPTED"
        s.end_time = datetime.utcnow()
    
    new_session = TrainingSession(
        status="ACTIVE",
        total_rounds=session_data.total_rounds
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    logger.info(f"Nueva sesión de entrenamiento iniciada: ID {new_session.id}")
    return {"session_id": new_session.id}

@app.post("/training/{session_id}/end")
def end_training_session(session_id: int, db: Session = Depends(get_db)):
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.status = "COMPLETED"
    session.end_time = datetime.utcnow()
    db.commit()
    logger.info(f"Sesión {session_id} completada")
    return {"status": "ok"}

@app.get("/training/sessions")
def get_all_sessions(db: Session = Depends(get_db)):
    """Listar todas las sesiones"""
    sessions = db.query(TrainingSession).order_by(TrainingSession.id.desc()).limit(20).all()
    return sessions

# 2. Gestión de Clientes
@app.post("/client/heartbeat")
def client_heartbeat(hb: ClientHeartbeat, db: Session = Depends(get_db)):
    """El cliente reporta su estado (ping)"""
    client = db.query(Client).filter(Client.user_id == hb.user_id).first()
    
    if not client:
        client = Client(user_id=hb.user_id)
        db.add(client)
    
    # Actualizar estado
    client.current_status = hb.status
    client.last_seen = datetime.utcnow()
    if hb.current_round is not None:
        client.current_round = hb.current_round
    if hb.ip_address:
        client.ip_address = hb.ip_address
    if hb.system_info:
        client.system_info = hb.system_info
    
    # Vincular a sesión si se proporciona session_id
    if hb.session_id:
        session = db.query(TrainingSession).filter(TrainingSession.id == hb.session_id).first()
        if session:
            # Verificar si ya está en la sesión
            if session not in client.sessions:
                client.sessions.append(session)
    
    db.commit()
    return {"status": "ok"}

@app.get("/clients")
def get_clients(db: Session = Depends(get_db)):
    """Obtener lista de todos los clientes históricos"""
    return db.query(Client).all()

@app.get("/training/{session_id}/clients")
def get_session_clients(session_id: int, db: Session = Depends(get_db)):
    """Obtener todos los clientes que participan en una sesión específica"""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session.clients

# 3. Métricas
@app.post("/training/{session_id}/round/global")
def log_global_metrics(session_id: int, log: GlobalMetricLog, db: Session = Depends(get_db)):
    """El servidor federado envía métricas agregadas de la ronda"""
    metric = GlobalRoundMetric(
        session_id=session_id,
        round_number=log.round_number,
        metrics=log.metrics
    )
    db.add(metric)
    db.commit()
    return {"status": "saved"}

@app.post("/training/{session_id}/client/metrics")
def log_client_metrics(session_id: int, log: ClientMetricLog, db: Session = Depends(get_db)):
    """El cliente envía métricas individuales"""
    client = db.query(Client).filter(Client.user_id == log.user_id).first()
    if not client:
        client = Client(user_id=log.user_id, current_status="UNKNOWN")
        db.add(client)
    
    # Asegurar relación con sesión
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if session and session not in client.sessions:
        client.sessions.append(session)
    
    metric = ClientRoundMetric(
        session_id=session_id,
        user_id=log.user_id,
        round_number=log.round_number,
        metrics=log.metrics
    )
    db.add(metric)
    db.commit()
    return {"status": "saved"}

@app.get("/client/{user_id}/metrics")
def get_client_metrics(user_id: str, session_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Obtener histórico de métricas de un cliente, opcionalmente filtrado por sesión"""
    query = db.query(ClientRoundMetric).filter(ClientRoundMetric.user_id == user_id)
    if session_id:
        query = query.filter(ClientRoundMetric.session_id == session_id)
    
    return query.order_by(ClientRoundMetric.round_number).all()

@app.get("/training/{session_id}/details")
def get_session_details(session_id: int, db: Session = Depends(get_db)):
    """
    Obtener detalles de una sesión: clientes participantes y métricas globales.
    """
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    global_metrics = db.query(GlobalRoundMetric)\
        .filter(GlobalRoundMetric.session_id == session_id)\
        .order_by(GlobalRoundMetric.round_number)\
        .all()
        
    return {
        "session": session,
        "clients": session.clients,
        "global_metrics": global_metrics
    }

@app.get("/training/{session_id}/client/{user_id}/metrics")
def get_session_client_metrics(session_id: int, user_id: str, db: Session = Depends(get_db)):
    """
    Filtrar métricas de un cliente en una sesión específica, procesarlas por época
    y obtener estadísticas (max, min, avg) y las mejores métricas de entrenamiento.
    """
    metrics_records = db.query(ClientRoundMetric)\
        .filter(ClientRoundMetric.session_id == session_id)\
        .filter(ClientRoundMetric.user_id == user_id)\
        .order_by(ClientRoundMetric.round_number)\
        .all()
    return metrics_records
    if not metrics_records:
        print("No metrics records found")
        return {"per_epoch": [], "best_train_metrics": {}}

    actor_loss_by_epoch = {}
    critic_loss_by_epoch = {}
    train_rewards_by_epoch = {}
    train_metrics_by_epoch = {} # {metric_name: {epoch_idx: [vals]}}
    
    all_train_metrics_keys = set()

    for record in metrics_records:
        m = record.metrics
        if not m or not isinstance(m, dict):
            continue
        
        # Procesar actor_loss
        al = m.get("actor_loss", [])
        if isinstance(al, list):
            for i, val in enumerate(al):
                actor_loss_by_epoch.setdefault(i, []).append(val)
            
        # Procesar critic_loss
        cl = m.get("critic_loss", [])
        if isinstance(cl, list):
            for i, val in enumerate(cl):
                critic_loss_by_epoch.setdefault(i, []).append(val)
            
        # Procesar train_rewards
        tr = m.get("train_rewards", [])
        if isinstance(tr, list):
            for i, val in enumerate(tr):
                train_rewards_by_epoch.setdefault(i, []).append(val)
            
        # Procesar train_metrics
        tm = m.get("train_metrics", {})
        if isinstance(tm, dict):
            for metric_name, values in tm.items():
                if isinstance(values, list):
                    all_train_metrics_keys.add(metric_name)
                    if metric_name not in train_metrics_by_epoch:
                        train_metrics_by_epoch[metric_name] = {}
                    for i, val in enumerate(values):
                        train_metrics_by_epoch[metric_name].setdefault(i, []).append(val)

    # Determinar el conjunto de todos los índices de época presentes
    all_epoch_indices = set(actor_loss_by_epoch.keys()) | \
                        set(critic_loss_by_epoch.keys()) | \
                        set(train_rewards_by_epoch.keys())
    
    for m_name in all_train_metrics_keys:
        all_epoch_indices |= set(train_metrics_by_epoch[m_name].keys())

    if not all_epoch_indices:
        return {"per_epoch": [], "best_train_metrics": {}}

    num_epochs = max(all_epoch_indices) + 1
    per_epoch_stats = []
    
    def calculate_stats(vals):
        if not vals:
            return {"max": 0, "min": 0, "avg": 0}
        return {
            "max": float(max(vals)),
            "min": float(min(vals)),
            "avg": float(sum(vals) / len(vals))
        }
            
    for i in range(num_epochs):
        epoch_stats = {
            "epoch": i,
            "actor_loss": calculate_stats(actor_loss_by_epoch.get(i, [])),
            "critic_loss": calculate_stats(critic_loss_by_epoch.get(i, [])),
            "train_rewards": calculate_stats(train_rewards_by_epoch.get(i, [])),
            "train_metrics": {
                m_name: calculate_stats(train_metrics_by_epoch.get(m_name, {}).get(i, []))
                for m_name in all_train_metrics_keys
            }
        }
        per_epoch_stats.append(epoch_stats)
        
    # Mejores train metrics (Best global value observed for each metric)
    best_train_metrics = {}
    for m_name in all_train_metrics_keys:
        all_vals = []
        for e_idx in train_metrics_by_epoch[m_name]:
            all_vals.extend(train_metrics_by_epoch[m_name][e_idx])
        if all_vals:
            # Para la mayoría de las métricas de recomendación el mejor es el max
            best_train_metrics[m_name] = float(max(all_vals))

    return {
        "per_epoch": per_epoch_stats,
        "best_train_metrics": best_train_metrics
    }

# 4. Dashboard Summary
@app.get("/dashboard/summary")
def get_dashboard_summary(session_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Datos para el dashboard. Si no se pasa session_id, usa la última activa."""
    
    if session_id:
        target_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    else:
        target_session = db.query(TrainingSession).filter(TrainingSession.status == "ACTIVE").order_by(TrainingSession.id.desc()).first()
        # Si no hay activa, tomar la última
        if not target_session:
             target_session = db.query(TrainingSession).order_by(TrainingSession.id.desc()).first()
    
    if not target_session:
        return {"session": None, "message": "No hay sesiones disponibles"}
    
    # Métricas globales de la sesión
    global_metrics = db.query(GlobalRoundMetric).filter(GlobalRoundMetric.session_id == target_session.id).order_by(GlobalRoundMetric.round_number).all()
    
    # Clientes participantes en esta sesión
    clients_in_session = target_session.clients
    
    # Calcular estado actual (resumido de clientes en esta sesión)
    active_now = 0
    # Consideramos 'activo' si ha dado heartbeat en los últimos 2 min Y está asociado a la sesión
    timeout = datetime.utcnow().timestamp() - 120 
    
    client_status_map = {}
    for c in clients_in_session:
        client_status_map[c.user_id] = c.current_status
        if c.last_seen and c.last_seen.timestamp() > timeout:
            active_now += 1
            
    return {
        "session": {
            "id": target_session.id,
            "status": target_session.status,
            "start_time": target_session.start_time,
            "total_rounds": target_session.total_rounds,
            "current_round": global_metrics[-1].round_number if global_metrics else 0
        },
        "stats": {
            "total_clients": len(clients_in_session),
            "active_now": active_now
        },
        "clients_status": client_status_map,
        "evolution": [
            {
                "round": m.round_number, 
                "avg_reward": m.metrics.get("avg_val_reward", 0),
                "avg_loss": m.metrics.get("avg_train_loss", 0),
                "metrics": m.metrics
            } for m in global_metrics
        ]
    }
