from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import enum

from sqlalchemy import Table

Base = declarative_base()

# Tabla de asociación Many-to-Many entre Sesiones y Clientes
session_clients = Table('session_clients', Base.metadata,
    Column('session_id', Integer, ForeignKey('training_sessions.id')),
    Column('client_id', String, ForeignKey('clients.user_id'))
)

class ClientStatus(str, enum.Enum):
    IDLE = "IDLE"
    LOADING_DATA = "LOADING_DATA"
    TRAINING = "TRAINING"
    EVALUATING = "EVALUATING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, default="ACTIVE") # ACTIVE, COMPLETED, FAILED
    total_rounds = Column(Integer, default=0)
    
    # Relaciones
    global_metrics = relationship("GlobalRoundMetric", back_populates="session")
    client_metrics = relationship("ClientRoundMetric", back_populates="session")
    clients = relationship("Client", secondary=session_clients, back_populates="sessions")

class Client(Base):
    __tablename__ = "clients"
    
    user_id = Column(String, primary_key=True, index=True) # "user_123"
    ip_address = Column(String, nullable=True)
    current_status = Column(String, default=ClientStatus.IDLE)
    last_seen = Column(DateTime, default=datetime.datetime.utcnow)
    current_round = Column(Integer, default=0)
    
    # Metadatos opcionales
    system_info = Column(JSON, nullable=True) # CPU, RAM, etc
    
    sessions = relationship("TrainingSession", secondary=session_clients, back_populates="clients")

class GlobalRoundMetric(Base):
    __tablename__ = "global_round_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"))
    round_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Métricas agregadas
    metrics = Column(JSON) # {"avg_val_reward": 0.5, ...}
    
    session = relationship("TrainingSession", back_populates="global_metrics")

class ClientRoundMetric(Base):
    __tablename__ = "client_round_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"))
    user_id = Column(String, ForeignKey("clients.user_id"))
    round_number = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Métricas locales
    metrics = Column(JSON) # {"val_reward": 0.4, "train_loss": ...}
    
    session = relationship("TrainingSession", back_populates="client_metrics")

# Configuración de base de datos
SQLALCHEMY_DATABASE_URL = "sqlite:///./monitoring.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
