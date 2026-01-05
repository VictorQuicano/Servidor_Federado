from fastapi import FastAPI, HTTPException
from user_distro_manager import UserDistributionManager
import os

app = FastAPI(title="User Distribution Service")

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_SUMMARY_PATH = os.path.join(BASE_DIR, "user_summary.json")

# Inicializar el manager como Singleton para mantener el estado
distro_manager = UserDistributionManager(
    user_summary_path=USER_SUMMARY_PATH,
    n_grupos=5,
    n_rondas=3
)

@app.get("/get_user")
async def get_user():
    """
    Entrega el siguiente user_id disponible según la lógica de distribución.
    """
    user_id = distro_manager.get_next_user()
    if not user_id:
        raise HTTPException(status_code=404, detail="No hay más usuarios disponibles en las rondas configuradas.")
    
    return {"user_id": user_id}

@app.get("/summary")
async def get_summary():
    """
    Muestra el resumen de la distribución actual por grupos y rondas.
    """
    return distro_manager.get_distribution_summary()

@app.post("/reset")
async def reset_distribution():
    """
    Reinicia la distribución de usuarios.
    """
    distro_manager.reset_distribution()
    return {"message": "Distribución reiniciada correctamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
