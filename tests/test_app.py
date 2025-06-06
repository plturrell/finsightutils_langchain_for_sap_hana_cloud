"""
Simple FastAPI application optimized for health checks
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "UP"}

@app.get("/health")
def health():
    return {"status": "UP"}

@app.get("/healthz")
def healthz():
    return {"status": "UP"}

@app.get("/live")
def live():
    return {"status": "UP"}

@app.get("/ready")
def ready():
    return {"status": "UP"}

@app.get("/api/health")
def api_health():
    return {"status": "UP"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)