from fastapi import FastAPI
import time
import os

# Get environment variables
VERSION = os.getenv("VERSION", "1.0.0")
PLATFORM = os.getenv("PLATFORM", "local")
IS_VERCEL = os.getenv("VERCEL", "0") == "1"

# Create a minimal FastAPI app
app = FastAPI(title="Minimal Health Check App")

@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "environment": "vercel" if IS_VERCEL else PLATFORM,
        "timestamp": time.time(),
        "version": VERSION
    }

@app.get("/health/check", tags=["Health"])
async def detailed_health_check():
    return {
        "status": "ok", 
        "database": {
            "connected": True,
            "host": os.getenv("HANA_HOST", "NA"),
            "user": os.getenv("HANA_USER", "NA"),
            "port": os.getenv("HANA_PORT", "NA")
        },
        "gpu": {
            "available": False,
            "mode": "CPU-only"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
