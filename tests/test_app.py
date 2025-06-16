"""
Simple FastAPI application to test the health of the API in Docker.
"""
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_app")

# Create the FastAPI application
app = FastAPI(
    title="SAP HANA Cloud LangChain Test API",
    description="Simple API to test Docker health",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: Basic API information
    """
    return {
        "status": "ok",
        "message": "SAP HANA Cloud LangChain Test API is running",
        "version": "1.0.0",
        "mode": "secure-docker",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Basic health check endpoint.
    """
    return {
        "status": "ok",
        "message": "API is healthy"
    }

@app.get("/ping", tags=["Health"])
async def ping():
    """
    Simple ping endpoint to verify API is running.
    """
    return {"ping": "pong"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
