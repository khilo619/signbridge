"""FastAPI application for the 100-class (full) sign recognition model.

Run with:
    uvicorn api.sign_full.main:app --reload --port 8000
"""

from fastapi import FastAPI

from api.common import health

from .routers import router as sign_router

app = FastAPI(
    title="SignBridge API (100-class Full Model)",
    version="1.0.0",
    description="Sign language recognition API using the 100-class I3D model.",
)


@app.get("/")
def read_root():
    return {
        "message": "SignBridge API (100-class Full Model) is running",
        "model": "100-class",
        "docs_url": "/docs",
    }


app.include_router(health.router)
app.include_router(sign_router)
