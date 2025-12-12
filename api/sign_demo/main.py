"""FastAPI application for the 55-class (demo) sign recognition model.

Run with:
    uvicorn api.sign_demo.main:app --reload --port 8001
"""

from fastapi import FastAPI

from api.common import health

from .routers import router as sign_router

app = FastAPI(
    title="SignBridge API (55-class Demo Model)",
    version="1.0.0",
    description="Sign language recognition API using the 55-class demo I3D model.",
)


@app.get("/")
def read_root():
    return {
        "message": "SignBridge API (55-class Demo Model) is running",
        "model": "55-class demo",
        "docs_url": "/docs",
    }


app.include_router(health.router)
app.include_router(sign_router)
