from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR
from .routers import data, eda, models, predict
from .service import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()  # build pipeline + train models once at boot
    yield


app = FastAPI(
    title="Smart Grid ML API",
    description="EDA + preprocessing pipeline + four ML models (Logistic, Ridge, Random Forest, XGBoost).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(eda.router)
app.include_router(models.router)
app.include_router(predict.router)


@app.get("/api/health")
def health():
    proj = service.project()
    return {
        "status": "ok",
        "source": "csv",
        "shape": list(proj.raw_df.shape),
        "models": list(proj.models.keys()),
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")