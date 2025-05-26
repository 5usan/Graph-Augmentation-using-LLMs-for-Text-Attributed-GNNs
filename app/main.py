from fastapi import FastAPI
from app.api.health_check_api import router as health_router
from app.api.data_preprocessed_api import router as data_preprocess_router
from app.api.graph_api import router as graph_router

app = FastAPI(title="Graph And LLM API")

app.include_router(health_router, tags=["Health Check"])
app.include_router(data_preprocess_router, tags=["Data Preprocessing"])
app.include_router(graph_router, tags=["Graph Related"])
