from fastapi import FastAPI

from api.routes import health, query

app = FastAPI(title="Healthcare RAG")

app.include_router(query.router)
app.include_router(health.router)
