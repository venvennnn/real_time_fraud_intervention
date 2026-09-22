"""FastAPI application exposing the real-time fraud intervention service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .engine import InterventionEngine
from .model import FraudModel
from .schemas import (
    DecisionStats,
    HealthResponse,
    InterventionResult,
    Transaction,
)


def create_app() -> FastAPI:
    state: dict[str, InterventionEngine] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model = FraudModel.train()
        state["engine"] = InterventionEngine(model)
        yield
        state.clear()

    app = FastAPI(
        title="Real-Time Fraud Intervention",
        version=__version__,
        lifespan=lifespan,
    )

    def engine() -> InterventionEngine:
        return state["engine"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_ready="engine" in state,
            version=__version__,
        )

    @app.post("/score", response_model=InterventionResult)
    def score(txn: Transaction) -> InterventionResult:
        return engine().score(txn)

    @app.get("/stats", response_model=DecisionStats)
    def stats() -> DecisionStats:
        return engine().stats

    return app


app = create_app()
