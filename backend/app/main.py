"""Punto de entrada FastAPI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handlers
from app.api.routes import router
from app.container import Services, build_services
from app.core.config import Settings, get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    """Crea la app. En tests se inyectan `services` con fakes (sin red ni base de datos)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        owned = services is None
        app.state.services = services or build_services(settings or get_settings())
        yield
        if owned:
            await app.state.services.aclose()

    app = FastAPI(
        title="Enterprise RAG Assistant",
        description="Asistente RAG sobre documentos PDF con fuentes y grounding verificable.",
        version="1.0.0",
        lifespan=lifespan,
    )
    cors = (settings or get_settings()).cors_origins
    app.add_middleware(CORSMiddleware, allow_origins=cors, allow_methods=["*"], allow_headers=["*"])
    register_error_handlers(app)
    app.include_router(router)
    return app


app = create_app()
