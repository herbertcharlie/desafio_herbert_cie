"""Hace cumplir la regla de dependencias de la arquitectura hexagonal.

    api ──► services ──► domain ◄── infra        (las flechas indican "importa")

El núcleo (`domain` y `services`) no puede conocer frameworks ni proveedores: si alguien importa
OpenAI, SQLAlchemy o FastAPI desde la lógica de negocio, esta prueba falla.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app"

# Paquetes de terceros/infraestructura que el núcleo no debe importar.
INFRASTRUCTURE = {"fastapi", "starlette", "openai", "sqlalchemy", "asyncpg", "pypdf", "pydantic"}

# capa -> capas de `app` que SÍ puede importar.
ALLOWED_INTERNAL = {
    "domain": {"domain"},
    "services": {"domain", "services"},
    "infra": {"domain", "core", "infra"},
    "api": {"domain", "services", "api", "container", "core"},
}
# Las capas del núcleo además no pueden usar infraestructura de terceros.
CORE_LAYERS = {"domain", "services"}


def imports_of(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def files_of(layer: str) -> list[Path]:
    return sorted((APP / layer).rglob("*.py"))


@pytest.mark.parametrize("layer", ALLOWED_INTERNAL)
def test_layers_only_import_allowed_internal_layers(layer):
    for path in files_of(layer):
        for module in imports_of(path):
            parts = module.split(".")
            if parts[0] == "app" and len(parts) > 1:
                assert parts[1] in ALLOWED_INTERNAL[layer], (
                    f"{path.relative_to(APP)} importa app.{parts[1]}, "
                    f"prohibido para la capa '{layer}'"
                )


@pytest.mark.parametrize("layer", sorted(CORE_LAYERS))
def test_core_layers_do_not_import_infrastructure_packages(layer):
    for path in files_of(layer):
        offending = {m.split(".")[0] for m in imports_of(path)} & INFRASTRUCTURE
        assert not offending, f"{path.relative_to(APP)} importa {sorted(offending)}"


def test_every_layer_has_python_files():
    assert all(files_of(layer) for layer in ALLOWED_INTERNAL)
