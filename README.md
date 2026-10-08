# Enterprise RAG Assistant

Asistente RAG (Retrieval-Augmented Generation) que responde preguntas sobre documentos PDF,
**cita las fuentes (documento y página)** y **se niega a responder cuando los documentos no
contienen la información**. Desafío técnico AI Engineer — CIE LAB.

- Backend: **Python 3.11 + FastAPI**
- Base de datos y búsqueda vectorial: **PostgreSQL + pgvector** (con búsqueda híbrida léxica)
- LLM y embeddings: **OpenAI** (`gpt-4o-mini`, `text-embedding-3-small`)
- Interfaz: HTML/JS mínimo servido por nginx, más Swagger en `/docs`
- Todo se ejecuta con **Docker Compose**

> Las decisiones técnicas, alternativas y limitaciones están en [DECISION_LOG.md](DECISION_LOG.md).

## Documentos utilizados

Tema: **agilidad** (Scrum y Kanban). Ambos son de acceso público, en español, y están en
[`data/pdfs/`](data/pdfs/).

| Archivo | Título | Fuente | Páginas | Descripción |
|---|---|---|---|---|
| `scrum-guide-2020-es.pdf` | *La Guía de Scrum* (versión 2020, español LATAM) | [scrumguides.org](https://www.scrumguides.org/docs/scrumguide/v2020/2020-Scrum-Guide-Spanish-Latin-South-American.pdf) · Ken Schwaber y Jeff Sutherland | 16 | Definición de Scrum: teoría, valores, roles, eventos y artefactos. Licencia CC BY-SA 4.0. |
| `kanban-guide-2020-es.pdf` | *Guía Kanban* (diciembre 2020, español) | [kanbanguides.org](https://kanbanguides.org/the-kanban-guide/translations/) · Orderly Disruption Ltd. y Daniel S. Vacanti, Inc. | 9 | Definición de Kanban: prácticas, DoW, WIP, SLE y métricas de flujo. El propio PDF declara CC BY-SA 4.0 (pág. 1) y CC BY 4.0 (pág. 8); ambas permiten este uso con atribución. |

Total: 25 páginas. Las "páginas" que cita la API son las del **PDF físico** (la numeración
impresa de la Guía Scrum va desplazada en 1).

## Requisitos

- Docker Desktop (Compose v2)
- Una clave de API de OpenAI **con crédito** y con acceso a los modelos `gpt-4o-mini` y
  `text-embedding-3-small` (los proyectos de OpenAI pueden restringir modelos; ver
  "Solución de problemas")
- Para desarrollo y pruebas locales: Python 3.11

## Configuración y ejecución

```bash
# 1. Variables de entorno (la clave NUNCA se versiona: .env está en .gitignore)
cp .env.example .env
#    edita .env y completa OPENAI_API_KEY

# 2. Levantar todo (la primera vez construye las imágenes)
docker compose up -d --build

# 3. Cargar los PDFs de ejemplo (o súbelos desde la interfaz)
python scripts/ingest_pdfs.py --base-url http://localhost:8000
```

| Servicio | URL |
|---|---|
| Interfaz web | http://localhost:8080 |
| API + Swagger | http://localhost:8000/docs |
| Health | http://localhost:8000/health |

Para empezar de cero (borra documentos e historial): `docker compose down -v`.

### Variables de entorno

Todas tienen valor por defecto salvo `OPENAI_API_KEY`. Ver [.env.example](.env.example).

| Variable | Defecto | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | — | **Obligatoria.** Sin ella la API arranca pero `/ask` y `/documents` responden 503 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Modelo de generación |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Modelo de embeddings (1536 dim; si cambias la dimensión, ajusta `vector(N)` en `db/init/001_schema.sql` y `EMBEDDING_DIM`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Caracteres por chunk y solape |
| `RETRIEVAL_CANDIDATES` | `10` | Candidatos por método (vectorial y léxico) |
| `CONTEXT_TOP_K` | `8` | Fragmentos enviados al LLM |
| `HYBRID_SEARCH` | `true` | Combina búsqueda vectorial y léxica (RRF) |
| `VECTOR_ANCHOR` | `2` | Mejores resultados vectoriales que siempre entran al contexto (ver DECISION_LOG §4) |
| `MIN_SIMILARITY` | `0.30` | Umbral de similitud coseno; por debajo no se llama al LLM |
| `LLM_MAX_CONCURRENCY` | `4` | Llamadas simultáneas al LLM por proceso |
| `LLM_TIMEOUT_S` | `30` | Timeout hacia OpenAI |
| `POSTGRES_USER/PASSWORD/DB` | `rag` | Credenciales de la base de datos |

## Arquitectura

```
Navegador ──► nginx :8080 ──/api/──► FastAPI :8000 ──► PostgreSQL + pgvector
                                         │
                                         └──► OpenAI (embeddings + LLM)
```

```
backend/app/
  api/        routers, esquemas Pydantic, manejo de errores
  services/   ingesta, RAG, chunking, prompts          (lógica de aplicación)
  domain/     modelos, errores y puertos (Protocols)
  infra/      OpenAI, Postgres/pgvector, pypdf         (implementaciones)
  container.py   composition root
db/init/      esquema SQL (se ejecuta al crear el volumen)
frontend/     index.html + nginx
eval/         dataset de evaluación + evaluador
scripts/      ingesta por línea de comandos
data/pdfs/    documentos de ejemplo
```

La lógica de negocio depende solo de interfaces (`Embedder`, `LLMProvider`, `KnowledgeStore`,
`ChatHistory`), no de OpenAI ni de Postgres.

### Flujo

**Ingesta** (`POST /documents`): validar PDF → extraer texto por página → chunks por oraciones
(~1000 caracteres, solape 150, con rango de páginas) → embeddings por lotes → guardar documento
y chunks en una transacción. El mismo archivo (SHA-256) no se procesa dos veces.

**Consulta** (`POST /ask`): embedding de la pregunta → top-10 vectorial + top-10 léxico →
fusión RRF anclada en el vectorial → 8 fragmentos → **puerta de relevancia** (si la similitud máxima < umbral, rechazo sin
llamar al LLM) → LLM con salida JSON `{sufficient, answer, citations}` → **verificación de
las citas contra el contexto enviado** → respuesta con fuentes. Se guarda en el historial.

### Trazabilidad

Cada respuesta incluye `sources`: `ref` (el `[n]` del texto), `filename`, `page_start`,
`page_end`, `chunk_id`, `similarity` y un `snippet`. Solo se devuelven las fuentes realmente
citadas y válidas. Detalle en [DECISION_LOG.md](DECISION_LOG.md) §7.

## API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/documents` | Sube uno o más PDF (`multipart`, campo `files`) |
| `GET` | `/documents` | Lista los documentos |
| `DELETE` | `/documents/{id}` | Borra un documento y sus chunks |
| `POST` | `/ask` | `{"question": "...", "session_id": "demo-1"}` |
| `GET` | `/history/{session_id}` | Historial de la sesión (persistido en PostgreSQL) |
| `GET` | `/health` | Estado de la API y la BD |

Ejemplo:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "¿Cuánto dura la Daily Scrum?", "session_id": "demo-1"}'
```

```jsonc
{
  "answer": "La Daily Scrum es un evento de 15 minutos para los Developers... [1]",
  "grounded": true,
  "sources": [{"ref": 1, "filename": "scrum-guide-2020-es.pdf", "page_start": 10, "page_end": 10, "similarity": 0.61, "snippet": "..."}],
  "meta": {"model": "gpt-4o-mini", "retrieval_ms": 310.2, "generation_ms": 1200.5, "total_ms": 1512.0, "prompt_tokens": 1450, "completion_tokens": 60}
}
```

Si los documentos no contienen la respuesta: `grounded: false`, `sources: []` y un mensaje que lo
dice explícitamente. Todos los errores usan `{"error": {"code": "...", "message": "..."}}`.

| Código | Cuándo |
|---|---|
| 413 `upload_too_large` | PDF mayor a 25 MB |
| 422 `invalid_document` / `no_text_extracted` / `validation_error` | No es PDF válido, PDF escaneado sin texto, entrada inválida |
| 503 `configuration_error` | Falta/errónea la clave, sin crédito o sin acceso al modelo |
| 503 `upstream_unavailable` / 504 `upstream_timeout` | Fallo temporal de OpenAI |
| 502 `invalid_llm_response` | El modelo devolvió un formato inválido |

## Pruebas y calidad de código

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements-dev.txt

pytest                                  # unitarias + API (con dobles, sin red ni BD)
pytest ../eval                          # lógica del evaluador + validación del dataset vs. los PDFs

ruff check . ../eval ../scripts         # lint
black --check .  &&  isort --check .   # formato
```

**Integración contra Postgres real** (con el stack levantado; se omiten si no hay
`TEST_DATABASE_URL`):

```bash
docker compose run --rm -v "$PWD/backend/tests:/srv/tests" -v "$PWD/backend/pyproject.toml:/srv/pyproject.toml" \
  -e TEST_DATABASE_URL=postgresql+asyncpg://rag:rag@db:5432/rag backend \
  sh -c "pip install -q pytest pytest-asyncio && python -m pytest tests/test_pg_integration.py"
```

Cobertura: chunking (tamaños, solape, páginas, ligaduras), pipeline RAG (rechazo por umbral,
verificación de citas, JSON inválido, historial, RRF, límite de concurrencia), ingesta
(duplicados, PDF sin texto), API (validaciones, códigos de error, historial), traducción de
errores de OpenAI y SQL real (búsqueda vectorial, léxica, cascada, JSONB).

## Evaluación del sistema

`eval/dataset.json`: **18 preguntas** sobre ambos documentos.

| Tipo | Cantidad | Qué mide |
|---|---|---|
| `direct` | 9 | La respuesta está en un fragmento |
| `multi_chunk` | 4 | Requiere combinar fragmentos (2 de ellas, ambos documentos) |
| `unanswerable` | 5 | No está en los documentos: debe rechazar. Incluye casos difíciles (cercanos al tema) |

Cada pregunta trae `expected_answer`, `expected_sources` (documento + páginas) y palabras
clave. Las páginas y palabras clave del dataset se **validan automáticamente contra los PDFs**
(`pytest ../eval`).

```bash
python eval/run_eval.py --base-url http://localhost:8000
```

Métricas: *source hit* (la fuente citada coincide con documento y página esperados), *keyword
recall* (la respuesta contiene los datos esperados) y *rechazo correcto* en las preguntas sin
respuesta. El detalle se guarda en `eval/results/` (ignorado por git).

### Resultados

Configuración por defecto, `gpt-4o-mini` con `temperature=0`. Medido sobre una **copia limpia
clonada desde GitHub** (ingesta desde cero), **18/18 en 5 de 5 corridas**:

| Métrica | Resultado |
|---|---|
| Preguntas aprobadas | **18 / 18** |
| Fuente correcta (preguntas con respuesta) | 100 % |
| Cobertura de datos esperados en la respuesta (*keyword recall*) | 0.95 |
| Rechazo correcto (preguntas sin respuesta) | 100 % (5/5) |
| Latencia media por pregunta | ≈ 1.6 s |

**Cómo leer estos números con honestidad:**
- El dataset es pequeño (18 preguntas) y se usó también para ajustar la configuración, por lo
  que **no demuestra generalización**; sí demuestra que el pipeline funciona de extremo a
  extremo y que las decisiones se midieron.
- Los resultados **dependen de la ingesta**: con una configuración anterior (`top_k=6`) se
  obtuvo 18/18, pero al re-ingestar desde cero bajó a 17/18 de forma estable. Eso llevó a
  subir el contexto a 8 fragmentos. El episodio completo (ablation, defecto de recuperación,
  corrección de una etiqueta del dataset y calibración del umbral) está en
  [DECISION_LOG.md](DECISION_LOG.md) §4 y §5.
- El *keyword recall* es una métrica burda: busca palabras clave, no evalúa la redacción.

## Solución de problemas

- **503 `configuration_error` — "no tiene crédito"**: la cuenta de OpenAI no tiene saldo.
- **503 `configuration_error` — "OpenAI denegó el acceso… does not have access to model"**: el
  *proyecto* de la clave tiene restringidos los modelos. En platform.openai.com → Project →
  Limits, habilita `text-embedding-3-small` y `gpt-4o-mini`, o usa una clave de otro proyecto.
- **El backend no arranca tras cambiar el esquema SQL**: el esquema solo se aplica al crear el
  volumen; usa `docker compose down -v`.
- **PDF "sin texto extraíble"**: es una imagen escaneada; este prototipo no hace OCR.

## Uso de herramientas de IA

Este proyecto se desarrolló con asistencia de Claude Code. El código fue revisado y las
decisiones están justificadas en [DECISION_LOG.md](DECISION_LOG.md).
