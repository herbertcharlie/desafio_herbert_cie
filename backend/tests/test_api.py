import pytest

from app.domain.errors import (
    ConfigurationError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
)
from tests.conftest import make_pdf
from tests.fakes import make_chunk


def upload(client, *files):
    return client.post(
        "/documents",
        files=[("files", (name, data, "application/pdf")) for name, data in files],
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_pdf_returns_created_document(client):
    response = upload(client, ("scrum.pdf", make_pdf(["Texto sobre Scrum y sus eventos."])))
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["status"] == "created"
    assert result["document"]["filename"] == "scrum.pdf"
    assert client.get("/documents").json()[0]["filename"] == "scrum.pdf"


def test_upload_same_pdf_twice_reports_duplicate(client):
    pdf = make_pdf(["Contenido idéntico en ambas cargas."])
    upload(client, ("a.pdf", pdf))
    assert upload(client, ("a.pdf", pdf)).json()["results"][0]["status"] == "duplicate"


def test_upload_non_pdf_is_rejected_with_consistent_error(client):
    response = upload(client, ("notas.txt", b"hola, no soy un pdf"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_document"


def test_upload_too_large_returns_413(client):
    big = b"%PDF-1.4" + b"0" * (2 * 1024 * 1024)
    response = upload(client, ("grande.pdf", big))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_batch_with_one_bad_file_reports_per_file_status(client):
    good = make_pdf(["Documento válido con texto suficiente."])
    response = upload(client, ("ok.pdf", good), ("mal.txt", b"nada"))
    assert response.status_code == 200
    statuses = {r["filename"]: r["status"] for r in response.json()["results"]}
    assert statuses == {"ok.pdf": "created", "mal.txt": "failed"}


def test_delete_document_and_not_found(client):
    upload(client, ("a.pdf", make_pdf(["Documento para borrar luego."])))
    doc_id = client.get("/documents").json()[0]["id"]
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    missing = client.delete(f"/documents/{doc_id}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "document_not_found"


def test_ask_returns_answer_sources_and_meta(client, store):
    store.vector_hits = [make_chunk("El Sprint dura un mes o menos.", page=5)]

    response = client.post("/ask", json={"question": "¿Cuánto dura un Sprint?", "session_id": "s1"})

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["sources"][0]["filename"] == "guia.pdf"
    assert body["sources"][0]["page_start"] == 5
    assert body["meta"]["model"] == "fake-llm"


def test_ask_without_evidence_says_so_explicitly(client, store):
    store.vector_hits = [make_chunk(similarity=0.02)]
    body = client.post(
        "/ask", json={"question": "¿Quién ganó el mundial?", "session_id": "s1"}
    ).json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert "No encontré información suficiente" in body["answer"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "session_id": "s1"},
        {"question": "   ", "session_id": "s1"},
        {"question": "hola"},
        {"question": "hola", "session_id": "con espacios!"},
        {"question": "x" * 1001, "session_id": "s1"},
    ],
)
def test_ask_validates_input(client, payload):
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_history_returns_session_messages_in_order(client, store):
    store.vector_hits = [make_chunk()]
    client.post("/ask", json={"question": "primera", "session_id": "abc"})
    client.post("/ask", json={"question": "segunda", "session_id": "abc"})
    client.post("/ask", json={"question": "otra sesión", "session_id": "zzz"})

    body = client.get("/history/abc").json()

    assert body["session_id"] == "abc"
    assert [m["question"] for m in body["messages"]] == ["primera", "segunda"]
    assert body["messages"][0]["sources"][0]["filename"] == "guia.pdf"


def test_history_of_unknown_session_is_empty(client):
    assert client.get("/history/nada").json()["messages"] == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (UpstreamTimeoutError("t"), 504, "upstream_timeout"),
        (UpstreamUnavailableError("u"), 503, "upstream_unavailable"),
        (ConfigurationError("c"), 503, "configuration_error"),
    ],
)
def test_upstream_failures_map_to_http_errors(client, services, error, status, code):
    async def boom(*_a, **_k):
        raise error

    services.rag.ask = boom
    response = client.post("/ask", json={"question": "hola", "session_id": "s1"})
    assert response.status_code == status
    assert response.json()["error"]["code"] == code


def test_unexpected_exception_returns_generic_500(services, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    async def boom(*_a, **_k):
        raise RuntimeError("secreto interno")

    services.rag.ask = boom
    with TestClient(
        create_app(services=services, settings=settings), raise_server_exceptions=False
    ) as c:
        response = c.post("/ask", json={"question": "hola", "session_id": "s1"})
    assert response.status_code == 500
    assert "secreto" not in response.text
