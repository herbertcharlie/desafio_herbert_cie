import asyncio

import pytest

from app.domain.errors import LLMResponseError
from app.services.prompts import NO_INFO_MESSAGE
from app.services.rag import (
    RAGConfig,
    RAGService,
    parse_llm_answer,
    reciprocal_rank_fusion,
    select_context,
)
from tests.fakes import FakeEmbedder, FakeHistory, FakeLLM, FakeStore, make_chunk


def build(store: FakeStore, llm: FakeLLM, **config) -> tuple[RAGService, FakeHistory]:
    history = FakeHistory()
    service = RAGService(
        embedder=FakeEmbedder(),
        store=store,
        llm=llm,
        history=history,
        config=RAGConfig(**{"min_similarity": 0.3, "context_top_k": 3, **config}),
    )
    return service, history


async def test_grounded_answer_returns_only_cited_sources():
    chunks = [make_chunk(f"contenido {i}", page=i) for i in range(1, 4)]
    llm = FakeLLM(answer="Sí [1][3].", citations=[1, 3])
    service, _ = build(FakeStore(vector_hits=chunks), llm)

    result = await service.ask("¿Qué es X?", "s1")

    assert result.grounded is True
    assert result.answer == "Sí [1][3]."
    assert [s.ref for s in result.sources] == [1, 3]
    assert [s.page_start for s in result.sources] == [1, 3]
    assert result.meta.model == "fake-llm"


async def test_context_sent_to_llm_is_numbered_with_document_and_page():
    service, _ = build(FakeStore(vector_hits=[make_chunk("hola", page=7)]), llm := FakeLLM())
    await service.ask("pregunta", "s1")
    user_prompt = llm.calls[0]["user"]
    assert "[1] (Documento: guia.pdf, pág. 7)" in user_prompt
    assert llm.calls[0]["json_mode"] is True


async def test_low_similarity_refuses_without_calling_llm():
    llm = FakeLLM()
    service, _ = build(FakeStore(vector_hits=[make_chunk(similarity=0.1)]), llm)

    result = await service.ask("¿Cuál es la capital de Francia?", "s1")

    assert result.grounded is False
    assert result.answer == NO_INFO_MESSAGE
    assert result.sources == []
    assert llm.calls == []


async def test_empty_knowledge_base_refuses_without_calling_llm():
    llm = FakeLLM()
    service, _ = build(FakeStore(), llm)
    result = await service.ask("hola", "s1")
    assert result.grounded is False
    assert llm.calls == []


async def test_llm_declaring_insufficient_info_is_a_refusal():
    llm = FakeLLM(sufficient=False, answer="", citations=[])
    service, _ = build(FakeStore(vector_hits=[make_chunk()]), llm)
    result = await service.ask("pregunta", "s1")
    assert result.grounded is False
    assert result.answer == NO_INFO_MESSAGE
    assert result.sources == []


async def test_citations_outside_context_are_discarded_and_cause_refusal():
    llm = FakeLLM(answer="Algo [9].", citations=[9, 0])
    service, _ = build(FakeStore(vector_hits=[make_chunk()]), llm)
    result = await service.ask("pregunta", "s1")
    assert result.grounded is False


async def test_answer_without_any_citation_is_not_accepted_as_grounded():
    llm = FakeLLM(answer="Respuesta sin citas.", citations=[])
    service, _ = build(FakeStore(vector_hits=[make_chunk()]), llm)
    result = await service.ask("pregunta", "s1")
    assert result.grounded is False


async def test_invalid_json_from_llm_raises_llm_response_error():
    service, _ = build(FakeStore(vector_hits=[make_chunk()]), FakeLLM(raw="no es json"))
    with pytest.raises(LLMResponseError):
        await service.ask("pregunta", "s1")


async def test_every_question_is_persisted_in_history_including_refusals():
    store = FakeStore(vector_hits=[make_chunk(similarity=0.05)])
    service, history = build(store, FakeLLM())
    await service.ask("pregunta 1", "sesion-a")
    store.vector_hits = [make_chunk(page=4)]
    await service.ask("pregunta 2", "sesion-a")

    assert [m.question for m in history.messages] == ["pregunta 1", "pregunta 2"]
    assert [m.grounded for m in history.messages] == [False, True]
    assert history.messages[1].sources[0]["page_start"] == 4


async def test_hybrid_search_can_be_disabled():
    store = FakeStore(vector_hits=[make_chunk()])
    service, _ = build(store, FakeLLM(), hybrid=False)
    await service.ask("pregunta", "s1")
    assert store.text_searches == 0


async def test_hybrid_search_adds_lexical_only_hits_to_context():
    vector = [make_chunk("vectorial", similarity=0.7)]
    lexical = [make_chunk("solo léxico", similarity=None)]
    llm = FakeLLM(citations=[1, 2])
    service, _ = build(FakeStore(vector_hits=vector, text_hits=lexical), llm)
    await service.ask("pregunta", "s1")
    assert "solo léxico" in llm.calls[0]["user"]


async def test_llm_concurrency_is_bounded_by_semaphore():
    llm = FakeLLM(delay=0.02)
    service, _ = build(FakeStore(vector_hits=[make_chunk()]), llm, max_llm_concurrency=2)
    await asyncio.gather(*(service.ask(f"pregunta {i}", "s1") for i in range(8)))
    assert len(llm.calls) == 8
    assert llm.max_active == 2


def test_noisy_lexical_hits_cannot_push_top_vector_results_out_of_context():
    """Regresión: un léxico poco selectivo daba doble voto a fragmentos mediocres (q02)."""
    vector = [make_chunk(f"v{i}", similarity=0.8 - i * 0.01) for i in range(10)]
    lexical = [vector[i] for i in (2, 3, 4, 5, 6)]  # solapa con los puestos 3-7 del vectorial
    # Sin ancla, RRF deja fuera al mejor y segundo resultado vectorial.
    unanchored = select_context(vector, lexical, top_k=5, anchor=0)
    assert vector[0].id not in {c.id for c in unanchored}
    anchored = select_context(vector, lexical, top_k=5, anchor=2)
    assert [c.id for c in anchored[:2]] == [vector[0].id, vector[1].id]
    assert len(anchored) == 5 and len({c.id for c in anchored}) == 5


def test_select_context_without_lexical_hits_is_the_vector_ranking():
    vector = [make_chunk(f"v{i}") for i in range(6)]
    assert select_context(vector, [], top_k=4, anchor=2) == vector[:4]


def test_anchor_is_capped_by_top_k():
    vector = [make_chunk(f"v{i}") for i in range(6)]
    assert len(select_context(vector, [], top_k=2, anchor=5)) == 2


def test_rrf_prefers_items_ranked_high_in_both_lists():
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    fused = reciprocal_rank_fusion([a, b], [c, b])
    assert fused[0].id == b.id
    assert {x.id for x in fused} == {a.id, b.id, c.id}


def test_rrf_keeps_vector_similarity_when_item_also_found_lexically():
    vec = make_chunk("a", similarity=0.6)
    lex = type(vec)(**{**vec.__dict__, "similarity": None})
    fused = reciprocal_rank_fusion([lex], [vec])
    assert fused[0].similarity == 0.6


def test_parse_accepts_well_formed_json():
    parsed = parse_llm_answer('{"sufficient": true, "answer": "ok [1]", "citations": [1]}')
    assert parsed.sufficient and parsed.answer == "ok [1]" and parsed.citations == [1]


@pytest.mark.parametrize(
    "raw", ["", "[]", '{"answer": "x"}', '{"sufficient": true, "citations": ["a"]}']
)
def test_parse_rejects_malformed_payloads(raw):
    with pytest.raises(LLMResponseError):
        parse_llm_answer(raw)
