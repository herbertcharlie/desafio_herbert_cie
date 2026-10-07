"""Pruebas de la lógica de métricas del evaluador (no requieren API)."""

from run_eval import fold, keyword_recall, source_hit, summarize


def test_fold_ignores_case_and_accents():
    assert fold("Teoría ÁGIL") == "teoria agil"


def test_source_hit_requires_same_document_and_overlapping_page():
    sources = [{"filename": "scrum-guide.pdf", "page_start": 3, "page_end": 4}]
    assert source_hit(sources, [{"document": "scrum-guide", "pages": [4]}])
    assert not source_hit(sources, [{"document": "scrum-guide", "pages": [9]}])
    assert not source_hit(sources, [{"document": "otro", "pages": [3]}])


def test_require_all_needs_every_expected_document():
    sources = [{"filename": "scrum-guide-2020-es.pdf", "page_start": 14, "page_end": 14}]
    expected = [
        {"document": "scrum-guide-2020-es", "pages": [14]},
        {"document": "kanban-guide-2020-es", "pages": [7]},
    ]
    assert source_hit(sources, expected)
    assert not source_hit(sources, expected, require_all=True)
    sources.append({"filename": "kanban-guide-2020-es.pdf", "page_start": 7, "page_end": 7})
    assert source_hit(sources, expected, require_all=True)


def test_keyword_recall_is_fraction_of_found_keywords():
    assert keyword_recall("El Sprint dura un mes", ["sprint", "mes", "semana"]) == 2 / 3
    assert keyword_recall("lo que sea", []) == 1.0


def test_summary_separates_answerable_from_unanswerable():
    rows = [
        {
            "type": "direct",
            "passed": True,
            "source_hit": True,
            "keyword_recall": 1.0,
            "latency_ms": 100,
        },
        {
            "type": "unanswerable",
            "passed": True,
            "correct_refusal": True,
            "latency_ms": 300,
        },
    ]
    summary = summarize(rows)
    assert summary["answerable_source_hit_rate"] == 1.0
    assert summary["unanswerable_correct_refusal_rate"] == 1.0
    assert summary["mean_latency_ms"] == 200
