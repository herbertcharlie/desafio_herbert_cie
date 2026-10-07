"""Valida el dataset contra los PDFs reales (sin API): páginas y palabras clave existentes."""

import json
import unicodedata
from pathlib import Path

import pytest
from pypdf import PdfReader
from run_eval import fold

ROOT = Path(__file__).resolve().parent.parent
DATASET = json.loads((Path(__file__).parent / "dataset.json").read_text(encoding="utf-8"))
QUESTIONS = DATASET["questions"]


def page_texts(document: str) -> list[str]:
    reader = PdfReader(ROOT / "data" / "pdfs" / f"{document}.pdf")
    return [unicodedata.normalize("NFKC", p.extract_text() or "") for p in reader.pages]


PAGES = {doc: page_texts(doc) for doc in DATASET["documents"]}


def test_dataset_meets_challenge_requirements():
    types = [q["type"] for q in QUESTIONS]
    assert len(QUESTIONS) >= 10
    assert types.count("multi_chunk") >= 2
    assert types.count("unanswerable") >= 2
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)


@pytest.mark.parametrize(
    "q", [q for q in QUESTIONS if q["type"] != "unanswerable"], ids=lambda q: q["id"]
)
def test_expected_pages_exist_and_contain_the_expected_keywords(q):
    text = ""
    for source in q["expected_sources"]:
        pages = PAGES[source["document"]]
        assert max(source["pages"]) <= len(pages)
        text += " ".join(pages[p - 1] for p in source["pages"])
    flat = " ".join(fold(text).split())  # el PDF de Kanban se extrae una palabra por línea
    for keyword in q["keywords"]:
        assert fold(keyword) in flat, f"'{keyword}' no está en las páginas esperadas"


@pytest.mark.parametrize(
    "q", [q for q in QUESTIONS if q["type"] == "unanswerable"], ids=lambda q: q["id"]
)
def test_unanswerable_topics_are_really_absent_from_the_documents(q):
    corpus = " ".join(fold(" ".join(" ".join(pages) for pages in PAGES.values())).split())
    for term in ("story point", "safe ", "psm", "francia", "salario"):
        assert term not in corpus
