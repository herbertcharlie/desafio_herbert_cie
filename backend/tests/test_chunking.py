import pytest

from app.domain.models import PageText
from app.services.chunking import chunk_pages, normalize_text


def pages_of(*texts: str) -> list[PageText]:
    return [PageText(number=i, text=t) for i, t in enumerate(texts, start=1)]


def sentences(n: int, prefix: str = "Oración") -> str:
    return " ".join(f"{prefix} número {i} del texto de prueba." for i in range(n))


def test_normalize_joins_hyphenated_words_and_collapses_whitespace():
    assert normalize_text("auto-\norganizado   equipo\n\nágil") == "autoorganizado equipo ágil"


def test_empty_pages_produce_no_chunks():
    assert chunk_pages(pages_of("", "   \n "), size=200, overlap=40) == []


def test_short_text_is_a_single_chunk_with_its_page():
    chunks = chunk_pages(pages_of("Hola mundo."), size=200, overlap=40)
    assert len(chunks) == 1
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 1)


def test_chunks_respect_max_size_and_are_sequentially_indexed():
    chunks = chunk_pages(pages_of(sentences(40)), size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.content) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_repeats_trailing_sentence_in_next_chunk():
    chunks = chunk_pages(pages_of(sentences(40)), size=200, overlap=60)
    first_tail = chunks[0].content.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].content


def test_no_overlap_means_no_repeated_text():
    chunks = chunk_pages(pages_of(sentences(30)), size=200, overlap=0)
    joined = " ".join(c.content for c in chunks)
    assert joined == normalize_text(sentences(30))


def test_page_range_is_tracked_across_pages():
    page1 = "Primera página con texto suficiente para llenar. " * 3
    page2 = "Segunda página con contenido distinto y también largo. " * 3
    chunks = chunk_pages(pages_of(page1, page2), size=300, overlap=50)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start != c.page_end for c in chunks)


def test_oversized_sentence_is_hard_split():
    chunks = chunk_pages(pages_of("x" * 450), size=200, overlap=0)
    assert [len(c.content) for c in chunks] == [200, 200, 50]


def test_last_chunk_is_not_just_the_overlap():
    chunks = chunk_pages(pages_of(sentences(40)), size=200, overlap=60)
    last, previous = chunks[-1].content, chunks[-2].content
    assert not previous.endswith(last)


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(pages_of("texto"), size=size, overlap=overlap)
