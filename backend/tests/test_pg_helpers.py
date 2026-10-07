from app.infra.pg_store import to_tsquery, to_vector_literal


def test_vector_literal_format():
    assert to_vector_literal([0.5, -1, 2.25]) == "[0.5,-1,2.25]"


def test_tsquery_is_or_of_unique_lowercase_terms_without_short_words():
    assert to_tsquery("¿Qué es el Sprint Backlog? sprint") == "backlog | qué | sprint"


def test_tsquery_strips_operators_so_it_cannot_inject_syntax():
    assert to_tsquery("scrum & (master) | !x ' ; DROP") == "drop | master | scrum"


def test_tsquery_empty_when_only_short_words():
    assert to_tsquery("a el ¿?") == ""
