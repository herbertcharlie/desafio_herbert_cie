"""Evaluación automatizada del RAG contra la API en ejecución (solo biblioteca estándar).

Uso:
    python eval/run_eval.py                      # contra http://localhost:8000
    python eval/run_eval.py --base-url http://localhost:8080/api

Métricas por pregunta (ver eval/README.md):
  - answerable : `grounded=true`, la fuente citada coincide con documento+página esperados
                 (`source_hit`) y la respuesta contiene las palabras clave esperadas
                 (`keyword_recall`).
  - unanswerable: el sistema debe rechazar (`grounded=false`, sin fuentes) -> `correct_refusal`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent


def fold(text: str) -> str:
    """Minúsculas y sin tildes, para comparar palabras clave."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise SystemExit(f"Error HTTP {exc.code} en {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"No se pudo conectar con {url}: {exc.reason}") from exc


def source_hit(sources: list[dict], expected: list[dict]) -> bool:
    """True si alguna fuente citada coincide en documento y solapa alguna página esperada."""
    for source in sources:
        for exp in expected:
            same_doc = fold(exp["document"]) in fold(source["filename"])
            pages = set(range(source["page_start"], source["page_end"] + 1))
            if same_doc and pages & set(exp["pages"]):
                return True
    return False


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    folded = fold(answer)
    return sum(fold(k) in folded for k in keywords) / len(keywords)


def evaluate(item: dict, base_url: str, session_id: str) -> dict:
    started = time.perf_counter()
    response = post_json(
        f"{base_url}/ask", {"question": item["question"], "session_id": session_id}
    )
    latency_ms = round((time.perf_counter() - started) * 1000)

    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "answer": response["answer"],
        "grounded": response["grounded"],
        "sources": [
            f'{s["filename"]} p.{s["page_start"]}-{s["page_end"]}' for s in response["sources"]
        ],
        "latency_ms": latency_ms,
    }
    if item["type"] == "unanswerable":
        row["correct_refusal"] = (not response["grounded"]) and not response["sources"]
        row["passed"] = row["correct_refusal"]
    else:
        row["source_hit"] = response["grounded"] and source_hit(
            response["sources"], item["expected_sources"]
        )
        row["keyword_recall"] = round(keyword_recall(response["answer"], item["keywords"]), 2)
        row["passed"] = bool(row["source_hit"]) and row["keyword_recall"] >= 0.5
    return row


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]

    def rate(items: list[dict], key: str) -> float | None:
        return round(sum(bool(r[key]) for r in items) / len(items), 2) if items else None

    return {
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "answerable_source_hit_rate": rate(answerable, "source_hit"),
        "answerable_mean_keyword_recall": (
            round(sum(r["keyword_recall"] for r in answerable) / len(answerable), 2)
            if answerable
            else None
        ),
        "unanswerable_correct_refusal_rate": rate(unanswerable, "correct_refusal"),
        "mean_latency_ms": (
            round(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.json")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    session_id = f"eval-{datetime.now():%Y%m%d%H%M%S}"
    base_url = args.base_url.rstrip("/")

    rows = []
    for item in dataset["questions"]:
        row = evaluate(item, base_url, session_id)
        rows.append(row)
        mark = "OK  " if row["passed"] else "FAIL"
        head = f'[{mark}] {row["id"]:<4} {row["type"]:<13} {row["latency_ms"]:>5} ms'
        print(f'{head}  {row["question"]}')
        if not row["passed"]:
            print(f'       -> grounded={row["grounded"]} fuentes={row["sources"]}')
            print(f'       -> {row["answer"][:160]}')

    summary = summarize(rows)
    print("\nResumen:", json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{session_id}.json"
    out_file.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Detalle guardado en {out_file}")
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
