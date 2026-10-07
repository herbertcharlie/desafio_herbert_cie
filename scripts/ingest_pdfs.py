"""Carga todos los PDF de data/pdfs a la API (equivale a usar POST /documents).

Uso:
    python scripts/ingest_pdfs.py                          # contra http://localhost:8000
    python scripts/ingest_pdfs.py --base-url http://localhost:8080/api
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def multipart(files: list[Path]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    body = b""
    for path in files:
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode()
        body += path.read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    files = sorted(PDF_DIR.glob("*.pdf"))
    if not files:
        print(f"No hay PDF en {PDF_DIR}")
        return 1

    body, content_type = multipart(files)
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/documents",
        data=body,
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            results = json.load(response)["results"]
    except urllib.error.HTTPError as exc:
        print(f"Error HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}")
        return 1
    except urllib.error.URLError as exc:
        print(f"No se pudo conectar: {exc.reason}")
        return 1

    for result in results:
        detail = result.get("document") or result.get("error")
        print(f'{result["status"]:<10} {result["filename"]}  {detail}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
