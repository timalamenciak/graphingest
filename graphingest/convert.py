"""Convert the manifest's PDFs to markdown.

    python -m graphingest.convert --manifest build/manifest.json \\
        --out-dir build/markdown
    python -m graphingest.convert --manifest build/manifest.json \\
        --out-dir build/markdown --converter pymupdf --limit 5

Two converters, because they fail in opposite directions:

* **marker** (the built-in default) is what you want for real work —
  layout-aware, reconstructs tables and reading order, and optionally uses the
  configured LLM to clean up tables and figure captions. It is also a large
  dependency, and slow: minutes per PDF on a CPU with no CUDA.
* **pymupdf** is the fast path: fast, no models, and it hands back the PDF's
  raw text layer with light heading heuristics. Converts a paper in well under
  a second. Poor on multi-column layouts and tables; fine for a quick corpus
  pass, a smoke test, or a clean born-digital PDF.

Which one runs is decided, in order: ``--converter`` on the command line, then
``convert.default_converter`` in ``config/pipeline.yaml``, then ``marker``. Set
the config default once instead of typing ``--converter pymupdf`` on every
invocation while you are iterating.

Output is written by the manifest slug, so ``build/markdown/<slug>.md`` lines
up with ``build/graphs/<slug>.yaml`` and a human can follow a paper through the
pipeline by name. Conversion is resumable: an existing output is left alone
unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .cli import configure_logging, configure_stdio
from .config import DEFAULT_LLM_CONFIG, load_yaml
from .graph_io import atomic_write, atomic_write_json
from .llm_client import LLMSettings

LOGGER = logging.getLogger("ingest.convert")

CONVERTERS = ("marker", "pymupdf")

#: Used when config/pipeline.yaml has no ``convert:`` block.
DEFAULT_CONVERT_SETTINGS = {"default_converter": "marker"}


def convert_settings(config_path: Optional[Path] = None) -> dict:
    """Read the ``convert:`` block, filling in the default it omits."""
    try:
        data = load_yaml(config_path or DEFAULT_LLM_CONFIG)
    except FileNotFoundError:
        return dict(DEFAULT_CONVERT_SETTINGS)
    settings = {**DEFAULT_CONVERT_SETTINGS, **(data.get("convert") or {})}
    if settings["default_converter"] not in CONVERTERS:
        raise ValueError(
            f"convert.default_converter {settings['default_converter']!r} in "
            f"{config_path or DEFAULT_LLM_CONFIG} must be one of {CONVERTERS}"
        )
    return settings


def resolve_converter(chosen: Optional[str], config_path: Optional[Path] = None) -> str:
    """``--converter`` if given, else the configured default, else marker."""
    return chosen or convert_settings(config_path)["default_converter"]


# ---------------------------------------------------------------------------
# Marker
# ---------------------------------------------------------------------------


def build_marker_converter(
    llm: Optional[dict] = None, force_ocr: bool = False
) -> Callable[[str], Any]:
    """Load Marker and return a callable that converts one PDF path.

    Marker's ``resolve_dependencies`` passes ``self.config`` to the LLM service
    class, so the key and base URL have to be in the config dict and not only
    in the environment — hence both are set. The two wiring strategies exist
    because Marker's service API has moved more than once; the older one is
    tried when the newer is refused, and LLM assist is dropped rather than
    failing the run.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
    except ImportError:
        return _legacy_marker_converter()

    base_config: dict[str, Any] = {"force_ocr": force_ocr}

    if not llm:
        from marker.config.parser import ConfigParser

        parser = ConfigParser(base_config)
        LOGGER.info("Loading Marker models (first run downloads weights)")
        return PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )

    from marker.config.parser import ConfigParser
    from marker.services.openai import OpenAIService

    LOGGER.info(
        "Loading Marker with LLM assist: %s @ %s", llm["model"], llm["endpoint"]
    )
    os.environ["OPENAI_BASE_URL"] = llm["endpoint"]
    os.environ["OPENAI_API_KEY"] = llm["api_key"]
    llm_config = {
        **base_config,
        "use_llm": True,
        "openai_api_key": llm["api_key"],
        "openai_base_url": llm["endpoint"],
        "openai_model": llm["model"],
    }

    # Strategy 1: name the service by dotted path.
    try:
        parser = ConfigParser(llm_config)
        return PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            llm_service="marker.services.openai.OpenAIService",
        )
    except Exception as error:  # noqa: BLE001 - Marker's API has moved repeatedly
        LOGGER.warning("Marker LLM strategy 1 failed (%s); trying strategy 2", error)

    # Strategy 2: monkeypatch the default service class.
    original = PdfConverter.default_llm_service
    try:
        PdfConverter.default_llm_service = OpenAIService
        parser = ConfigParser(llm_config)
        return PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )
    except Exception as error:  # noqa: BLE001
        PdfConverter.default_llm_service = original
        LOGGER.warning(
            "Marker LLM strategy 2 failed (%s); continuing without LLM assist", error
        )
        parser = ConfigParser(base_config)
        return PdfConverter(
            config=parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )


def _legacy_marker_converter() -> Callable[[str], Any]:
    """Marker < 0.3 exposed convert_single_pdf instead of PdfConverter."""
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models

    LOGGER.info("Using the legacy Marker API")
    models = load_all_models()

    def convert(pdf_path: str):
        text, _images, _metadata = convert_single_pdf(pdf_path, models)

        class Result:
            markdown = text

        return Result()

    return convert


def _render_marker(converter: Callable[[str], Any], pdf_path: Path) -> str:
    rendered = converter(str(pdf_path))
    if hasattr(rendered, "markdown"):
        return rendered.markdown
    from marker.output import text_from_rendered

    text, _metadata, _images = text_from_rendered(rendered)
    return text


# ---------------------------------------------------------------------------
# PyMuPDF fallback
# ---------------------------------------------------------------------------


def build_pymupdf_converter() -> Callable[[str], str]:
    """Return a text-layer extractor that emits per-page markdown sections.

    Page headings are not cosmetic: the chunker keys sections off markdown
    headings, so a document converted this way still chunks sensibly if the
    article is too long to send whole.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as error:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyMuPDF is required for --converter pymupdf: pip install pymupdf"
        ) from error

    def convert(pdf_path: str) -> str:
        parts: list[str] = []
        with fitz.open(pdf_path) as document:
            for number, page in enumerate(document, 1):
                text = page.get_text("text").strip()
                if text:
                    parts.append(f"## Page {number}\n\n{text}")
        return "\n\n".join(parts)

    return convert


def build_converter(
    kind: str, llm: Optional[dict] = None, force_ocr: bool = False
) -> Callable[[str], Any]:
    if kind == "pymupdf":
        return build_pymupdf_converter()
    return build_marker_converter(llm, force_ocr)


def render(kind: str, converter: Callable[[str], Any], pdf_path: Path) -> str:
    if kind == "pymupdf":
        return converter(str(pdf_path))
    return _render_marker(converter, pdf_path)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


def convert_documents(
    documents: list[dict],
    out_dir: Path,
    converter: Callable[[str], Any],
    kind: str = "marker",
    force: bool = False,
    min_words: int = 200,
) -> list[dict]:
    """Convert each document's PDF, skipping ones already done.

    Mutates each row in place with ``markdown_path``/``word_count`` so the
    caller can write an updated manifest and the next stage needs no second
    lookup.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for index, document in enumerate(documents, 1):
        slug = document["slug"]
        destination = out_dir / f"{slug}.md"
        record = {
            "slug": slug,
            "document_id": document.get("document_id"),
            "markdown_path": str(destination),
        }

        pdf_path = document.get("pdf_path")
        if not pdf_path or not Path(pdf_path).exists():
            record.update({"status": "failed", "error": "PDF not found on disk"})
            document["markdown_path"] = None
            results.append(record)
            LOGGER.error("[%d/%d] %s: PDF missing", index, len(documents), slug)
            continue

        if destination.exists() and not force:
            text = destination.read_text(encoding="utf-8", errors="replace")
            record.update({"status": "skipped_existing", "word_count": len(text.split())})
            document["markdown_path"] = str(destination)
            document["word_count"] = record["word_count"]
            results.append(record)
            LOGGER.info("[%d/%d] %s: already converted", index, len(documents), slug)
            continue

        started = time.time()
        try:
            text = render(kind, converter, Path(pdf_path))
        except Exception as error:  # noqa: BLE001 - one bad PDF must not stop the run
            record.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            document["markdown_path"] = None
            results.append(record)
            LOGGER.exception("[%d/%d] %s: conversion failed", index, len(documents), slug)
            continue

        words = len(text.split())
        record["seconds"] = round(time.time() - started, 1)
        if words < min_words:
            # A near-empty conversion usually means a scanned PDF with no text
            # layer. Recording it as suspect beats shipping a stub article.
            record.update(
                {
                    "status": "suspect_short",
                    "word_count": words,
                    "error": f"only {words} words; likely a scan needing --force-ocr",
                }
            )
            LOGGER.warning("[%d/%d] %s: only %d words", index, len(documents), slug, words)
        else:
            record.update({"status": "converted", "word_count": words})
            LOGGER.info(
                "[%d/%d] %s: %d words in %.1fs",
                index, len(documents), slug, words, record["seconds"],
            )
        atomic_write(destination, text)
        document["markdown_path"] = str(destination)
        document["word_count"] = words
        results.append(record)

    return results


def tally(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in results:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument(
        "--converter", choices=CONVERTERS, default=None,
        help="Default: convert.default_converter in config/pipeline.yaml, else marker",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Reconvert existing output")
    parser.add_argument("--no-llm", action="store_true", help="Disable Marker LLM assist")
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on every page")
    parser.add_argument("--min-words", type=int, default=200)
    parser.add_argument(
        "--report", type=Path, help="Where to write the conversion report JSON"
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_stdio()
    configure_logging(args.log_level)
    args.converter = resolve_converter(args.converter, args.llm_config)
    LOGGER.info("Converter: %s", args.converter)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = [row for row in manifest.get("documents") or [] if row.get("pdf_path")]
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print("Nothing to convert: no manifest row has a PDF.")
        return 0

    llm = None
    if args.converter == "marker" and not args.no_llm:
        llm = LLMSettings.from_config(args.llm_config).resolved_marker_llm()

    try:
        converter = build_converter(args.converter, llm, args.force_ocr)
    except ImportError as error:
        hint = (
            "pip install marker-pdf"
            if args.converter == "marker"
            else "pip install pymupdf"
        )
        print(f"{args.converter} is not installed ({error}).\nInstall it with:  {hint}",
              file=sys.stderr)
        return 2

    results = convert_documents(
        documents, args.out_dir, converter, args.converter, args.force, args.min_words
    )

    counts = tally(results)
    report_path = args.report or args.out_dir.parent / "conversion_report.json"
    atomic_write_json(
        report_path,
        {"converter": args.converter, "total": len(results),
         "by_status": counts, "results": results},
    )
    # The manifest now knows where each markdown file landed.
    atomic_write_json(args.manifest, manifest)

    print(f"\nConverted {len(results)} document(s) with {args.converter}:")
    for status, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"    {count:5d}  {status}")
    print(f"  wrote {report_path}")
    print(f"  updated {args.manifest}")
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
