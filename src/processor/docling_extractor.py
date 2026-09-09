"""Docling-backed PDF extraction: layout-aware conversion to markdown.

Docling runs layout analysis, table-structure recognition and OCR in a single
pass, so it replaces the raw PyMuPDF text dump followed by a separate OCR step.
Output is markdown, which keeps headings, tables and reading order that a flat
text dump discards.

Model loading is expensive, so converters are memoised per configuration.
"""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OCR_LANGUAGES = ["en"]

# Config `ocr_engine` value -> the docling options class implementing it.
OCR_ENGINES = {
    "easyocr": "EasyOcrOptions",
    "tesseract": "TesseractOcrOptions",
    "tesseract_cli": "TesseractCliOcrOptions",
    "rapidocr": "RapidOcrOptions",
    "ocrmac": "OcrMacOptions",
}

_CONVERTERS: dict = {}


class DoclingUnavailable(RuntimeError):
    """Raised when docling is not installed, so callers can fall back."""


def _ocr_options(engine: str, languages: list[str]):
    """Instantiate the docling OCR options class named by `engine`."""
    if engine not in OCR_ENGINES:
        raise ValueError(f"Unknown ocr_engine '{engine}'. Known: {', '.join(sorted(OCR_ENGINES))}")
    from docling.datamodel import pipeline_options as po

    cls = getattr(po, OCR_ENGINES[engine], None)
    if cls is None:
        raise ValueError(f"docling has no OCR backend for '{engine}' in this version")
    return cls(lang=list(languages))


def build_converter(do_ocr: bool = True, ocr_engine: str = "easyocr", languages: list[str] | None = None):
    """Return a DocumentConverter for this configuration, memoised across calls.

    Each converter loads layout and table models, so rebuilding one per document
    would dominate runtime on a corpus this size.
    """
    languages = list(languages or DEFAULT_OCR_LANGUAGES)
    key = (do_ocr, ocr_engine, tuple(languages))
    if key in _CONVERTERS:
        return _CONVERTERS[key]

    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
    except ImportError as e:
        raise DoclingUnavailable(
            "docling is not installed. Run `pip install -e .`, or pass --no-docling "
            "to fall back to the PyMuPDF extractor."
        ) from e

    opts = PdfPipelineOptions()
    opts.do_ocr = do_ocr
    opts.do_table_structure = True
    opts.table_structure_options.do_cell_matching = True
    if do_ocr:
        opts.ocr_options = _ocr_options(ocr_engine, languages)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    _CONVERTERS[key] = converter
    return converter


def extract_markdown_from_pdf(
    path: Path,
    do_ocr: bool = True,
    ocr_engine: str = "easyocr",
    languages: list[str] | None = None,
) -> dict:
    """Convert a PDF to markdown.

    Returns text, page_count and ocr_used. `ocr_used` records whether OCR was
    *enabled* for the conversion — docling decides per region whether to run it
    and does not report that back, so this is not a claim that OCR fired.

    Raises DoclingUnavailable if docling is missing, and lets docling's own
    conversion errors propagate so the caller can fall back.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    converter = build_converter(do_ocr=do_ocr, ocr_engine=ocr_engine, languages=languages)
    result = converter.convert(str(path))
    document = result.document
    markdown = document.export_to_markdown()

    return {
        "text": markdown,
        "page_count": len(getattr(document, "pages", []) or []),
        "ocr_used": do_ocr,
        "text_format": "markdown",
    }
