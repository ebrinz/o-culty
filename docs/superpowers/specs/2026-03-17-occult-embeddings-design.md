# Occult Primary Source Literature: Scraping & Embeddings Analysis

## Overview

Scrape all available occult primary-source literature from public domain sources, extract and clean text (with OCR for scanned documents), generate embeddings using open-weight models on Apple Silicon, and provide modular analysis tools for semantic search, clustering, similarity mapping, and interactive visualization.

Phase 2 (future) will add image/symbol extraction via SAM segmentation and multimodal embeddings (CLIP/SigLIP).

## Goals

- Build the largest feasible corpus of public domain occult primary sources
- Map conceptual similarity across traditions (Hermetic, Kabbalistic, Alchemical, etc.)
- Enable semantic search across the full corpus
- Discover hidden thematic groupings via clustering
- Generate tradition-level and text-level similarity matrices
- Interactive UMAP/t-SNE visualizations for exploration
- Modular design: each pipeline stage runs independently, future image phase slots in cleanly

## Project Structure

```
o-culty/
├── pyproject.toml
├── config.yaml                  # sources list, model names, chunk params
├── src/
│   ├── scraper/                 # source-specific scrapers
│   │   ├── sacred_texts.py
│   │   ├── internet_archive.py
│   │   ├── gutenberg.py
│   │   └── base.py             # shared scraper interface
│   ├── processor/               # text extraction & cleaning
│   │   ├── pdf_extractor.py    # PDF -> text (pymupdf)
│   │   ├── html_extractor.py   # HTML -> text
│   │   ├── ocr.py              # scanned pages (easyocr)
│   │   └── cleaner.py          # normalize unicode, strip boilerplate
│   ├── chunker/                 # text -> chunks with metadata
│   │   └── chunker.py          # sliding window, chapter-aware splits
│   ├── embedder/                # chunks -> vectors
│   │   └── embedder.py         # sentence-transformers, batch processing
│   ├── analysis/                # all downstream analysis
│   │   ├── cluster.py          # HDBSCAN clustering + auto-labeling
│   │   ├── search.py           # semantic search over embeddings
│   │   ├── similarity.py       # text/tradition similarity matrices
│   │   └── visualize.py        # UMAP/t-SNE projections (plotly)
│   └── utils.py                 # logging, metadata helpers
├── data/
│   ├── raw/                     # scraped files as-is (HTML, PDF)
│   ├── processed/               # extracted clean text + metadata JSON
│   ├── chunks/                  # chunked text with provenance
│   ├── embeddings/              # numpy arrays + metadata parquet
│   └── results/                 # cluster labels, similarity matrices, plots
└── scripts/
    ├── scrape.py                # CLI entry: run scrapers
    ├── process.py               # CLI entry: extract & clean
    ├── chunk.py                 # CLI entry: chunk texts
    ├── embed.py                 # CLI entry: generate embeddings
    └── analyze.py               # CLI entry: run analyses
```

## Scraping Layer

### Sources

- **sacred-texts.com** -- HTML scraping. Crawl index pages per tradition, follow links to full texts, save raw HTML. Rate-limited (1.5s delay).
- **Internet Archive** -- `internetarchive` Python package. Search and download occult/esoteric texts. Prioritize PDF and DjVu. Up to 50 results per search term.
- **Project Gutenberg** -- Mirror/API for public domain plaintext.

### Search Terms / Traditions

hermetic, kabbalah, kabbalistic, alchemy, alchemical, grimoire, enochian, rosicrucian, thelema, thelemic, gnostic, theurgy, occult, esoteric, magic

### Metadata Per Source

- Title, author (if known), date (if known)
- Tradition/category
- Source URL, download date
- File type (HTML, PDF, plaintext)

### Behavior

- Idempotent -- skips already-downloaded files
- `manifest.json` per source tracking fetched items
- Respects robots.txt
- Config-driven: traditions, search terms, delays all in `config.yaml`

## Processing & OCR Layer

### Text Extraction

- **HTML** -- BeautifulSoup, strip nav/boilerplate, preserve chapter/section structure
- **PDF (text-based)** -- PyMuPDF for fast extraction with layout awareness
- **PDF (scanned/image-based)** -- EasyOCR (open weights, MPS-friendly). Tesseract fallback.
- **Plaintext** -- Normalize encoding only

### Cleaning Pipeline

- Normalize Unicode (NFKC)
- Strip page numbers, headers/footers, OCR artifacts
- Normalize whitespace
- Detect and tag language (some texts have Latin, Hebrew, Greek passages)
- Preserve paragraph and chapter boundaries as structural markers

### Output Per Text

Cleaned `.txt` file plus JSON sidecar:

```json
{
  "id": "sacred-texts_hermetic_corpus-hermeticum",
  "title": "Corpus Hermeticum",
  "author": "Hermes Trismegistus (trans. G.R.S. Mead)",
  "tradition": "hermetic",
  "source": "sacred-texts.com",
  "source_url": "https://...",
  "file_type": "html",
  "ocr_used": false,
  "language": "en",
  "chapters": ["I. Poemandres", "II. To Asclepius"],
  "char_count": 128450,
  "processed_date": "2026-03-17"
}
```

## Chunking Strategy

- Chapter-aware: split on chapter/section boundaries first
- Within chapters: sliding window, ~512 tokens, ~64 token overlap
- Short chapters (< 512 tokens) stay as single chunks
- Each chunk inherits parent metadata plus: `chunk_id`, `chapter`, `position_in_chapter`, raw text
- Storage: one parquet file per source text

## Embedding Layer

- **Model:** `nomic-ai/nomic-embed-text-v1.5` (768 dims, 8192 token context, Matryoshka support)
- Configurable in `config.yaml` -- swap to BGE, E5, etc. without code changes
- Batch embed via sentence-transformers
- **Device priority:** MPS (Apple Silicon) -> CUDA -> CPU

### Output

```
data/embeddings/
├── vectors.npy              # (N, 768) float32 array, all chunks
├── metadata.parquet         # chunk_id, text, title, tradition, chapter, etc.
└── model_info.json          # model name, dimensions, date generated
```

Single flat files for full-corpus analysis. Metadata parquet supports slicing by tradition, author, text, etc.

## Analysis Layer

### Semantic Search (`search.py`)

- Encode query with same model
- Cosine similarity against all chunk vectors
- Top-k results with text, source, tradition, score
- Optional filtering by tradition/text

### Clustering (`cluster.py`)

- HDBSCAN (no need to predefine k)
- Auto-label clusters via top TF-IDF terms per cluster
- Chunk-level or document-level aggregation
- Output: cluster assignments + labels in metadata parquet

### Similarity Matrices (`similarity.py`)

- Mean embeddings per text or per tradition
- Cosine similarity matrix
- Labeled heatmap output (plotly)

### Visualization (`visualize.py`)

- UMAP (default) or t-SNE to 2D/3D
- Plotly interactive scatter plots colored by tradition, text, or cluster
- Hover shows chunk text, source, chapter
- Standalone HTML files in `data/results/`

### CLI

All via `scripts/analyze.py` with subcommands:
- `python scripts/analyze.py search "the nature of the soul"`
- `python scripts/analyze.py cluster`
- `python scripts/analyze.py similarity --level tradition`
- `python scripts/analyze.py visualize --color-by tradition`

## Configuration

```yaml
scraping:
  sacred_texts:
    enabled: true
    traditions: [hermetic, kabbalah, alchemy, grimoire, enochian, rosicrucian, thelema, gnostic]
    delay_seconds: 1.5
  internet_archive:
    enabled: true
    search_terms: [occult, hermetic, alchemy, grimoire, kabbalah, theurgy, enochian, esoteric]
    max_results_per_term: 50
    formats: [pdf, txt, djvu]
  gutenberg:
    enabled: true
    search_terms: [occult, alchemy, hermetic, magic]

processing:
  ocr_engine: easyocr
  ocr_languages: [en, la]

chunking:
  chunk_size_tokens: 512
  overlap_tokens: 64

embedding:
  model: nomic-ai/nomic-embed-text-v1.5
  batch_size: 64
  device: auto  # mps -> cuda -> cpu

analysis:
  umap_n_neighbors: 15
  umap_min_dist: 0.1
  hdbscan_min_cluster_size: 10
```

## Dependencies

- `requests`, `beautifulsoup4` -- web scraping
- `internetarchive` -- Internet Archive API
- `pymupdf` -- PDF text extraction
- `easyocr` -- OCR for scanned pages
- `sentence-transformers` -- embedding models
- `numpy`, `pandas`, `pyarrow` -- data handling
- `hdbscan` -- clustering
- `umap-learn` -- dimensionality reduction
- `plotly` -- interactive visualizations
- `pyyaml` -- config
- `tqdm` -- progress bars

## Future: Phase 2 (Image/Symbol Analysis)

- Extract images from PDFs and scanned pages
- SAM segmentation to isolate individual symbols, sigils, diagrams
- CLIP/SigLIP embeddings for images in the same vector space as text
- Cross-modal search: query with text, find relevant symbols (and vice versa)
- Symbol clustering and tradition-level visual motif analysis
