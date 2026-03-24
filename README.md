<p align="center">
  <img src="assets/banner-github.svg" alt="o-culty banner" width="100%"/>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/ebrinz/text-cult">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-text--cult-blue" alt="HuggingFace Dataset"/>
  </a>
</p>

Scrape, extract, clean, and structure occult and esoteric texts from public domain sources into a research-ready corpus.

## Pipeline

```
scrape → extract → clean → embed → cluster → serve
```

**Sources:** Project Gutenberg, Sacred Texts, Internet Archive, Hermetic Library, Esoteric Archives

**Output:** Deduplicated Parquet dataset on [HuggingFace](https://huggingface.co/datasets/ebrinz/text-cult) — 8,334 texts, 1.45B characters across 27 traditions.

## Setup

```bash
pip install -e .
```

## Usage

```bash
# Scrape sources
python scripts/scrape_gutenberg.py
python scripts/scrape_sacred_texts.py
python scripts/scrape_internet_archive.py

# Process raw files into clean text
python scripts/process.py              # full run with OCR
python scripts/process.py --no-ocr     # fast text-only pass

# Export to Parquet
python scripts/export_parquet.py
```

## Corpus

```python
from datasets import load_dataset

ds = load_dataset("ebrinz/text-cult")
```

| Source | Documents | Characters |
|--------|-----------|------------|
| sacred-texts | 5,873 | 85.7M |
| internet-archive | 2,231 | 1.3G |
| hermetic-library | 171 | 37.7M |
| gutenberg | 59 | 51.9M |

| Tradition | Docs | | Tradition | Docs | | Tradition | Docs |
|-----------|-----:|-|-----------|-----:|-|-----------|-----:|
| kabbalah | 1,690 | | rpg-fiction | 328 | | mysticism | 21 |
| alchemy | 1,438 | | christian-esoteric | 217 | | islamic-esoteric | 17 |
| grimoire | 1,155 | | freemasonry | 92 | | renaissance-magic | 15 |
| hermetic | 1,026 | | enochian | 88 | | theosophy | 14 |
| occult-general | 804 | | thelema | 78 | | witchcraft | 13 |
| gnostic | 572 | | ancient-near-east | 61 | | divination | 13 |
| rosicrucian | 517 | | angelology | 51 | | egyptian-magic | 6 |
| | | | neoplatonic | 27 | | mesopotamian-magic | 4 |
