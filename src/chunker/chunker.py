import re
from tokenizers import Tokenizer

_tokenizer_cache: dict[str, Tokenizer] = {}

def _get_tokenizer(name: str) -> Tokenizer:
    if name not in _tokenizer_cache:
        _tokenizer_cache[name] = Tokenizer.from_pretrained(name)
    return _tokenizer_cache[name]

def split_into_chapters(text: str) -> list[dict]:
    pattern = r"\n(?=(?:Chapter|CHAPTER|Section|SECTION)\s+[\dIVXLCDM]+[.:]\s*\S)"
    parts = re.split(pattern, text)
    if len(parts) <= 1:
        return [{"title": "", "text": text.strip()}]
    chapters = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.split("\n")[0].strip()
        if re.match(r"(?:Chapter|CHAPTER|Section|SECTION)\s+[\dIVXLCDM]+", first_line):
            title = first_line
            body = part[len(first_line):].strip()
        else:
            title = ""
            body = part
        chapters.append({"title": title, "text": body})
    return chapters if chapters else [{"title": "", "text": text.strip()}]

def _sliding_window(tokens: list[int], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    if len(tokens) <= chunk_size:
        return [(0, len(tokens))]
    windows = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        windows.append((start, end))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return windows

def chunk_text(
    text: str,
    metadata: dict,
    chunk_size: int = 512,
    overlap: int = 64,
    tokenizer_name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> list[dict]:
    tokenizer = _get_tokenizer(tokenizer_name)
    chapters = split_into_chapters(text)
    chunks = []
    chunk_idx = 0
    for chapter in chapters:
        chapter_text = chapter["text"]
        if not chapter_text.strip():
            continue
        encoding = tokenizer.encode(chapter_text)
        token_ids = encoding.ids
        windows = _sliding_window(token_ids, chunk_size, overlap)
        for pos, (start, end) in enumerate(windows):
            chunk_tokens = token_ids[start:end]
            chunk_text_decoded = tokenizer.decode(chunk_tokens)
            chunk = {
                **{k: v for k, v in metadata.items() if k != "chapters"},  # IMPORTANT: exclude 'chapters' list to avoid column collision with 'chapter' string
                "chunk_id": f"{metadata.get('id', 'unknown')}_{chunk_idx:05d}",
                "chapter": chapter["title"],
                "position_in_chapter": pos,
                "text": chunk_text_decoded,
            }
            chunks.append(chunk)
            chunk_idx += 1
    return chunks
