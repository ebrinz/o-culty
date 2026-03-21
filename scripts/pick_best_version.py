"""Compare PDF-extracted vs _djvu.txt versions and move the worse one to data/overflow/.

For each document that has both a PDF-extracted and a _djvu.txt processed version,
scores both on text length, weird-char ratio, and average word length, then moves
the lower-scoring version to data/overflow/.
"""
import json
import shutil
from pathlib import Path


def score_text(text: str) -> float:
    """Score text quality: higher is better."""
    if not text.strip():
        return 0.0
    sample = text[:5000]
    length = len(text.strip())
    weird = sum(1 for c in sample if ord(c) > 127 or c in "£¬|»«®©™{}[]<>\\^~`") / len(sample)
    words = sample.split()
    avg_word = sum(len(w) for w in words) / max(len(words), 1)
    return length * (1 - weird) * min(avg_word / 5.0, 1.0)


def main():
    processed_dir = Path("data/processed")
    overflow_dir = Path("data/overflow")
    overflow_dir.mkdir(parents=True, exist_ok=True)

    # Find all djvu files and check for a non-djvu counterpart
    pairs = []
    for djvu_txt in sorted(processed_dir.glob("*_djvu.txt")):
        base_name = djvu_txt.stem.replace("_djvu", "")
        pdf_txt = processed_dir / f"{base_name}.txt"
        if pdf_txt.exists():
            pairs.append((pdf_txt, djvu_txt))

    print(f"Found {len(pairs)} PDF/DJVU pairs to compare")

    stats = {"pdf_wins": 0, "djvu_wins": 0, "tie": 0}

    for pdf_path, djvu_path in pairs:
        pdf_text = pdf_path.read_text(encoding="utf-8", errors="replace")
        djvu_text = djvu_path.read_text(encoding="utf-8", errors="replace")

        pdf_score = score_text(pdf_text)
        djvu_score = score_text(djvu_text)

        if pdf_score > djvu_score:
            loser = djvu_path
            winner_tag = "PDF"
            stats["pdf_wins"] += 1
        elif djvu_score > pdf_score:
            loser = pdf_path
            winner_tag = "DJVU"
            stats["djvu_wins"] += 1
        else:
            loser = pdf_path
            winner_tag = "TIE->DJVU"
            stats["tie"] += 1

        # Move loser txt + json to overflow
        loser_json = loser.with_suffix(".json")
        shutil.move(str(loser), str(overflow_dir / loser.name))
        if loser_json.exists():
            shutil.move(str(loser_json), str(overflow_dir / loser_json.name))

    print(f"\nResults:")
    print(f"  PDF version better:  {stats['pdf_wins']}")
    print(f"  DJVU version better: {stats['djvu_wins']}")
    print(f"  Tie (kept DJVU):     {stats['tie']}")
    print(f"  Total moved to overflow: {sum(stats.values())}")


if __name__ == "__main__":
    main()
