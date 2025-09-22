"""Simple OCR text metrics (CER/WER) and helpers."""

from __future__ import annotations

from typing import Tuple
import re


def _normalize_text(s: str) -> str:
    # Lowercase, collapse whitespace; keep alnum and common punctuation
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str, normalize: bool = True) -> Tuple[float, int, int]:
    """Character Error Rate = edit_distance / len(ref).

    Returns (cer, distance, ref_len). If ref is empty, cer is 0 if hyp empty else 1.
    """
    if normalize:
        ref, hyp = _normalize_text(ref), _normalize_text(hyp)
    if len(ref) == 0:
        return (0.0 if len(hyp) == 0 else 1.0, len(hyp), 0)
    d = levenshtein(ref, hyp)
    return d / len(ref), d, len(ref)


def wer(ref: str, hyp: str, normalize: bool = True) -> Tuple[float, int, int]:
    if normalize:
        ref, hyp = _normalize_text(ref), _normalize_text(hyp)
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    if not ref_tokens:
        return (0.0 if not hyp_tokens else 1.0, len(hyp_tokens), 0)
    d = levenshtein(" ".join(ref_tokens), " ".join(hyp_tokens))
    return d / len(ref_tokens), d, len(ref_tokens)

