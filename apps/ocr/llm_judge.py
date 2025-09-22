"""Optional LLM-based judge for OCR outputs via OpenRouter-compatible APIs.

This module scores OCR candidates and can optionally normalize/fix typos.

Environment:
  - OPENROUTER_API_KEY: required for network calls
  - OPENROUTER_BASE: optional, defaults to https://openrouter.ai/api/v1

Usage:
  from apps.ocr.llm_judge import LlmJudge
  judge = LlmJudge(model="google/gemini-2.5-flash")
  result = judge.score(["Creates a Tarot card when discarded"], context="UI tooltip")
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional
    OpenAI = None  # type: ignore


class LlmJudge:
    def __init__(self, model: str = "google/gemini-2.5-flash", base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.enabled = bool(self.api_key and OpenAI is not None)
        self._client = None
        if self.enabled:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def score(self, candidates: List[str], context: str = "") -> Dict[str, Any]:
        """Return best candidate with a score in [0,1].

        If LLM is unavailable, falls back to a heuristic: longest non-empty.
        """
        cands = [c for c in candidates if isinstance(c, str)]
        if not cands:
            return {"best": "", "score": 0.0, "explanation": "no candidates", "source": "fallback"}

        if not self.enabled:
            best = max(cands, key=lambda s: len(s.strip()))
            return {"best": best, "score": 0.5 if best.strip() else 0.0, "explanation": "length heuristic", "source": "fallback"}

        prompt = (
            "You are an OCR result judge for a video game UI tooltip. "
            "Given several OCR candidates of the same small text region, select the most accurate, "
            "correct obvious OCR typos (like 'Tarpt'->'Tarot'), and return JSON with keys: best, score (0..1), explanation."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Context: {context}\nCandidates:\n" + "\n".join(f"- {c}" for c in cands)},
        ]

        try:
            resp = self._client.chat.completions.create(model=self.model, messages=messages, response_format={"type": "json_object"})
            content = resp.choices[0].message.content  # type: ignore[attr-defined]
            import json as _json

            parsed = _json.loads(content) if content else {}
            best = parsed.get("best", "")
            score = float(parsed.get("score", 0))
            explanation = parsed.get("explanation", "")
            return {"best": best, "score": score, "explanation": explanation, "source": "llm"}
        except Exception as e:  # pragma: no cover - network variability
            best = max(cands, key=lambda s: len(s.strip()))
            return {"best": best, "score": 0.4, "explanation": f"fallback due to error: {e}", "source": "fallback"}

