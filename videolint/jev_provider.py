"""Jev judgment adapter for compatible typed Decisions APIs."""
from __future__ import annotations

import json
import math
import urllib.request

from .ai import AIProviderError, Judgment


class JevProvider:
    name = "jev"

    def __init__(self, api_key: str, model: str | None = None, opener=None,
                 openrouter: bool = False, beatapi: bool = False):
        if not api_key:
            raise AIProviderError("Jev API key not configured")
        if openrouter and beatapi:
            raise ValueError("Choose one Jev API backend")
        self._api_key = api_key
        self.model = model or ("jev-1.13-free" if beatapi else
                               "typesafe/jev-1.13" if openrouter else "jev-latest")
        self.url = ("https://api.beatapi.io/v1/systemone" if beatapi else
                    "https://openrouter.ai/api/alpha/decisions" if openrouter else
                    "https://api.typesafe.ai/v1/systemone")
        self.last_cost_usd = None
        self._opener = opener or urllib.request.urlopen

    def judge(self, question: str, context: dict) -> Judgment:
        payload = {
            "model": self.model,
            "state": context,
            "questions": {"finding": {"type": "noul", "instructions": question}},
        }
        request = urllib.request.Request(
            self.url,
            json.dumps(payload, ensure_ascii=False).encode(),
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=45) as response:
                data = json.load(response)
        except Exception as exc:
            status = getattr(exc, "code", None)
            detail = f" (HTTP {status})" if isinstance(status, int) else ""
            raise AIProviderError(f"Jev {type(exc).__name__}{detail}; checker could not complete") from exc
        answers = data.get("answers") if isinstance(data, dict) else None
        answer = answers.get("finding") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise AIProviderError("Jev returned an invalid decision response")
        probability = answer.get("noul")
        if (answer.get("type") != "noul" or isinstance(probability, bool) or
                not isinstance(probability, (int, float)) or
                not math.isfinite(probability) or not 0 <= probability <= 1):
            raise AIProviderError("Jev returned an invalid Noul probability")
        probability = float(probability)
        usage = data.get("usage", {})
        cost = usage.get("cost") if isinstance(usage, dict) else None
        self.last_cost_usd = (float(cost) if isinstance(cost, (int, float)) and
                              not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
                              else 0.0 if self.model == "jev-1.13-free" and
                              self.url == "https://api.beatapi.io/v1/systemone" else None)
        verdict = "yes" if probability >= .80 else "no" if probability <= .20 else "uncertain"
        confidence = probability if verdict == "yes" else 1 - probability if verdict == "no" else max(probability, 1 - probability)
        reason = f"Jev estimated a {probability:.0%} probability that this focused finding is present. Review the transcript evidence."
        return Judgment.validate({"verdict": verdict, "confidence": confidence, "reason": reason})

    def connectivity_test(self) -> None:
        self.judge("Does the state contain the word VideoLint?", {"text": "VideoLint"})
