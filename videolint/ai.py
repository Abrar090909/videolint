"""Provider contract for focused editorial decisions and transcript preparation."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

SKIPPED_NO_KEY = "AI editorial checks skipped — Gemini API key not configured."


class AIProviderError(RuntimeError):
    """An external AI operation could not produce a trustworthy result."""


@dataclass(frozen=True)
class Judgment:
    verdict: Literal["yes", "no", "uncertain"]
    confidence: float
    reason: str

    @classmethod
    def validate(cls, value: Mapping) -> "Judgment":
        if not isinstance(value, Mapping):
            raise AIProviderError("AI response is not an object")
        verdict = value.get("verdict")
        confidence = value.get("confidence")
        reason = value.get("reason")
        if verdict not in ("yes", "no", "uncertain"):
            raise AIProviderError("AI response has an invalid verdict")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise AIProviderError("AI response has an invalid confidence")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 300:
            raise AIProviderError("AI response has an invalid reason")
        return cls(verdict, float(confidence), reason.strip())


class AIJudgmentProvider(Protocol):
    name: str
    model: str

    def judge(self, question: str, context: dict) -> Judgment: ...
    def connectivity_test(self) -> None: ...


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[dict]: ...


class FallbackJudgmentProvider:
    """Use Gemini only if a configured Jev request cannot make a decision."""

    def __init__(self, primary: AIJudgmentProvider, fallback: AIJudgmentProvider):
        self.primary = primary
        self.fallback = fallback
        self.active = primary

    @property
    def name(self) -> str:
        return self.active.name

    @property
    def model(self) -> str:
        return self.active.model

    def judge(self, question: str, context: dict) -> Judgment:
        self.active = self.primary
        try:
            return self.primary.judge(question, context)
        except AIProviderError:
            self.active = self.fallback
            return self.fallback.judge(question, context)

    def connectivity_test(self) -> None:
        self.primary.connectivity_test()


def selected_provider_name() -> str:
    return os.environ.get("VIDEOLINT_AI_PROVIDER", "gemini").strip().lower()


def gemini_configured() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def jev_configured() -> bool:
    return bool(os.environ.get("BEATAPI_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or
                os.environ.get("TYPESAFE_API_KEY"))


def provider_configured() -> bool:
    name = selected_provider_name()
    return (name == "gemini" and gemini_configured()) or (name == "jev" and jev_configured())


def get_gemini_provider():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise AIProviderError(SKIPPED_NO_KEY)
    from .gemini_provider import GeminiProvider
    return GeminiProvider(api_key=key)


def get_transcriber() -> SpeechTranscriber:
    return get_gemini_provider()


def get_jev_provider() -> AIJudgmentProvider:
    """Prefer the explicitly free BeatAPI route when more than one Jev key exists."""
    from .jev_provider import JevProvider
    key = os.environ.get("BEATAPI_API_KEY")
    if key:
        return JevProvider(api_key=key, beatapi=True)
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return JevProvider(api_key=key, openrouter=True)
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return JevProvider(api_key=key)
    raise AIProviderError("AI editorial checks skipped — Jev API key not configured (set BEATAPI_API_KEY).")


def get_provider() -> AIJudgmentProvider:
    name = selected_provider_name()
    if name == "gemini":
        return get_gemini_provider()
    if name == "jev":
        try:
            jev = get_jev_provider()
        except AIProviderError:
            return get_gemini_provider()
        try:
            gemini = get_gemini_provider()
        except AIProviderError:
            return jev
        return FallbackJudgmentProvider(jev, gemini)
    raise AIProviderError(f"AI provider '{name}' is not installed")
