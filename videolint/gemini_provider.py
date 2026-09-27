"""Gemini implementation of the provider contract. No video is sent for judgments."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from google import genai
from google.genai import types

from .ai import AIProviderError, Judgment
from . import media


JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "description": "A short explanation grounded in the supplied transcript."},
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}


def _safe_error(exc: Exception) -> AIProviderError:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    suffix = f" (HTTP {code})" if isinstance(code, int) else ""
    return AIProviderError(f"Gemini {type(exc).__name__}{suffix}; checker could not complete")


def _offset_ms(value: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(value))
    if not match:
        raise AIProviderError("Gemini transcription returned an invalid word timestamp")
    return round(float(match.group(1)) * 1000)


def _audio_duration_ms(path: Path) -> int:
    result = media.run("-i", str(path), timeout=30)
    output = result.stderr.decode("utf-8", "replace")
    match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", output)
    if not match:
        raise AIProviderError("Could not determine the duration of transcribed audio")
    return round((int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))) * 1000)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-3.6-flash", client=None):
        if not api_key:
            raise AIProviderError("Gemini API key not configured")
        self.model = model
        self._api_key = api_key
        self.client = client or genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=45000))

    def judge(self, question: str, context: dict) -> Judgment:
        # The provider owns the API shape; checkers supply only a bounded question and context.
        prompt = f"Question: {question}\n\nEvidence (JSON): {json.dumps(context, ensure_ascii=False)}\nBase your decision only on this evidence. If it is insufficient, choose uncertain."
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json", "response_json_schema": JUDGMENT_SCHEMA,
                        "temperature": 0, "max_output_tokens": 1024,
                        "thinking_config": {"thinking_level": "LOW"}},
            )
        except Exception as exc:
            raise _safe_error(exc) from exc
        try:
            return Judgment.validate(json.loads(response.text or ""))
        except (ValueError, TypeError) as exc:
            candidates = getattr(response, "candidates", None) or []
            finish = getattr(candidates[0], "finish_reason", None) if candidates else None
            reason = getattr(finish, "name", None) or str(finish or "unknown")
            raise AIProviderError(f"Gemini returned incomplete structured output ({reason[:40]})") from exc

    def _transcription_response(self, uri: str, mime_type: str) -> dict:
        # The SDK currently discards word_info annotations from InteractionContent.
        # Use the documented REST response for this endpoint so timestamps survive.
        payload = {"model": "gemini-3.5-transcribe",
                   "input": [{"type": "audio", "uri": uri, "mime_type": mime_type}],
                   "generation_config": {"transcription_config": {"mode": {
                       "type": "verbatim", "timestamp_granularities": ["word"]}}}}
        request = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            json.dumps(payload).encode(),
            {"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    def transcribe(self, audio_path: Path) -> list[dict]:
        uploaded = None
        try:
            uploaded = self.client.files.upload(file=str(audio_path))
            interaction = self._transcription_response(uploaded.uri, uploaded.mime_type)
            annotations = []
            for step in interaction.get("steps", []):
                for content in step.get("content", []):
                    for annotation in content.get("annotations", []):
                        if annotation.get("type") != "word_info":
                            continue
                        text = (annotation.get("text") or "").strip()
                        if not text:
                            continue
                        annotations.append({"startMs": _offset_ms(annotation.get("start_offset")),
                                            "endOffset": annotation.get("end_offset"), "text": text})
            words = []
            duration_ms = None
            for index, annotation in enumerate(annotations):
                start = annotation["startMs"]
                raw_end = annotation["endOffset"]
                inferred = raw_end is None
                if inferred:
                    if index + 1 < len(annotations):
                        end = annotations[index + 1]["startMs"]
                    else:
                        duration_ms = duration_ms or _audio_duration_ms(audio_path)
                        end = min(duration_ms, start + 1000)
                else:
                    end = _offset_ms(raw_end)
                if end <= start:
                    raise AIProviderError("Gemini transcription returned a nonpositive word range")
                word = {"startMs": start, "endMs": end, "text": annotation["text"]}
                if inferred:
                    word["endInferred"] = True
                words.append(word)
            if not words and (interaction.get("output_text") or "").strip():
                raise AIProviderError("Gemini returned speech without usable timestamps")
            words.sort(key=lambda word: word["startMs"])
            return words
        except AIProviderError:
            raise
        except Exception as exc:
            raise _safe_error(exc) from exc
        finally:
            if uploaded is not None and getattr(uploaded, "name", None):
                try:
                    self.client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    def connectivity_test(self) -> None:
        self.judge("Does the evidence text contain the word VideoLint?", {"text": "VideoLint"})
