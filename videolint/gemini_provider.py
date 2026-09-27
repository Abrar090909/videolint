"""Gemini implementation of the provider contract. No video is sent for judgments."""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types

from .ai import AIProviderError, Judgment
from . import media

log = logging.getLogger(__name__)


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

GOAL_IMAGE_SCHEMA = {
    "type": "object", "properties": {
        "observations": {"type": "array", "items": {"type": "object", "properties": {
            "timestampMs": {"type": "integer", "minimum": 0}, "observation": {"type": "string"}},
            "required": ["timestampMs", "observation"], "additionalProperties": False}},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "claimId": {"type": "string"},
            "verdict": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}},
            "required": ["claimId", "verdict", "confidence", "reason"],
            "additionalProperties": False}}},
    "required": ["observations", "claims"], "additionalProperties": False,
}

GOAL_AUDIO_SCHEMA = {
    "type": "object", "properties": {"claims": {"type": "array", "items": {
        "type": "object", "properties": {
            "claimId": {"type": "string"},
            "verdict": {"type": "string", "enum": ["yes", "no", "uncertain"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}},
        "required": ["claimId", "verdict", "confidence", "reason"],
        "additionalProperties": False}}},
    "required": ["claims"], "additionalProperties": False,
}


def _safe_error(exc: Exception) -> AIProviderError:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    suffix = f" (HTTP {code})" if isinstance(code, int) else ""
    # Include the original exception message so the real cause is visible in job reports
    original = str(exc).strip()
    detail = f": {original}" if original else ""
    if os.environ.get("VIDEOLINT_DEV_MODE") == "1":
        log.exception("Gemini %s%s%s", type(exc).__name__, suffix, detail)
    return AIProviderError(f"Gemini {type(exc).__name__}{suffix}; checker could not complete{detail}")


def _offset_ms(value) -> Optional[int]:
    """Parse a Gemini timestamp like '1.200s' into milliseconds.

    Returns None for missing or unparseable values so callers can skip
    malformed annotations instead of crashing the entire transcription.
    """
    if value is None:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(value))
    if not match:
        log.warning("Unparseable Gemini timestamp: %r", value)
        return None
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
                        start_ms = _offset_ms(annotation.get("start_offset"))
                        if start_ms is None:
                            # Skip annotations with missing start timestamps
                            log.debug("Skipping word with missing start: %r", text)
                            continue
                        annotations.append({"startMs": start_ms,
                                            "endOffset": annotation.get("end_offset"), "text": text})
            words = []
            duration_ms = None
            for index, annotation in enumerate(annotations):
                start = annotation["startMs"]
                raw_end = annotation["endOffset"]
                end_ms = _offset_ms(raw_end)
                inferred = end_ms is None
                if inferred:
                    if index + 1 < len(annotations):
                        end = annotations[index + 1]["startMs"]
                    else:
                        duration_ms = duration_ms or _audio_duration_ms(audio_path)
                        end = min(duration_ms, start + 1000)
                else:
                    end = end_ms
                if end <= start:
                    # Skip rather than crash on zero/negative ranges
                    log.debug("Skipping word with nonpositive range: %s (%d-%d)",
                              annotation["text"], start, end)
                    continue
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

    def judge_with_images(self, question: str, context: dict,
                          image_paths: list[Path]) -> Judgment:
        """Like judge(), but includes images as inline parts for visual analysis."""
        parts = []
        for img_path in image_paths:
            if not img_path.is_file():
                continue
            data = base64.b64encode(img_path.read_bytes()).decode()
            parts.append(types.Part.from_bytes(data=base64.b64decode(data),
                                               mime_type="image/jpeg"))
        text_part = (f"Question: {question}\n\n"
                     f"Evidence (JSON): {json.dumps(context, ensure_ascii=False)}\n"
                     f"Base your decision only on the images and evidence above. "
                     f"If it is insufficient, choose uncertain.")
        parts.append(text_part)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=parts,
                config={"response_mime_type": "application/json",
                        "response_json_schema": JUDGMENT_SCHEMA,
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
            raise AIProviderError(
                f"Gemini returned incomplete structured output ({reason[:40]})"
            ) from exc

    def analyze_goal_with_images(self, goal: str, claims: list[dict], image_paths: list[Path],
                                 timestamps_ms: list[int], audio_evidence: dict) -> dict:
        """Describe sampled frames and assess all visual claims in one model request."""
        parts = []
        included = []
        for timestamp, img_path in zip(timestamps_ms, image_paths):
            if not img_path.is_file():
                continue
            parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"))
            included.append(timestamp)
        prompt = {"goal": goal, "claims": claims, "frameTimestampsMs": included,
                  "audioEvidence": audio_evidence}
        parts.append(
            "For each frame, describe only visible content and preserve its timestamp. "
            "Then judge each claim using only the supplied frames and stated audio evidence. "
            "Do not infer movement from one still frame unless the sequence supports it. "
            "Use uncertain when evidence is insufficient. Return every claim exactly once.\n"
            + json.dumps(prompt, ensure_ascii=False)
        )
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=parts,
                config={"response_mime_type": "application/json",
                        "response_json_schema": GOAL_IMAGE_SCHEMA,
                        "temperature": 0, "max_output_tokens": 2048,
                        "thinking_config": {"thinking_level": "LOW"}},
            )
            result = json.loads(response.text or "")
            allowed = set(included)
            result["observations"] = [row for row in result.get("observations", [])
                if isinstance(row, dict) and row.get("timestampMs") in allowed
                and isinstance(row.get("observation"), str)]
            expected = {claim["id"] for claim in claims}
            result["claims"] = [row for row in result.get("claims", [])
                if isinstance(row, dict) and row.get("claimId") in expected]
            return result
        except Exception as exc:
            raise _safe_error(exc) from exc

    def analyze_goal_with_audio(self, goal: str, claims: list[dict], audio_path: Path,
                                audio_evidence: dict) -> dict:
        """Assess all audio claims from the audio bytes in one model request."""
        prompt = {"goal": goal, "claims": claims, "audioEvidence": audio_evidence}
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=audio_path.read_bytes(), mime_type="audio/mp4"),
                          "Listen to the supplied audio. Judge each claim only from audible content; "
                          "use uncertain when evidence is insufficient. Return every claim exactly once.\n"
                          + json.dumps(prompt, ensure_ascii=False)],
                config={"response_mime_type": "application/json",
                        "response_json_schema": GOAL_AUDIO_SCHEMA,
                        "temperature": 0, "max_output_tokens": 1024,
                        "thinking_config": {"thinking_level": "LOW"}},
            )
            result = json.loads(response.text or "")
            expected = {claim["id"] for claim in claims}
            result["claims"] = [row for row in result.get("claims", [])
                if isinstance(row, dict) and row.get("claimId") in expected]
            return result
        except Exception as exc:
            raise _safe_error(exc) from exc

    def connectivity_test(self) -> None:
        self.judge("Does the evidence text contain the word VideoLint?", {"text": "VideoLint"})
