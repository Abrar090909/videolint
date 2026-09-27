"""Candidate selection and focused, provider-independent editorial checkers."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .ai import AIJudgmentProvider, AIProviderError, Judgment, SpeechTranscriber
from .media import extract_speech_audio

log = logging.getLogger(__name__)

FINDING_THRESHOLD = 0.80
STOPWORDS = set("a an and are as at be been but by for from have in into is it its of on or our so that the their there these this to was were what when why will with you your".split())


def transcribe(video: Path, workdir: Path, provider: SpeechTranscriber) -> list[dict]:
    """Extract speech audio then transcribe it.

    Raises AIProviderError only for genuine provider (API) failures.
    Audio extraction failures are surfaced as-is so callers can distinguish
    infrastructure problems from transcription problems.
    """
    audio = workdir / "speech.m4a"
    try:
        try:
            extract_speech_audio(video, audio)
        except Exception as exc:
            # ffmpeg extraction failure — do NOT blame the AI provider
            log.error("Audio extraction failed before transcription: %s", exc)
            raise AIProviderError(f"Audio extraction failed: {exc}") from exc
        return provider.transcribe(audio)
    finally:
        audio.unlink(missing_ok=True)


def validate_transcript(segments: list[dict], duration_ms: int) -> list[dict]:
    """Reject unusable timestamps before checkers make decisions from them."""
    if not isinstance(segments, list):
        raise ValueError("Transcript is not a list of timestamped segments")
    validated = []
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Transcript contains a malformed segment")
        start, end, text = segment.get("startMs"), segment.get("endMs"), segment.get("text")
        if (type(start) is not int or type(end) is not int or start < 0 or
                end <= start or end > duration_ms + 1000 or
                not isinstance(text, str) or not text.strip()):
            raise ValueError("Transcript contains an invalid timestamp or empty text")
        validated.append({**segment, "text": text.strip()})
    validated.sort(key=lambda item: item["startMs"])
    return validated


def duration_check(goal: str, duration_ms: int) -> list[dict]:
    # Only explicit numeric targets are treated as measurable constraints.
    match = re.search(r"\b(\d{1,3})\s*[- ]?\s*(second|sec|minute|min)s?\b", goal, re.I)
    if not match:
        return []
    target = int(match.group(1)) * (60 if match.group(2).lower().startswith("min") else 1)
    actual = duration_ms / 1000
    tolerance = max(2, target * 0.1)
    if abs(actual - target) <= tolerance:
        return []
    return [{"id": "duration_goal", "type": "duration_goal", "category": "goal",
             "startMs": 0, "endMs": duration_ms, "severity": "warning", "confidence": 1,
             "title": "Duration differs from goal", "explanation": f"Goal specifies {target}s; export is {actual:.1f}s.",
             "source": "rule", "evidence": {"metrics": {"targetSeconds": target, "actualSeconds": round(actual, 2)}},
             "suggestedAction": "Adjust the edit length or revise the goal."}]


def _finding(kind: str, category: str, start: int, end: int, judgment: Judgment,
             title: str, transcript: str, metrics: dict, action: str,
             provider: AIJudgmentProvider) -> dict:
    return {"id": f"{kind}_{start}", "type": kind, "category": category,
            "startMs": start, "endMs": end, "severity": "warning",
            "confidence": round(judgment.confidence, 3), "title": title,
            "explanation": judgment.reason, "source": provider.name,
            "evidence": {"transcript": transcript, "metrics": {**metrics, "model": provider.model}},
            "suggestedAction": action}


def _is_finding(judgment: Judgment) -> bool:
    return judgment.verdict == "yes" and judgment.confidence >= FINDING_THRESHOLD


def _words(segments: list[dict], start: int, end: int) -> str:
    return " ".join(s["text"] for s in segments
                    if s["endMs"] > start and s["startMs"] < end).strip()


def _chunks(segments: list[dict], max_ms: int = 8000) -> list[dict]:
    """Build bounded transcript sections from word or sentence timestamps."""
    chunks, current = [], []
    for segment in sorted(segments, key=lambda s: s["startMs"]):
        if current and (segment["startMs"] - current[-1]["endMs"] > 2500 or
                        segment["endMs"] - current[0]["startMs"] > max_ms):
            chunks.append({"startMs": current[0]["startMs"], "endMs": current[-1]["endMs"],
                           "text": " ".join(s["text"] for s in current)})
            current = []
        current.append(segment)
        if len(current) >= 3 and re.search(r"[.!?][\"']?$", segment["text"]):
            chunks.append({"startMs": current[0]["startMs"], "endMs": current[-1]["endMs"],
                           "text": " ".join(s["text"] for s in current)})
            current = []
    if current:
        chunks.append({"startMs": current[0]["startMs"], "endMs": current[-1]["endMs"],
                       "text": " ".join(s["text"] for s in current)})
    return chunks


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[\w']{3,}", text.lower()) if word not in STOPWORDS}


def _similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / max(1, len(left | right))


class InterruptedThoughtChecker:
    id = "interrupted_thought"

    def run(self, segments: list[dict], shots: list[dict], provider: AIJudgmentProvider) -> list[dict]:
        issues = []
        for shot in shots[1:]:
            cut = shot["startMs"]
            before_segments = [s for s in segments if s["endMs"] > cut - 4000 and
                               s["startMs"] < cut and (s["startMs"] + s["endMs"]) / 2 < cut]
            after_segments = [s for s in segments if s["startMs"] < cut + 4000 and
                              s["endMs"] > cut and (s["startMs"] + s["endMs"]) / 2 >= cut]
            if not before_segments or not after_segments:
                continue
            if cut - before_segments[-1]["endMs"] > 1500 or after_segments[0]["startMs"] - cut > 1500:
                continue
            before = " ".join(s["text"] for s in before_segments)
            after = " ".join(s["text"] for s in after_segments)
            if len(before.split()) < 3 or len(after.split()) < 3:
                continue
            judgment = provider.judge(
                "Does this edit interrupt a single semantic thought? YES only if the words across the cut reveal missing content or an abrupt, confusing semantic jump. A visual cut between words in a fluent sentence is NO, even when it falls mid-sentence. Repetition is a separate check. If the transcript cannot establish an interruption, answer UNCERTAIN.",
                {"cutTimestampMs": cut, "transcriptBeforeCut": before, "transcriptAfterCut": after,
                 "speechGapAtCutMs": max(0, after_segments[0]["startMs"] - before_segments[-1]["endMs"])},
            )
            if _is_finding(judgment):
                issues.append(_finding(self.id, "cut", cut, cut, judgment,
                    "Possible interrupted thought", f"Before: {before}\nAfter: {after}",
                    {"cutMs": cut}, "Review the cut with audio and restore missing context if needed.", provider))
        return issues


class RepetitionChecker:
    id = "repetition"

    def run(self, segments: list[dict], provider: AIJudgmentProvider) -> list[dict]:
        chunks = _chunks(segments)
        candidates = []
        for i, current in enumerate(chunks):
            for earlier in chunks[:i]:
                if current["startMs"] - earlier["startMs"] < 1500:
                    continue
                similarity = _similarity(earlier["text"], current["text"])
                if similarity >= 0.18:
                    candidates.append((similarity, earlier, current))
        candidates.sort(key=lambda item: item[0], reverse=True)
        issues, seen = [], set()
        for similarity, earlier, current in candidates[:12]:
            if current["startMs"] in seen:
                continue
            judgment = provider.judge(
                "Does the current passage substantially repeat the same specific idea already communicated in the earlier passage without adding meaningful new information?",
                {"earlierStartMs": earlier["startMs"], "currentStartMs": current["startMs"],
                 "earlierPassage": earlier["text"], "currentPassage": current["text"]},
            )
            if _is_finding(judgment):
                seen.add(current["startMs"])
                issues.append(_finding(self.id, "repetition", current["startMs"], current["endMs"],
                    judgment, "Possible repeated idea",
                    f"Earlier: {earlier['text']}\nCurrent: {current['text']}",
                    {"earlierStartMs": earlier["startMs"], "candidateSimilarity": round(similarity, 3)},
                    "Compare both sections and remove the restatement if it adds no value.", provider))
        return issues


class EndingCompletenessChecker:
    id = "ending"

    def run(self, segments: list[dict], duration_ms: int, provider: AIJudgmentProvider) -> list[dict]:
        if not segments:
            return []
        tail = _words(segments, max(0, duration_ms - 8000), duration_ms)
        if not tail:
            return []
        gap = max(0, duration_ms - segments[-1]["endMs"])
        judgment = provider.judge(
            "Does this video end with an unfinished spoken sentence or semantic thought? A complete thought followed by a natural ending is no.",
            {"endTimestampMs": duration_ms, "finalTranscript": tail, "gapAfterSpeechMs": gap},
        )
        if not _is_finding(judgment):
            return []
        return [_finding("incomplete_ending", "cut", max(0, duration_ms - 3000), duration_ms,
            judgment, "Possible incomplete ending", tail, {"gapAfterSpeechMs": gap},
            "Listen to the ending and restore the missing conclusion if needed.", provider)]


class GoalAlignmentChecker:
    id = "topic_alignment"

    def run(self, goal: str, segments: list[dict], duration_ms: int,
            provider: AIJudgmentProvider) -> list[dict]:
        chunks = _chunks(segments)
        if not chunks:
            return []
        ranked = sorted(chunks, key=lambda chunk: _similarity(goal, chunk["text"]), reverse=True)
        selected = sorted(ranked[:3], key=lambda chunk: chunk["startMs"])
        context = [{"startMs": c["startMs"], "endMs": c["endMs"], "text": c["text"][:600]}
                   for c in selected]
        judgment = provider.judge(
            "Do these most relevant transcript sections fail to address the main subject or informational request in the editing goal? Ignore style, visual content, and duration. If the samples are insufficient to tell, answer uncertain.",
            {"timestampMs": selected[0]["startMs"], "editingGoal": goal,
             "selectedTranscriptSections": context},
        )
        if not _is_finding(judgment):
            return []
        return [_finding("topic_mismatch", "goal", 0, duration_ms, judgment,
            "Possible topic mismatch", "\n".join(c["text"] for c in context),
            {"sectionCount": len(context)},
            "Compare the edit against the original request and include the missing subject if needed.", provider)]


def interrupted_thoughts(segments: list[dict], shots: list[dict], provider: AIJudgmentProvider) -> list[dict]:
    return InterruptedThoughtChecker().run(segments, shots, provider)


def repetition(segments: list[dict], provider: AIJudgmentProvider) -> list[dict]:
    return RepetitionChecker().run(segments, provider)


def ending(segments: list[dict], duration_ms: int, provider: AIJudgmentProvider) -> list[dict]:
    return EndingCompletenessChecker().run(segments, duration_ms, provider)


def topic_alignment(goal: str, segments: list[dict], duration_ms: int,
                    provider: AIJudgmentProvider) -> list[dict]:
    return GoalAlignmentChecker().run(goal, segments, duration_ms, provider)

