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

    def extract_requirements(self, goal: str) -> list[dict]:
        """Split common visual, audio, duration, and subjective constraints into claims.

        Unknown goal language is retained as an unverified claim instead of silently
        disappearing from the alignment report.
        """
        text = goal.lower()
        claims = []
        def add(claim_id: str, label: str, source: str, question: str = "") -> None:
            claims.append({"id": claim_id, "requirement": label, "source": source,
                           "question": question})

        if re.search(r"\b(jungle|rainforest|forest)\b", text):
            add("jungle_environment", "Jungle environment present", "visual",
                "Do the sampled video frames visibly show a jungle or dense tropical forest?")
        if re.search(r"\b(continuous movement|continuous motion|moving through|movement through|move through)\b", text):
            add("continuous_movement", "Continuous movement through the environment", "visual",
                "Do the sampled frames support continuous movement through the environment? Only say yes when visible evidence supports motion; still frames alone may be insufficient.")
        if re.search(r"\b(multiple people|group|people|person|people present)\b", text):
            add("multiple_people_present", "Multiple people present", "visual",
                "Are at least two people visibly present in the sampled video frames?")
        if re.search(r"\b(voices?|audible|speaking|people.*sound|group.*voice)\b", text):
            add("people_audible", "People or speech audible", "speech_audio",
                "Does the audio contain clearly audible human speech or vocal activity?")
        if re.search(r"\b(natural|environmental|ambient|nature)\b.*\b(sound|audio|noise)\b|\b(sound|audio)\b.*\b(natural|environmental|ambient|nature)\b|\bnatural environment\b.*\b(audible|clearly audible)\b", text):
            add("natural_environmental_audio", "Natural environmental audio present", "audio_semantic",
                "Does the audio contain natural environmental sounds (such as wind, water, insects, birds, or foliage), rather than only speech or music?")
        duration = re.search(r"\b(\d{1,3})\s*[- ]?\s*(second|sec|minute|min)s?\b", text)
        if duration:
            target = int(duration.group(1)) * (60 if duration.group(2).startswith("min") else 1)
            add("duration", f"Approximately {target} seconds", "duration_rule")
            claims[-1]["targetSeconds"] = target
        if re.search(r"\b(energetic|immersive|exciting|adventure feel|adventurous)\b", text):
            add("energetic_immersive_feel", "Energetic, immersive adventure feel", "multimodal_semantic",
                "Taken together, do the sampled visuals and available audio create an energetic, immersive adventure feel? Base this only on supplied evidence.")
        if not claims:
            add("goal_content", goal.strip(), "unmapped")
        return claims

    def verify(self, goal: str, segments: list[dict], duration_ms: int,
               audio_windows: list[dict], frames: list[dict],
               vision_provider: AIJudgmentProvider | None,
               audio_provider: AIJudgmentProvider | None = None,
               audio_path: Path | None = None,
               transcript_status: str | None = None) -> dict:
        """Evaluate goal claims against only the evidence sources that can support them."""
        claims = self.extract_requirements(goal)
        frame_evidence = [{"timestampMs": frame["timestampMs"]} for frame in frames]
        observations = []
        image_decisions = {}
        image_error = None
        visual_claims = [claim for claim in claims if claim["source"] in
                         ("visual", "multimodal_semantic")]
        if vision_provider and frames:
            try:
                analysis = vision_provider.analyze_goal_with_images(
                    goal, visual_claims, [f["path"] for f in frames],
                    [f["timestampMs"] for f in frames],
                    {"transcript": " ".join(s["text"] for s in segments)[:1200],
                     "nonSilentWindowCount": sum(w.get("peak", 0) > 0 for w in audio_windows)})
                observations = analysis.get("observations", [])
                image_decisions = {row["claimId"]: row for row in analysis.get("claims", [])}
            except Exception as exc:
                image_error = str(exc)
                log.warning("Could not analyze goal against sampled frames: %s", exc)
        audible_windows = [w for w in audio_windows if w.get("peak", 0) > 0]
        speech = " ".join(s["text"] for s in segments if s.get("text", "").strip())
        audio_claims = [claim for claim in claims if claim["source"] == "audio_semantic" or
                        (claim["source"] == "speech_audio" and not speech)]
        audio_decisions = {}
        audio_error = None
        if audio_provider and audio_path and audible_windows and audio_claims:
            try:
                analysis = audio_provider.analyze_goal_with_audio(
                    goal, audio_claims, audio_path,
                    {"nonSilentWindowCount": len(audible_windows),
                     "totalWindowCount": len(audio_windows)})
                audio_decisions = {row["claimId"]: row for row in analysis.get("claims", [])}
            except Exception as exc:
                audio_error = str(exc)
                log.warning("Could not analyze goal against audio: %s", exc)
        results = []
        for claim in claims:
            result = {k: v for k, v in claim.items() if k != "question"}
            result.update({"status": "UNVERIFIED", "component": None,
                           "evidence": {}, "reason": "Required evidence is unavailable."})
            if claim["source"] == "duration_rule":
                target = claim["targetSeconds"]
                actual = duration_ms / 1000
                tolerance = max(2, target * 0.1)
                result.update(status="PASS" if abs(actual-target) <= tolerance else "FAIL",
                              component="Rule", reason=f"Measured duration is {actual:.2f}s; target is approximately {target}s.",
                              evidence={"durationMs": duration_ms, "targetSeconds": target,
                                        "toleranceSeconds": tolerance})
            elif claim["source"] == "visual":
                result["evidence"] = {"sampledFrames": frame_evidence,
                                      "visualObservations": observations}
                decision = image_decisions.get(claim["id"])
                if decision:
                    result.update(status=("PASS" if decision["verdict"] == "yes" else
                                          "FAIL" if decision["verdict"] == "no" else "UNVERIFIED"),
                                  component="Gemini", reason=decision["reason"],
                                  confidence=decision["confidence"])
                elif vision_provider and frames:
                    result.update(component="Gemini", reason=image_error or "Gemini did not return a decision for this claim.")
                else:
                    result["reason"] = "No sampled frames or vision provider are available."
            elif claim["source"] == "speech_audio":
                result["evidence"] = {"transcript": speech[:1200],
                    "transcriptStatus": transcript_status or ("SUCCESS" if speech else "NO_SPEECH"),
                    "nonSilentWindowCount": len(audible_windows)}
                if speech:
                    result.update(status="PASS", component="Gemini", reason="Timestamped speech is present in the transcript.")
                elif not audible_windows:
                    result.update(status="FAIL", component="Rule", reason="FFmpeg measured no audible audio in the analyzed windows.")
                elif claim["id"] in audio_decisions:
                    decision = audio_decisions[claim["id"]]
                    result.update(status=("PASS" if decision["verdict"] == "yes" else
                                          "FAIL" if decision["verdict"] == "no" else "UNVERIFIED"),
                                  component="Gemini", reason=decision["reason"],
                                  confidence=decision["confidence"])
                elif audio_provider and audio_path and audible_windows:
                    result.update(component="Gemini", reason=audio_error or "Gemini did not return a decision for this claim.")
                else:
                    result["reason"] = "Audio is present, but no transcript or audio classifier can establish human vocal activity."
            elif claim["source"] == "audio_semantic":
                result["evidence"] = {"nonSilentWindowCount": len(audible_windows),
                    "totalWindowCount": len(audio_windows)}
                if not audible_windows:
                    result.update(status="FAIL", component="Rule",
                                  reason="FFmpeg found no non-silent audio windows, so natural environmental audio is absent.")
                elif claim["id"] in audio_decisions:
                    decision = audio_decisions[claim["id"]]
                    result.update(status=("PASS" if decision["verdict"] == "yes" else
                                          "FAIL" if decision["verdict"] == "no" else "UNVERIFIED"),
                                  component="Gemini", reason=decision["reason"],
                                  confidence=decision["confidence"])
                elif audio_provider and audio_path and audible_windows:
                    result.update(component="Gemini", reason=audio_error or "Gemini did not return a decision for this claim.")
                else:
                    result["reason"] = "FFmpeg can measure signal levels but cannot identify natural environmental sounds."
            elif claim["source"] == "multimodal_semantic":
                result["evidence"] = {"sampledFrames": frame_evidence,
                    "visualObservations": observations, "transcript": speech[:1200],
                    "nonSilentWindowCount": len(audible_windows)}
                decision = image_decisions.get(claim["id"])
                if decision:
                    result.update(status=("PASS" if decision["verdict"] == "yes" else
                                          "FAIL" if decision["verdict"] == "no" else "UNVERIFIED"),
                                  component="Gemini", reason=decision["reason"],
                                  confidence=decision["confidence"])
                elif vision_provider and frames:
                    result.update(component="Gemini", reason=image_error or "Gemini did not return a decision for this claim.")
                else:
                    result["reason"] = "Sampled visuals are unavailable for the subjective judgment."
            results.append(result)
        return {"requirements": results, "visualObservations": observations,
                "sampledFrameCount": len(frames)}

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
