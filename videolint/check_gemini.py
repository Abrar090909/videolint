"""Run one small Gemini request to verify local key, model access, and JSON parsing."""
from __future__ import annotations

import sys

from .ai import AIProviderError, get_gemini_provider


def main() -> int:
    try:
        provider = get_gemini_provider()
        provider.connectivity_test()
    except AIProviderError as exc:
        print(f"Gemini connection failed: {exc}")
        return 1
    print(f"Gemini connection OK ({provider.model}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
