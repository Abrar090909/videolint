"""Optional Jev credential check; does not upload a video."""
from __future__ import annotations

import sys

from .ai import AIProviderError, get_jev_provider


def main() -> int:
    try:
        provider = get_jev_provider()
        provider.connectivity_test()
    except AIProviderError as exc:
        print(f"Jev connection failed: {exc}")
        return 1
    print(f"Jev connection OK ({provider.model}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
