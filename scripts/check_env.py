"""Phase 0 check: print GPU/CUDA status and which API keys are present in .env.

Key values are never printed, only whether each one is set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_key  # noqa: E402

REQUIRED_KEYS = ["GCP_PROJECT_ID", "OPENALEX_API_KEY", "GEMINI_API_KEY", "DATABASE_URL"]


def main() -> int:
    """Print the environment report; return 0 if GPU and all keys are available, else 1."""
    ok = True
    print(f"Python: {sys.version.split()[0]}")

    try:
        import torch

        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: yes (CUDA {torch.version.cuda})")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {vram:.1f} GB")
        else:
            print("CUDA available: NO")
            ok = False
    except ImportError:
        print("PyTorch: not installed")
        ok = False

    print("\nAPI keys in .env:")
    for name in REQUIRED_KEYS:
        present = get_key(name) is not None
        ok &= present
        print(f"  {name:<22} {'found' if present else 'MISSING'}")

    print("\nResult:", "all checks passed" if ok else "some checks failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
