"""Redeploy the live services on Render and wait until they are live.

The services were created from the public repository URL, so Render does not redeploy on push;
run this after pushing. Needs RENDER_API_KEY in .env.

Usage:
    python scripts/deploy.py            # API and website
    python scripts/deploy.py site       # website only
    python scripts/deploy.py api        # API only
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_key  # noqa: E402

SERVICES = {"api": "srv-dbk44iff3r2c738kuaa0", "site": "srv-dbk46cflk1mc73aq2h7g"}
URLS = {"api": "https://innovation-intelligence-api.onrender.com", "site": "https://innovation-intelligence.onrender.com"}
DONE = {"live", "build_failed", "update_failed", "canceled", "deactivated"}


def main() -> int:
    """Trigger a deploy of the requested services and report the final status of each."""
    key = get_key("RENDER_API_KEY")
    if not key:
        raise SystemExit("RENDER_API_KEY is missing from .env")
    h = {"Authorization": f"Bearer {key}", "Accept": "application/json", "Content-Type": "application/json"}
    names = sys.argv[1:] or list(SERVICES)
    deploys = {}
    for n in names:
        r = requests.post(f"https://api.render.com/v1/services/{SERVICES[n]}/deploys", headers=h,
                          json={"clearCache": "do_not_clear"}, timeout=30)
        r.raise_for_status()
        deploys[n] = r.json()["id"]
        print(f"{n}: deploy {deploys[n]} started")
    status = {}
    for _ in range(60):
        for n, dep in deploys.items():
            if status.get(n) not in DONE:
                status[n] = requests.get(f"https://api.render.com/v1/services/{SERVICES[n]}/deploys/{dep}",
                                         headers=h, timeout=30).json()["status"]
        print(time.strftime("%H:%M:%S"), status, flush=True)
        if all(s in DONE for s in status.values()):
            break
        time.sleep(20)
    for n in names:
        print(f"{n}: {status[n]}  {URLS[n]}")
    return 0 if all(s == "live" for s in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
