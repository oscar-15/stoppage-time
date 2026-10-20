"""Tiny HTTP layer. Set FIXTURES=/some/dir to read canned files instead (for tests)."""
import json
import os
import time
from pathlib import Path

import requests

_session = requests.Session()
_session.headers["User-Agent"] = "stoppage-time-bot/1.0 (+github)"
last_headers: dict = {}


def _fixture(name):
    fx = os.environ.get("FIXTURES")
    if not fx:
        return None
    p = Path(fx) / name
    if not p.exists():
        raise FileNotFoundError(f"missing fixture {p}")
    return p.read_bytes()


def get(url, params=None, fixture=None, as_json=True, retries=3, headers=None):
    """Return parsed JSON, or raw bytes when as_json=False. Returns None on 404."""
    if os.environ.get("FIXTURES"):
        try:
            raw = _fixture(fixture)
        except FileNotFoundError:
            return None
        return json.loads(raw) if as_json else raw
    err = None
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=40, headers=headers)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            last_headers.clear()
            last_headers.update({k.lower(): v for k, v in r.headers.items()})
            return r.json() if as_json else r.content
        except requests.RequestException as e:   # network blip
            err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"GET failed {url}: {err}")


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")
