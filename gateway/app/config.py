"""Environment-driven settings for the dub gateway.

Nothing here talks to bigboy. The gateway is meant to run as its own service
(eventually on bigboy, today just locally) and dispatch dub jobs to the
bazzite worker once that worker exists.
"""

from __future__ import annotations

import os

# --- storage -----------------------------------------------------------
DB_PATH = os.getenv("GATEWAY_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "gateway.sqlite3"))

# --- bazzite target (dub worker host) -----------------------------------
# Real deployment target. MAC is the real NIC MAC for the magic packet.
BAZZITE_MAC = os.getenv("BAZZITE_MAC", "b0:82:e2:44:ea:bd")
BAZZITE_HOST = os.getenv("BAZZITE_HOST", "192.168.2.107")

# Readiness probe: the real dub worker HTTP endpoint doesn't exist yet, so we
# probe a port we know is alive on bazzite (sshd) as a stand-in "is the box
# up" check. One-line swap later: point this at the worker's own /health URL
# once it's built, and switch _is_ready() to the HTTP GET used by
# wake_bazzite.py's _is_serving() instead of a bare TCP connect.
BAZZITE_PROBE_PORT = int(os.getenv("BAZZITE_PROBE_PORT", "22"))

# Real relay-based WOL (see wake_bazzite.py) only works when the magic packet
# originates from bigboy's source IP via the fermentation station's DNAT
# rule. We are not touching bigboy for this task, so WAKE_SIMULATE is forced
# on here. Flip it off (and supply RELAY_HOST/RELAY_PORT) once this code runs
# on bigboy for real.
WAKE_SIMULATE = os.getenv("WAKE_SIMULATE", "true").lower() == "true"
RELAY_HOST = os.getenv("WAKE_RELAY_HOST", "192.168.1.99")
RELAY_PORT = int(os.getenv("WAKE_RELAY_PORT", "9"))

WAKE_PROBE_TIMEOUT = float(os.getenv("WAKE_PROBE_TIMEOUT", "3"))
WAKE_TIMEOUT = float(os.getenv("WAKE_TIMEOUT", "120"))
WAKE_POLL_INTERVAL = float(os.getenv("WAKE_POLL_INTERVAL", "2"))

# --- the (not yet built) dub worker API on bazzite ----------------------
WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", f"http://{BAZZITE_HOST}:8420")

# --- Jellyfin auth check -------------------------------------------------
JELLYFIN_BASE_URL = os.getenv("JELLYFIN_BASE_URL", "https://jellyfin.osmosis.page")
JELLYFIN_API_KEY = os.getenv("JELLYFIN_API_KEY")  # unset on this dev machine
# This dev machine CAN reach jellyfin.osmosis.page (verified: 401 on
# /Users/Me with no key, 200 on /System/Info/Public), but has no API key
# checked into this repo/host, so we simulate the auth check by default.
# One-line swap: set JELLYFIN_API_KEY (or JELLYFIN_SIMULATE_AUTH=false with
# a key available) once this runs somewhere that has the real key.
JELLYFIN_SIMULATE_AUTH = os.getenv("JELLYFIN_SIMULATE_AUTH", "true").lower() == "true" or not JELLYFIN_API_KEY

# --- dispatch -------------------------------------------------------------
DISPATCH_POLL_INTERVAL = float(os.getenv("DISPATCH_POLL_INTERVAL", "1"))
GLOBAL_CONCURRENCY = 1  # fixed: only one dub job may run at a time, by design

# --- CORS (unauthenticated API, called directly from injected browser JS) --
# "*" by default since there's no auth anyway at this stage; set to
# "https://jellyfin.osmosis.page" (comma-separated for more than one) once
# the real Jellyfin domain is the only caller that should be allowed.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("GATEWAY_ALLOWED_ORIGINS", "*").split(",")]
