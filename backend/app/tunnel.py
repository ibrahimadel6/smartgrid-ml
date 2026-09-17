"""Expose the local server to the internet automatically at startup.

Provider cascade (first that works wins):
1. Cloudflare quick tunnel (`cloudflared`) - no account needed, most reliable.
   The binary is downloaded once into `backend/bin/` if it is not on PATH.
2. pyngrok - used when `NGROK_AUTHTOKEN` is configured.
3. py-localtunnel (`pylt`) - last-resort fallback.

The public URL is printed to the terminal in a prominent banner once it is up.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request

from .config import BACKEND_DIR

try:  # avoid a flashing console window on Windows
    CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
except Exception:
    CREATE_NO_WINDOW = 0

_URL_RE = re.compile(
    r"https?://[\w.\-]+\.(?:trycloudflare\.com|localtunnel\.me|loca\.lt|ngrok[\w\-]*\.(?:app|io))",
    re.I,
)
_WARN_SECONDS = 45

_BIN_DIR = BACKEND_DIR / "bin"
_PROCS: list = []


# ----------------------------------------------------------------------------
# output helpers
# ----------------------------------------------------------------------------
def _print_line(text: str) -> None:
    try:
        print(text, flush=True)
    except Exception:
        pass


def _print_banner(url: str, port: int) -> None:
    bar = "=" * 68
    _print_line("")
    _print_line(bar)
    _print_line("  PUBLIC URL (open it from any device on the internet):")
    _print_line(f"      {url.strip()}")
    _print_line(bar)
    _print_line("")
    _print_line(f"  Local:      http://127.0.0.1:{port}")
    _print_line("  Keep this window open while sharing the link.")
    _print_line("  Stop:       Ctrl+C  (the tunnel closes with the server).")
    _print_line("")


def print_local_urls(port: int) -> None:
    _print_line(f"  Local server:  http://127.0.0.1:{port}")
    try:
        _print_line(f"  LAN access:    http://{_lan_ip()}:{port}")
    except Exception:
        pass


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _extract_url(line: str):
    m = _URL_RE.search(line or "")
    return m.group(0).rstrip(' .,;:()[]"\'') if m else None


def _cleanup() -> None:
    for proc in _PROCS:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


atexit.register(_cleanup)


# ----------------------------------------------------------------------------
# 1) Cloudflare quick tunnel
# ----------------------------------------------------------------------------
def _cloudflared_asset() -> str | None:
    if sys.platform.startswith("win"):
        return "cloudflared-windows-amd64.exe"
    if sys.platform.startswith("linux"):
        return "cloudflared-linux-amd64"
    if sys.platform == "darwin":
        return "cloudflared-darwin-amd64.tgz"  # needs extra unpacking; skipped
    return None


def _ensure_cloudflared() -> str | None:
    found = shutil.which("cloudflared")
    if found:
        return found
    exe_name = "cloudflared.exe" if sys.platform.startswith("win") else "cloudflared"
    local = _BIN_DIR / exe_name
    if local.exists():
        return str(local)
    asset = _cloudflared_asset()
    if not asset or asset.endswith(".tgz"):
        return None
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
    try:
        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        _print_line(f"[tunnel] downloading cloudflared ({asset}) ...")
        tmp = local.with_suffix(local.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(local)
        try:
            local.chmod(0o755)
        except Exception:
            pass
        _print_line("[tunnel] cloudflared ready.")
        return str(local)
    except Exception as exc:
        _print_line(f"[tunnel] cloudflared download failed: {exc}")
        return None


def _spawn(cmd, on_url) -> None:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=CREATE_NO_WINDOW,
        stdin=subprocess.DEVNULL,
    )
    _PROCS.append(proc)

    def _read():
        for line in proc.stdout:
            url = _extract_url(line)
            if url:
                on_url(url)

    threading.Thread(target=_read, daemon=True).start()


def _start_cloudflared(port: int, on_url) -> bool:
    exe = _ensure_cloudflared()
    if not exe:
        return False
    _spawn(
        [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        on_url,
    )
    return True


# ----------------------------------------------------------------------------
# 2) ngrok (needs NGROK_AUTHTOKEN)
# ----------------------------------------------------------------------------
def _ngrok_available() -> bool:
    if os.environ.get("NGROK_AUTHTOKEN"):
        return True
    return os.path.exists(os.path.join(os.path.expanduser("~"), ".config", "ngrok", "ngrok.yml"))


def _start_ngrok(port: int, on_url) -> None:
    from pyngrok import conf, ngrok

    token = os.environ.get("NGROK_AUTHTOKEN")
    if token:
        conf.get_default().auth_token = token
    ngrok.connect(port)
    for tunnel in ngrok.get_tunnels():
        on_url(str(tunnel.public_url))


# ----------------------------------------------------------------------------
# 3) localtunnel (pylt)
# ----------------------------------------------------------------------------
def _start_localtunnel(port: int, on_url) -> bool:
    exe = shutil.which("pylt")
    cmd = [exe, "port", str(port)] if exe else [sys.executable, "-m", "py_localtunnel.cli", "port", str(port)]
    try:
        _spawn(cmd, on_url)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------
def start_public_tunnel(port: int = 8000) -> None:
    """Open a public tunnel to `port` and print its URL. Never crashes the app."""
    state: dict = {"url": None}

    def on_url(url: str) -> None:
        if state["url"]:
            return
        state["url"] = url
        _print_banner(url, port)

    def _warn_if_slow() -> None:
        def check():
            if not state["url"]:
                _print_line(
                    f"[tunnel] no public URL after {_WARN_SECONDS}s - the app is still "
                    "running locally (check your internet connection)."
                )

        timer = threading.Timer(_WARN_SECONDS, check)
        timer.daemon = True
        timer.start()

    # 1) Cloudflare quick tunnel (no account, most reliable)
    try:
        if _start_cloudflared(port, on_url):
            _warn_if_slow()
            return
        _print_line("[tunnel] cloudflared unavailable, trying other providers...")
    except Exception as exc:
        _print_line(f"[tunnel] cloudflared failed ({exc}), trying other providers...")

    # 2) ngrok when a token is configured
    if _ngrok_available():
        try:
            threading.Thread(target=_start_ngrok, args=(port, on_url), daemon=True).start()
            _warn_if_slow()
            return
        except Exception as exc:
            _print_line(f"[tunnel] ngrok failed ({exc}), trying localtunnel...")

    # 3) localtunnel fallback
    try:
        if _start_localtunnel(port, on_url):
            _warn_if_slow()
            return
    except Exception as exc:
        _print_line(f"[tunnel] localtunnel failed: {exc}")

    _print_line("[tunnel] could not open a public tunnel; serving locally only.")


__all__ = ["start_public_tunnel", "print_local_urls"]