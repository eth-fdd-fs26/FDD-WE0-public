"""One-command launcher for the Nuclear Central Manager panel.

    python workshop/launch.py

Starts the FastAPI backend (which also serves the built React front end) and
opens your browser. If port 8000 is already taken (e.g. an earlier launch is
still running), it automatically moves to the next free port instead of failing
silently. If the front end hasn't been built yet, the backend still runs and the
page explains how to build it.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser

HOST = "127.0.0.1"
PORT = int(os.environ.get("NCM_PORT", "8000"))

# Make `workshop.backend...` importable no matter where this is launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def _find_free_port(start: int, tries: int = 20) -> int:
    for p in range(start, start + tries):
        if _port_is_free(p):
            return p
    return start  # give up; uvicorn will report the bind error clearly


def _open_browser(port: int) -> None:
    webbrowser.open(f"http://{HOST}:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Nuclear Central Manager panel.")
    parser.add_argument(
        "--allow-fallback", action="store_true",
        help="Run the built-in reference solution for any part missing a "
             "solutions/solution_part{N}.py file. Without this flag (default), a "
             "missing solution file melts the plant down at that part.")
    args = parser.parse_args()
    if args.allow_fallback:
        os.environ["NCM_ALLOW_FALLBACK"] = "1"
        print("ℹ️  Reference fall-back ENABLED — missing solution files run the reference.")
    else:
        print("ℹ️  Strict mode — a missing solutions/solution_part{N}.py melts the plant down "
              "(use --allow-fallback to run the reference instead).")

    try:
        import uvicorn
    except ImportError:
        sys.exit(
            "uvicorn is not installed. Launch with uv (recommended):\n"
            "    uv run --project workshop python workshop/launch.py\n"
            "or install the deps yourself:  pip install -r workshop/requirements.txt"
        )

    from workshop.backend.main import app  # noqa: WPS433 (import after path setup)

    port = _find_free_port(PORT)
    if port != PORT:
        print(f"\n⚠️  Port {PORT} is busy (another launch still running?). Using {port} instead.")

    url = f"http://{HOST}:{port}"
    print(f"\n☢️  Nuclear Central Manager → {url}")
    print("    (press Ctrl+C to stop)\n")
    threading.Timer(1.5, _open_browser, args=(port,)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="info")


if __name__ == "__main__":
    main()
