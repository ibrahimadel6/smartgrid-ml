import os
import socket

from app.main import app
from app.tunnel import print_local_urls, start_public_tunnel


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
            return False
        except OSError:
            return True


def main():
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    reload_enabled = os.environ.get("RELOAD", "0") == "1"

    if _port_in_use(port):
        print(
            f"\n[run] ERROR: port {port} is already in use — another server instance "
            "is running.\n"
            "      Close the other terminal/window that is running the server, or\n"
            "      start this one on a different port with a local tunnel:\n"
            "         PORT=8001 python run.py"
        )
        raise SystemExit(1)

    print_local_urls(port)

    # 1) Open a public tunnel (localtunnel, or ngrok if NGROK_AUTHTOKEN is set).
    #    It runs in a daemon thread and prints the public URL to the terminal.
    try:
        start_public_tunnel(port)
    except Exception as exc:  # never block startup because of tunneling
        print(f"[run] public tunnel unavailable: {exc}")

    # 2) Start FastAPI (models build once at startup via the lifespan hook).
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()