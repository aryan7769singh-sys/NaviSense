"""Ride collection server for the local network.

Run this on the laptop; phones on the same Wi-Fi upload their recordings to it.

    python collect_server.py

Prints the exact endpoint to paste into the app. No Cloudflare account, no card,
no internet required.

Timing note: phones do NOT upload while driving — uploads are Wi-Fi only, to protect
the collector's mobile data. They upload when they get back and rejoin your network,
so this only needs to be running when people return, not while they are out.
"""
from __future__ import annotations

import argparse
import http.server
import re
import socket
import secrets
import sys
from datetime import datetime
from pathlib import Path

SAFE_NAME = re.compile(r"^[\w.\-]+\.csv$")


def lan_ip() -> str:
    """Best-effort local address that other devices on the network can reach."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # no packets sent; just picks the right interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Handler(http.server.BaseHTTPRequestHandler):
    token = ""
    root = Path("data/collected")

    def _reply(self, code: int, msg: str) -> None:
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_PUT(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        if len(parts) != 2:
            return self._reply(400, "expected /<token>/<filename>.csv")

        tok, name = parts
        if tok != self.token:
            print(f"  REJECTED bad token from {self.client_address[0]}")
            return self._reply(403, "bad token")
        if not SAFE_NAME.match(name):
            return self._reply(400, "bad filename")

        device = self.headers.get("X-Device", "unknown")
        device = re.sub(r"[^\w\-]", "_", device)[:60]
        # Group by device so several collectors never collide, and so vehicle
        # diversity is visible in the folder layout itself.
        out_dir = self.root / device
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / name

        if dest.exists():
            # Never silently overwrite: a repeated name from a different drive would
            # destroy data that cannot be recollected.
            print(f"  already have {device}/{name}")
            return self._reply(200, "already stored")

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return self._reply(400, "empty body")

        tmp = dest.with_suffix(".csv.part")
        written = 0
        with tmp.open("wb") as fh:
            while written < length:
                chunk = self.rfile.read(min(1 << 16, length - written))
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)

        if written != length:
            # A truncated upload must not look like a complete ride.
            tmp.unlink(missing_ok=True)
            print(f"  TRUNCATED {device}/{name}: {written}/{length} bytes")
            return self._reply(400, "incomplete upload")

        tmp.rename(dest)
        stamp = datetime.now().strftime("%H:%M:%S")
        print(f"  [{stamp}] {device}/{name}  {written/1e6:.1f} MB")
        return self._reply(200, "ok")

    def do_GET(self) -> None:
        files = list(self.root.rglob("*.csv"))
        total = sum(f.stat().st_size for f in files)
        self._reply(200, f"IDR collector: {len(files)} rides, {total/1e6:.0f} MB\n")

    def log_message(self, *args) -> None:
        pass   # our own prints are more useful than the default access log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default="")
    ap.add_argument("--dir", default="data/collected")
    args = ap.parse_args()

    Handler.token = args.token or secrets.token_urlsafe(9)
    Handler.root = Path(args.dir)
    Handler.root.mkdir(parents=True, exist_ok=True)

    ip = lan_ip()
    endpoint = f"http://{ip}:{args.port}/{Handler.token}"

    print("=" * 66)
    print("  IDR ride collector")
    print("=" * 66)
    print(f"  Paste this into the app's SENSORS tab -> Collection endpoint:\n")
    print(f"      {endpoint}\n")
    print(f"  Saving to: {Handler.root.resolve()}")
    print(f"  Phones must be on the SAME Wi-Fi as this laptop to upload.")
    print(f"  Leave this running. Ctrl+C to stop.")
    print("=" * 66, flush=True)

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
