# Simple no-cache HTTP server for local development of this demo.
# Usage:  python serve.py [port]
#
# Avoids the standard http.server problem where browsers aggressively cache
# .js files and don't refetch even with hard reloads.

from __future__ import annotations

import http.server
import socketserver
import sys


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    with socketserver.TCPServer(("127.0.0.1", port), NoCacheHandler) as httpd:
        print(f"serving fire-bandwidth-extension demo at http://127.0.0.1:{port}/  (Ctrl+C to stop)")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
