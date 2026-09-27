"""Vercel health endpoint for the browser-hosted review mode."""
from http.server import BaseHTTPRequestHandler
import json

from videolint import media


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        payload = {
            "ok": True,
            "hosted": True,
            "maxUploadBytes": 4 * 1024 * 1024,
            "ffmpeg": media.ffmpeg(),
            "aiProvider": "none",
            "aiConfigured": False,
            "transcriptionConfigured": False,
        }
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
