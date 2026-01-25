"""
LoRA File Server

A simple HTTP server that serves LoRA files to remote ComfyUI instances during training.
Runs in a background thread alongside training.
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import unquote, urlparse

from toolkit.print import print_acc


class LoraFileHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving LoRA files."""

    def log_message(self, format, *args):
        """Override to use print_acc for logging."""
        # Only log actual requests, not every connection
        if args and '200' in str(args):
            print_acc(f"LoRA Server: {args[0]}")

    def do_GET(self):
        """Handle GET requests for LoRA files."""
        parsed_path = urlparse(self.path)
        path = unquote(parsed_path.path)

        # Route: /lora/<filename>
        if path.startswith('/lora/'):
            filename = path[6:]  # Remove '/lora/' prefix
            self._serve_lora_file(filename)

        # Route: /health - health check endpoint
        elif path == '/health':
            self._send_response(200, b'OK', 'text/plain')

        # Route: /list - list available LoRA files
        elif path == '/list':
            self._list_lora_files()

        else:
            self._send_response(404, b'Not Found', 'text/plain')

    def _serve_lora_file(self, filename: str):
        """Serve a LoRA file from the lora directory."""
        # Security: prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            self._send_response(400, b'Invalid filename', 'text/plain')
            return

        lora_dir = self.server.lora_directory
        file_path = os.path.join(lora_dir, filename)

        if not os.path.exists(file_path):
            self._send_response(404, b'LoRA file not found', 'text/plain')
            return

        if not file_path.endswith('.safetensors'):
            self._send_response(400, b'Invalid file type', 'text/plain')
            return

        try:
            with open(file_path, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(content)

        except Exception as e:
            print_acc(f"Error serving LoRA file: {e}")
            self._send_response(500, b'Internal Server Error', 'text/plain')

    def _list_lora_files(self):
        """List available LoRA files."""
        import json

        lora_dir = self.server.lora_directory
        try:
            files = [f for f in os.listdir(lora_dir) if f.endswith('.safetensors')]
            files.sort(reverse=True)  # Most recent first

            response = json.dumps({
                'files': files,
                'directory': lora_dir
            })
            self._send_response(200, response.encode('utf-8'), 'application/json')
        except Exception as e:
            self._send_response(500, str(e).encode('utf-8'), 'text/plain')

    def _send_response(self, code: int, content: bytes, content_type: str):
        """Send a simple HTTP response."""
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class LoraHTTPServer(HTTPServer):
    """HTTP server with LoRA directory configuration."""

    def __init__(self, server_address, handler_class, lora_directory: str):
        super().__init__(server_address, handler_class)
        self.lora_directory = lora_directory


class LoraServer:
    """
    Background HTTP server for serving LoRA files to ComfyUI.

    Usage:
        server = LoraServer(
            host="0.0.0.0",
            port=8765,
            lora_directory="/path/to/loras",
            external_url="http://192.168.1.50:8765"
        )
        server.start()
        # ... training ...
        server.stop()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        lora_directory: str = "./remote_loras",
        external_url: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.lora_directory = os.path.abspath(lora_directory)
        self.external_url = external_url or f"http://{host}:{port}"

        self._server: Optional[LoraHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the LoRA server in a background thread."""
        if self._running:
            print_acc("LoRA server already running")
            return

        # Ensure lora directory exists
        os.makedirs(self.lora_directory, exist_ok=True)

        try:
            self._server = LoraHTTPServer(
                (self.host, self.port),
                LoraFileHandler,
                self.lora_directory
            )
            self._running = True

            self._thread = threading.Thread(
                target=self._serve_forever,
                daemon=True,
                name="LoraServer"
            )
            self._thread.start()

            print_acc(f"LoRA server started at http://{self.host}:{self.port}")
            print_acc(f"External URL: {self.external_url}")
            print_acc(f"Serving files from: {self.lora_directory}")

        except Exception as e:
            print_acc(f"Failed to start LoRA server: {e}")
            self._running = False
            raise

    def _serve_forever(self):
        """Server loop running in background thread."""
        try:
            while self._running:
                self._server.handle_request()
        except Exception as e:
            if self._running:
                print_acc(f"LoRA server error: {e}")

    def stop(self):
        """Stop the LoRA server."""
        if not self._running:
            return

        self._running = False

        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass

        if self._thread:
            self._thread.join(timeout=2.0)

        print_acc("LoRA server stopped")

    def get_file_url(self, filename: str) -> str:
        """Get the URL for a LoRA file."""
        return f"{self.external_url}/lora/{filename}"

    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
