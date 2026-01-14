#!/usr/bin/env python3
"""
Serve HTML detection viewer with Cloudflare tunnel.

Starts a local HTTP server and creates a Cloudflare tunnel for remote access.

Usage:
    python serve_html.py /path/to/output/html
    python serve_html.py /path/to/output/html --port 8081
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
import http.server
import socketserver
import threading
import time


def start_http_server(directory, port):
    """Start a simple HTTP server in a thread."""
    os.chdir(directory)
    handler = http.server.SimpleHTTPRequestHandler
    handler.extensions_map.update({
        '.html': 'text/html',
        '.json': 'application/json',
        '.js': 'application/javascript',
        '.css': 'text/css',
    })

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        httpd.serve_forever()


def start_cloudflare_tunnel(port):
    """Start cloudflared tunnel."""
    cloudflared = Path.home() / "cloudflared"
    if not cloudflared.exists():
        cloudflared = "cloudflared"  # Try system path

    cmd = [str(cloudflared), "tunnel", "--url", f"http://localhost:{port}"]

    print(f"\nStarting Cloudflare tunnel...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Read output and look for the URL
        for line in process.stdout:
            print(line, end='')
            if "trycloudflare.com" in line:
                # URL found - continue running
                pass

        process.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        process.terminate()
    except FileNotFoundError:
        print("ERROR: cloudflared not found!")
        print("Install with: curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/cloudflared && chmod +x ~/cloudflared")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Serve HTML viewer with Cloudflare tunnel')
    parser.add_argument('directory', help='Path to HTML directory')
    parser.add_argument('--port', type=int, default=8080, help='Port for HTTP server (default: 8080)')
    parser.add_argument('--no-tunnel', action='store_true', help='Only start HTTP server, no Cloudflare tunnel')

    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        sys.exit(1)

    # Find html subdirectory if needed
    if (directory / "html").exists():
        directory = directory / "html"

    index_file = directory / "index.html"
    if not index_file.exists():
        print(f"ERROR: No index.html found in {directory}")
        sys.exit(1)

    print(f"Serving: {directory}")
    print(f"Port: {args.port}")

    # Start HTTP server in background thread
    server_thread = threading.Thread(
        target=start_http_server,
        args=(str(directory), args.port),
        daemon=True
    )
    server_thread.start()

    time.sleep(1)  # Give server time to start

    if args.no_tunnel:
        print(f"\nView at: http://localhost:{args.port}")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
    else:
        # Start Cloudflare tunnel (blocking)
        start_cloudflare_tunnel(args.port)


if __name__ == '__main__':
    main()
