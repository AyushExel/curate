"""Tiny HTTP front for ClaudeProposer, for when the GPU box has no Anthropic
credentials but this machine has the `claude` CLI. Run here, reach it from the
box through an ssh remote forward:  ssh -R 8765:localhost:8765 rook
POST {"history": [...], "context": {...}} -> {"where", "rationale"} or {"stop": true}."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from curate.research import ClaudeProposer

proposer = ClaudeProposer()


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = proposer(body["history"], body["context"]) or {"stop": True}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)
        print("proposed:", out, flush=True)


HTTPServer(("127.0.0.1", 8765), H).serve_forever()
