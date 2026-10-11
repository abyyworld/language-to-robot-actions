"""The OpenAI-compatible planner (Ollama, vLLM, ...) against a local stub server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from lang2act.plan import parse_plan
from lang2act.planners import make_planner
from lang2act.planners.base import PlanningRequest

PLAN = {"feasible": False, "reason": "no blue block", "steps": []}


@pytest.fixture()
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = {
                "model": body["model"],
                "choices": [{"message": {"content": json.dumps(PLAN)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen
    httpd.shutdown()


def test_ollama_request_is_seeded_and_structured(server, monkeypatch, tmp_path):
    url, seen = server
    monkeypatch.setenv("OLLAMA_BASE_URL", url)
    planner = make_planner("ollama:qwen2.5:7b", cache_dir=str(tmp_path))
    c = planner.session(
        PlanningRequest("Put the blue block on the red one.", {"objects": []}), sample=3
    ).propose()
    assert parse_plan(c.text).feasible is False
    assert c.input_tokens == 50
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["temperature"] == 0.0 and body["seed"] == 3
    assert body["messages"][0]["role"] == "system"
    assert body["response_format"]["type"] == "json_schema"


def test_connection_errors_become_api_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9/v1")
    planner = make_planner("ollama:x", cache_dir=str(tmp_path))
    c = planner.session(PlanningRequest("x", {"objects": []})).propose()
    assert c.error and c.error.startswith("connection_error")
