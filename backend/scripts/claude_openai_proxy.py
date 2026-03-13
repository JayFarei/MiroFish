"""
Claude Code OpenAI-compatible API proxy.

Exposes `claude -p` (headless mode) as an OpenAI-compatible
/v1/chat/completions endpoint so that libraries like CAMEL-AI
can use Claude Code as their LLM backend.

Includes a concurrency limiter to avoid hitting Claude's session limit.
Requests beyond the limit are queued and processed in order.

Usage:
    uv run python scripts/claude_openai_proxy.py [--port 8082] [--model claude-sonnet-4-6] [--max-concurrent 2]

Then set in .env:
    LLM_API_KEY=sk-claude-proxy
    LLM_BASE_URL=http://localhost:8082/v1
    LLM_MODEL_NAME=claude-sonnet-4-6
"""

import argparse
import json
import logging
import subprocess
import threading
import time
import uuid
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("claude-proxy")

app = Flask(__name__)

CLAUDE_BIN = "claude"
DEFAULT_MODEL = "claude-sonnet-4-6"
TIMEOUT = 600
MAX_CONCURRENT = 2

# Semaphore to limit concurrent claude -p calls
_semaphore = None
_queue_depth = 0
_queue_lock = threading.Lock()


def call_claude(messages, model=None):
    """Call claude -p with the given messages and return the response text."""
    global _queue_depth
    model = model or DEFAULT_MODEL

    system_prompt = None
    user_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        else:
            user_parts.append(content)

    prompt = "\n\n".join(user_parts)

    cmd = [
        CLAUDE_BIN, "-p",
        "--dangerously-skip-permissions",
        "--model", model,
        "--output-format", "json",
        "--tools", "",
        "--strict-mcp-config",
        "--mcp-config", '{"mcpServers":{}}',
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    # Wait for a slot
    with _queue_lock:
        _queue_depth += 1
        queue_pos = _queue_depth

    if queue_pos > MAX_CONCURRENT:
        logger.info(f"Queued request (position {queue_pos}, waiting for slot, prompt_len={len(prompt)})")

    _semaphore.acquire()
    try:
        with _queue_lock:
            _queue_depth -= 1

        logger.info(f"Calling claude -p (model={model}, prompt_len={len(prompt)})")
        start = time.time()

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )

        elapsed = time.time() - start
        logger.info(f"Claude responded in {elapsed:.1f}s (exit={result.returncode})")

        if result.returncode != 0:
            raise RuntimeError(f"claude -p failed (exit {result.returncode}): {result.stderr.strip()}")

        try:
            envelope = json.loads(result.stdout)
            return envelope.get("result", result.stdout)
        except json.JSONDecodeError:
            return result.stdout.strip()
    finally:
        _semaphore.release()


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.get_json()
    messages = data.get("messages", [])
    model = data.get("model", DEFAULT_MODEL)

    try:
        content = call_claude(messages, model=model)
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": {"message": str(e), "type": "server_error"}}), 500

    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    return jsonify(response)


@app.route("/v1/models", methods=["GET"])
def list_models():
    """Return a minimal model list so clients can verify connectivity."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL,
                "object": "model",
                "owned_by": "anthropic",
            }
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "max_concurrent": MAX_CONCURRENT,
        "queued": _queue_depth,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude Code OpenAI proxy")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--max-concurrent", type=int, default=2,
                        help="Max concurrent claude -p sessions (default: 2)")
    args = parser.parse_args()

    DEFAULT_MODEL = args.model
    MAX_CONCURRENT = args.max_concurrent
    _semaphore = threading.Semaphore(MAX_CONCURRENT)

    logger.info(f"Starting Claude OpenAI proxy on {args.host}:{args.port}")
    logger.info(f"  model={DEFAULT_MODEL}, max_concurrent={MAX_CONCURRENT}, timeout={TIMEOUT}s")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
