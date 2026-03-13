"""
LLM client wrapper
Supports two backends:
1. OpenAI-compatible API (default)
2. Claude Code headless mode (claude -p)
"""

import json
import re
import subprocess
import logging
import threading
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')

# Global semaphore to limit concurrent claude -p sessions (shared across all ClaudeCodeClient instances)
_claude_semaphore = threading.Semaphore(int(Config.CLAUDE_CODE_MAX_CONCURRENT if hasattr(Config, 'CLAUDE_CODE_MAX_CONCURRENT') else 1))


class ClaudeCodeClient:
    """
    Claude Code headless client
    Calls via `claude -p` command with --dangerously-skip-permissions and --model flags.
    Uses a global semaphore to limit concurrent sessions.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model or Config.CLAUDE_CODE_MODEL
        self.claude_bin = Config.CLAUDE_CODE_BIN

    def _build_command(
        self,
        system_prompt: Optional[str] = None,
        json_schema: Optional[Dict] = None,
    ) -> List[str]:
        cmd = [
            self.claude_bin, "-p",
            "--dangerously-skip-permissions",
            "--model", self.model,
            "--output-format", "json",
            "--tools", "",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
        ]
        if system_prompt:
            cmd.extend(["--system-prompt", system_prompt])
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        return cmd

    def _run(self, prompt: str, system_prompt: Optional[str] = None,
             json_schema: Optional[Dict] = None) -> str:
        cmd = self._build_command(system_prompt=system_prompt, json_schema=json_schema)
        logger.debug(f"Claude Code command: {' '.join(cmd[:6])}...")

        logger.info("Waiting for claude -p session slot...")
        _claude_semaphore.acquire()
        try:
            logger.info(f"Claude -p session acquired (prompt_len={len(prompt)})")
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=Config.CLAUDE_CODE_TIMEOUT,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(f"Claude Code failed (exit {result.returncode}): {stderr}")

            # --output-format json wraps the response in a JSON envelope with a "result" field
            try:
                envelope = json.loads(result.stdout)
                return envelope.get("result", result.stdout)
            except json.JSONDecodeError:
                return result.stdout.strip()
        finally:
            _claude_semaphore.release()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        # Extract system prompt and build a single user prompt from messages
        system_prompt = None
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_parts.append(msg["content"])

        prompt = "\n\n".join(user_parts)
        return self._run(prompt, system_prompt=system_prompt)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Clean markdown code fences
        cleaned = response.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n?```\s*$', '', cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Claude Code returned invalid JSON: {cleaned[:500]}")

    def create_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        OpenAI-compatible interface for services that call client.chat.completions.create() directly.
        Returns a mock response object with .choices[0].message.content and .choices[0].finish_reason.
        """
        system_prompt = None
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_parts.append(msg["content"])

        prompt = "\n\n".join(user_parts)

        if response_format and response_format.get("type") == "json_object":
            if system_prompt:
                system_prompt += "\n\nIMPORTANT: You must respond with valid JSON only, no markdown."
            else:
                system_prompt = "You must respond with valid JSON only, no markdown."

        content = self._run(prompt, system_prompt=system_prompt)
        return _MockCompletion(content)


class _MockMessage:
    def __init__(self, content: str):
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMessage(content)
        self.finish_reason = "stop"


class _MockCompletion:
    def __init__(self, content: str):
        self.choices = [_MockChoice(content)]


class _ClaudeCodeCompletionsProxy:
    """Proxy that mimics openai.chat.completions so services can call .chat.completions.create()"""

    def __init__(self, claude_client: ClaudeCodeClient):
        self._client = claude_client

    def create(self, **kwargs):
        return self._client.create_completion(**kwargs)


class _ClaudeCodeChatProxy:
    """Proxy that mimics openai.chat so services can call .chat.completions.create()"""

    def __init__(self, claude_client: ClaudeCodeClient):
        self.completions = _ClaudeCodeCompletionsProxy(claude_client)


class ClaudeCodeOpenAIShim:
    """
    Drop-in replacement for openai.OpenAI() that routes through Claude Code headless.
    Services that do `self.client = OpenAI(...)` then `self.client.chat.completions.create(...)`
    can use this instead without any other code changes.
    """

    def __init__(self, model: Optional[str] = None, **kwargs):
        self._claude = ClaudeCodeClient(model=model)
        self.chat = _ClaudeCodeChatProxy(self._claude)


class LLMClient:
    """LLM client - auto-selects backend (Claude Code or OpenAI-compatible API)"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.use_claude_code = Config.LLM_BACKEND == "claude_code"

        if self.use_claude_code:
            self._claude = ClaudeCodeClient(model=model)
        else:
            self.api_key = api_key or Config.LLM_API_KEY
            self.base_url = base_url or Config.LLM_BASE_URL
            self.model = model or Config.LLM_MODEL_NAME

            if not self.api_key:
                raise ValueError("LLM_API_KEY not configured")

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Send a chat request

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count
            response_format: Response format (e.g. JSON mode)

        Returns:
            Model response text
        """
        if self.use_claude_code:
            return self._claude.chat(messages, temperature, max_tokens, response_format)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models (e.g. MiniMax M2.5) include <think> tags in content, need to remove
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return JSON

        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Max token count

        Returns:
            Parsed JSON object
        """
        if self.use_claude_code:
            return self._claude.chat_json(messages, temperature, max_tokens)

        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Clean markdown code fence markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format returned by LLM: {cleaned_response}")
