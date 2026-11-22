import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import APIConnectionError, AsyncOpenAI

from .config import ModelConfig

RECONNECT_ATTEMPTS = 10
POLL_SECONDS = 15


@dataclass
class Reply:
    text: str
    reasoning: str | None
    finish_reason: str
    input_tokens: int
    output_tokens: int
    latency_s: float


async def resolve_local(name: str, endpoints: Path, wait: bool) -> str:
    path = endpoints / name
    while not path.exists():
        if not wait:
            raise RuntimeError(f"no running server named {name} (expected {path}); start it with bv_llm_host or pass --wait")
        print(f"waiting for server {name} ...", flush=True)
        await asyncio.sleep(POLL_SECONDS)
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return fields["URL"]


class ChatClient:
    def __init__(self, config: ModelConfig, endpoints: Path, wait: bool):
        self.config = config
        self.endpoints = endpoints
        self.wait = wait
        self.local = config.base_url.removeprefix("local:") if config.base_url.startswith("local:") else None
        self.api_key = os.environ[config.api_key_env] if config.api_key_env else "none"
        self.lock = asyncio.Lock()
        self.base_url = None
        self.client = None
        self.model = config.model

    @classmethod
    async def connect(cls, config: ModelConfig, endpoints: Path, wait: bool = False) -> "ChatClient":
        chat = cls(config, endpoints, wait)
        await chat.open(config.base_url if chat.local is None else await resolve_local(chat.local, endpoints, wait))
        chat.model = chat.model or (await chat.client.models.list()).data[0].id
        return chat

    async def open(self, base_url: str) -> None:
        self.base_url = base_url
        self.client = AsyncOpenAI(base_url=base_url, api_key=self.api_key, timeout=900, max_retries=5)

    async def reconnect(self, failed_url: str) -> None:
        async with self.lock:
            if self.base_url != failed_url:
                return
            for _ in range(RECONNECT_ATTEMPTS):
                url = await resolve_local(self.local, self.endpoints, wait=True)
                if url != failed_url:
                    print(f"{self.local} moved to {url}", flush=True)
                    await self.open(url)
                    return
                await asyncio.sleep(POLL_SECONDS)

    async def __call__(self, messages: list[dict]) -> Reply:
        for attempt in range(RECONNECT_ATTEMPTS + 1):
            url = self.base_url
            start = time.perf_counter()
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model, messages=messages, extra_body=self.config.params
                )
                break
            except APIConnectionError:
                if self.local is None or attempt == RECONNECT_ATTEMPTS:
                    raise
                await self.reconnect(url)
        choice = resp.choices[0]
        extra = choice.message.model_extra or {}
        return Reply(
            text=choice.message.content or "",
            reasoning=extra.get("reasoning") or extra.get("reasoning_content"),
            finish_reason=choice.finish_reason,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            latency_s=round(time.perf_counter() - start, 2),
        )
