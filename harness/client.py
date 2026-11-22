import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from .config import ModelConfig


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
        await asyncio.sleep(30)
    fields = dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)
    return fields["URL"]


class ChatClient:
    def __init__(self, config: ModelConfig, client: AsyncOpenAI, model: str):
        self.config = config
        self.client = client
        self.model = model

    @classmethod
    async def connect(cls, config: ModelConfig, endpoints: Path, wait: bool = False) -> "ChatClient":
        base_url = config.base_url
        if base_url.startswith("local:"):
            base_url = await resolve_local(base_url.removeprefix("local:"), endpoints, wait)
        api_key = os.environ[config.api_key_env] if config.api_key_env else "none"
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=900, max_retries=5)
        model = config.model or (await client.models.list()).data[0].id
        return cls(config, client, model)

    async def __call__(self, messages: list[dict]) -> Reply:
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, extra_body=self.config.params
        )
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
