import os
from typing import List, Dict, Any, Generator
from openai import OpenAI
from dotenv import load_dotenv

# --- Загружаем .env безопасно ---
# Определяем путь к файлу .env (на уровень выше src/)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(BASE_DIR, ".env")

if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    # можно добавить лог: print(f".env не найден по пути {env_path}")
    pass

# --- Клиент OpenAI ---
class OpenAILLM:
    """Клиент OpenAI с обычным и потоковым режимом (chat.completions)."""

    def __init__(self, model: str = "gpt-4o-mini", max_output_tokens: int = 400):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def build_messages(self, system: str, messages: List[Dict[str, str]], context_blocks: str):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        if context_blocks:
            msgs.append({"role": "user", "content": "Полезный контекст:\n" + context_blocks})
        for m in messages:
            role = "user" if m.get("role") == "user" else "assistant"
            content = m.get("content", "")
            if content:
                msgs.append({"role": role, "content": content})
        return msgs

    def chat(self, system: str, messages: List[Dict[str, str]], context_blocks: str, temperature: float = 0.7) -> str:
        msgs = self.build_messages(system, messages, context_blocks)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=self.max_output_tokens
        )
        return resp.choices[0].message.content

    def chat_stream(
        self,
        system: str,
        messages: List[Dict[str, str]],
        context_blocks: str,
        temperature: float = 0.7
    ) -> Generator[str, None, None]:
        msgs = self.build_messages(system, messages, context_blocks)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=self.max_output_tokens,
            stream=True
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

