from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LocalLLM:
    """Обёртка вокруг transformers.pipeline('text-generation')."""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def build_prompt(self, system: str, messages, context_blocks: str) -> str:
        prompt = system.strip() + "\n\n"
        if context_blocks:
            prompt += "Ниже — полезный контекст, используй его по возможности:\n" + context_blocks + "\n\n"
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                prompt += f"Пользователь: {content}\n"
            else:
                prompt += f"Ассистент: {content}\n"
        prompt += "Ассистент:"
        return prompt

    def chat(self, system: str, messages, context_blocks: str, temperature: float = 0.7):
        prompt = self.build_prompt(system, messages, context_blocks)
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )[0]["generated_text"]
        return out.split("Ассистент:", maxsplit=1)[-1].strip()
