"""OpenRouter API client for knowledge distillation"""

from openrouter import OpenRouter
from typing import List

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate(self, prompt: str, model: str = "anthropic/claude-opus-4-6",
                 max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """Generate response from external model"""
        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

    def batch_generate(self, prompts: List[str], model: str = "anthropic/claude-opus-4-6") -> List[str]:
        """Generate responses for multiple prompts"""
        return [self.generate(p, model) for p in prompts]
