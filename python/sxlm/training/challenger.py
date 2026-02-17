import random
from typing import List, Dict
from openrouter import OpenRouter

class ChallengerAnswerGenerator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.models = [
            "anthropic/claude-opus-4-6",
            "openai/gpt-4-turbo",
            "google/gemini-pro-1.5"
        ]

    def generate_challenge(self, topic: str, difficulty: str = "hard") -> Dict[str, str]:
        model = random.choice(self.models)
        prompt = f"Generate a {difficulty} question about: {topic}\nProvide both question and detailed answer."

        with OpenRouter(api_key=self.api_key) as client:
            response = client.chat.send(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.choices[0].message.content

            parts = content.split("\n\n", 1)
            return {
                "question": parts[0] if len(parts) > 0 else content,
                "answer": parts[1] if len(parts) > 1 else "",
                "source_model": model
            }

    def generate_batch(self, topics: List[str], batch_size: int = 10) -> List[Dict[str, str]]:
        challenges = []
        for _ in range(batch_size):
            topic = random.choice(topics)
            try:
                challenge = self.generate_challenge(topic)
                challenges.append(challenge)
            except Exception as e:
                print(f"Error generating challenge: {e}")
                continue
        return challenges
