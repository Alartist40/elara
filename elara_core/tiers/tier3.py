import openai
import os

class Tier3Engine:
    """
    When local models fail, call the cloud.
    Not "deep reasoning"—just bigger models.
    """

    def __init__(self):
        # Use OpenRouter for cheap access to many models
        # Or Together AI, or direct OpenAI
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model = os.getenv("TIER3_MODEL", "meta-llama/llama-3.3-70b-instruct")

        if self.api_key:
            self.client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        else:
            self.client = None

        self.system_prompt = """You are Elara, a helpful AI assistant.
Be concise, accurate, and helpful. If unsure, say so."""

    def generate(self, prompt: str, max_tokens: int = 1024, system_prompt: str = None) -> str:
        if not self.client:
            return "Error: Tier 3 API key not set."

        try:
            messages = [
                {"role": "system", "content": system_prompt or self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            return f"Error in Tier 3 generation: {e}"

    def is_available(self) -> bool:
        return self.api_key is not None
