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
        """
        Initialize the Tier3Engine by configuring the external API client, model selection, and the default system prompt.
        
        Reads these environment variables:
            - OPENROUTER_API_KEY: API key for the OpenRouter/OpenAI-compatible service (if unset, no client is created).
            - OPENROUTER_BASE_URL: Base URL for the API (default: "https://openrouter.ai/api/v1").
            - TIER3_MODEL: Model identifier to use for generation (default: "meta-llama/llama-3.3-70b-instruct").
        
        Behavior:
            - If `OPENROUTER_API_KEY` is present, creates an OpenAI-compatible client configured with the base URL and API key; otherwise sets the client to `None`.
            - Sets `self.model` and `self.system_prompt` to their configured or default values. The default system prompt is:
              "You are Elara, a helpful AI assistant. Be concise, accurate, and helpful. If unsure, say so."
        """
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
        """
        Generate a completion from the configured tier-3 cloud model using the provided prompt.
        
        Parameters:
        	prompt (str): User prompt to send to the model.
        	max_tokens (int): Maximum tokens to generate in the model response.
        	system_prompt (str | None): Optional system prompt to override the engine's default system prompt.
        
        Returns:
        	str: The model's response text; an empty string if the response content is None. If the engine is not configured or an error occurs, returns an error message string beginning with "Error".
        """
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