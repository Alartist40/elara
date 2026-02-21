import os
from llama_cpp import Llama
from typing import Dict, Any, Optional

class Tier1Engine:
    """
    Direct inference with quantized Gemma.
    No tricks, no recursion, no diffusion.
    Just a model that works.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_threads: Optional[int] = None
    ):
        """
        Initialize the Tier1Engine by resolving configuration and attempting to load the quantized Gemma model.
        
        Parameters:
            model_path (Optional[str]): Path to the model file. If None, uses the `ELARA_MODEL_PATH` environment variable or the default "models/gemma-3-1b-it-q4_0.gguf".
            n_threads (Optional[int]): Number of CPU threads to use for the model. If None, defaults to the system CPU count or 4.
        
        Notes:
            - Attempts to instantiate the underlying Llama model; on failure `self.model` is set to `None` and an error message is printed.
            - Sets `self.system_prompt` to a concise assistant instruction used as the default system prompt for generations.
        """
        if model_path is None:
            model_path = os.getenv("ELARA_MODEL_PATH", "models/gemma-3-1b-it-q4_0.gguf")

        if n_threads is None:
            n_threads = os.cpu_count() or 4

        # llama-cpp-python handles quantization, GPU offloading
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=n_threads,
                n_gpu_layers=0,       # Set to -1 if you have GPU
                chat_format="gemma",
                verbose=False,
            )
        except Exception as e:
            print(f"Error loading Tier 1 model: {e}")
            self.model = None

        self.system_prompt = """You are Elara, a helpful AI assistant.
Be concise, accurate, and helpful. If unsure, say so."""

    def generate(self, prompt: str, max_tokens: int = 256, system_prompt: Optional[str] = None) -> str:
        """
        Generate a chat completion from the loaded Gemma model using the provided user prompt.
        
        Parameters:
            prompt (str): The user's prompt to send to the model.
            max_tokens (int): Maximum number of tokens to generate for the completion.
            system_prompt (Optional[str]): Optional override for the system message; if omitted, the instance's default system prompt is used.
        
        Returns:
            str: The assistant's reply text (trimmed). If the model is not loaded, returns the string "Error: Tier 1 model not loaded.".
        """
        if self.model is None:
            return "Error: Tier 1 model not loaded."

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        output = self.model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["User:", "Assistant:"],
        )

        return output["choices"][0]["message"]["content"].strip()

    def get_stats(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model": "gemma-3-1b-it-q4_0",
            "ram_mb": self.model.n_ctx * 0.5,  # Rough estimate
            "gpu_layers": self.model.n_gpu_layers,
            "n_threads": self.model.context_params.n_threads,
        }