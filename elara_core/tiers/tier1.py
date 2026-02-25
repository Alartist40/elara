import os
from llama_cpp import Llama
from typing import Optional, Any

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
        if model_path is None:
            model_path = os.getenv("ELARA_MODEL_PATH", "models/gemma-3-1b-it-q4_0.gguf")

        if n_threads is None:
            n_threads = os.cpu_count() or 4

        from elara_core.utils import check_memory
        mem_status = check_memory()
        if not mem_status['can_load_model']:
            print(f"Skipping Tier 1 load: insufficient memory ({mem_status['used_gb']:.1f}GB used)")
            self.model = None
            return

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
        Generate a chat response from the loaded Gemma model using the given prompt and optional system prompt.
        
        Parameters:
            prompt (str): The user's prompt to send to the model.
            max_tokens (int): Maximum number of tokens the model may generate.
            system_prompt (Optional[str]): Optional system message to override the default system prompt.
        
        Returns:
            str: The model's reply text with surrounding whitespace trimmed; if the model is not loaded, returns "Error: Tier 1 model not loaded."
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

    def get_stats(self) -> dict[str, Any]:
        """
        Return runtime statistics for the loaded Tier1 model or indicate that no model is loaded.
        
        Returns:
            dict[str, Any]: If the model is not loaded, returns {"status": "not_loaded"}. Otherwise returns a dictionary with:
                - "model" (str): Model identifier.
                - "ram_mb" (float): Estimated RAM usage in megabytes (approximate, derived from model context size).
                - "gpu_layers" (int): Number of GPU layers allocated to the model.
                - "n_threads" (int): Number of CPU threads configured for the model.
        """
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model": "gemma-3-1b-it-q4_0",
            "ram_mb": self.model.n_ctx * 0.5,  # Rough estimate
            "gpu_layers": self.model.n_gpu_layers,
            "n_threads": self.model.context_params.n_threads,
        }
