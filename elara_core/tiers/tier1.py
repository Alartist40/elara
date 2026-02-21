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
