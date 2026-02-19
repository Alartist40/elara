from llama_cpp import Llama
from typing import Dict, Any

class Tier1Engine:
    """
    Direct inference with quantized Gemma.
    No tricks, no recursion, no diffusion.
    Just a model that works.
    """

    def __init__(self, model_path: str = "models/gemma-3-1b-it-q4_0.gguf"):
        # llama-cpp-python handles quantization, GPU offloading
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=4096,           # Context window
                n_threads=4,          # CPU threads (adjust to your cores)
                n_gpu_layers=0,       # Set to -1 if you have GPU
                verbose=False,
            )
        except Exception as e:
            print(f"Error loading Tier 1 model: {e}")
            self.model = None

        self.system_prompt = """You are Elara, a helpful AI assistant.
Be concise, accurate, and helpful. If unsure, say so."""

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if self.model is None:
            return "Error: Tier 1 model not loaded."

        full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"

        output = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["User:", "Assistant:"],
        )

        return output["choices"][0]["text"].strip()

    def get_stats(self) -> Dict[str, Any]:
        if self.model is None:
            return {"status": "not_loaded"}

        return {
            "model": "gemma-3-1b-it-q4_0",
            "ram_mb": self.model.n_ctx * 0.5,  # Rough estimate
            "gpu_layers": self.model.n_gpu_layers,
        }
