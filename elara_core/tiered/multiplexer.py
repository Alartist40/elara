"""
InputMultiplexer - Normalizes text, image, and voice inputs into token embeddings.
"""

import torch
from enum import Enum
from typing import Union, Optional, Tuple, Dict, Any
from pathlib import Path


class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    MULTIMODAL = "multimodal"


class InputMultiplexer:
    """
    Normalizes all input types (text, image, voice) into token sequences.
    Voice is always converted to text first (via Whisper STT),
    so the model always reasons over text.
    """

    def __init__(self, tokenizer_path: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer_path = tokenizer_path
        self._tokenizer = None

    def _ensure_tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is not None:
            return

        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            self._tokenizer = MistralTokenizer.from_file(self.tokenizer_path)
        except (ImportError, FileNotFoundError):
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            except ImportError:
                # Minimal fallback
                self._tokenizer = _SimpleTokenizer()

    def process(
        self,
        input_data: Union[str, dict],
        preferred_modality: Optional[ModalityType] = None,
    ) -> Tuple[torch.Tensor, ModalityType, Dict[str, Any]]:
        """
        Process input into tokens.

        Args:
            input_data: Text string or multimodal dict
            preferred_modality: Force a specific modality

        Returns:
            tokens: [L] token tensor
            modality: detected modality type
            metadata: dict with processing info
        """
        self._ensure_tokenizer()

        if isinstance(input_data, str):
            return self._process_text(input_data)
        elif isinstance(input_data, dict):
            return self._process_multimodal(input_data)
        else:
            # Assume text for anything else
            return self._process_text(str(input_data))

    def _process_text(
        self, text: str
    ) -> Tuple[torch.Tensor, ModalityType, Dict[str, Any]]:
        """Tokenize text input."""
        if hasattr(self._tokenizer, 'encode'):
            if hasattr(self._tokenizer, 'bos'):
                token_ids = self._tokenizer.encode(text, bos=True, eos=False)
            else:
                result = self._tokenizer.encode(text)
                token_ids = result if isinstance(result, list) else result.ids
        else:
            token_ids = self._tokenizer.encode(text)

        tokens = torch.tensor(token_ids, dtype=torch.long)

        return tokens, ModalityType.TEXT, {
            "length": len(token_ids),
            "original_text": text[:200],
        }

    def _process_multimodal(
        self, data: dict
    ) -> Tuple[torch.Tensor, ModalityType, Dict[str, Any]]:
        """Process multimodal input dict."""
        text = data.get("text", "")
        image_path = data.get("image")

        tokens, _, meta = self._process_text(text)
        meta["has_image"] = image_path is not None

        modality = ModalityType.MULTIMODAL if image_path else ModalityType.TEXT
        return tokens, modality, meta

    def detokenize(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to text."""
        self._ensure_tokenizer()
        if hasattr(self._tokenizer, 'decode'):
            return self._tokenizer.decode(tokens.tolist())
        return "[detokenize unavailable]"


class _SimpleTokenizer:
    """Minimal fallback tokenizer for testing."""

    def encode(self, text: str) -> list:
        return [ord(c) % 32000 for c in text]

    def decode(self, ids: list) -> str:
        return "".join(chr(i % 128) for i in ids)
