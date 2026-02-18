"""
TieredInferenceEngine - Central orchestrator for the Elara multi-tier system.
Routes queries through Constitutional Layer → Tier Selection → Generation → Post-filter → Voice.
Based on Strategy.md specification.
"""

import time
import yaml
try:
    import torch
except ImportError:
    torch = None
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, Dict, Any

from elara_core.constitutional.layer import ConstitutionalLayer
try:
    from elara_core.voice.gateway import VoiceGateway
except ImportError:
    VoiceGateway = None
from elara_core.tiered.multiplexer import InputMultiplexer, ModalityType
from elara_core.tiered.router import TierRouter
from elara_core.tiered.metrics import MetricsTracker
from elara_core.tools.router import ToolRouter


@dataclass
class GenerationResult:
    """Result of a generation request."""
    text: Optional[str]
    audio: Optional[object]  # np.ndarray when voice
    format: str              # "text" or "audio"
    metadata: Dict[str, Any]


class TieredInferenceEngine:
    """
    Central orchestrator for Elara's multi-tier inference system.

    Pipeline:
        Input → Voice STT → Multiplexer → Constitutional Pre-Filter →
        Tier Selection → Tier Execution → Constitutional Post-Filter →
        Voice TTS → Output

    Tiers:
        1 (95%): Direct Mistral generation
        2 (4%):  CLaRa retrieval + augmented generation
        3 (1%):  TRM reasoning + TiDAR generation + tools
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path:
            self.config = self._load_config(config_path)

        # Get sub-configs
        const_cfg = self.config.get("constitutional", {})
        voice_cfg = self.config.get("voice", {})
        tools_cfg = self.config.get("tools", {})
        tier_cfg = self.config.get("tiering", {}).get("thresholds", {})

        # Core components
        principles_path = const_cfg.get("principles_path", "config/biblical_principles.yaml")
        self.constitutional = ConstitutionalLayer(
            principles_path=principles_path,
            audit_log_path=const_cfg.get("audit_log_path", "logs/constitution.log"),
            strict_mode=const_cfg.get("strict_mode", True),
            watermark_voice=const_cfg.get("watermark_synthetic_voice", True),
        )

        stt_cfg = voice_cfg.get("stt", {})
        if VoiceGateway:
            self.voice_gateway = VoiceGateway(
                stt_model=stt_cfg.get("model", "base"),
                tts_speaker=voice_cfg.get("tts", {}).get("default_speaker", 0),
                device=stt_cfg.get("device", "auto"),
            )
        else:
            self.voice_gateway = None

        self.multiplexer = InputMultiplexer(
            tokenizer_path=self.config.get("input", {}).get(
                "tokenizer_path", "mistralai/Mistral-7B-Instruct-v0.3"
            )
        )

        self.router = TierRouter(
            tier_2_complexity=tier_cfg.get("tier_2_complexity", 0.5),
            tier_3_complexity=tier_cfg.get("tier_3_complexity", 0.8),
            voice_query_boost=tier_cfg.get("voice_query_boost", 1),
            force_tier=self.config.get("tiering", {}).get("force_tier"),
        )

        self.tool_router = ToolRouter(
            schema_path=tools_cfg.get("schema_path", "config/tool_schema.json"),
            max_iterations=tools_cfg.get("max_iterations", 3),
            allowed_tools=tools_cfg.get("allowed_tools"),
        ) if tools_cfg.get("enabled", True) else None

        self.metrics = MetricsTracker()

        # Tier components (lazy loaded)
        self._mistral = None       # Tier 1
        self._clara_store = None   # Tier 2
        self._compressor = None    # Tier 2
        self._query_reasoner = None  # Tier 2
        self._trm_core = None     # Tier 3
        self._tidar_gen = None    # Tier 3
        self._airllm = None       # Tier 3 fallback

    def _load_config(self, path: str) -> dict:
        """Load YAML configuration."""
        config_path = Path(path)
        if not config_path.exists():
            return {}
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def generate(
        self,
        input_data: Union[str, bytes, dict],
        voice_input: bool = False,
        voice_output: bool = False,
        max_tokens: int = 100,
        temperature: float = 0.7,
        force_tier: Optional[int] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Main entry point for all generation requests.

        Args:
            input_data: Text, audio bytes, or multimodal dict.
            voice_input: If True, treat bytes as audio.
            voice_output: If True, return audio.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            force_tier: Override tier selection.

        Returns:
            GenerationResult with text or audio and metadata.
        """
        start_time = time.time()

        # === Step 1: Voice STT ===
        if voice_input and isinstance(input_data, (bytes, str)):
            if self.voice_gateway:
                try:
                    input_data = self.voice_gateway.listen(input_data)
                except Exception as e:
                    return GenerationResult(
                        text=f"[Voice transcription failed: {e}]",
                        audio=None, format="text",
                        metadata={"error": str(e), "tier": 0},
                    )
            else:
                input_data = "[Voice STT unavailable]"

        # Ensure text
        text = input_data if isinstance(input_data, str) else str(input_data)

        # === Step 2: Constitutional Pre-Filter ===
        pre_result = self.constitutional.pre_filter(
            text, {"modality": "voice" if voice_input else "text"}
        )

        if not pre_result.allowed:
            self.metrics.record_request(0, (time.time() - start_time) * 1000, blocked=True)
            return self._format_output(
                pre_result.text, voice_output,
                {"blocked": True, "tier": 0, "principles": pre_result.triggered_principles},
            )

        filtered_text = pre_result.text

        # === Step 3: Tier Selection ===
        has_memory = self._clara_store is not None and self._clara_store.doc_count > 0
        tier = force_tier or self.router.select_tier(
            text=filtered_text,
            modality="voice" if voice_input else "text",
            has_tools=kwargs.get("tool_use", False),
            has_memory=has_memory,
        )

        # === Step 4: Tier Execution ===
        if tier == 1:
            output_text = self._tier_1_generate(filtered_text, max_tokens, temperature)
        elif tier == 2:
            output_text = self._tier_2_generate(filtered_text, max_tokens, temperature)
        else:
            output_text = self._tier_3_generate(filtered_text, max_tokens, temperature, kwargs)

        # === Step 5: Constitutional Post-Filter ===
        post_result = self.constitutional.post_filter(
            output_text,
            {"tier": tier, "voice_output": voice_output},
        )

        if not post_result.allowed:
            output_text = post_result.text
        else:
            output_text = post_result.text

        # === Step 6: Metrics ===
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.record_request(tier, latency_ms, voice=voice_input)

        # === Step 7: Format output ===
        return self._format_output(
            output_text, voice_output,
            {
                "tier": tier,
                "latency_ms": round(latency_ms, 2),
                "pre_filter": pre_result.to_dict(),
                "post_filter": post_result.to_dict(),
            },
        )

    def _tier_1_generate(
        self, text: str, max_tokens: int, temperature: float
    ) -> str:
        """Tier 1: Direct Mistral generation."""
        # When mistral-inference is loaded, use it. Otherwise echo with context.
        if self._mistral is not None:
            try:
                tokens, _, _ = self.multiplexer.process(text)
                output = self._mistral.generate(
                    tokens.unsqueeze(0),
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                return self.multiplexer.detokenize(output[0])
            except Exception as e:
                return f"[Tier 1 generation error: {e}]"

        return f"[Tier 1] Elara response to: {text[:200]}"

    def _tier_2_generate(
        self, text: str, max_tokens: int, temperature: float
    ) -> str:
        """Tier 2: CLaRa retrieval-augmented generation."""
        if self._clara_store is not None and self._query_reasoner is not None:
            try:
                tokens, _, _ = self.multiplexer.process(text)
                query_embed = self._query_reasoner(tokens.unsqueeze(0))
                retrieved, scores, meta = self._clara_store.retrieve(query_embed)
                context = f"[Retrieved context from {len(meta[0])} documents]\n"
                return self._tier_1_generate(context + text, max_tokens, temperature)
            except Exception as e:
                return f"[Tier 2 retrieval error: {e}]"

        return f"[Tier 2 - CLaRa augmented] Elara response to: {text[:200]}"

    def _tier_3_generate(
        self, text: str, max_tokens: int, temperature: float, kwargs: dict
    ) -> str:
        """Tier 3: TRM reasoning + TiDAR generation + tools."""
        output_parts = []

        # TRM recursive reasoning
        if self._trm_core is not None:
            try:
                tokens, _, _ = self.multiplexer.process(text)
                result = self._trm_core(input_ids=tokens.unsqueeze(0))
                output_parts.append(f"[TRM: {result['n_recursions']} recursions]")
            except Exception as e:
                output_parts.append(f"[TRM error: {e}]")

        # TiDAR generation
        if self._tidar_gen is not None:
            try:
                tokens, _, _ = self.multiplexer.process(text)
                generated = self._tidar_gen.generate(tokens.unsqueeze(0), max_new_tokens=max_tokens)
                output_parts.append(self.multiplexer.detokenize(generated[0]))
            except Exception as e:
                output_parts.append(f"[TiDAR error: {e}]")

        # Tool execution
        if self.tool_router and kwargs.get("tool_use"):
            tool_results = self.tool_router.execute(text)
            for tr in tool_results:
                output_parts.append(f"[Tool {tr.name}: {tr.output}]")

        if not output_parts:
            return f"[Tier 3 - Deep reasoning] Elara response to: {text[:200]}"

        return "\n".join(output_parts)

    def _format_output(
        self, text: str, voice_output: bool, metadata: dict
    ) -> GenerationResult:
        """Format final output with optional TTS."""
        if voice_output and self.voice_gateway:
            try:
                audio = self.voice_gateway.speak(text)
                return GenerationResult(
                    text=None, audio=audio, format="audio", metadata=metadata
                )
            except Exception as e:
                metadata["tts_error"] = str(e)
                return GenerationResult(
                    text=text, audio=None, format="text", metadata=metadata
                )

        return GenerationResult(
            text=text, audio=None, format="text", metadata=metadata
        )

    def load_tier_models(
        self,
        tiers: Optional[list] = None,
    ) -> Dict[str, bool]:
        """
        Lazy-load tier-specific models.

        Args:
            tiers: List of tier numbers to load. None = load all available.

        Returns:
            Dict mapping component names to load success.
        """
        status = {}
        tiers = tiers or [1, 2, 3]

        if 1 in tiers:
            # Tier 1: Mistral
            try:
                from mistral_inference.model import Transformer
                from mistral_inference.generate import generate
                # Would need model path/checkpoint to actually load
                status["mistral"] = False  # Needs weights
            except ImportError:
                status["mistral"] = False

        if 2 in tiers:
            # Tier 2: CLaRa
            try:
                from elara_core.clara.compressor import SCPCompressor, SCPCompressorConfig
                from elara_core.clara.query_reasoner import QueryReasoner
                from elara_core.clara.store import CLaRaStore

                cfg = SCPCompressorConfig()
                self._compressor = SCPCompressor(cfg)
                self._query_reasoner = QueryReasoner(cfg)

                clara_cfg = self.config.get("clara", {})
                self._clara_store = CLaRaStore(
                    store_path=clara_cfg.get("store_path", "data/clara_store"),
                    d_model=cfg.d_model,
                    n_memory_tokens=cfg.n_memory_tokens,
                )
                status["clara"] = True
            except Exception as e:
                status["clara"] = False

        if 3 in tiers:
            # Tier 3: TRM + TiDAR
            try:
                from elara_core.trm.core import TRMCore, TRMConfig
                self._trm_core = TRMCore(TRMConfig())
                status["trm"] = True
            except Exception as e:
                status["trm"] = False

            try:
                from elara_core.tidar.generator import TiDARGenerator, TiDARConfig
                self._tidar_gen = TiDARGenerator(TiDARConfig())
                status["tidar"] = True
            except Exception as e:
                status["tidar"] = False

            # AirLLM fallback
            try:
                from elara_core.airllm_fallback import AirLLMFallback
                self._airllm = AirLLMFallback()
                status["airllm"] = self._airllm.is_available()
            except Exception:
                status["airllm"] = False

        return status

    def get_system_status(self) -> Dict[str, Any]:
        """Return complete system status."""
        return {
            "version": "2.0",
            "components": {
                "constitutional": True,
                "voice_stt": getattr(self.voice_gateway, "_stt_initialized", False) if self.voice_gateway else False,
                "voice_tts": getattr(self.voice_gateway, "_tts_initialized", False) if self.voice_gateway else False,
                "mistral": self._mistral is not None,
                "clara": self._clara_store is not None,
                "trm": self._trm_core is not None,
                "tidar": self._tidar_gen is not None,
                "airllm": self._airllm is not None,
                "tools": self.tool_router is not None,
            },
            "metrics": self.metrics.get_stats(),
            "constitutional_stats": self.constitutional.get_stats(),
        }
