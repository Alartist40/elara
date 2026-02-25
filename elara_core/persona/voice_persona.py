"""
Voice persona system adapted from PersonaPlex.
Maintains consistent speaker identity across sessions.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import torch
import logging

logger = logging.getLogger(__name__)

@dataclass
class VoicePersona:
    """Configuration for a voice persona."""
    name: str
    voice_sample: str  # Path to reference audio
    speaking_style: str = "neutral"
    speed: float = 1.0
    pitch_shift: float = 0.0
    text_style: str = "You are a helpful assistant."

class VoicePersonaManager:
    """
    Manages multiple voice personas.
    """

    def __init__(self, tts_engine, config_path: Optional[str] = None):
        self.tts = tts_engine
        self.config_path = Path(config_path or os.getenv("ELARA_PERSONA_CONFIG", "config/personas.yaml"))
        self.personas: dict[str, VoicePersona] = {}
        self.active_persona: Optional[str] = None

        self._load_config()

    def _load_config(self):
        """Load persona definitions from YAML."""
        try:
            if not self.config_path.exists():
                self._create_default_config()

            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            for name, cfg in data.get("personas", {}).items():
                try:
                    self.personas[name] = VoicePersona(
                        name=name,
                        voice_sample=cfg["voice_sample"],
                        speaking_style=cfg.get("style", "neutral"),
                        speed=cfg.get("speed", 1.0),
                        pitch_shift=cfg.get("pitch", 0.0),
                        text_style=cfg.get("text_style", "You are a helpful assistant.")
                    )
                except KeyError as e:
                    logger.warning(f"Skipping persona '{name}': missing required field {e}")
        except Exception as e:
            logger.error(f"Failed to load persona config: {e}")
            # Fallback to a basic persona if everything fails
            self.personas["elara"] = VoicePersona(
                name="elara",
                voice_sample="data/voices/elara_sample.wav",
                text_style="You are Elara, a helpful AI assistant."
            )

    def _create_default_config(self):
        """Create default persona configuration."""
        default = {
            "personas": {
                "elara": {
                    "voice_sample": "data/voices/elara_sample.wav",
                    "style": "neutral",
                    "speed": 1.0,
                    "text_style": "You are Elara, a helpful AI assistant. Be concise and clear."
                },
                "professional": {
                    "voice_sample": "data/voices/professional_sample.wav",
                    "style": "professional",
                    "speed": 0.9,
                    "text_style": "You are a professional consultant. Be thorough and formal."
                }
            }
        }
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding="utf-8") as f:
            yaml.dump(default, f)

    def load_persona(self, name: str) -> VoicePersona:
        """Activate a persona, loading voice if needed."""
        if name not in self.personas:
            logger.warning(f"Unknown persona: {name}. Falling back to 'elara'.")
            name = "elara"
            if name not in self.personas:
                 # Ensure 'elara' exists
                 try:
                     self._create_default_config()
                     self._load_config()
                 except Exception as e:
                     logger.error(f"Failed to create default persona config: {e}")

        persona = self.personas.get(name)
        if persona is None:
            logger.error(f"Persona '{name}' could not be loaded. Creating inline fallback.")
            persona = VoicePersona(
                name="elara",
                voice_sample="data/voices/elara_sample.wav",
                text_style="You are Elara, a helpful AI assistant."
            )
            self.personas["elara"] = persona

        # Load voice into TTS engine
        if hasattr(self.tts, 'load_voice'):
            voice_path = Path(persona.voice_sample)
            # MimiTTS.load_voice handles missing files by random initialization
            self.tts.load_voice(name, str(voice_path))

        self.active_persona = name
        return persona

    def get_system_prompt(self) -> str:
        """Get text style prompt for current persona."""
        if self.active_persona is None:
            return "You are a helpful assistant."
        persona = self.personas.get(self.active_persona)
        return persona.text_style if persona else "You are a helpful assistant."

    def synthesize(self, text: str, override_persona: Optional[str] = None):
        """Synthesize with current persona's voice settings."""
        voice = override_persona or self.active_persona or "default"

        if hasattr(self.tts, 'synthesize'):
            persona = self.personas.get(voice)
            speed = persona.speed if persona else 1.0
            return self.tts.synthesize(text, voice=voice, speed=speed)

        return None
