"""
Voice persona system adapted from PersonaPlex.
Maintains consistent speaker identity across sessions.
"""

import yaml
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
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
        """
        Initialize the manager, configure the TTS engine reference, determine the persona config path, and load personas.
        
        Parameters:
            config_path (Optional[str]): Path to the personas YAML file. If omitted, uses the ELARA_PERSONA_CONFIG environment variable or "config/personas.yaml".
        """
        self.tts = tts_engine
        self.config_path = Path(config_path or os.getenv("ELARA_PERSONA_CONFIG", "config/personas.yaml"))
        self.personas: Dict[str, VoicePersona] = {}
        self.active_persona: Optional[str] = None

        self._load_config()

    def _load_config(self):
        """
        Load voice persona definitions from the configured YAML file into self.personas.
        
        If the configured file does not exist, a default configuration is created. Parses the top-level "personas" mapping and populates the manager's personas dictionary with VoicePersona instances. On error, logs the failure and falls back to a minimal "elara" persona so the system remains usable.
        """
        if not self.config_path.exists():
            self._create_default_config()

        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            for name, cfg in data.get("personas", {}).items():
                self.personas[name] = VoicePersona(
                    name=name,
                    voice_sample=cfg["voice_sample"],
                    speaking_style=cfg.get("style", "neutral"),
                    speed=cfg.get("speed", 1.0),
                    pitch_shift=cfg.get("pitch", 0.0),
                    text_style=cfg.get("text_style", "You are a helpful assistant.")
                )
        except Exception as e:
            logger.error(f"Failed to load persona config: {e}")
            # Fallback to a basic persona if everything fails
            self.personas["elara"] = VoicePersona(
                name="elara",
                voice_sample="data/voices/elara_sample.wav",
                text_style="You are Elara, a helpful AI assistant."
            )

    def _create_default_config(self):
        """
        Create a default persona configuration file at self.config_path.
        
        Writes a YAML document defining at least two personas ("elara" and "professional") with fields for voice_sample, style, speed, and text_style. Ensures the parent directory exists and overwrites any existing config file at that path.
        """
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
        """
        Activate the named persona and make it the current active persona.
        
        If the requested persona is unknown, falls back to the "elara" persona and will create and reload the default configuration if "elara" is also missing. If the TTS engine exposes a `load_voice` method, it is invoked with the persona's name and voice sample path to prepare the voice. The manager's `active_persona` is set to the chosen persona.
        
        Parameters:
            name (str): The identifier of the persona to activate.
        
        Returns:
            VoicePersona: The activated persona configuration.
        """
        if name not in self.personas:
            logger.warning(f"Unknown persona: {name}. Falling back to 'elara'.")
            name = "elara"
            if name not in self.personas:
                 # Ensure 'elara' exists
                 self._create_default_config()
                 self._load_config()

        persona = self.personas[name]

        # Load voice into TTS engine
        if hasattr(self.tts, 'load_voice'):
            voice_path = Path(persona.voice_sample)
            # MimiTTS.load_voice handles missing files by random initialization
            self.tts.load_voice(name, str(voice_path))

        self.active_persona = name
        return persona

    def get_system_prompt(self) -> str:
        """
        Provide the system prompt text for the currently active persona.
        
        Returns:
            The active persona's `text_style` string, or "You are a helpful assistant." if no persona is active.
        """
        if self.active_persona is None:
            return "You are a helpful assistant."
        return self.personas[self.active_persona].text_style

    def synthesize(self, text: str, override_persona: Optional[str] = None):
        """
        Synthesize speech for the given text using the active persona or a specified override.
        
        Parameters:
            text (str): Text to synthesize.
            override_persona (Optional[str]): Persona name to use instead of the currently active persona.
        
        Returns:
            The synthesized audio output as produced by the TTS engine, or `None` if the TTS engine does not support synthesis.
        """
        voice = override_persona or self.active_persona or "default"

        if hasattr(self.tts, 'synthesize'):
            persona = self.personas.get(voice)
            speed = persona.speed if persona else 1.0
            return self.tts.synthesize(text, voice=voice, speed=speed)

        return None