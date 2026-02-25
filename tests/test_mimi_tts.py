import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from elara_core.voice.mimi_tts import MimiTTS

class TestMimiTTS(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.tts = MimiTTS()
        # Bolt: Stub _load_model to avoid invoking the real loader which depends on moshi
        self.tts._load_model = MagicMock()
        self.tts._model = self.mock_model
        self.tts.num_codebooks = 8

    def test_synthesize_broadcasting(self):
        # Setup voice cache with a mock embedding [1, K, 1]
        voice_emb = torch.randn(1, 8, 1)
        self.tts._voice_cache['test'] = voice_emb

        # Mock _estimate_frames and _generate_prosody
        self.tts._estimate_frames = MagicMock(return_value=100)
        # _generate_prosody returns [1, 8, 100]
        variation = torch.randn(1, 8, 100)
        self.tts._generate_prosody = MagicMock(return_value=variation)

        # Mock model.decode to return a tensor
        fake_audio = torch.randn(1, 1, 192000)
        self.mock_model.decode.return_value = fake_audio

        # Call synthesize
        pcm = self.tts.synthesize("Hello world", voice="test")

        # Bolt: Verify exactly one call to decode
        self.mock_model.decode.assert_called_once()

        # Verify model.decode was called with correct shape [1, 8, 100]
        # and correct data (broadcasting verification)
        args, _ = self.mock_model.decode.call_args
        codes = args[0]

        expected_codes = (voice_emb + variation).clamp(0, 2047).long().to(codes.device)
        torch.testing.assert_close(codes, expected_codes)

        self.assertEqual(codes.shape, (1, 8, 100))
        self.assertIsInstance(pcm, np.ndarray)
        self.assertEqual(pcm.shape, (192000,))

if __name__ == "__main__":
    unittest.main()
