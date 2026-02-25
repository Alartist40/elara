import unittest
from unittest.mock import MagicMock, patch
import torch
import numpy as np
from elara_core.voice.mimi_tts import MimiTTS

class TestMimiTTS(unittest.TestCase):
    def setUp(self):
        # Mock the model loading to avoid heavy dependencies
        """
        Prepare a MimiTTS instance for tests by patching model loading and injecting a mocked model.
        
        Patches moshi.models.loaders.get_mimi to return a MagicMock, assigns that mock to self.tts._model, and configures self.tts.num_codebooks to 8 so tests run without loading the real model.
        """
        with patch('moshi.models.loaders.get_mimi') as mock_get_mimi:
            self.mock_model = MagicMock()
            mock_get_mimi.return_value = self.mock_model
            self.tts = MimiTTS()
            self.tts._model = self.mock_model
            self.tts.num_codebooks = 8

    def test_synthesize_broadcasting(self):
        # Setup voice cache with a mock embedding [1, K, 1]
        """
        Verifies that synthesize broadcasts a voice embedding across codebooks and produces code tensors with the expected shape and a NumPy PCM output.
        
        Sets a voice embedding shaped (1, K, 1), mocks internal frame/prosody estimation to produce (1, K, frames), and stubs the model decode result. Asserts that the model.decode input has shape (1, num_codebooks, frames) and that the returned PCM is a numpy.ndarray.
        """
        self.tts._voice_cache['test'] = torch.randn(1, 8, 1)

        # Mock _estimate_frames and _generate_prosody
        self.tts._estimate_frames = MagicMock(return_value=100)
        # _generate_prosody returns [1, 8, 100]
        self.tts._generate_prosody = MagicMock(return_value=torch.randn(1, 8, 100))

        # Mock model.decode to return a tensor
        self.mock_model.decode.return_value = torch.randn(1, 1, 192000)

        # Call synthesize
        pcm = self.tts.synthesize("Hello world", voice="test")

        # Verify model.decode was called with correct shape [1, 8, 100]
        # This confirms broadcasting worked and produced the expected shape for addition
        args, _ = self.mock_model.decode.call_args
        codes = args[0]
        self.assertEqual(codes.shape, (1, 8, 100))
        self.assertIsInstance(pcm, np.ndarray)

if __name__ == "__main__":
    unittest.main()
