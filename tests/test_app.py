from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import asr
import app


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = str(Path(self.folder.name) / "clip.wav")

    def write(self, seconds=1, rate=16000, stereo=False):
        t = np.arange(int(seconds * rate)) / rate
        wave = 0.2 * np.sin(2 * np.pi * 440 * t)
        if stereo:
            wave = np.stack([wave, wave * 0.5], axis=1)
        sf.write(self.path, wave, rate)

    def test_stereo_resampled_to_mono_16k(self):
        self.write(rate=44100, stereo=True)
        waveform, duration = asr.read_audio(self.path)
        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(duration, 1)
        self.assertAlmostEqual(float(np.max(waveform)), 0.15, places=2)

    def test_missing_audio(self):
        with self.assertRaisesRegex(ValueError, "Record or upload"):
            asr.read_audio(None)

    def test_silent_audio(self):
        sf.write(self.path, np.zeros(16000), 16000)
        with self.assertRaisesRegex(ValueError, "silent"):
            asr.read_audio(self.path)

    def test_duration_limits(self):
        for seconds in (0.1, asr.MAX_SECONDS + 0.1):
            self.write(seconds=seconds)
            with self.assertRaisesRegex(ValueError, "between"):
                asr.read_audio(self.path)

    def test_broken_file(self):
        Path(self.path).write_text("not audio")
        with self.assertRaisesRegex(ValueError, "could not be read"):
            asr.read_audio(self.path)

    def test_unicode_transcript_download(self):
        self.write()
        expected = "ẹ káàárọ̀ ndị mmadụ"
        with patch("app.load_model", return_value=object()), patch("app.decode", return_value=expected):
            stream = app.transcribe(self.path, language="Yoruba", progress=lambda *a, **k: None)
            next(stream)
            result = next(stream)
            try:
                self.assertEqual(result[0], expected)
                self.assertEqual(Path(result[1]).read_text(encoding="utf-8").strip(), expected)
                self.assertEqual(result[5], "4")
            finally:
                stream.close()
            self.assertFalse(Path(result[1]).exists())

    def test_model_failure_clears_result(self):
        self.write()
        with patch("app.load_model", side_effect=OSError("private repo")), self.assertLogs("app", level="ERROR"):
            result = list(app.transcribe(self.path, language="Yoruba", progress=lambda *a, **k: None))[-1]
        self.assertEqual(result[0], "")
        self.assertIsNone(result[1])
        self.assertIn("HF_TOKEN", result[2])

    def test_blank_decode_has_no_download(self):
        self.write()
        with patch("app.load_model", return_value=object()), patch("app.decode", return_value=""):
            result = list(app.transcribe(self.path, language="Yoruba", progress=lambda *a, **k: None))[-1]
        self.assertIsNone(result[1])
        self.assertIn("No words", result[2])

    def test_interface_and_reset(self):
        self.assertTrue(app.demo.config["components"])
        self.assertEqual(len(app.reset_results()), 7)
        self.assertIsNone(app.reset_results()[1])

    def test_model_override(self):
        with patch.dict("os.environ", {"MODEL_ID": "owner/checkpoint"}):
            self.assertEqual(asr.model_source(), "owner/checkpoint")

    def test_nested_auth_error_classification(self):
        from types import SimpleNamespace
        inner = RuntimeError("authentication failed")
        inner.response = SimpleNamespace(status_code=401)
        outer = OSError("wrapped by transformers")
        outer.__cause__ = inner
        self.assertTrue(asr.is_access_error(outer))
        inner.response.status_code = 500
        self.assertFalse(asr.is_access_error(outer))

    def test_access_failure_is_actionable_and_skips_detection(self):
        self.write()
        with patch("app.load_model", side_effect=asr.ModelAccessError("Model access denied: configure HF_TOKEN")), patch("app.detect_language") as detector:
            result = list(app.transcribe(self.path, progress=lambda *a, **k: None))[-1]
        detector.assert_not_called()
        self.assertIn("Model access denied", result[2])
        self.assertIsNone(result[1])

    def test_long_recording_accepted(self):
        self.write(seconds=25)
        waveform, duration = asr.read_audio(self.path)
        self.assertEqual(duration, 25)
        self.assertEqual(len(waveform), 25 * asr.SAMPLE_RATE)

    def test_detection_summary(self):
        labels = {0: "yor", 1: "hau", 2: "ibo", 3: "eng"}
        self.assertIn("Hausa", asr.summarize_language([[0.1, 0.8, 0.05, 0.05]], labels))
        self.assertIn("Outside", asr.summarize_language([[0.05, 0.05, 0.1, 0.8]], labels))
        self.assertIn("Uncertain", asr.summarize_language([[0.3, 0.4, 0.2, 0.1]], labels))

    def test_manual_language_skips_detector(self):
        self.write()
        with patch("app.detect_language") as detector, patch("app.load_model", return_value=object()), patch("app.decode", return_value=""):
            result = list(app.transcribe(self.path, "Igbo", progress=lambda *a, **k: None))[-1]
        detector.assert_not_called()
        self.assertIn("Igbo", result[6])

    def test_detection_failure_still_transcribes(self):
        self.write()
        with patch("app.detect_language", side_effect=OSError("offline")), patch("app.load_model", return_value=object()), patch("app.decode", return_value=""), self.assertLogs("app", level="ERROR"):
            result = list(app.transcribe(self.path, progress=lambda *a, **k: None))[-1]
        self.assertIn("Detection unavailable", result[6])

    def test_upload_source_routing(self):
        with patch("app.transcribe", return_value=iter([])) as callback:
            list(app.transcribe_source("mic.wav", "upload.wav", "upload", "Hausa"))
        self.assertEqual(callback.call_args.args[:2], ("upload.wav", "Hausa"))

    def test_chunked_pipeline_configuration(self):
        from types import SimpleNamespace
        processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
        with patch("transformers.pipelines.pipeline") as factory:
            factory.return_value.return_value = {"text": " long transcript "}
            result = asr.decode(np.zeros(16000 * 45), (processor, object(), "cpu"))
        self.assertEqual(result, "long transcript")
        self.assertEqual(factory.return_value.call_args.kwargs["chunk_length_s"], 20)
        self.assertEqual(factory.return_value.call_args.kwargs["stride_length_s"], 3)

    def test_real_ctc_pipeline_with_small_local_model(self):
        import json
        from transformers import Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
        vocab = Path(self.folder.name) / "vocab.json"
        vocab.write_text(json.dumps({"[PAD]": 0, "[UNK]": 1, "|": 2, "a": 3, "b": 4}), encoding="utf-8")
        tokenizer = Wav2Vec2CTCTokenizer(str(vocab), pad_token="[PAD]", unk_token="[UNK]", word_delimiter_token="|")
        processor = Wav2Vec2Processor(Wav2Vec2FeatureExtractor(sampling_rate=16000, return_attention_mask=True), tokenizer)
        config = Wav2Vec2Config(vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16, conv_dim=(8, 8, 8, 8), conv_stride=(5, 4, 4, 4), conv_kernel=(10, 8, 4, 4), num_conv_pos_embedding_groups=2, num_conv_pos_embeddings=16, pad_token_id=0)
        model = Wav2Vec2ForCTC(config).eval()
        self.write(seconds=25)
        audio, _ = asr.read_audio(self.path)
        result = asr.decode(audio, (processor, model, "cpu"))
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
