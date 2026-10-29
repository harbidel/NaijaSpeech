"""Inference for the XLS-R CTC checkpoint from the training notebook."""
from functools import lru_cache
from math import gcd
import os
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "true", "yes")
if ZERO_GPU:
    import spaces  # Initialize CUDA emulation before importing torch.

SAMPLE_RATE = 16000
MAX_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "600"))
CHUNK_SECONDS = 20
LANGUAGES = {"yor": "Yoruba", "hau": "Hausa", "ibo": "Igbo"}
LANGUAGE_CHOICES = ["Auto-detect", *LANGUAGES.values()]
DEFAULT_MODEL = "Harbidel/yoruba-hausa-igbo-xls-r-300m-ctc"


class ModelAccessError(RuntimeError):
    """The host lacks permission to download the configured ASR checkpoint."""


def is_access_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if getattr(getattr(error, "response", None), "status_code", None) in (401, 403):
            return True
        error = error.__cause__ or error.__context__
    return False


def model_source():
    configured = os.environ.get("MODEL_ID")
    if configured:
        return configured
    for name in ("final_model", "model"):
        folder = Path(__file__).parent / name
        if (folder / "config.json").is_file():
            return str(folder)
    return DEFAULT_MODEL


def read_audio(path):
    if not path:
        raise ValueError("Record or upload some audio first.")
    try:
        with sf.SoundFile(path) as source:
            rate = source.samplerate
            duration = source.frames / rate
            if not 0.25 <= duration <= MAX_SECONDS:
                raise ValueError(f"Use a recording between 0.25 and {MAX_SECONDS} seconds. This clip is {duration:.1f} seconds.")
            if source.channels > 2:
                raise ValueError("Please upload a mono or stereo recording.")
            if source.frames * source.channels > 60_000_000:
                raise ValueError("This audio has an unusually high sample rate. Export it as mono WAV at 16 kHz or upload a shorter clip.")
            audio = source.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError("This audio could not be read. Try uploading a WAV or MP3 file again.") from exc
    if not np.isfinite(audio).all():
        raise ValueError("This recording contains invalid audio samples. Please record it again.")
    if float(np.max(np.abs(audio))) < 1e-5:
        raise ValueError("This recording is silent. Check your microphone and try again.")
    if rate != SAMPLE_RATE:
        divisor = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    return np.asarray(audio, dtype=np.float32), duration


@lru_cache(maxsize=1)
def load_model():
    # Lazy loading keeps the interface available while the checkpoint downloads.
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    source = model_source()
    kwargs = {"token": os.environ.get("HF_TOKEN") or None}
    try:
        processor = Wav2Vec2Processor.from_pretrained(source, **kwargs)
        model = Wav2Vec2ForCTC.from_pretrained(source, **kwargs)
    except Exception as exc:
        if is_access_error(exc):
            raise ModelAccessError(
                "Model access denied by Hugging Face (401/403). This app needs permission to read the private ASR model. "
                "Locally, run .\\.venv\\Scripts\\hf.exe auth login with a token that can read the model repository. "
                "On Spaces, add HF_TOKEN under Settings > Variables and secrets as a secret, then restart the Space. "
                "If credentials are already configured, check the token's repository access."
            ) from exc
        raise
    device = "cuda" if ZERO_GPU or torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return processor, model, device


def decode(audio, bundle):
    if ZERO_GPU:
        return _decode_zero_gpu(audio)
    return _decode_pipeline(audio, bundle)


def _decode_pipeline(audio, bundle):
    from transformers.pipelines import pipeline
    processor, model, device = bundle
    recognizer = pipeline(
        "automatic-speech-recognition", model=model,
        tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
        device=device,
    )
    # CTC pipeline trims overlapping token frames and decodes the joined token
    # sequence once, avoiding duplicated boundary words from string joining.
    return recognizer(audio, chunk_length_s=CHUNK_SECONDS, stride_length_s=3, batch_size=1)["text"].strip()


if ZERO_GPU:
    # ZeroGPU requires CUDA placement during startup, outside its GPU worker.
    _zero_gpu_bundle = load_model()

    @spaces.GPU(duration=120)
    def _decode_zero_gpu(audio):
        return _decode_pipeline(audio, _zero_gpu_bundle)


@lru_cache(maxsize=1)
def load_language_detector():
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    repo = os.environ.get("LID_MODEL_ID", "facebook/mms-lid-126")
    kwargs = {"token": os.environ.get("HF_TOKEN") or None}
    extractor = AutoFeatureExtractor.from_pretrained(repo, **kwargs)
    # Keep LID on CPU so it does not compete with ASR for GPU memory.
    model = Wav2Vec2ForSequenceClassification.from_pretrained(repo, **kwargs).eval()
    return extractor, model


def summarize_language(probabilities, labels):
    scores = np.mean(probabilities, axis=0)
    winner = int(np.argmax(scores))
    code = labels.get(winner, labels.get(str(winner), "unknown"))
    score = float(scores[winner])
    if code not in LANGUAGES:
        return f"Outside supported languages or uncertain ({code}); select a language if needed."
    if score < 0.5:
        return f"Uncertain — possibly {LANGUAGES[code]}. Select a language if you know it."
    return f"{LANGUAGES[code]} · automatically detected"


def detect_language(audio):
    import torch

    extractor, model = load_language_detector()
    # Sample up to three windows across the recording, avoiding reliance only
    # on an opening greeting. This reports an overall language, not segments.
    size = min(len(audio), SAMPLE_RATE * 10)
    starts = np.unique(np.linspace(0, max(0, len(audio) - size), min(3, max(1, len(audio) // size)), dtype=int))
    probabilities = []
    for start in starts:
        clip = audio[start:start + size]
        if np.sqrt(np.mean(clip ** 2)) < 1e-4:
            continue
        inputs = extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            probabilities.append(model(**inputs).logits.softmax(dim=-1)[0].cpu().numpy())
    if not probabilities:
        return "Uncertain — not enough audible speech."
    return summarize_language(probabilities, model.config.id2label)
