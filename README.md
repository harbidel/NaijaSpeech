---
title: NaijaSpeech
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.9.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Yoruba, Hausa and Igbo speech, in their own words.
---

# NaijaSpeech

A Gradio speech-to-text demo for the multilingual XLS-R 300M CTC model in `Multilingual_Yoruba_Hausa_Igbo_XLSR_CTC.ipynb`.

Features: separate microphone and upload tabs, recordings up to 10 minutes, automatic language identification or manual Yoruba/Hausa/Igbo selection, a copyable transcript, UTF-8 download, duration, inference timing, word count, and progress feedback. All three languages use the same ASR decoder; manual selection labels the recording and skips identification, but does not force the decoder to a language. There is no translation.

## Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space on Hugging Face.
2. Upload `app.py`, `asr.py`, `style.css`, `requirements.txt`, `packages.txt`, and this `README.md` to its root. The training notebook is not needed for deployment.
3. In **Settings → Variables and secrets**, add a **secret** named `HF_TOKEN` with read access to `Harbidel/yoruba-hausa-igbo-xls-r-300m-ctc` if that repository is private (the notebook creates it as private). Never put a token in source code or in a public variable.
4. Optionally set the `MODEL_ID` variable to a different compatible repository ID or local checkpoint path.
5. Wait for the build, then use **Record** or **Upload recording**, choose a language mode, and select **Transcribe audio**. The active tab determines which audio is processed. First use downloads the ASR checkpoint and, in Auto-detect mode, the separate language detector. CPU is supported; an available GPU is used for ASR. Long recordings can take several minutes on CPU.

Optional variables: `MAX_AUDIO_SECONDS` changes the default 600-second limit; `LID_MODEL_ID` changes the language detector (must be compatible with Wav2Vec2ForSequenceClassification and ISO 639-3 labels). The upload limit is 100 MB. Consider memory and shared queue wait times before raising the duration limit.

ZeroGPU is supported: the ASR model is placed on CUDA at startup and inference uses `spaces.GPU` with a 120-second allocation. Language identification remains on CPU. GPU availability and user quotas can affect requests. Other environments continue to use a dedicated GPU when available, otherwise CPU.

## Language detection

Auto-detect uses [facebook/mms-lid-126](https://huggingface.co/facebook/mms-lid-126), a separate 1B-parameter audio language classifier supporting `yor`, `hau`, and `ibo`. It runs on CPU and adds roughly 4 GB of model weights plus runtime overhead; allow sufficient host RAM alongside the ASR model. The checkpoint is **CC-BY-NC-4.0 (noncommercial)**. Check that license before deploying commercially, or replace the detector with one appropriately licensed for your use.

The app samples up to three 10-second sections across the recording and averages their class probabilities. It reports the winning language among all classifier languages, rather than forcing unsupported speech into Yoruba/Hausa/Igbo. Scores below 0.5 are shown as uncertain; this is a heuristic, not calibrated confidence. It does not label individual speakers or language switches. Manual selection skips the detector. A detector loading failure does not prevent ASR transcription.

The trained model and processor must already be saved/uploaded. A notebook alone is not a checkpoint. Required files include `config.json`, model weights (`model.safetensors` or `pytorch_model.bin`, plus any shards/index), `preprocessor_config.json`, `vocab.json`, and saved tokenizer configuration files.

## Run locally

Use Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
# If the model is private, set HF_TOKEN in your environment before launching.
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\python app.py
```

Open the localhost URL printed by Gradio. Install FFmpeg and put it on PATH for browser recordings and compressed audio conversion; Hugging Face installs it via `packages.txt`.

**401 Unauthorized / model access denied:** browser login to huggingface.co does not authenticate the Python app. Run the login command above in your own terminal and enter a read-capable token for the private model. You can decline saving it as a Git credential. Alternatively configure `HF_TOKEN` in the app process's environment. For a hosted Space, add `HF_TOKEN` as a Space secret, then restart. Never paste the token into chat, source files, or a public Space variable.

To use a local model, put the complete exported `final_model` folder beside `app.py`, or set `MODEL_ID` to its path. Resolution order: `MODEL_ID`, `./final_model`, `./model`, then the notebook's Hugging Face repository. Restart the app after changing the model or credentials.

## Inference and limits

- Recordings must be 0.25–600 seconds by default. Uploads are capped at 100 MB. Limits are checked explicitly; long recordings are not silently truncated.
- The ASR pipeline processes 20-second windows with 3-second left/right context and batch size 1. It removes overlap in token space and decodes the joined sequence, avoiding naive concatenation of duplicated boundary words. The notebook's 20-second training limit is a window size, not a hard limit on recording duration.
- Audio is downmixed to mono and resampled to 16 kHz using polyphase resampling. The saved processor handles normalization. Greedy CTC decoding preserves the trained vocabulary.
- One inference request runs at a time, with up to 12 queued requests. The model is loaded lazily and reused, so the UI can start even before model access is configured.
- Inference time excludes language detection, checkpoint download/loading, and audio preparation. No fabricated accuracy figures are displayed.
- Transcripts generally contain lowercase text without punctuation. Accuracy depends on accents, noise, domain and code-switching.
- Audio is processed on the host. The app does not train on uploads or log transcripts. Gradio temporary uploads/downloads are cleaned hourly when older than one hour, while the server is running. Reset clears the UI; it does not promise instant deletion of cached files.

## Checks

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The automated checks cover audio validation, recordings beyond 20 seconds, stereo/resampling, Unicode download, inference errors, language decisions, manual mode, detector failure, upload routing, and chunked pipeline configuration without downloading model checkpoints. Real transcription and detection accuracy require the actual weights and representative speech.

References: [Gradio Audio](https://www.gradio.app/docs/gradio/audio), [Spaces configuration](https://huggingface.co/docs/hub/spaces-config-reference), [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2).
