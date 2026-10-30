"""NaijaSpeech — a Gradio demo for Yoruba, Hausa and Igbo ASR."""
import logging
from pathlib import Path
import tempfile
import time

import gradio as gr

from asr import LANGUAGE_CHOICES, MAX_SECONDS, ModelAccessError, decode, detect_language, load_model, read_audio

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).parent
CSS = (ROOT / "style.css").read_text(encoding="utf-8")
LOGO = (ROOT / "assets" / "logo.svg").read_text(encoding="utf-8")
IDLE = "Ready when you are · Add a recording to begin."


def styled_html(content):
    # Gradio 6 scopes HTML styling; provide component CSS explicitly instead
    # of depending on global selectors overriding its default typography.
    return gr.HTML(content, css_template=CSS, apply_default_css=False)


def build_theme():
    theme = gr.themes.Soft(primary_hue="emerald", neutral_hue="stone", font=["Arial", "sans-serif"])
    theme.set(
        body_background_fill="#f7f6f0", body_text_color="#183c32",
        body_text_color_subdued="#526454", background_fill_primary="#fffefa",
        background_fill_secondary="#eef1e6", block_background_fill="#fffefa",
        block_label_background_fill="#fffefa", block_label_text_color="#304e3b",
        block_title_text_color="#244738", input_background_fill="#f5f6ee",
        input_border_color="#c7d1bf", input_placeholder_color="#64745e",
        button_primary_background_fill="#254f3b", button_primary_text_color="#ffffff",
        button_secondary_background_fill="#edf1e5", button_secondary_text_color="#254f3b",
    )
    # Keep the intentionally light studio palette coherent when the embedding
    # HF page or the visitor's OS chooses dark mode.
    values = theme.to_dict()["theme"]
    theme.set(**{key: values[key[:-5]] for key in values if key.endswith("_dark") and key[:-5] in values})
    return theme


def transcribe(path, language="Auto-detect", progress=gr.Progress()):
    # Clear previous results even when validation or inference fails.
    yield "", None, "Preparing your audio…", "—", "—", "—", "—"
    try:
        progress(0.05, desc="Checking audio")
        audio, duration = read_audio(path)
        if language not in LANGUAGE_CHOICES:
            raise ValueError("Choose Auto-detect, Yoruba, Hausa, or Igbo.")
        progress(0.1, desc="Getting ready · the first request may take a moment")
        bundle = load_model()
        language_result = f"{language} · manually selected"
        if language == "Auto-detect":
            progress(0.25, desc="Identifying the language")
            try:
                language_result = detect_language(audio)
            except Exception:
                LOG.exception("Language detection failed")
                language_result = "Detection unavailable · select a language manually. Transcription is still available."
        progress(0.4, desc="Transcribing your recording · longer recordings take more time")
        start = time.perf_counter()
        transcript = decode(audio, bundle)
        elapsed = time.perf_counter() - start
        if not transcript:
            yield "", None, "No words recognized. Try a clearer recording with less background noise.", f"{duration:.1f}s", f"{elapsed:.1f}s", "0", language_result
            return
        with tempfile.TemporaryDirectory(prefix="speech-transcript-") as folder:
            download = Path(folder) / "transcript.txt"
            download.write_text(transcript + "\n", encoding="utf-8")
            progress(1, desc="Complete")
            # Gradio copies this output into its managed cache before resuming
            # the generator; the original is removed as the context exits.
            yield transcript, str(download), "Transcription complete · Your words, in your language.", f"{duration:.1f}s", f"{elapsed:.1f}s", str(len(transcript.split())), language_result
    except (ValueError, ModelAccessError) as exc:
        yield "", None, str(exc), "—", "—", "—", "—"
    except Exception:
        LOG.exception("Transcription failed")
        yield "", None, "The model could not complete this request. Try again shortly. If this persists, the Space owner should check model access, HF_TOKEN, and the server logs.", "—", "—", "—", "—"


def reset_results():
    return "", None, IDLE, "—", "—", "—", "—"


def transcribe_source(recording, upload, source, language, progress=gr.Progress()):
    yield from transcribe(recording if source == "record" else upload, language, progress)


def build_demo():
    with gr.Blocks(title="NaijaSpeech · Multilingual ASR", delete_cache=(3600, 3600)) as demo:
        styled_html("""
        <header class="masthead"><a class="brand" href="#" aria-label="NaijaSpeech home">""" + LOGO + """<span>Naija<span class="brand-light">Speech</span></span></a>
        <span class="edition">THE MULTILINGUAL SPEECH LAB <span class="live-dot"></span></span></header>
        <section class="hero">
          <div class="hero-copy"><div class="eyebrow">THREE LANGUAGES. COUNTLESS STORIES.</div>
            <h1>Every voice.<br><em>In its own words.</em></h1>
            <p>Turn Yoruba, Hausa, and Igbo speech into text.<br>Speak naturally. Let your language take the lead.</p>
            <div class="language-tags"><span>Yorùbá</span><span>Hausa</span><span>Igbo</span></div>
          </div>
          <div class="sound-art" aria-hidden="true"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
            <div class="wave">""" + "".join(f'<i style="--h:{h}px"></i>' for h in [24,44,32,70,104,60,126,170,110,76,144,190,132,88,154,110,64,98,52,34,56,24]) + """</div>
            <span class="art-caption">DIFFERENT SOUNDS. SHARED POSSIBILITIES.</span>
          </div>
        </section>
        <div class="workspace-title"><h2>The listening room</h2><span>YOUR VOICE → YOUR TEXT</span></div>
        """)
        with gr.Row(equal_height=True, elem_id="studio"):
            with gr.Column(scale=1, min_width=300, elem_classes="studio-panel"):
                styled_html(f'<div class="panel-title"><span class="step">01</span><h3>Add your voice</h3><span class="panel-note">UP TO {MAX_SECONDS // 60} MIN</span></div>')
                gr.Markdown(f"Record here or upload an existing recording. Up to {MAX_SECONDS // 60} minutes / 100 MB.", elem_classes="subtle")
                source = gr.State("record")
                with gr.Tabs():
                    with gr.Tab("Record", id="record") as record_tab:
                        audio = gr.Audio(sources=["microphone"], type="filepath", format="wav", label="Microphone recording", elem_id="audio-input")
                    with gr.Tab("Upload recording", id="upload") as upload_tab:
                        upload = gr.Audio(sources=["upload"], type="filepath", format="wav", label="Upload WAV, MP3, FLAC, M4A or OGG", elem_id="upload-input")
                language = gr.Dropdown(LANGUAGE_CHOICES, value="Auto-detect", label="Recording language", info="Estimate the recording's language automatically, or label it yourself. This choice does not translate the audio.")
                gr.Markdown("**A little quiet goes a long way.** Speak clearly and keep your microphone close.", elem_classes="recording-tip")
                with gr.Row():
                    submit = gr.Button("Transcribe audio  →", variant="primary", scale=3, elem_id="transcribe-button")
                    clear = gr.Button("Reset", scale=1, variant="secondary")
            with gr.Column(scale=1, min_width=300, elem_classes="studio-panel"):
                styled_html('<div class="panel-title"><span class="step">02</span><h3>See your words</h3><span class="panel-note">TRANSCRIPT</span></div>')
                transcript = gr.Textbox(label="Transcription", show_label=False, placeholder="Your words will appear here…", lines=7, interactive=False, buttons=["copy"], elem_id="transcript")
                with gr.Row(elem_id="metrics"):
                    duration = gr.Textbox(value="—", label="AUDIO LENGTH", interactive=False, min_width=85)
                    speed = gr.Textbox(value="—", label="INFERENCE TIME", interactive=False, min_width=85)
                    words = gr.Textbox(value="—", label="WORDS", interactive=False, min_width=65)
                download = gr.DownloadButton("Download transcript ↓", value=None, variant="secondary")
                detected_language = gr.Textbox(value="—", label="Language", interactive=False)
        status = gr.Textbox(value=IDLE, label="Status", interactive=False, elem_id="status")
        styled_html('''<section class="notes-grid">
          <div><span class="note-index">01 / RECORD OR UPLOAD</span><h3>A voice, not a script.</h3><p>Record with your microphone or upload a saved conversation, interview, or voice note.</p></div>
          <div><span class="note-index">02 / TRANSCRIBE</span><h3>Your language, recognized.</h3><p>Let the app estimate the language, or select Yoruba, Hausa, or Igbo yourself.</p></div>
          <div><span class="note-index">03 / EXPLORE</span><h3>Keep the conversation going.</h3><p>Copy or download your transcript, then try another recording. Accents and audio quality can affect results.</p></div>
        </section>''')
        with gr.Accordion("Using NaijaSpeech", open=False):
            gr.Markdown("""**Speak or upload.** Turn Yoruba, Hausa, and Igbo recordings into text. NaijaSpeech transcribes your words; it does not translate them.

**For better results.** Use clear audio with little background noise. Automatic language detection can be uncertain, especially in mixed-language recordings. Always review your transcript for errors.

**Your privacy.** Audio is sent to this service for transcription and stored temporarily during processing. Uploads are not used to train the app. Only share recordings you have permission to use.""")
        styled_html('<footer><span>NaijaSpeech <span class="footer-separator">/</span> Made for the way we speak.</span><span>YORÙBÁ · HAUSA · IGBO</span></footer>')
        outputs = [transcript, download, status, duration, speed, words, detected_language]
        submit.click(
            transcribe_source, inputs=[audio, upload, source, language], outputs=outputs,
            concurrency_limit=1, concurrency_id="model", trigger_mode="once",
            show_progress="full", show_progress_on=[status],
        )
        # Queue reset/input changes with inference so a late result cannot overwrite a reset.
        audio.change(reset_results, outputs=outputs, concurrency_id="model", concurrency_limit=1)
        upload.change(reset_results, outputs=outputs, concurrency_id="model", concurrency_limit=1)
        language.change(reset_results, outputs=outputs, concurrency_id="model", concurrency_limit=1)
        record_tab.select(lambda: ("record", *reset_results()), outputs=[source, *outputs], concurrency_id="model", concurrency_limit=1)
        upload_tab.select(lambda: ("upload", *reset_results()), outputs=[source, *outputs], concurrency_id="model", concurrency_limit=1)
        clear.click(lambda: (None, None, *reset_results()), outputs=[audio, upload, *outputs], concurrency_id="model", concurrency_limit=1)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.queue(max_size=12, default_concurrency_limit=1).launch(
        theme=build_theme(),
        css=CSS,
        max_file_size="100mb",
        show_error=False,
    )
