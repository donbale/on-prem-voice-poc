import os
import time
import tempfile
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import base64

from faster_whisper import WhisperModel
from kokoro import KPipeline
from llama_cpp import Llama

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
KOKORO_SR = 24000

# UI typing cadence
WORDS_PER_TICK = 3
TYPE_TICK_SLEEP = 0.015

# LLM output controls
SYSTEM_PROMPT = (
    "You are a concise, friendly voice assistant. "
    "Answer in 1–2 sentences (<= 35 words total)."
)
MAX_TOKENS = 160
MAX_CHARS_TOTAL = 360


# =========================
# Cached loaders
# =========================
@st.cache_resource
def load_vad():
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    model.to(DEVICE)
    if isinstance(utils, dict):
        get_speech_timestamps = utils["get_speech_timestamps"]
        collect_chunks = utils["collect_chunks"]
    else:
        get_speech_timestamps, _, _, _, collect_chunks = utils
    return model, get_speech_timestamps, collect_chunks


@st.cache_resource
def load_whisper():
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    return WhisperModel("base.en", device=DEVICE, compute_type=compute_type)


@st.cache_resource
def load_kokoro(lang_code="a"):
    return KPipeline(lang_code=lang_code)


@st.cache_resource
def load_llm():
    return Llama(
        model_path="./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_gpu_layers=-1 if DEVICE == "cuda" else 0,
        n_ctx=4096,
        seed=42,
        verbose=False,
    )


# =========================
# Helpers
# =========================
def ensure_mono_16k_from_path(p: str | os.PathLike) -> torch.Tensor:
    wav, sr = torchaudio.load(p)
    if wav.dim() == 2 and wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def transcribe(whisper_model: WhisperModel, audio_path: str) -> str:
    segs, _info = whisper_model.transcribe(audio_path, vad_filter=True)
    return " ".join([s.text.strip() for s in segs]).strip()


def kokoro_speak_full(kokoro: KPipeline, text: str, voice: str, speed: float) -> Tuple[np.ndarray, int]:
    """Synthesize one single clip for the entire final LLM text."""
    chunks = []
    for _, _, audio in kokoro(text, voice=voice, speed=speed):
        chunks.append(audio)
    if not chunks:
        return np.zeros(0, dtype=np.float32), KOKORO_SR
    audio = np.concatenate(chunks, axis=0).astype(np.float32)
    return audio, KOKORO_SR


def llm_answer_stream(history: List[dict]) -> Iterable[str]:
    llm = load_llm()
    msgs = []
    if not history or history[0].get("role") != "system":
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.extend(history)

    stream = llm.create_chat_completion(
        messages=msgs,
        stream=True,
        max_tokens=MAX_TOKENS,
        temperature=0.5,
        top_p=0.9,
    )
    for chunk in stream:
        if "choices" in chunk:
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece:
                yield piece


# =========================
# Audio pipeline
# =========================
def process_audio_file(audio_path: str) -> str | None:
    vad_model, get_speech_timestamps, _ = load_vad()
    whisper_model = load_whisper()

    wav_16k = ensure_mono_16k_from_path(audio_path).to(DEVICE)
    ts = get_speech_timestamps(wav_16k, vad_model, sampling_rate=SAMPLE_RATE)
    if not ts:
        st.warning("🔇 No speech detected.")
        return None

    with st.spinner("🔎 Transcribing..."):
        text = transcribe(whisper_model, audio_path)

    if not text:
        st.warning("🤷 Couldn’t transcribe any text.")
        return None

    return text


def stream_assistant_reply_exact_tts(
    history: List[dict],
    kokoro: KPipeline,
    voice: str,
    speed: float,
    ui_text_ph: st.delta_generator.DeltaGenerator,
) -> str:
    """Stream text tokens to UI, then synthesize TTS once for the full text."""
    acc_text = ""
    char_budget = MAX_CHARS_TOTAL
    words_since_tick = 0

    for tok in llm_answer_stream(history):
        if char_budget <= 0:
            break
        tok = tok.replace("\n", " ")
        acc_text += tok
        char_budget -= len(tok)

        words_since_tick += tok.count(" ")
        if words_since_tick >= WORDS_PER_TICK:
            ui_text_ph.markdown(acc_text.strip())
            words_since_tick = 0
            time.sleep(TYPE_TICK_SLEEP)

    ui_text_ph.markdown(acc_text.strip())

    # Generate audio
    audio_np, sr = kokoro_speak_full(kokoro, acc_text.strip(), voice, speed)
    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(out_wav, audio_np, sr)

    # Play once (no replay bar)
    st.markdown(
        f"""
        <audio autoplay>
            <source src="data:audio/wav;base64,{base64.b64encode(open(out_wav,'rb').read()).decode()}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )

    return acc_text.strip()


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="On-Prem Voice Chat", page_icon="🎤")
st.title("🎤 On-Prem Voice Assistant")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello 👋, how can I help you today?"}
    ]

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    voice = st.selectbox("Voice", ["af_bella", "af_sarah", "am_adam"], index=0)
    speed = st.slider("Speech speed", 0.6, 1.4, 1.0, 0.05)

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ========== Fixed Mic Bar at Bottom ==========
st.markdown(
    """
    <style>
    .mic-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: #111;
        padding: 12px 20px;
        border-top: 1px solid #333;
        box-shadow: 0 -2px 6px rgba(0,0,0,0.4);
        z-index: 1000;
    }
    .mic-bar label { color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="mic-bar">', unsafe_allow_html=True)
rec_file = st.audio_input("🎙 Speak here…", key="mic_input")
st.markdown('</div>', unsafe_allow_html=True)


# Handle new user input
if rec_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(rec_file.read())  # ✅ FIX
        audio_path = tmp.name

    user_text = process_audio_file(audio_path)
    if user_text:
        st.session_state["messages"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            text_ph = st.empty()

        kokoro = load_kokoro()
        final_text = stream_assistant_reply_exact_tts(
            st.session_state["messages"], kokoro, voice, speed, text_ph
        )

        st.session_state["messages"].append({"role": "assistant", "content": final_text})