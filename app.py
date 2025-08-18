# app.py — Fast Windows PoC (LLM via llama.cpp GGUF, Whisper STT, Chatterbox TTS, Gradio UI)
import os
import tempfile
from typing import List, Dict, Optional

import torch
import torchaudio
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ---------- Config ----------
GGUF_PATH = os.environ.get("GGUF_PATH", r"C:\models\tinyllama-1.1b-chat-q4_k_m.gguf")
N_CTX = int(os.environ.get("N_CTX", "4096"))
N_THREADS = int(os.environ.get("N_THREADS", max(1, os.cpu_count() - 1)))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "160"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base.en")
TTS_CFG_WEIGHT = float(os.environ.get("TTS_CFG_WEIGHT", "0.4"))
TTS_EXAGGERATION = float(os.environ.get("TTS_EXAGGERATION", "0.35"))

GRADIO_PORT = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "false").lower() == "true"
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- STT ----------
from faster_whisper import WhisperModel
print(f"[INIT] Loading Whisper ({WHISPER_SIZE}) on {DEVICE} …")
whisper = WhisperModel(
    WHISPER_SIZE,
    device=DEVICE,
    compute_type="float16" if DEVICE == "cuda" else "int8"
)

def transcribe_audio(wav_path: str) -> str:
    # small beam_size for speed, VAD on
    segments, _ = whisper.transcribe(wav_path, beam_size=1, vad_filter=True)
    return " ".join(seg.text for seg in segments).strip()

# ---------- LLM (llama.cpp) ----------
from llama_cpp import Llama, LogitsProcessorList

print(f"[INIT] Loading GGUF model from: {GGUF_PATH}")
llm = Llama(
    model_path=GGUF_PATH,
    n_ctx=N_CTX,
    n_threads=N_THREADS,
    n_gpu_layers=N_GPU_LAYERS,   # 0 = CPU; >0 offloads to GPU (if cuBLAS wheel)
    use_mlock=True,              # lock in RAM for speed (if permitted)
    use_mmap=True,               # memory-mapped
    verbose=False
)

SYSTEM_PROMPT = (
    "You are a concise, friendly on-prem voice assistant. "
    "Answer briefly and clearly. If a task requires the internet, say you are offline."
)

# Maintain a short rolling history to keep prompt small/fast
def clamp_history(history: List[Dict[str, str]], max_turns: int = 4) -> List[Dict[str, str]]:
    # keep last N user/assistant pairs
    trimmed: List[Dict[str, str]] = []
    count_pairs = 0
    for msg in reversed(history):
        trimmed.append(msg)
        if msg["role"] == "user":
            count_pairs += 1
            if count_pairs >= max_turns:
                break
    trimmed.reverse()
    return trimmed

def build_prompt(history: List[Dict[str, str]], user_text: str) -> str:
    # Simple chat template for speed
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + clamp_history(history) + [
        {"role": "user", "content": user_text}
    ]
    lines = []
    for m in msgs:
        if m["role"] == "system":
            lines.append(f"<|system|>\n{m['content']}")
        elif m["role"] == "user":
            lines.append(f"<|user|>\n{m['content']}")
        else:
            lines.append(f"<|assistant|>\n{m['content']}")
    lines.append("<|assistant|>\n")
    return "\n".join(lines)

def generate_stream(prompt: str):
    """Yield tokens for streaming text in the UI."""
    acc = ""
    for out in llm.create_completion(
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
        stop=["<|user|>", "<|system|>", "<|assistant|>"],
    ):
        tok = out["choices"][0]["text"]
        acc += tok
        yield acc

# ---------- TTS (Chatterbox) ----------
from chatterbox.tts import ChatterboxTTS
print(f"[INIT] Loading Chatterbox TTS on {DEVICE} …")
tts = ChatterboxTTS.from_pretrained(device=DEVICE)

def synthesize_speech(text: str, voice_wav: Optional[str] = None) -> str:
    wav_tensor = tts.generate(
        text,
        audio_prompt_path=voice_wav,
        cfg_weight=TTS_CFG_WEIGHT,
        exaggeration=TTS_EXAGGERATION,
    )
    sr = tts.sr
    out_path = tempfile.mktemp(suffix=".wav")
    if wav_tensor.dtype != torch.float32:
        wav_tensor = wav_tensor.to(torch.float32)
    torchaudio.save(out_path, wav_tensor.cpu(), sr)
    return out_path

# ---------- Pipeline ----------
def handle_once(input_audio_path: Optional[str], reference_voice_path: Optional[str], history: List[Dict[str, str]]):
    if not input_audio_path:
        return history, "", "", None, None, "Please record or upload audio."
    # 1) STT
    try:
        user_text = transcribe_audio(input_audio_path)
    except Exception as e:
        return history, "", "", None, None, f"STT error: {e}"
    if not user_text:
        return history, "", "", None, None, "I didn't catch that. Please try again."

    # 2) LLM (streaming text)
    prompt = build_prompt(history, user_text)
    stream = generate_stream(prompt)

    # We’ll stream text in the UI; when it’s done, we generate TTS once.
    final_text = ""
    for partial in stream:
        final_text = partial
        yield history, user_text, partial, None, None, ""  # stream updates

    # 3) Update history with final text
    new_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": final_text},
    ]

    # 4) TTS (after text is done)
    try:
        tts_wav = synthesize_speech(final_text, voice_wav=reference_voice_path)
    except Exception as e:
        yield new_history, user_text, final_text, None, None, f"TTS error: {e}"
        return

    yield new_history, user_text, final_text, tts_wav, None, ""

# ---------- UI ----------
with gr.Blocks(title="On-Prem Voice Agent (Fast)") as demo:
    gr.Markdown(
        "# ⚡ On-Prem Voice Agent (Fast)\n"
        "- **LLM:** GGUF via llama.cpp (CPU/GPU), streamed\n"
        "- **STT:** Whisper (faster-whisper, INT8 on CPU)\n"
        "- **TTS:** Chatterbox (local)\n"
        "Tip: keep utterances ~3–5s for lowest latency."
    )

    state = gr.State([])

    with gr.Row():
        mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak or upload")
        ref = gr.Audio(sources=["upload"], type="filepath", label="(Optional) Reference Voice")

    with gr.Row():
        stt_box = gr.Textbox(label="Transcribed (STT)", interactive=False)
    with gr.Row():
        llm_box = gr.Textbox(label="Assistant (streaming)", interactive=False, lines=6)
    with gr.Row():
        tts_audio = gr.Audio(label="Assistant (TTS)", interactive=False)
    err_box = gr.Markdown("")

    go = gr.Button("Process")
    clear = gr.Button("Clear")

    go.click(
        fn=handle_once,
        inputs=[mic, ref, state],
        outputs=[state, stt_box, llm_box, tts_audio, ref, err_box],
        api_name="process",
        show_api=False,
        queue=True,
    )

    def reset():
        return [], "", "", None, None, ""

    clear.click(reset, None, [state, stt_box, llm_box, tts_audio, ref, err_box], queue=False)

if __name__ == "__main__":
    # Gradio v5: either omit queue() or call it without args
    demo.queue()
    demo.launch(server_name=GRADIO_SERVER_NAME, server_port=GRADIO_PORT, share=GRADIO_SHARE)

