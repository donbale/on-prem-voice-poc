# On-Prem Voice Assistant POC

This proof-of-concept demonstrates a **fully on-premise voice assistant** that runs without cloud dependencies.  
It combines:

- 🎤 **Microphone input** (via Streamlit)  
- 🧠 **Local LLM inference** using [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) with [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen) in GGUF format  
- 🗣️ **Text-to-Speech** via [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)  
- 📦 **Streamlit UI** with a fixed “Speak Here” bar at the bottom  
- ⚡ Support for both **CPU** and **GPU (CUDA 12.4)** installs  

---

## Features

- Click **Speak Here** to start/stop recording.  
- Audio is transcribed → passed to the LLM → response is spoken back with Kokoro TTS.  
- Everything runs **locally** (no external APIs).  
- GPU acceleration available if supported.  

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourname/on-prem-voice-poc.git
cd on-prem-voice-poc
```

### 2. Create a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
#### CPU version (portable)
```bash
pip install -r requirements.txt
```

#### GPU version (CUDA 12.4)
```bash
pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 4. Model Setup
```bash
mkdir -p models
wget -O models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 5. Run the App
```bash
streamlit run app.py
```

## Usage

- Hold **spacebar** or click **Speak Here** 🎤 to record  
- Release to stop recording  
- Assistant reply will be:  
  - Printed in chat log  
  - Spoken back via Kokoro  

---

## Requirements Files

- `requirements.txt` → minimal CPU version (works everywhere)  
- `requirements-gpu.txt` → GPU accelerated version (CUDA 12.4 builds of PyTorch & llama-cpp-python)  

---

## Known Issues

- 🔇 Audio autoplay may be blocked in some browsers (Chrome/Edge). Press play if needed.  
- 🔁 Long LLM responses can overlap TTS playback (**chunking WIP**).  
- 🎨 UI styling may shift slightly between browsers.  

---

## Roadmap

- [ ] Fix overlapping TTS playback with streaming queue  
- [ ] Improve error handling for mic permissions  
- [ ] Add option to swap in other GGUF models (e.g. LLaMA, Mistral)  