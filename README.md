# Moshi – Speech-Text Foundation Model for Real-Time Voice Dialogue | AI Voice Assistant

![precommit badge](https://github.com/kyutai-labs/moshi/workflows/precommit/badge.svg)
![rust ci badge](https://github.com/kyutai-labs/moshi/workflows/Rust%20CI/badge.svg)

**Moshi** is an open-source speech-text foundation model for **real-time full-duplex voice dialogue**. It powers natural voice conversations with low latency using the Mimi neural audio codec. Try the [live demo](https://moshi.chat) or use [Hugging Face models](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd).

---

## Author & Contact

**Maintainer:** **KuchikiRenji**

| Contact | Details |
|--------|---------|
| **Email** | [KuchikiRenji@outlook.com](mailto:KuchikiRenji@outlook.com) |
| **GitHub** | [github.com/KuchikiRenji](https://github.com/KuchikiRenji) |
| **Discord** | `kuchiki_renji` |

For questions, contributions, or collaboration, reach out via the channels above.

---

## Table of Contents

- [What is Moshi?](#what-is-moshi)
- [Key Features](#key-features)
- [Repository Structure](#organisation-of-the-repository)
- [Models](#models)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
  - [Python (PyTorch)](#python-pytorch)
  - [Python (MLX) for macOS](#python-mlx-for-local-inference-on-macos)
  - [Rust Backend](#rust)
- [Clients](#clients)
- [Development](#development)
- [FAQ](#faq)
- [License](#license)
- [Citation](#citation)

---

## What is Moshi?

**Moshi** is a **speech-text foundation model** and **full-duplex** spoken dialogue framework. It enables real-time voice AI conversations with:

- **Mimi** – A state-of-the-art **streaming neural audio codec** that processes 24 kHz audio down to a **12.5 Hz** representation at **1.1 kbps**, with **80 ms** frame latency, in a fully streaming way. It outperforms non-streaming codecs such as [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) (50 Hz, 4 kbps) and [SemantiCodec](https://github.com/haoheliu/SemantiCodec-inference) (50 Hz, 1.3 kbps).

- **Dual audio streams** – Moshi models **two streams**: one for the user (from the microphone) and one for Moshi (from the model). It also predicts **text tokens** for its own speech (inner monologue), which improves generation quality.

- **Low latency** – A small **Depth Transformer** handles inter-codebook dependencies per time step; a **7B-parameter Temporal Transformer** handles time. Theoretical latency is **160 ms** (80 ms Mimi frame + 80 ms acoustic); practical end-to-end latency can be as low as **~200 ms** on an L4 GPU.

**[Talk to Moshi](https://moshi.chat)** on the live demo.

<p align="center">
<img src="./moshi.png" alt="Moshi architecture: two audio streams (user and Moshi), text tokens, Depth and Temporal Transformers for real-time voice dialogue."
width="650px"></p>

### About Mimi (Neural Codec)

Mimi builds on [SoundStream](https://arxiv.org/abs/2107.03312) and [EnCodec](https://github.com/facebookresearch/encodec), adding Transformers in both encoder and decoder and using strides for a **12.5 Hz** frame rate—closer to text token rates (~3–4 Hz)—reducing autoregressive steps in Moshi. Like SpeechTokenizer, Mimi uses a distillation loss so the first codebook tokens match [WavLM](https://arxiv.org/abs/2110.13900) representations, modeling both semantic and acoustic content. Mimi is causal and streaming yet matches non-causal WavLM well. Like [EBEN](https://arxiv.org/pdf/2210.14090), it uses **only an adversarial training loss** with feature matching for strong subjective quality at low bitrate.

<p align="center">
<img src="./mimi.png" alt="Mimi neural codec: Transformer encoder and decoder, 12.5 Hz frame rate, streaming audio compression for Moshi."
width="800px"></p>

---

## Key Features

- **Real-time full-duplex voice dialogue** – Speak and hear responses with low latency.
- **Streaming neural codec (Mimi)** – 24 kHz → 12.5 Hz, 1.1 kbps, 80 ms frames.
- **Multiple backends** – PyTorch, MLX (Apple Silicon), and Rust/Candle.
- **Voice variants** – Moshika (female) and Moshiko (male); multiple quantizations (bf16, int8, int4 for MLX).
- **Web UI and CLI** – Easy local or remote use.

---

## Organisation of the repository

| Directory | Description |
|-----------|-------------|
| [`moshi/`](moshi/) | **Python (PyTorch)** – Moshi and Mimi inference. |
| [`moshi_mlx/`](moshi_mlx/) | **Python (MLX)** – Moshi on Apple M-series Macs. |
| [`rust/`](rust/) | **Rust** – Production backend; includes Mimi in Rust and `rustymimi` Python bindings. |
| [`client/`](client/) | **Web UI** – Frontend for the live demo. |

---

## Models

Released models:

- **Mimi** – Speech codec (included in each Moshi repo).
- **Moshika** – Moshi with female voice.
- **Moshiko** – Moshi with male voice.

Formats and quantization depend on the backend. All are on Hugging Face (CC-BY 4.0):

| Model | Backend | Variants |
|-------|---------|----------|
| Moshika | PyTorch | [kyutai/moshika-pytorch-bf16](https://huggingface.co/kyutai/moshika-pytorch-bf16) (bf16) |
| Moshiko | PyTorch | [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (bf16) |
| Moshika | MLX | [q4](https://huggingface.co/kyutai/moshika-mlx-q4) / [q8](https://huggingface.co/kyutai/moshika-mlx-q8) / [bf16](https://huggingface.co/kyutai/moshika-mlx-bf16) |
| Moshiko | MLX | [q4](https://huggingface.co/kyutai/moshiko-mlx-q4) / [q8](https://huggingface.co/kyutai/moshiko-mlx-q8) / [bf16](https://huggingface.co/kyutai/moshiko-mlx-bf16) |
| Moshika | Rust/Candle | [q8](https://huggingface.co/kyutai/moshika-candle-q8) / [bf16](https://huggingface.co/kyutai/moshika-candle-bf16) |
| Moshiko | Rust/Candle | [q8](https://huggingface.co/kyutai/moshiko-candle-q8) / [bf16](https://huggingface.co/kyutai/moshiko-candle-bf16) |

---

## Requirements

- **Python** – 3.10 minimum; **3.12 recommended**. See each backend’s directory for details.
- **PyTorch / MLX** – Install via PyPI (see below). MLX and `rustymimi` may need **Python 3.12** or a [Rust toolchain](https://rustup.rs/).
- **Rust backend** – [Rust toolchain](https://rustup.rs/); for GPU: [CUDA](https://developer.nvidia.com/cuda-toolkit) with `nvcc`.
- **GPU** – PyTorch: ~24 GB VRAM (no quantization). MLX tested on MacBook Pro M3. Windows is not officially supported.

### Install from PyPI

```bash
pip install moshi        # PyTorch
pip install moshi_mlx    # MLX (Python 3.12 recommended)
pip install rustymimi    # Mimi in Rust (Python bindings)

# Bleeding edge from this repo
pip install -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi"
pip install -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi_mlx&subdirectory=moshi_mlx"
```

---

## Quick Start

### Python (PyTorch)

Start the server, then open the web UI at **http://localhost:8998**:

```bash
python -m moshi.server [--gradio-tunnel] [--hf-repo kyutai/moshika-pytorch-bf16]
```

- Use `--gradio-tunnel` for a public URL (e.g. remote GPU). Latency may increase (e.g. +500 ms from Europe).
- Use `--gradio-tunnel-token` for a fixed secret token and stable URL.
- Use `--hf-repo` to pick another Hugging Face model.

CLI client (no echo cancellation):

```bash
python -m moshi.client [--url URL_TO_GRADIO]
```

More details and API: [moshi/README.md](moshi/README.md).

### Python (MLX) for local inference on macOS

```bash
python -m moshi_mlx.local -q 4   # 4-bit quantization
python -m moshi_mlx.local -q 8   # 8-bit
python -m moshi_mlx.local -q 4 --hf-repo kyutai/moshika-mlx-q4
```

Web UI:

```bash
python -m moshi_mlx.local_web
# → http://localhost:8998
```

Match `-q` and `--hf-repo` (e.g. q4 with `*-mlx-q4`).

### Rust

From the `rust` directory:

```bash
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```

On macOS use `--features metal` instead of `--features cuda`. For int8 use `config-q8.json`. Set `"hf_repo"` in the config for Moshika/Moshiko.

When you see **"standalone worker listening"**, open the web UI at **https://localhost:8998** (browser may show a warning; you can proceed to localhost).

---

## Clients

- **Web UI** (recommended) – Echo cancellation and best experience; usually served automatically at the URLs above.
- **Rust CLI** – From `rust/`: `cargo run --bin moshi-cli -r -- tui --host localhost`
- **Python CLI** – `python -m moshi.client`

### Building the Web UI

```bash
cd client
npm install
npm run build
```

Output is in `client/dist`.

---

## Development

From the repo root:

```bash
pip install -e 'moshi[dev]'
pip install -e 'moshi_mlx[dev]'
pre-commit install
```

Build `rustymimi` locally (with Rust installed):

```bash
pip install maturin
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

---

## FAQ

See [FAQ.md](FAQ.md) before opening an issue. Common topics: training code, dataset, multilingual support, voice/personality, M1/small GPU, PyTorch quantization.

---

## License

- **Code** – MIT (Python, client); Apache (Rust backend). Some code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft) (MIT).
- **Model weights** – CC-BY 4.0.

---

## Citation

If you use Mimi or Moshi, please cite:

```bibtex
@techreport{kyutai2024moshi,
    author = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and
              Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year = {2024},
    month = {September},
    url = {http://kyutai.org/Moshi.pdf},
}
```

**Paper:** [Moshi: a speech-text foundation model for real-time dialogue](http://kyutai.org/Moshi.pdf)
