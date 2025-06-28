# Nano-vLLM (macOS/MPS Fork)

> **Note:** This is a community fork of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) specifically optimized for **macOS** with Metal Performance Shaders (MPS). All code related to CUDA, Linux, and distributed training has been removed to keep it lightweight. This fork is intended for Apple Silicon and Intel Macs using PyTorch MPS.

A lightweight vLLM implementation, now macOS/MPS only.

## Key Features

* 🚀 **Fast offline inference on macOS** – Optimized for MPS, competitive speeds on Apple hardware
* 📖 **Readable codebase** – Clean implementation in ~1,200 lines of Python
* ⚡ **MPS-optimized** – Efficient KV cache, correct causal attention, eager execution

## Installation

```bash
pip install git+https://github.com/jacko06v/nano-vllm-macos.git
```

## Manual Download

If you prefer to download model weights manually:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example_macos.py` for usage on macOS. The API is similar to vLLM, but now optimized for MPS:


**Test Configuration:**
- Hardware: MacBook Pro M1/M2/M3 (16GB)
- Model: Qwen3-0.6B
- Input: Example prompt, output ~13-15 tokens/s on MPS

**Note:** Performance on NVIDIA/CUDA GPUs or Linux is not supported in this fork.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=jacko06v/nano-vllm-macos&type=Date)](https://www.star-history.com/#jacko06v/nano-vllm-macos&Date)
