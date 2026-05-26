# hexonit-llm 🚀

<div align="center">

**Ultra-fast local LLM inference — zero config, one import, maximum tokens/sec.**

[![CI](https://github.com/met1yann/hexonit-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/met1yann/hexonit-llm/actions)
[![PyPI version](https://badge.fury.io/py/hexonit-llm.svg)](https://pypi.org/project/hexonit-llm/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/hexonit-llm)](https://pepy.tech/project/hexonit-llm)

</div>

---

## 🔍 Can I Run This Model?

Check **before downloading** whether your hardware supports a model:

```python
from hexonit_llm import UltraInference

# Static check — no model loading required
advice = UltraInference.check("meta-llama/Meta-Llama-3-70B-Instruct")
print(advice)
# ✅ Can run | Recommended: Q4_K_M | Est. VRAM: 38.5GB / 80.0GB available (52% headroom)
#    70B parameter model at Q4_K_M uses ~38.5GB including KV cache overhead.

# Or if you don't have enough VRAM:
# ❌ Cannot run | Need 38.5GB, have 8.0GB (deficit: 30.5GB)
#    💡 Try instead: meta-llama/Meta-Llama-3-8B-Instruct (8B) fits at Q4_K_M
```

---

## Philosophy

> **"One import. That's all."**

`hexonit-llm` is an intelligent orchestrator that:

1. **Inspects your hardware** — OS, VRAM, system RAM, CPU
2. **Selects the fastest engine** — vLLM (Linux, ≥16GB VRAM) or llama.cpp (Windows/macOS/Linux)
3. **Enables speculative decoding** — automatically downloads the matching draft model
4. **Delivers maximum tokens/sec** — hardcoded, battle-tested optimisation presets

All with **zero configuration**.

---

## Quick Start

### Installation

```bash
pip install hexonit-llm        # core dependencies only
pip install hexonit-llm[vllm]      # + vLLM (Linux only)
pip install hexonit-llm[llamacpp]  # + llama.cpp (Windows/macOS/Linux)
pip install hexonit-llm[cloud]     # + httpx for cloud draft
```

### Usage

```python
from hexonit_llm import UltraInference

# That's it. One line.
pipe = UltraInference("meta-llama/Meta-Llama-3-70B-Instruct")

# Generate text
response = pipe.generate("What is the meaning of life?")
print(response)

# Batch generation
responses = pipe.generate_batch([
    "Tell me a joke",
    "What is 2+2?",
])

# Chat interface
reply = pipe.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
])
```

### Check what's running

```python
pipe = UltraInference("meta-llama/Meta-Llama-3-70B-Instruct")
print(pipe.engine_name)     # "vllm" or "llamacpp"
print(pipe.draft_model)     # "meta-llama/Llama-3.2-3B-Instruct"
print(pipe.hardware_info)
```

---

## ⚡ Benchmarks

Run your own benchmark:

```python
pipe = UltraInference("meta-llama/Meta-Llama-3-8B-Instruct")
stats = pipe.benchmark(runs=10)
# 🔥 Benchmarking llamacpp with 10 runs...
#   Run 1/10: 47.3 tok/s
#   ...
# 📊 Results: 45.8 tok/s average (llamacpp)
```

> Community benchmark results welcome! Open a PR to add yours to [docs/benchmarks.md](docs/benchmarks.md).

---

## Supported Model Families

| Family | Target Model | Auto-selected Draft |
|--------|-------------|-------------------|
| **Meta LLaMA 3** | `Meta-Llama-3-70B-Instruct` | `Llama-3.2-3B-Instruct` |
| **Meta LLaMA 3** | `Meta-Llama-3-8B-Instruct` | `Llama-3.2-1B-Instruct` |
| **Qwen 2.5** | `Qwen2.5-72B-Instruct` | `Qwen2.5-1.5B-Instruct` |
| **Mixtral** | `Mixtral-8x22B-Instruct` | `Ministral-8B-Instruct` |
| **Gemma 2** | `gemma-2-27b-it` | `gemma-2-2b-it` |
| **DeepSeek** | `DeepSeek-V2.5` | `deepseek-llm-7b-chat` |
| **Phi-3** | `Phi-3-medium-4k-instruct` | `Phi-3-mini-4k-instruct` |
| *… and many more* | *See [model_mappings.py](hexonit_llm/config/model_mappings.py)* |

---

## Architecture

```
hexonit_llm/
├── __init__.py              # UltraInference – the public API
├── orchestrator.py          # The brain: hardware routing + engine factory
├── engines/
│   ├── base.py              # Abstract base engine
│   ├── vllm_engine.py       # vLLM backend (PagedAttention, FlashAttention-2)
│   └── llamacpp_engine.py   # llama.cpp backend (GGUF offloading)
├── config/
│   └── model_mappings.py    # 30+ target→draft model mappings
└── utils/
    ├── hardware_detector.py # OS, VRAM, RAM detection
    ├── model_mapper.py      # HF Hub download & caching
    └── quantization_advisor.py  # Pre-download VRAM analysis
```

### Routing Logic

```text
UltraInference(model)
    │
    ├── OS = Linux & VRAM ≥ 16GB  ──>  vLLM  (FlashAttention-2, PagedAttention)
    │
    └── OS = Windows / macOS
        or VRAM < 16GB           ──>  llama.cpp  (GGUF, GPU offloading)
```

Speculative decoding is **always enabled** when a matching draft model exists.

---

## 🆚 Compared to Alternatives

| Feature | hexonit-llm | Ollama | vLLM direct | llama.cpp direct |
|---------|------------|--------|-------------|-----------------|
| Zero config | ✅ | ✅ | ❌ | ❌ |
| Auto engine selection | ✅ | ❌ | ❌ | ❌ |
| Speculative decoding auto | ✅ | ❌ | Manual | ❌ |
| Pre-download VRAM check | ✅ | ❌ | ❌ | ❌ |
| Python-native API | ✅ | Via REST | ✅ | Via binding |
| Windows support | ✅ | ✅ | ❌ | ✅ |
| Benchmark built-in | ✅ | ❌ | ❌ | ❌ |

---

## Performance

The engines ship with **hardcoded, max-throughput presets**:

| Setting | vLLM | llama.cpp |
|---------|------|-----------|
| GPU Memory Utilisation | 95% | All layers (-1) |
| Batch Size | 256 sequences | 2048 tokens |
| Flash Attention | ✅ v2 | ✅ |
| Prefix Caching | ✅ | N/A |
| CUDA Graphs | ✅ | N/A |

---

## License

MIT © 2026 [Hexonithy Studios](https://github.com/met1yann/hexonit-llm)

---

## Contributing

PRs welcome! Please ensure your code passes our checks:

```bash
pip install -e ".[dev]"
ruff check .
mypy hexonit_llm
pytest tests/