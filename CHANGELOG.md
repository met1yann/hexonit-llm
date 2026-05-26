# Changelog

All notable changes to hexonit-llm are documented here.

Format: [Semantic Versioning](https://semver.org/)

---

## [0.1.0] — 2026-05-26

### Added
- `UltraInference.check()` — static pre-download VRAM compatibility check
- `UltraInference.can_run()` — instance method for custom models
- `UltraInference.benchmark()` — tokens/sec measurement utility
- `QuantizationAdvisor` — recommends optimal quantization before downloading
- Abstract `BaseEngine` interface for extensibility
- Full pytest test suite (hardware, quantization, orchestrator)
- GitHub Actions CI across Python 3.10-3.12 on Linux/Windows/macOS
- GitHub Actions publish workflow (tag-triggered)
- `LICENSE` file (MIT)

### Changed
- License: MIT (was AGPL-3.0)
- Build backend: hatchling (was setuptools)
- Lighter core dependencies (removed torch, transformers, sentencepiece)
- Version bumped to 0.1.0

### Fixed
- License inconsistency between pyproject.toml and README
- Missing LICENSE file
- Missing CHANGELOG.md

---

## [0.0.2] — 2026-05-26

### Added
- Universal OpenAI-compatible cloud draft provider
- `draft_base_url` parameter for any API endpoint

### Changed
- Email: metesezer54@gmail.com

---

## [0.0.1] — 2026-05-26

### Added
- Initial release
- `UltraInference` class with auto hardware routing
- vLLM and llama.cpp engine backends
- 31 model mappings for speculative decoding
- Hardware detection (OS, VRAM, RAM, CPU)
- Triple-mode operation: cloud, explicit pair, auto local