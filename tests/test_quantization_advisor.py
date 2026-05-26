"""Tests for the quantization advisor."""
from hexonit_llm.utils.quantization_advisor import advise, QuantizationAdvice


def test_advise_large_model_no_vram():
    advice = advise("meta-llama/Meta-Llama-3-70B-Instruct", available_vram_gb=4.0)
    assert advice.can_run is False
    assert advice.deficit_gb > 0
    assert advice.fallback_suggestion is not None


def test_advise_small_model_ample_vram():
    advice = advise("meta-llama/Meta-Llama-3-8B-Instruct", available_vram_gb=24.0)
    assert advice.can_run is True
    assert advice.surplus_gb > 0
    assert advice.recommended_quant in ("fp16", "q8_0", "q6_k", "q5_k_m", "q4_k_m")


def test_advise_returns_quantization_advice():
    advice = advise("meta-llama/Meta-Llama-3-8B-Instruct", available_vram_gb=8.0)
    assert isinstance(advice, QuantizationAdvice)


def test_advise_unknown_model_heuristic():
    advice = advise("some-org/some-model-13b", available_vram_gb=16.0)
    assert isinstance(advice, QuantizationAdvice)


def test_advise_no_gpu():
    advice = advise("meta-llama/Meta-Llama-3-70B-Instruct", available_vram_gb=0.0)
    assert advice.can_run is False
    assert "No GPU" in advice.fallback_suggestion or "GPU" in advice.explanation


def test_advice_str_output_can_run():
    advice = advise("meta-llama/Meta-Llama-3-8B-Instruct", available_vram_gb=16.0)
    output = str(advice)
    assert "✅" in output


def test_advice_str_output_cannot_run():
    advice = advise("meta-llama/Meta-Llama-3-70B-Instruct", available_vram_gb=2.0)
    output = str(advice)
    assert "❌" in output


def test_advise_default_model():
    advice = advise("unknown/model", available_vram_gb=16.0)
    assert isinstance(advice, QuantizationAdvice)
    assert advice.can_run is True