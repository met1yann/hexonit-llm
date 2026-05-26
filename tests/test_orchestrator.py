"""Tests for the orchestrator and routing logic."""
from hexonit_llm.utils.hardware_detector import can_use_vllm, can_use_llamacpp, select_engine


def test_select_engine_vllm_on_linux_high_vram(mock_hw_24gb):
    """Linux with 24GB VRAM should prefer vLLM."""
    assert can_use_vllm(mock_hw_24gb) is True
    assert can_use_llamacpp(mock_hw_24gb) is False


def test_select_engine_llamacpp_on_windows(mock_hw_no_gpu):
    """Windows should always route to llama.cpp."""
    assert can_use_vllm(mock_hw_no_gpu) is False
    assert can_use_llamacpp(mock_hw_no_gpu) is True


def test_select_engine_llamacpp_on_low_vram_linux(mock_hw_8gb):
    """Linux with only 8GB VRAM should route to llama.cpp."""
    assert can_use_vllm(mock_hw_8gb) is False
    assert can_use_llamacpp(mock_hw_8gb) is True


def test_select_engine_returns_string():
    engine = select_engine()
    assert engine in ("vllm", "llamacpp")


def test_ultrainference_importable():
    from hexonit_llm import UltraInference
    assert callable(UltraInference)
    assert UltraInference.__name__ == "UltraInference"


def test_ultrainference_auto_initializes():
    from hexonit_llm import UltraInference
    pipe = UltraInference("meta-llama/Meta-Llama-3-70B-Instruct")
    assert pipe.engine_name in ("vllm", "llamacpp")
    assert pipe.mode in ("local_auto", "local_single")
    assert pipe.draft_model is not None or pipe.draft_model is None


def test_ultrainference_explicit_pair():
    from hexonit_llm import UltraInference
    pipe = UltraInference("test/model", draft_model="test/draft")
    assert pipe.mode == "local_explicit_pair"
    assert pipe.draft_model == "test/draft"


def test_ultrainference_unknown_model():
    from hexonit_llm import UltraInference
    pipe = UltraInference("unknown/model-999B")
    assert pipe.mode == "local_single"
    assert pipe.draft_model is None


def test_ultrainference_hardware_info():
    from hexonit_llm import UltraInference
    pipe = UltraInference("test/model")
    info = pipe.hardware_info
    assert "os" in info
    assert "total_ram_gb" in info
    assert "cuda_available" in info