"""Tests for hardware detection."""
from hexonit_llm.utils.hardware_detector import detect_hardware, select_engine, can_use_vllm, can_use_llamacpp


def test_detect_hardware_returns_hardware_profile():
    result = detect_hardware()
    assert result.os_type in ("Windows", "Linux", "Darwin")
    assert result.total_ram_gb > 0
    assert result.cpu_count > 0
    assert isinstance(result.cuda_available, bool)


def test_os_is_detected():
    result = detect_hardware()
    assert isinstance(result.os_type, str)
    assert len(result.os_type) > 0


def test_vram_is_non_negative():
    result = detect_hardware()
    assert result.total_vram_gb >= 0


def test_select_engine_returns_string():
    engine = select_engine()
    assert engine in ("vllm", "llamacpp")


def test_can_use_vllm_linux_low_vram(mock_hw_8gb):
    assert can_use_vllm(mock_hw_8gb) is False


def test_can_use_vllm_linux_high_vram(mock_hw_24gb):
    assert can_use_vllm(mock_hw_24gb) is True


def test_can_use_vllm_windows_no_gpu(mock_hw_no_gpu):
    assert can_use_vllm(mock_hw_no_gpu) is False


def test_can_use_llamacpp_windows(mock_hw_no_gpu):
    assert can_use_llamacpp(mock_hw_no_gpu) is True


def test_can_use_llamacpp_linux_low_vram(mock_hw_8gb):
    assert can_use_llamacpp(mock_hw_8gb) is True


def test_can_use_llamacpp_linux_high_vram(mock_hw_24gb):
    assert can_use_llamacpp(mock_hw_24gb) is False