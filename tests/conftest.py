"""
Pytest configuration and fixtures for agent tests.

Конфигурация и фикстуры для тестирования агентов.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_complaint_text():
    """Пример текста жалобы для тестирования."""
    return """
    Вчера, 15 января, я пришёл в поликлинику №5 на приём к врачу Петрову. 
    Ждал в очереди 2 часа! Это безобразие. Медсестра была грубой и хамила. 
    Кабинет 215 был грязный. Требую разобраться в ситуации.
    """


@pytest.fixture
def sample_short_text():
    """Короткий текст для тестирования edge cases."""
    return "Плохо."


@pytest.fixture
def sample_toxic_text():
    """Токсичный текст для тестирования антифрод."""
    return """
    Этот идиот врач Сидоров полный дурак! Я его убью если он ещё раз 
    так со мной обойдётся. Требую компенсацию! Буду жаловаться везде!
    """


@pytest.fixture
def sample_positive_text():
    """Положительный текст для тестирования."""
    return """
    Хочу выразить благодарность доктору Ивановой. Она отлично помогла 
    и быстро поставила диагноз. Спасибо!
    """


@pytest.fixture
def mock_ollama_provider():
    """Mock для OllamaProvider."""
    with patch('complaintagents.llm_provider.OllamaProvider') as MockProvider:
        mock_instance = MagicMock()
        mock_instance.get_chat_model.return_value = MagicMock()
        mock_instance.get_llm.return_value = MagicMock()
        mock_instance.check_health = AsyncMock(return_value={"status": "ok"})
        MockProvider.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_whisper_model():
    """Mock для WhisperModel."""
    with patch('faster_whisper.WhisperModel') as MockWhisper:
        mock_model = MagicMock()
        
        # Mock transcription result
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.text = "Тестовая транскрипция"
        mock_segment.words = []
        
        mock_info = MagicMock()
        mock_info.language = "ru"
        mock_info.duration = 5.0
        mock_info.language_probability = 0.95
        
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        MockWhisper.return_value = mock_model
        
        yield mock_model


@pytest.fixture
def temp_audio_file():
    """Создать временный аудио файл для тестирования."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Write minimal WAV header (empty audio)
        f.write(b'RIFF')
        f.write((36).to_bytes(4, 'little'))  # File size - 8
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write((16).to_bytes(4, 'little'))  # Subchunk1 size
        f.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
        f.write((1).to_bytes(2, 'little'))   # Num channels
        f.write((16000).to_bytes(4, 'little'))  # Sample rate
        f.write((32000).to_bytes(4, 'little'))  # Byte rate
        f.write((2).to_bytes(2, 'little'))   # Block align
        f.write((16).to_bytes(2, 'little'))  # Bits per sample
        f.write(b'data')
        f.write((0).to_bytes(4, 'little'))   # Data size
        
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        os.unlink(temp_path)


@pytest.fixture
def reset_config():
    """Сбросить глобальную конфигурацию перед тестом."""
    from ..complaintagents.config import reset_agent_config
    from ..complaintagents.llm_provider import reset_ollama_provider
    
    reset_agent_config()
    reset_ollama_provider()
    yield
    reset_agent_config()
    reset_ollama_provider()


@pytest.fixture
def test_config():
    """Тестовая конфигурация."""
    from ..complaintagents.config import AgentSystemConfig, OllamaConfig, WhisperConfig, AntifraudConfig
    
    return AgentSystemConfig(
        ollama=OllamaConfig(
            base_url="http://localhost:11434",
            model="qwen3-vl:8b",
            temperature=0.3,
            timeout=60,
        ),
        whisper=WhisperConfig(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="ru",
        ),
        antifraud=AntifraudConfig(
            rate_limit_per_hour=10,
            rate_limit_per_day=50,
            min_text_length=10,
        ),
        debug=True,
        verbose=True,
    )

