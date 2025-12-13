"""
Agent System Configuration

Конфигурация мультиагентной системы.
Независима от основного приложения для возможности
выделения в отдельный репозиторий.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
import os


@dataclass
class OllamaConfig:
    """Конфигурация Ollama LLM."""
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-vl:8b"
    temperature: float = 0.3
    timeout: int = 120
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("LLM_MODEL", "qwen3-vl:8b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        )


@dataclass
class WhisperConfig:
    """Конфигурация Whisper ASR."""
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str = "ru"
    
    @classmethod
    def from_env(cls) -> "WhisperConfig":
        return cls(
            model_size=os.getenv("WHISPER_MODEL", "base"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            language=os.getenv("WHISPER_LANGUAGE", "ru"),
        )


@dataclass
class AntifraudConfig:
    """Конфигурация антифрод системы."""
    rate_limit_per_hour: int = 5
    rate_limit_per_day: int = 20
    min_text_length: int = 20
    suspicious_patterns: List[str] = field(default_factory=lambda: [
        r"^\d+$",
        r"^[a-zA-Z]+$",
        r"(.)\1{10,}",
    ])
    
    @classmethod
    def from_env(cls) -> "AntifraudConfig":
        return cls(
            rate_limit_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "5")),
            rate_limit_per_day=int(os.getenv("RATE_LIMIT_PER_DAY", "20")),
            min_text_length=int(os.getenv("MIN_COMPLAINT_LENGTH", "20")),
        )


@dataclass
class AgentSystemConfig:
    """Полная конфигурация агентной системы."""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    antifraud: AntifraudConfig = field(default_factory=AntifraudConfig)
    
    # Директории
    temp_dir: Path = Path("uploads/temp")
    audio_dir: Path = Path("uploads/audio")
    
    # Режим отладки
    debug: bool = False
    verbose: bool = False
    
    @classmethod
    def from_env(cls) -> "AgentSystemConfig":
        return cls(
            ollama=OllamaConfig.from_env(),
            whisper=WhisperConfig.from_env(),
            antifraud=AntifraudConfig.from_env(),
            temp_dir=Path(os.getenv("TEMP_DIR", "uploads/temp")),
            audio_dir=Path(os.getenv("AUDIO_DIR", "uploads/audio")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
        )


# Глобальная конфигурация (lazy load)
_config: Optional[AgentSystemConfig] = None


def get_agent_config() -> AgentSystemConfig:
    """Получить конфигурацию агентной системы."""
    global _config
    if _config is None:
        _config = AgentSystemConfig.from_env()
    return _config


def set_agent_config(config: AgentSystemConfig):
    """Установить конфигурацию агентной системы."""
    global _config
    _config = config

