"""
Complaint Processing Agents - LangChain агенты для обработки жалоб.

Этот модуль содержит специализированных агентов для обработки жалоб:
- TranscriptionAgent: ASR через faster-whisper
- AnalyzerAgent: NLU анализ через LLM (qwen3-vl:8b)
- SummarizerAgent: Суммаризация и нормализация текста
- RouterAgent: Маршрутизация по отделам
- AntifraudAgent: Скоринг достоверности и защита от спама
- AgentOrchestrator: Координация всех агентов

Использует LangChain Features:
- Structured Output (Pydantic парсеры)
- Tools (инструменты анализа)
- Chains (цепочки обработки)
- Memory (память сессии)
"""

from .base import BaseAgent, AgentResult
from .transcription import TranscriptionAgent
from .analyzer import AnalyzerAgent
from .summarizer import SummarizerAgent
from .router import RouterAgent
from .antifraud import AntifraudAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    # Base
    "BaseAgent",
    "AgentResult",
    # Agents
    "TranscriptionAgent",
    "AnalyzerAgent",
    "SummarizerAgent",
    "RouterAgent",
    "AntifraudAgent",
    "AgentOrchestrator",
]

