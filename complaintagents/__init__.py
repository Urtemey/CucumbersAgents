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

from complaintagents.base import BaseAgent, AgentResult
from complaintagents.transcription import TranscriptionAgent
from complaintagents.analyzer import AnalyzerAgent
from complaintagents.summarizer import SummarizerAgent
from complaintagents.router import RouterAgent
from complaintagents.antifraud import AntifraudAgent
from complaintagents.orchestrator import AgentOrchestrator

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

