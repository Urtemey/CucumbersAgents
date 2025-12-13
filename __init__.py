"""
Multi-Agent System for Complaint Processing

Мультиагентная система на базе LangChain + Ollama.
Может быть выделена в отдельный репозиторий.

Агенты:
- TranscriptionAgent: ASR через Whisper (локально)
- AnalyzerAgent: NLU анализ через LLM (qwen3-vl:8b)
- SummarizerAgent: Суммаризация через LLM
- RouterAgent: Маршрутизация (правила)
- AntifraudAgent: Скоринг достоверности
- AgentOrchestrator: Координация всех агентов

LangChain Features:
- Structured Output (Pydantic парсеры)
- Tools (инструменты анализа)
- Chains (цепочки обработки)
- Memory (память сессии)
"""

__version__ = "1.0.0"

from agents.base import BaseAgent, AgentResult
from agents.llm_provider import OllamaProvider, get_ollama_provider
from agents.tools import (
    get_analysis_tools,
    get_search_tools,
    get_all_tools,
    extract_entities,
    classify_category,
    analyze_sentiment,
    check_toxicity,
)
from agents.transcription import TranscriptionAgent
from agents.analyzer import AnalyzerAgent
from agents.summarizer import SummarizerAgent
from agents.router import RouterAgent
from agents.antifraud import AntifraudAgent
from agents.orchestrator import AgentOrchestrator

__all__ = [
    # Version
    "__version__",
    # Base
    "BaseAgent",
    "AgentResult",
    # Provider
    "OllamaProvider",
    "get_ollama_provider",
    # Tools
    "get_analysis_tools",
    "get_search_tools", 
    "get_all_tools",
    "extract_entities",
    "classify_category",
    "analyze_sentiment",
    "check_toxicity",
    # Agents
    "TranscriptionAgent",
    "AnalyzerAgent", 
    "SummarizerAgent",
    "RouterAgent",
    "AntifraudAgent",
    "AgentOrchestrator",
]

