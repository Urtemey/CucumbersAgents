"""
CucumbersAgents - Multi-Agent System for Complaint Processing

Мультиагентная система на базе LangChain + Ollama.
Может быть выделена в отдельный репозиторий.

Структура:
- complaintagents/  - Основные агенты обработки жалоб
- tests/            - Тесты для агентов

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

Usage:
    from CucumbersAgents import AgentOrchestrator
    
    orchestrator = AgentOrchestrator()
    await orchestrator.initialize()
    result = await orchestrator.process_text(text="Жалоба...")
"""

__version__ = "1.0.0"

# Re-export from complaintagents submodule
from complaintagents import (
    # Base
    BaseAgent,
    AgentResult,
    # Agents
    TranscriptionAgent,
    AnalyzerAgent,
    SummarizerAgent,
    RouterAgent,
    AntifraudAgent,
    AgentOrchestrator,
)

# Config and providers - import directly for backward compatibility
from complaintagents.config import (
    AgentSystemConfig,
    OllamaConfig,
    WhisperConfig,
    AntifraudConfig,
    get_agent_config,
    set_agent_config,
)

from complaintagents.llm_provider import (
    OllamaProvider,
    get_ollama_provider,
)

from complaintagents.tools import (
    get_analysis_tools,
    get_search_tools,
    get_all_tools,
    extract_entities,
    classify_category,
    analyze_sentiment,
    check_toxicity,
    calculate_urgency,
)

from complaintagents.models import (
    # Enums
    ComplaintCategory,
    SentimentLevel,
    UrgencyLevel,
    VerificationLevel,
    IntakeChannel,
    CasePriority,
    # Data models
    TextArtifacts,
    ComplaintMetrics,
    TranscriptionData,
    RoutingDecision,
    FraudScore,
    ProcessingResult,
)

__all__ = [
    # Version
    "__version__",
    # Base
    "BaseAgent",
    "AgentResult",
    # Config
    "AgentSystemConfig",
    "OllamaConfig",
    "WhisperConfig",
    "AntifraudConfig",
    "get_agent_config",
    "set_agent_config",
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
    "calculate_urgency",
    # Enums
    "ComplaintCategory",
    "SentimentLevel",
    "UrgencyLevel",
    "VerificationLevel",
    "IntakeChannel",
    "CasePriority",
    # Data models
    "TextArtifacts",
    "ComplaintMetrics",
    "TranscriptionData",
    "RoutingDecision",
    "FraudScore",
    "ProcessingResult",
    # Agents
    "TranscriptionAgent",
    "AnalyzerAgent",
    "SummarizerAgent",
    "RouterAgent",
    "AntifraudAgent",
    "AgentOrchestrator",
]
