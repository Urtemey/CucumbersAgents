# Claude AI Instructions for CucumbersAgents

## Package Overview

This is **CucumbersAgents** - a LangChain-based multi-agent system for processing citizen complaints. This package can be extracted as a standalone repository and used independently from the main application.

## Tech Stack

- **Framework**: LangChain
- **LLM**: Ollama with qwen3-vl:4b
- **ASR**: faster-whisper
- **Language**: Python 3.10+
- **Build**: Poetry / pyproject.toml

## Package Structure (LangChain Template Style)

```
CucumbersAgents/
├── complaintagents/        # 🤖 Core agent module
│   ├── __init__.py         # Agent exports
│   ├── base.py             # BaseAgent & AgentResult
│   ├── config.py           # Configuration
│   ├── models.py           # Domain models
│   ├── llm_provider.py     # Ollama integration
│   ├── tools.py            # LangChain tools
│   ├── transcription.py    # Whisper ASR agent
│   ├── analyzer.py         # NLU analysis agent
│   ├── summarizer.py       # Text summarization agent
│   ├── router.py           # Routing agent
│   ├── antifraud.py        # Fraud detection agent
│   ├── orchestrator.py     # Pipeline coordinator
│   └── claude.md           # Agent-specific instructions
│
├── tests/                  # 🧪 Test suite
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_tools.py       # Tool tests
│   ├── test_models.py      # Model tests
│   └── test_agents.py      # Agent tests
│
├── __init__.py             # Package re-exports
├── pyproject.toml          # Package configuration
├── README.md               # Documentation
└── claude.md               # This file
```

## Quick Start

### Installation
```bash
cd CucumbersAgents
pip install -e .
# or
poetry install
```

### Basic Usage
```python
from CucumbersAgents import AgentOrchestrator

orchestrator = AgentOrchestrator()
await orchestrator.initialize()

# Process text
result = await orchestrator.process_text(text="Жалоба на качество обслуживания...")

# Process audio
result = await orchestrator.process_audio(audio_path=Path("complaint.wav"))
```

### Using Individual Agents
```python
from CucumbersAgents import AnalyzerAgent, ComplaintMetrics

analyzer = AnalyzerAgent()
await analyzer.initialize()
result = await analyzer.process("Врач был груб...")

metrics: ComplaintMetrics = result.data
print(f"Category: {metrics.category}, Urgency: {metrics.urgency}")
```

### Using Tools
```python
from CucumbersAgents import get_analysis_tools, classify_category

# Get all analysis tools
tools = get_analysis_tools()

# Use individual tool
category = classify_category.invoke("Жалоба на врача в поликлинике")
# Returns: "medical"
```

## Agent Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
┌───▼───┐           ┌───────▼───────┐       ┌───────▼───────┐
│Whisper│           │  AnalyzerAgent │       │SummarizerAgent│
│  ASR  │──text────▶│   (LLM NLU)   │       │  (LLM Text)   │
└───────┘           └───────┬───────┘       └───────┬───────┘
                            │                       │
                    ┌───────▼───────┐       ┌───────▼───────┐
                    │AntifraudAgent │       │  RouterAgent  │
                    │   (Rules)     │       │   (Rules)     │
                    └───────────────┘       └───────────────┘
```

## LangChain Features Used

| Feature | Usage |
|---------|-------|
| **Chains** | `prompt \| llm \| parser` in analyzer/summarizer |
| **Tools** | `@tool` decorated functions for analysis |
| **Structured Output** | `PydanticOutputParser` for JSON responses |
| **Memory** | `ConversationBufferMemory` in orchestrator |
| **Chat Models** | `ChatOllama` with JSON mode |

## Configuration

### Environment Variables
```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3-vl:4b
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
RATE_LIMIT_PER_HOUR=5
DEBUG=true
```

### Programmatic Config
```python
from CucumbersAgents import AgentSystemConfig, set_agent_config

config = AgentSystemConfig()
config.ollama.model = "llama3:8b"
config.whisper.model_size = "small"
set_agent_config(config)
```

## Key Models

### Input/Output
- **ProcessingResult** - Final result with all artifacts
- **ComplaintMetrics** - Analysis metrics (category, sentiment, urgency)
- **TextArtifacts** - Three text versions (original, normalized, neutral)
- **RoutingDecision** - Department and escalation info
- **FraudScore** - Spam detection results

### Enums
- **ComplaintCategory** - medical, school, housing, service, etc.
- **SentimentLevel** - positive, neutral, negative, very_negative
- **UrgencyLevel** - low, medium, high, critical
- **VerificationLevel** - anonymous, identified, employee

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agents.py::TestRouterAgent -v

# With coverage
pytest tests/ --cov=complaintagents --cov-report=html
```

## Extending

### Add New Agent
1. Create `complaintagents/my_agent.py`
2. Inherit `BaseAgent`
3. Implement `initialize()`, `process()`, `health_check()`
4. Export in `complaintagents/__init__.py`
5. Add tests in `tests/test_agents.py`

### Add New Tool
1. Add `@tool` function in `complaintagents/tools.py`
2. Register in `get_*_tools()` function
3. Add tests in `tests/test_tools.py`

## Dependencies

Core:
- langchain >= 0.1.0
- langchain-community >= 0.0.10
- faster-whisper >= 0.10.0
- pydantic >= 2.0.0

Dev:
- pytest >= 7.0.0
- pytest-asyncio >= 0.21.0

## Model: qwen3-vl:4b

All LLM agents use the same model for consistency:
- `AnalyzerAgent.MODEL_NAME = "qwen3-vl:4b"`
- `SummarizerAgent.MODEL_NAME = "qwen3-vl:4b"`
- `OllamaProvider.DEFAULT_MODEL = "qwen3-vl:4b"`

To change globally:
```python
from CucumbersAgents import AgentOrchestrator
orchestrator = AgentOrchestrator(model_name="llama3:8b")
```

