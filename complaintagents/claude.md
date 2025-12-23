# Claude AI Instructions for ComplaintAgents

## Module Overview

This is **complaintagents** - the core LangChain agent module for processing citizen complaints. It contains specialized agents that work together to analyze, categorize, and route complaints.

## Tech Stack

- **LLM Framework**: LangChain with Ollama
- **LLM Model**: qwen3-vl:4b (unified for all agents)
- **ASR**: faster-whisper (local)
- **Language**: Python 3.10+
- **Parsing**: Pydantic v1 (for LangChain compatibility)

## Module Structure

```
complaintagents/
├── __init__.py         # Package exports
├── base.py             # BaseAgent class & AgentResult
├── config.py           # Agent system configuration
├── models.py           # Domain models & enums
├── llm_provider.py     # OllamaProvider singleton
├── tools.py            # LangChain @tool functions
├── transcription.py    # TranscriptionAgent (Whisper ASR)
├── analyzer.py         # AnalyzerAgent (NLU via LLM)
├── summarizer.py       # SummarizerAgent (text normalization)
├── router.py           # RouterAgent (rule-based routing)
├── antifraud.py        # AntifraudAgent (spam detection)
└── orchestrator.py     # AgentOrchestrator (coordination)
```

## Agent Architecture

### Base Components

**BaseAgent** (`base.py`):
- Abstract base class for all agents
- Provides logging, initialization, memory management
- LangChain integration: tools, executor, memory

**AgentResult[T]** (`base.py`):
- Generic wrapper for agent outputs
- Includes: success, data, error, processing_time, warnings
- Factory methods: `AgentResult.ok()`, `AgentResult.fail()`

### LLM Agents (use qwen3-vl:4b)

**AnalyzerAgent** (`analyzer.py`):
- NLU analysis via LangChain chains
- Structured output with `AnalysisOutput` Pydantic model
- Extracts: category, sentiment, urgency, entities, toxicity
- Fallback to tools if parsing fails

**SummarizerAgent** (`summarizer.py`):
- Creates 3 text artifacts: original, normalized, neutral
- Uses JSON mode for structured output
- Cleans speech artifacts, normalizes punctuation

### Rule-Based Agents

**RouterAgent** (`router.py`):
- Maps categories to departments
- Escalation rules based on metrics
- Calculates SLA based on urgency
- Adds additional departments (legal, HR) if needed

**AntifraudAgent** (`antifraud.py`):
- Rate limiting per source fingerprint
- Content validation (length, patterns)
- Spam probability calculation
- Coordinated attack detection

### ASR Agent

**TranscriptionAgent** (`transcription.py`):
- Wrapper for faster-whisper
- Word-level timestamps
- Confidence scoring
- Custom vocabulary support

### Orchestrator

**AgentOrchestrator** (`orchestrator.py`):
- Coordinates all agents in pipeline
- Handles both audio and text input
- Maintains session memory
- Aggregates warnings from all agents

## LangChain Usage Patterns

### Structured Output Chain
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | llm | parser
result = await chain.ainvoke({"text": text})
```

### Tools
```python
from langchain.tools import tool

@tool
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from text."""
    ...
```

### LLM Provider
```python
from complaintagents.llm_provider import get_ollama_provider

provider = get_ollama_provider()
llm = provider.get_llm(temperature=0.3)
chat = provider.get_chat_model(format="json")
```

## Key Data Models

**ComplaintMetrics** (`models.py`):
- Core analysis result
- Contains: category, sentiment, urgency, entities, scores, flags

**ProcessingResult** (`models.py`):
- Final pipeline output
- Contains: text_artifacts, metrics, routing, fraud_score, timings

## Configuration

```python
from complaintagents.config import get_agent_config, AgentSystemConfig

config = get_agent_config()
# Access: config.ollama.model, config.whisper.model_size, etc.
```

Environment variables:
- `OLLAMA_BASE_URL` - Ollama server URL
- `LLM_MODEL` - LLM model name (default: qwen3-vl:4b)
- `WHISPER_MODEL` - Whisper model size (default: base)
- `RATE_LIMIT_PER_HOUR` - Antifraud rate limit

## Adding New Agent

1. Create `my_agent.py` in this folder
2. Inherit from `BaseAgent`
3. Implement:
   - `async initialize() -> bool`
   - `async process(*args) -> AgentResult[T]`
   - `async health_check() -> Dict`
4. Use `get_ollama_provider()` for LLM access
5. Register in orchestrator if needed

## Adding New Tool

1. Add to `tools.py`
2. Use `@tool` decorator from langchain
3. Add to registry function (`get_analysis_tools()` etc.)

## Error Handling

- All agents return `AgentResult` (never raise exceptions to caller)
- Use `AgentResult.fail(error)` for errors
- Add warnings via `result.add_warning(msg)`
- Fallback mechanisms in analyzer (tools fallback)

## Testing

```bash
# From CucumbersAgents directory
pytest tests/ -v

# Specific test file
pytest tests/test_agents.py -v

# With coverage
pytest tests/ --cov=complaintagents
```

## Import Patterns

```python
# From parent package
from CucumbersAgents import AgentOrchestrator, ComplaintMetrics

# Direct import
from complaintagents.orchestrator import AgentOrchestrator
from complaintagents.models import ComplaintMetrics
```

