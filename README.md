# 🤖 CucumbersAgents

Мультиагентная система на базе **LangChain + Ollama** для обработки жалоб.

> Следует шаблону [LangChain Templates](https://github.com/langchain-ai/langchain/tree/v0.2/templates/gemini-functions-agent)

## 🎯 Модель

Все текстовые агенты используют единую модель: **qwen3-vl:4b**

## 📁 Структура

```
CucumbersAgents/
├── complaintagents/       # 🤖 Core agents module
│   ├── __init__.py        # Agent exports
│   ├── base.py            # BaseAgent & AgentResult
│   ├── config.py          # Configuration
│   ├── models.py          # Domain models & enums
│   ├── llm_provider.py    # Ollama integration
│   ├── tools.py           # LangChain @tool functions
│   ├── transcription.py   # Whisper ASR agent
│   ├── analyzer.py        # NLU analysis agent (LLM)
│   ├── summarizer.py      # Text summarization agent (LLM)
│   ├── router.py          # Routing agent (rules)
│   ├── antifraud.py       # Fraud detection agent (rules)
│   ├── orchestrator.py    # Pipeline coordinator
│   └── claude.md          # Agent-specific instructions
│
├── tests/                 # 🧪 Test suite
│   ├── conftest.py        # Pytest fixtures
│   ├── test_tools.py      # Tool tests
│   ├── test_models.py     # Model tests
│   └── test_agents.py     # Agent tests
│
├── __init__.py            # Package re-exports
├── pyproject.toml         # Package configuration
├── README.md              # This file
└── claude.md              # AI instructions
```

## 🚀 Установка

```bash
cd CucumbersAgents
pip install -e .
# или
poetry install
```

## 🚀 Использование

```python
from CucumbersAgents import AgentOrchestrator

# Создание оркестратора
orchestrator = AgentOrchestrator()
await orchestrator.initialize()

# Обработка текста
result = await orchestrator.process_text(
    text="Жалоба на качество обслуживания...",
)

# Обработка аудио
result = await orchestrator.process_audio(
    audio_path=Path("complaint.wav"),
)

# Результат
if result.success:
    data = result.data
    print(f"Category: {data.metrics.category}")
    print(f"Sentiment: {data.metrics.sentiment}")
    print(f"Neutral text: {data.text_artifacts.neutral}")
```

## 🛠️ Отдельные агенты

```python
from CucumbersAgents import AnalyzerAgent, RouterAgent

# Анализатор
analyzer = AnalyzerAgent()
await analyzer.initialize()
result = await analyzer.process("Врач был груб...")

# Маршрутизатор
router = RouterAgent()
await router.initialize()
routing = await router.process(result.data)
```

## 🔧 Конфигурация

```python
from CucumbersAgents import AgentSystemConfig, set_agent_config, OllamaConfig

config = AgentSystemConfig()
config.ollama.model = "llama3:8b"
set_agent_config(config)
```

Или через переменные окружения:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3-vl:4b
WHISPER_MODEL=base
```

## 🛠️ LangChain Features

### Tools
```python
from CucumbersAgents import get_analysis_tools

tools = get_analysis_tools()
# [extract_entities, classify_category, analyze_sentiment, check_toxicity, calculate_urgency]
```

### Structured Output
```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
chain = prompt | llm | parser
```

### Ollama Provider
```python
from CucumbersAgents import get_ollama_provider

provider = get_ollama_provider()
llm = provider.get_llm(temperature=0.3)
chat = provider.get_chat_model(format="json")
```

## 🧪 Тестирование

```bash
pytest tests/ -v
pytest tests/ --cov=complaintagents
```

## 📦 Зависимости

```txt
langchain>=0.1.0
langchain-core>=0.1.0
langchain-community>=0.0.10
faster-whisper>=0.10.0
pydantic>=2.0.0
```

## 🔒 Независимость

Модуль может быть выделен в отдельный репозиторий:
- Собственная конфигурация
- Собственные модели
- Собственные тесты
- pyproject.toml для установки

## 📄 Лицензия

MIT
