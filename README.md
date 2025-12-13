# 🤖 Multi-Agent System

Мультиагентная система на базе **LangChain + Ollama** для обработки жалоб.

> Может быть выделена в отдельный репозиторий

## 🎯 Модель

Все текстовые агенты используют единую модель: **qwen3-vl:8b**

## 📁 Структура

```
agents/
├── __init__.py       # Экспорты
├── config.py         # Независимая конфигурация
├── models.py         # Доменные модели
├── base.py           # Базовый агент
├── llm_provider.py   # Ollama провайдер
├── tools.py          # LangChain инструменты
├── transcription.py  # ASR (Whisper)
├── analyzer.py       # NLU анализ (qwen3-vl:8b)
├── summarizer.py     # Суммаризация (qwen3-vl:8b)
├── router.py         # Маршрутизация
├── antifraud.py      # Антифрод
└── orchestrator.py   # Координация
```

## 🚀 Использование

```python
from agents import AgentOrchestrator

# Создание оркестратора
orchestrator = AgentOrchestrator()
await orchestrator.initialize()

# Обработка текста
result = await orchestrator.process_text(
    text="Жалоба на качество обслуживания...",
    intake_channel=IntakeChannel.WEB_FORM,
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

## 🔧 Конфигурация

```python
from agents.config import AgentSystemConfig, set_agent_config

config = AgentSystemConfig(
    ollama=OllamaConfig(
        base_url="http://localhost:11434",
        model="qwen3-vl:8b",
    ),
    whisper=WhisperConfig(
        model_size="base",
        device="cpu",
    ),
)

set_agent_config(config)
```

Или через переменные окружения:

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=qwen3-vl:8b
WHISPER_MODEL=base
```

## 🛠️ LangChain Features

### Tools
```python
from agents.tools import get_analysis_tools

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
from agents.llm_provider import get_ollama_provider

provider = get_ollama_provider()
llm = provider.get_llm(temperature=0.3)
chat = provider.get_chat_model(format="json")
```

## 📦 Зависимости

```txt
langchain>=0.1.16
langchain-core>=0.1.40
langchain-community>=0.0.29
faster-whisper>=1.0.0
```

## 🔒 Независимость

Модуль `agents/` не зависит от `app/`:
- Собственная конфигурация (`agents/config.py`)
- Собственные модели (`agents/models.py`)
- Можно использовать отдельно

