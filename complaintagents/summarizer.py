"""
Summarizer Agent - Суммаризация с LangChain.

Использует:
- qwen3-vl:8b через Ollama
- Chain для последовательной обработки
- Structured Output для трёх артефактов
"""

import time
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from .base import BaseAgent, AgentResult
from .llm_provider import get_ollama_provider
from .models import TextArtifacts


class SummarizationOutput(BaseModel):
    """Структурированный вывод суммаризации."""
    normalized: str = Field(description="Нормализованная версия с исправленной пунктуацией, без слов-паразитов")
    neutral: str = Field(description="Нейтральная управленческая версия без эмоций")
    summary: str = Field(description="Краткое резюме в 1-2 предложениях")


class SummarizerAgent(BaseAgent):
    """
    Агент суммаризации с LangChain.
    
    Использует модель: qwen3-vl:8b
    
    Создает три версии текста:
    1. Нормализованная транскрипция
    2. Нейтральная управленческая выжимка
    3. Краткое резюме
    """
    
    # Модель для суммаризации
    MODEL_NAME = "qwen3-vl:8b"
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.3,
    ):
        super().__init__("SummarizerAgent")
        self.model_name = model_name or self.MODEL_NAME
        self.temperature = temperature
        
        self._provider = None
        self._chain = None
    
    async def initialize(self) -> bool:
        """Инициализация LangChain компонентов."""
        try:
            self.log_start(f"Initializing LangChain Summarizer with {self.model_name}")
            
            self._provider = get_ollama_provider()
            
            self.llm = self._provider.get_chat_model(
                model=self.model_name,
                temperature=self.temperature,
                format="json",
            )
            
            self.output_parser = PydanticOutputParser(pydantic_object=SummarizationOutput)
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", "{text}"),
            ])
            
            self._chain = self.prompt | self.llm | self.output_parser
            
            self.logger.info(f"Summarizer initialized with {self.model_name}")
            return True
            
        except Exception as e:
            self.log_error("initialize", e)
            return False
    
    def _get_system_prompt(self) -> str:
        """Системный промпт для суммаризации."""
        return """Ты - эксперт по обработке обращений граждан. Обработай текст жалобы и создай три версии.

Правила обработки:

1. NORMALIZED (нормализованная версия):
   - Исправь орфографию и пунктуацию
   - Убери слова-паразиты (ну, вот, как бы, типа)
   - Убери повторы и заминки речи
   - СОХРАНИ все факты и смысл дословно
   - Это должна быть читаемая версия оригинала

2. NEUTRAL (нейтральная версия):
   - Перепиши формальным деловым языком
   - Убери все эмоции и оценочные суждения
   - Замени оскорбления на нейтральные формулировки
   - Замени "хам", "грубиян" -> "проявил некорректное поведение"
   - Замени "ужасно", "безобразие" -> "выявлены недостатки"
   - СОХРАНИ все факты и важную информацию

3. SUMMARY (краткое резюме):
   - 1-2 предложения для руководителя
   - Суть проблемы
   - Категория обращения
   - Требуемые действия

Отвечай ТОЛЬКО валидным JSON:
{
  "normalized": "нормализованный текст",
  "neutral": "нейтральный текст", 
  "summary": "краткое резюме"
}"""
    
    async def process(
        self,
        original_text: str,
        language: str = "ru",
        audio_duration: float = None,
        transcription_time: float = None,
    ) -> AgentResult[TextArtifacts]:
        """
        Создать три артефакта текста.
        """
        start_time = time.time()
        
        try:
            await self.ensure_initialized()
            
            self.log_start(f"Summarizing text ({len(original_text)} chars)")
            
            if not original_text or len(original_text.strip()) < 10:
                artifacts = TextArtifacts(
                    original=original_text,
                    normalized=original_text,
                    neutral=original_text,
                    language=language,
                    audio_duration=audio_duration,
                    transcription_time=transcription_time,
                )
                return AgentResult.ok(data=artifacts, processing_time=0.0)
            
            # Вызов chain
            try:
                output: SummarizationOutput = await self._chain.ainvoke({"text": original_text})
                
                artifacts = TextArtifacts(
                    original=original_text,
                    normalized=output.normalized,
                    neutral=output.neutral,
                    language=language,
                    audio_duration=audio_duration,
                    transcription_time=transcription_time,
                )
                
            except Exception as parse_error:
                self.logger.warning(f"Summarization parsing failed: {parse_error}")
                artifacts = TextArtifacts(
                    original=original_text,
                    normalized=original_text,
                    neutral=original_text,
                    language=language,
                    audio_duration=audio_duration,
                    transcription_time=transcription_time,
                )
            
            processing_time = time.time() - start_time
            self.log_complete("summarization", processing_time)
            
            result = AgentResult.ok(
                data=artifacts,
                processing_time=processing_time,
            )
            
            if len(artifacts.neutral) < len(original_text) * 0.3:
                result.add_warning("Neutral version significantly shorter than original")
            
            return result
            
        except Exception as e:
            self.log_error("summarization", e)
            
            artifacts = TextArtifacts(
                original=original_text,
                normalized=original_text,
                neutral=original_text,
                language=language,
            )
            return AgentResult.ok(data=artifacts, processing_time=time.time() - start_time)
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния агента."""
        return {
            "name": self.name,
            "status": "ok" if self._chain else "not_initialized",
            "model": self.model_name,
            "initialized": self._initialized,
        }

