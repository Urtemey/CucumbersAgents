"""
Analyzer Agent - NLU анализ с использованием LangChain.

Использует:
- qwen3-vl:8b через Ollama
- Structured Output для извлечения данных
- Tools для анализа
"""

import time
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from complaintagents.base import BaseAgent, AgentResult
from complaintagents.llm_provider import get_ollama_provider
from complaintagents.tools import get_analysis_tools
from complaintagents.models import (
    ComplaintMetrics,
    ComplaintCategory,
    SentimentLevel,
    UrgencyLevel,
)


class AnalysisOutput(BaseModel):
    """Структурированный вывод анализа."""
    category: str = Field(description="Категория: medical, school, housing, service, hotel, retail, government, other")
    sentiment: str = Field(description="Тональность: positive, neutral, negative, very_negative")
    urgency: str = Field(description="Срочность: low, medium, high, critical")
    toxicity_score: float = Field(description="Токсичность от 0.0 до 1.0")
    severity_score: float = Field(description="Серьезность от 0.0 до 1.0")
    keywords: list = Field(description="Ключевые слова")
    mentioned_persons: list = Field(description="Упомянутые лица")
    mentioned_locations: list = Field(description="Упомянутые места")
    mentioned_dates: list = Field(description="Упомянутые даты")
    contains_pii: bool = Field(description="Содержит персональные данные")
    contains_accusations: bool = Field(description="Содержит обвинения")
    reasoning: str = Field(description="Обоснование анализа")


class AnalyzerAgent(BaseAgent):
    """
    Агент анализа жалоб с LangChain.
    
    Использует модель: qwen3-vl:8b
    
    Выполняет:
    - Классификацию по категориям
    - Определение тональности
    - Извлечение сущностей (NER)
    - Оценку срочности и токсичности
    """
    
    # Модель для анализа
    MODEL_NAME = "qwen3-vl:8b"
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.2,
        use_tools: bool = True,
    ):
        super().__init__("AnalyzerAgent")
        self.model_name = model_name or self.MODEL_NAME
        self.temperature = temperature
        self.use_tools = use_tools
        
        self._provider = None
        self._chain = None
    
    async def initialize(self) -> bool:
        """Инициализация LangChain компонентов."""
        try:
            self.log_start(f"Initializing LangChain Analyzer with {self.model_name}")
            
            self._provider = get_ollama_provider()
            
            # Получаем ChatOllama с JSON mode
            self.llm = self._provider.get_chat_model(
                model=self.model_name,
                temperature=self.temperature,
                format="json",
            )
            
            # Настраиваем парсер вывода
            self.output_parser = PydanticOutputParser(pydantic_object=AnalysisOutput)
            
            # Промпт с форматированием
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", "{text}"),
            ])
            
            # Создаём chain
            self._chain = self.prompt | self.llm | self.output_parser
            
            # Если нужны инструменты
            if self.use_tools:
                self.tools = get_analysis_tools()
            
            self.logger.info(f"Analyzer initialized with {self.model_name}")
            return True
            
        except Exception as e:
            self.log_error("initialize", e)
            return False
    
    def _get_system_prompt(self) -> str:
        """Системный промпт для анализа."""
        return """Ты - эксперт по анализу обращений граждан. Твоя задача - проанализировать текст жалобы и извлечь структурированную информацию.

Проанализируй текст и верни JSON со следующими полями:
- category: категория жалобы (medical, school, housing, service, hotel, retail, government, other)
- sentiment: тональность (positive, neutral, negative, very_negative)
- urgency: срочность (low, medium, high, critical)
  * critical: угроза жизни/здоровью, требует немедленной реакции
  * high: серьезное нарушение, нужна реакция в течение дня
  * medium: стандартная жалоба
  * low: пожелание, незначительное замечание
- toxicity_score: уровень токсичности от 0.0 до 1.0 (оскорбления, угрозы, мат)
- severity_score: серьезность ситуации от 0.0 до 1.0
- keywords: список ключевых слов (3-7 слов)
- mentioned_persons: список упомянутых лиц (ФИО, должности)
- mentioned_locations: список мест (кабинеты, адреса, отделения)
- mentioned_dates: список дат и времени из текста
- contains_pii: true если есть персональные данные (ФИО, телефоны, адреса)
- contains_accusations: true если есть прямые обвинения конкретных лиц
- reasoning: краткое обоснование твоего анализа (1-2 предложения)

Отвечай ТОЛЬКО валидным JSON без дополнительного текста."""
    
    async def process(self, text: str) -> AgentResult[ComplaintMetrics]:
        """
        Анализировать текст жалобы.
        """
        start_time = time.time()
        
        try:
            await self.ensure_initialized()
            
            self.log_start(f"Analyzing text ({len(text)} chars)")
            
            if not text or len(text.strip()) < 10:
                return AgentResult.fail("Text too short for analysis")
            
            # Вызов chain
            try:
                analysis: AnalysisOutput = await self._chain.ainvoke({"text": text})
                metrics = self._convert_to_metrics(analysis)
                
            except Exception as parse_error:
                self.logger.warning(f"Structured parsing failed: {parse_error}")
                metrics = await self._fallback_analysis(text)
            
            processing_time = time.time() - start_time
            self.log_complete("analysis", processing_time)
            
            result = AgentResult.ok(
                data=metrics,
                processing_time=processing_time,
            )
            
            if metrics.contains_accusations:
                result.add_warning("Contains direct accusations - requires verification")
            
            if metrics.contains_pii:
                result.add_warning("Contains personally identifiable information")
            
            if metrics.toxicity_score > 0.7:
                result.add_warning(f"High toxicity detected: {metrics.toxicity_score:.2f}")
            
            return result
            
        except Exception as e:
            self.log_error("analysis", e)
            return AgentResult.fail(str(e))
    
    def _convert_to_metrics(self, analysis: AnalysisOutput) -> ComplaintMetrics:
        """Конвертация AnalysisOutput в ComplaintMetrics."""
        return ComplaintMetrics(
            sentiment=SentimentLevel(analysis.sentiment),
            toxicity_score=analysis.toxicity_score,
            urgency=UrgencyLevel(analysis.urgency),
            category=ComplaintCategory(analysis.category),
            keywords=analysis.keywords,
            mentioned_persons=analysis.mentioned_persons,
            mentioned_locations=analysis.mentioned_locations,
            mentioned_dates=analysis.mentioned_dates,
            severity_score=analysis.severity_score,
            contains_pii=analysis.contains_pii,
            contains_accusations=analysis.contains_accusations,
        )
    
    async def _fallback_analysis(self, text: str) -> ComplaintMetrics:
        """Fallback анализ без структурированного парсера."""
        from complaintagents.tools import (
            classify_category,
            analyze_sentiment,
            check_toxicity,
            extract_entities,
        )
        
        category = classify_category.invoke(text)
        sentiment = analyze_sentiment.invoke(text)
        toxicity = check_toxicity.invoke(text)
        entities = extract_entities.invoke(text)
        
        return ComplaintMetrics(
            sentiment=SentimentLevel(sentiment["sentiment"]),
            toxicity_score=toxicity["toxicity_score"],
            urgency=UrgencyLevel.MEDIUM,
            category=ComplaintCategory(category),
            keywords=[],
            mentioned_persons=entities.get("persons", []),
            mentioned_locations=entities.get("locations", []),
            mentioned_dates=entities.get("dates", []),
            severity_score=0.5,
            contains_pii=False,
            contains_accusations=len(entities.get("persons", [])) > 0,
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния агента."""
        return {
            "name": self.name,
            "status": "ok" if self._chain else "not_initialized",
            "model": self.model_name,
            "use_tools": self.use_tools,
            "initialized": self._initialized,
        }

