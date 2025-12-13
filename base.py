"""
Base agent class using LangChain framework.

Использует LangChain для:
- Создания агентов с инструментами
- Управления памятью разговора
- Структурированного вывода
- Цепочек обработки
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, TypeVar, Optional, List

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class AgentResult(Generic[T]):
    """
    Результат работы агента.
    
    Унифицированная обертка для результатов всех агентов,
    включающая метаданные обработки и информацию об ошибках.
    """
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # LangChain specific
    intermediate_steps: List[Any] = field(default_factory=list)
    tokens_used: int = 0
    
    @classmethod
    def ok(cls, data: T, processing_time: float = 0.0, **metadata) -> "AgentResult[T]":
        """Создать успешный результат."""
        return cls(
            success=True,
            data=data,
            processing_time=processing_time,
            metadata=metadata
        )
    
    @classmethod
    def fail(cls, error: str, **metadata) -> "AgentResult[T]":
        """Создать результат с ошибкой."""
        return cls(
            success=False,
            error=error,
            metadata=metadata
        )
    
    def add_warning(self, warning: str):
        """Добавить предупреждение."""
        self.warnings.append(warning)


class BaseAgent(ABC):
    """
    Базовый класс для LangChain агентов.
    
    Предоставляет:
    - Интеграцию с Ollama LLM
    - Память разговора
    - Систему инструментов
    - Структурированный вывод
    """
    
    def __init__(
        self,
        name: str = None,
        llm: BaseLLM = None,
        memory: ConversationBufferMemory = None,
        verbose: bool = False,
    ):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"agents.{self.name}")
        self._initialized = False
        
        # LangChain components
        self.llm = llm
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.verbose = verbose
        
        # Tools and executor
        self.tools: List[Any] = []
        self.agent_executor: Optional[AgentExecutor] = None
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Инициализация агента (загрузка моделей, настройка tools).
        
        Returns:
            True если инициализация успешна
        """
        pass
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> AgentResult:
        """
        Основной метод обработки.
        
        Returns:
            AgentResult с результатом обработки
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка состояния агента.
        
        Returns:
            Словарь с информацией о состоянии
        """
        pass
    
    def get_tools(self) -> List[Any]:
        """Получить список инструментов агента."""
        return self.tools
    
    async def ensure_initialized(self):
        """Убедиться, что агент инициализирован."""
        if not self._initialized:
            success = await self.initialize()
            if success:
                self._initialized = True
            else:
                raise RuntimeError(f"Failed to initialize agent: {self.name}")
    
    def log_start(self, operation: str):
        """Логирование начала операции."""
        self.logger.info(f"[{self.name}] Starting: {operation}")
    
    def log_complete(self, operation: str, duration: float):
        """Логирование завершения операции."""
        self.logger.info(f"[{self.name}] Completed: {operation} ({duration:.2f}s)")
    
    def log_error(self, operation: str, error: Exception):
        """Логирование ошибки."""
        self.logger.error(f"[{self.name}] Error in {operation}: {error}", exc_info=True)
    
    def clear_memory(self):
        """Очистить память разговора."""
        if self.memory:
            self.memory.clear()

