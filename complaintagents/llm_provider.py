"""
LLM Provider - настройка Ollama для LangChain.

Централизованная конфигурация LLM для всех агентов.
Поддерживает:
- Локальный Ollama
- Модель qwen3-vl:4b (единая для всех агентов)
- Кастомные параметры генерации
"""

import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from .config import get_agent_config

logger = logging.getLogger(__name__)


class OllamaProvider:
    """
    Провайдер Ollama LLM для LangChain.
    
    Использует qwen3-vl:4b как основную модель для всех агентов.
    
    Управляет:
    - Созданием LLM инстансов
    - Кэшированием моделей
    - Конфигурацией параметров
    """
    
    # Модель по умолчанию для всех агентов
    DEFAULT_MODEL = "qwen3-vl:4b"
    
    def __init__(
        self,
        base_url: str = None,
        default_model: str = None,
        timeout: int = None,
    ):
        config = get_agent_config()
        self.base_url = base_url or config.ollama.base_url
        self.default_model = default_model or self.DEFAULT_MODEL
        self.timeout = timeout or config.ollama.timeout
        
        self._llm_cache: Dict[str, Ollama] = {}
        self._chat_cache: Dict[str, ChatOllama] = {}
        
        logger.info(f"OllamaProvider initialized: {self.base_url}, model: {self.default_model}")
    
    def get_llm(
        self,
        model: str = None,
        temperature: float = 0.3,
        streaming: bool = False,
        **kwargs,
    ) -> Ollama:
        """
        Получить Ollama LLM инстанс.
        
        Args:
            model: Название модели (по умолчанию qwen3-vl:4b)
            temperature: Температура генерации
            streaming: Включить стриминг
            **kwargs: Дополнительные параметры
            
        Returns:
            Настроенный Ollama LLM
        """
        model_name = model or self.default_model
        cache_key = f"{model_name}_{temperature}_{streaming}"
        
        if cache_key not in self._llm_cache:
            self._llm_cache[cache_key] = Ollama(
                base_url=self.base_url,
                model=model_name,
                temperature=temperature,
                **kwargs,
            )
            
            logger.debug(f"Created LLM: {model_name} (temp={temperature})")
        
        return self._llm_cache[cache_key]
    
    def get_chat_model(
        self,
        model: str = None,
        temperature: float = 0.3,
        streaming: bool = False,
        format: str = None,  # "json" for JSON mode
        **kwargs,
    ) -> ChatOllama:
        """
        Получить ChatOllama для диалоговых агентов.
        
        Args:
            model: Название модели (по умолчанию qwen3-vl:4b)
            temperature: Температура
            streaming: Стриминг
            format: Формат вывода ("json" для JSON mode)
            
        Returns:
            ChatOllama инстанс
        """
        model_name = model or self.default_model
        cache_key = f"chat_{model_name}_{temperature}_{format}"
        
        if cache_key not in self._chat_cache:
            self._chat_cache[cache_key] = ChatOllama(
                base_url=self.base_url,
                model=model_name,
                temperature=temperature,
                format=format,
                **kwargs,
            )
            
            logger.debug(f"Created ChatModel: {model_name}")
        
        return self._chat_cache[cache_key]
    
    def get_embeddings(self, model: str = None) -> OllamaEmbeddings:
        """
        Получить Ollama Embeddings для векторного поиска.
        
        Args:
            model: Модель для эмбеддингов
            
        Returns:
            OllamaEmbeddings
        """
        return OllamaEmbeddings(
            base_url=self.base_url,
            model=model or self.default_model,
        )
    
    async def check_health(self) -> Dict[str, Any]:
        """Проверить доступность Ollama."""
        try:
            llm = self.get_llm()
            # Простой тест
            response = llm.invoke("test")
            return {
                "status": "ok",
                "base_url": self.base_url,
                "default_model": self.default_model,
            }
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                "status": "error",
                "base_url": self.base_url,
                "error": str(e),
            }
    
    async def list_models(self) -> List[str]:
        """Получить список доступных моделей."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def clear_cache(self):
        """Очистить кэш моделей."""
        self._llm_cache.clear()
        self._chat_cache.clear()
        logger.debug("LLM cache cleared")


# Глобальный провайдер
_provider: Optional[OllamaProvider] = None


def get_ollama_provider() -> OllamaProvider:
    """Получить singleton провайдера Ollama."""
    global _provider
    if _provider is None:
        _provider = OllamaProvider()
    return _provider


def reset_ollama_provider():
    """Сбросить провайдер (для тестов)."""
    global _provider
    _provider = None

