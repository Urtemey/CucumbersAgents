"""
LangChain Tools для агентов обработки жалоб.

Набор инструментов, которые агенты могут использовать:
- Извлечение сущностей
- Классификация текста
- Поиск по базе жалоб
- Проверка персоналий
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


# ==================== Pydantic Models for Tools ====================

class EntityExtractionInput(BaseModel):
    """Входные данные для извлечения сущностей."""
    text: str = Field(description="Текст для анализа")
    entity_types: List[str] = Field(
        default=["person", "location", "date", "organization"],
        description="Типы сущностей для извлечения"
    )


class ClassificationInput(BaseModel):
    """Входные данные для классификации."""
    text: str = Field(description="Текст для классификации")
    categories: List[str] = Field(description="Список возможных категорий")


class SimilarComplaintsInput(BaseModel):
    """Входные данные для поиска похожих жалоб."""
    text: str = Field(description="Текст жалобы")
    limit: int = Field(default=5, description="Максимум результатов")


class ToxicityCheckInput(BaseModel):
    """Входные данные для проверки токсичности."""
    text: str = Field(description="Текст для проверки")


# ==================== Tool Functions ====================

@tool
def extract_entities(text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Извлечь сущности из текста (имена, места, даты).
    
    Args:
        text: Текст для анализа
        entity_types: Типы сущностей для извлечения
        
    Returns:
        Словарь с извлечёнными сущностями по типам
    """
    # Простая эвристика (в проде - NER модель)
    entities = {
        "persons": [],
        "locations": [],
        "dates": [],
        "organizations": [],
    }
    
    # Ключевые слова для персон
    person_keywords = ["доктор", "врач", "медсестра", "учитель", "директор", "администратор"]
    location_keywords = ["кабинет", "этаж", "корпус", "здание", "отделение", "палата"]
    
    words = text.lower().split()
    
    for i, word in enumerate(words):
        # Персоны
        if word in person_keywords and i + 1 < len(words):
            entities["persons"].append(f"{word} {words[i+1]}")
        
        # Локации
        if word in location_keywords and i + 1 < len(words):
            entities["locations"].append(f"{word} {words[i+1]}")
    
    return entities


@tool
def classify_category(text: str) -> str:
    """
    Определить категорию жалобы.
    
    Args:
        text: Текст жалобы
        
    Returns:
        Категория (medical, school, housing, service, other)
    """
    text_lower = text.lower()
    
    # Ключевые слова по категориям
    categories = {
        "medical": ["врач", "больница", "поликлиника", "медсестра", "лечение", "прием", "диагноз"],
        "school": ["учитель", "школа", "директор", "урок", "ученик", "класс", "образование"],
        "housing": ["жкх", "подъезд", "лифт", "отопление", "вода", "управляющая", "дом"],
        "service": ["обслуживание", "сервис", "персонал", "клиент", "услуга"],
        "hotel": ["отель", "гостиница", "номер", "ресепшн", "бронь"],
    }
    
    scores = {cat: 0 for cat in categories}
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                scores[category] += 1
    
    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else "other"


@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Анализ тональности текста.
    
    Args:
        text: Текст для анализа
        
    Returns:
        Результат анализа с sentiment и score
    """
    text_lower = text.lower()
    
    negative_words = ["ужасно", "плохо", "отвратительно", "безобразие", "хам", "грубо", "долго", "никто", "невозможно"]
    positive_words = ["хорошо", "отлично", "спасибо", "благодарю", "помогли", "быстро"]
    
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    
    if neg_count > pos_count + 2:
        return {"sentiment": "very_negative", "score": -0.9}
    elif neg_count > pos_count:
        return {"sentiment": "negative", "score": -0.5}
    elif pos_count > neg_count:
        return {"sentiment": "positive", "score": 0.5}
    else:
        return {"sentiment": "neutral", "score": 0.0}


@tool
def check_toxicity(text: str) -> Dict[str, Any]:
    """
    Проверить текст на токсичность.
    
    Args:
        text: Текст для проверки
        
    Returns:
        Результат с toxicity_score и flags
    """
    text_lower = text.lower()
    
    toxic_patterns = [
        "идиот", "дурак", "тупой", "урод", "мразь", "скотина",
        "убью", "уволю", "засужу", "разнесу", "пожалею",
    ]
    
    threat_patterns = ["убью", "уничтожу", "найду", "накажу"]
    
    flags = []
    toxic_count = 0
    
    for pattern in toxic_patterns:
        if pattern in text_lower:
            toxic_count += 1
            flags.append(f"toxic_word: {pattern}")
    
    for pattern in threat_patterns:
        if pattern in text_lower:
            flags.append(f"threat: {pattern}")
    
    # Проверка на CAPS LOCK
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.5 and len(text) > 20:
        flags.append("excessive_caps")
    
    toxicity_score = min(1.0, toxic_count * 0.3)
    
    return {
        "toxicity_score": toxicity_score,
        "is_toxic": toxicity_score > 0.5,
        "flags": flags,
    }


@tool
def calculate_urgency(
    sentiment_score: float,
    toxicity_score: float,
    has_health_keywords: bool,
    has_safety_keywords: bool,
) -> str:
    """
    Рассчитать срочность обращения.
    
    Args:
        sentiment_score: Оценка тональности (-1 to 1)
        toxicity_score: Оценка токсичности (0 to 1)
        has_health_keywords: Есть ли ключевые слова о здоровье
        has_safety_keywords: Есть ли ключевые слова о безопасности
        
    Returns:
        Уровень срочности (low, medium, high, critical)
    """
    score = 0.0
    
    # Негативность добавляет срочности
    if sentiment_score < -0.7:
        score += 0.3
    elif sentiment_score < -0.3:
        score += 0.15
    
    # Токсичность
    score += toxicity_score * 0.2
    
    # Здоровье и безопасность - критично
    if has_health_keywords:
        score += 0.3
    if has_safety_keywords:
        score += 0.4
    
    if score >= 0.7:
        return "critical"
    elif score >= 0.5:
        return "high"
    elif score >= 0.25:
        return "medium"
    return "low"


@tool
def search_similar_complaints(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Поиск похожих жалоб в истории.
    
    Args:
        text: Текст жалобы для поиска
        limit: Максимум результатов
        
    Returns:
        Список похожих жалоб
    """
    # В проде - векторный поиск через embeddings
    # Сейчас - заглушка
    return []


@tool  
def validate_person_mention(person_name: str, organization_id: str = None) -> Dict[str, Any]:
    """
    Проверить упоминание персоны.
    
    Args:
        person_name: Имя/должность упомянутого лица
        organization_id: ID организации
        
    Returns:
        Информация о валидации
    """
    # В проде - проверка по базе сотрудников
    return {
        "found": False,
        "verified": False,
        "note": "Требуется верификация",
    }


# ==================== Tool Registry ====================

def get_analysis_tools() -> List[BaseTool]:
    """Получить инструменты для анализа жалоб."""
    return [
        extract_entities,
        classify_category,
        analyze_sentiment,
        check_toxicity,
        calculate_urgency,
    ]


def get_search_tools() -> List[BaseTool]:
    """Получить инструменты для поиска."""
    return [
        search_similar_complaints,
        validate_person_mention,
    ]


def get_all_tools() -> List[BaseTool]:
    """Получить все доступные инструменты."""
    return get_analysis_tools() + get_search_tools()

