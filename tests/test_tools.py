"""
Tests for LangChain tools.

Тесты для инструментов LangChain.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ..complaintagents.tools import (
    extract_entities,
    classify_category,
    analyze_sentiment,
    check_toxicity,
    calculate_urgency,
    get_analysis_tools,
    get_search_tools,
    get_all_tools,
)


class TestExtractEntities:
    """Тесты для извлечения сущностей."""
    
    def test_extract_persons(self):
        """Тест извлечения упоминаний персон."""
        text = "Врач Петров был груб. Медсестра Иванова помогла."
        result = extract_entities.invoke(text)
        
        assert "persons" in result
        assert len(result["persons"]) >= 1
    
    def test_extract_locations(self):
        """Тест извлечения локаций."""
        text = "В кабинете 215 на этаже 3 было очень грязно."
        result = extract_entities.invoke(text)
        
        assert "locations" in result
        assert len(result["locations"]) >= 1
    
    def test_empty_text(self):
        """Тест с пустым текстом."""
        result = extract_entities.invoke("")
        
        assert "persons" in result
        assert "locations" in result
        assert len(result["persons"]) == 0


class TestClassifyCategory:
    """Тесты для классификации категорий."""
    
    def test_medical_category(self):
        """Тест определения медицинской категории."""
        text = "Врач в поликлинике плохо провёл приём"
        result = classify_category.invoke(text)
        
        assert result == "medical"
    
    def test_school_category(self):
        """Тест определения школьной категории."""
        text = "Учитель в школе неправильно ведёт урок"
        result = classify_category.invoke(text)
        
        assert result == "school"
    
    def test_housing_category(self):
        """Тест определения категории ЖКХ."""
        text = "В подъезде не работает лифт, отопление холодное"
        result = classify_category.invoke(text)
        
        assert result == "housing"
    
    def test_hotel_category(self):
        """Тест определения гостиничной категории."""
        text = "В отеле плохой номер, ресепшн не помог"
        result = classify_category.invoke(text)
        
        assert result == "hotel"
    
    def test_other_category(self):
        """Тест для неопределённой категории."""
        text = "Просто какой-то текст без ключевых слов"
        result = classify_category.invoke(text)
        
        assert result == "other"


class TestAnalyzeSentiment:
    """Тесты для анализа тональности."""
    
    def test_negative_sentiment(self):
        """Тест негативной тональности."""
        text = "Ужасно! Плохо! Безобразие!"
        result = analyze_sentiment.invoke(text)
        
        assert result["sentiment"] in ["negative", "very_negative"]
        assert result["score"] < 0
    
    def test_positive_sentiment(self):
        """Тест позитивной тональности."""
        text = "Отлично! Спасибо! Помогли быстро!"
        result = analyze_sentiment.invoke(text)
        
        assert result["sentiment"] == "positive"
        assert result["score"] > 0
    
    def test_neutral_sentiment(self):
        """Тест нейтральной тональности."""
        text = "Я был на приёме у врача сегодня."
        result = analyze_sentiment.invoke(text)
        
        assert result["sentiment"] == "neutral"
        assert result["score"] == 0.0


class TestCheckToxicity:
    """Тесты для проверки токсичности."""
    
    def test_toxic_text(self):
        """Тест токсичного текста."""
        text = "Этот идиот дурак полный!"
        result = check_toxicity.invoke(text)
        
        assert result["toxicity_score"] > 0
        assert result["is_toxic"] is True
        assert len(result["flags"]) > 0
    
    def test_non_toxic_text(self):
        """Тест нетоксичного текста."""
        text = "Пожалуйста, помогите решить проблему."
        result = check_toxicity.invoke(text)
        
        assert result["toxicity_score"] == 0.0
        assert result["is_toxic"] is False
    
    def test_all_caps_flag(self):
        """Тест флага для CAPS LOCK."""
        text = "Я ОЧЕНЬ НЕДОВОЛЕН КАЧЕСТВОМ ОБСЛУЖИВАНИЯ В ВАШЕМ МАГАЗИНЕ!"
        result = check_toxicity.invoke(text)
        
        assert "excessive_caps" in result["flags"]


class TestCalculateUrgency:
    """Тесты для расчёта срочности."""
    
    def test_critical_urgency(self):
        """Тест критической срочности."""
        result = calculate_urgency.invoke(
            sentiment_score=-0.9,
            toxicity_score=0.8,
            has_health_keywords=True,
            has_safety_keywords=True,
        )
        
        assert result == "critical"
    
    def test_low_urgency(self):
        """Тест низкой срочности."""
        result = calculate_urgency.invoke(
            sentiment_score=0.0,
            toxicity_score=0.0,
            has_health_keywords=False,
            has_safety_keywords=False,
        )
        
        assert result == "low"
    
    def test_medium_urgency(self):
        """Тест средней срочности."""
        result = calculate_urgency.invoke(
            sentiment_score=-0.5,
            toxicity_score=0.3,
            has_health_keywords=False,
            has_safety_keywords=False,
        )
        
        assert result in ["low", "medium"]


class TestToolRegistry:
    """Тесты для реестра инструментов."""
    
    def test_get_analysis_tools(self):
        """Тест получения инструментов анализа."""
        tools = get_analysis_tools()
        
        assert len(tools) == 5
        tool_names = [t.name for t in tools]
        assert "extract_entities" in tool_names
        assert "classify_category" in tool_names
        assert "analyze_sentiment" in tool_names
        assert "check_toxicity" in tool_names
        assert "calculate_urgency" in tool_names
    
    def test_get_search_tools(self):
        """Тест получения инструментов поиска."""
        tools = get_search_tools()
        
        assert len(tools) == 2
        tool_names = [t.name for t in tools]
        assert "search_similar_complaints" in tool_names
        assert "validate_person_mention" in tool_names
    
    def test_get_all_tools(self):
        """Тест получения всех инструментов."""
        tools = get_all_tools()
        
        assert len(tools) == 7

