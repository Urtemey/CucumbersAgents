"""
Tests for domain models.

Тесты для доменных моделей.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from complaintagents.models import (
    ComplaintCategory,
    SentimentLevel,
    UrgencyLevel,
    VerificationLevel,
    IntakeChannel,
    CasePriority,
    TextArtifacts,
    ComplaintMetrics,
    TranscriptionData,
    RoutingDecision,
    FraudScore,
    ProcessingResult,
)


class TestEnums:
    """Тесты для перечислений."""
    
    def test_complaint_category_display_names(self):
        """Тест отображаемых имён категорий."""
        assert ComplaintCategory.MEDICAL.display_name == "Медицинская"
        assert ComplaintCategory.SCHOOL.display_name == "Образовательная"
        assert ComplaintCategory.HOUSING.display_name == "ЖКХ"
        assert ComplaintCategory.OTHER.display_name == "Прочее"
    
    def test_sentiment_level_display_names(self):
        """Тест отображаемых имён тональности."""
        assert SentimentLevel.POSITIVE.display_name == "Положительная"
        assert SentimentLevel.NEGATIVE.display_name == "Негативная"
        assert SentimentLevel.VERY_NEGATIVE.display_name == "Очень негативная"
    
    def test_urgency_level_sla_hours(self):
        """Тест SLA часов для срочности."""
        assert UrgencyLevel.LOW.sla_hours == 72
        assert UrgencyLevel.MEDIUM.sla_hours == 24
        assert UrgencyLevel.HIGH.sla_hours == 8
        assert UrgencyLevel.CRITICAL.sla_hours == 2
    
    def test_verification_level_trust_weight(self):
        """Тест весов доверия для верификации."""
        assert VerificationLevel.ANONYMOUS.trust_weight == 0.3
        assert VerificationLevel.IDENTIFIED.trust_weight == 0.9
        assert VerificationLevel.EMPLOYEE.trust_weight == 1.0
    
    def test_intake_channel_values(self):
        """Тест значений каналов поступления."""
        assert IntakeChannel.WEB_FORM.value == "web_form"
        assert IntakeChannel.API.value == "api"
        assert IntakeChannel.MOBILE_APP.value == "mobile_app"


class TestTextArtifacts:
    """Тесты для TextArtifacts."""
    
    def test_create_text_artifacts(self):
        """Тест создания артефактов текста."""
        artifacts = TextArtifacts(
            original="Оригинальный текст",
            normalized="Нормализованный текст",
            neutral="Нейтральный текст",
            language="ru",
        )
        
        assert artifacts.original == "Оригинальный текст"
        assert artifacts.normalized == "Нормализованный текст"
        assert artifacts.neutral == "Нейтральный текст"
        assert artifacts.language == "ru"
    
    def test_text_artifacts_with_audio_info(self):
        """Тест артефактов с информацией об аудио."""
        artifacts = TextArtifacts(
            original="Текст",
            normalized="Текст",
            neutral="Текст",
            language="ru",
            audio_duration=10.5,
            transcription_time=2.3,
        )
        
        assert artifacts.audio_duration == 10.5
        assert artifacts.transcription_time == 2.3


class TestComplaintMetrics:
    """Тесты для ComplaintMetrics."""
    
    def test_create_complaint_metrics(self):
        """Тест создания метрик жалобы."""
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.3,
            urgency=UrgencyLevel.MEDIUM,
            category=ComplaintCategory.MEDICAL,
        )
        
        assert metrics.sentiment == SentimentLevel.NEGATIVE
        assert metrics.toxicity_score == 0.3
        assert metrics.urgency == UrgencyLevel.MEDIUM
        assert metrics.category == ComplaintCategory.MEDICAL
    
    def test_calculate_priority_urgent(self):
        """Тест расчёта срочного приоритета."""
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.VERY_NEGATIVE,
            toxicity_score=0.9,
            urgency=UrgencyLevel.CRITICAL,
            category=ComplaintCategory.MEDICAL,
            severity_score=0.95,
            credibility_score=0.9,
        )
        
        priority = metrics.calculate_priority()
        assert priority == CasePriority.URGENT
    
    def test_calculate_priority_low(self):
        """Тест расчёта низкого приоритета."""
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEUTRAL,
            toxicity_score=0.1,
            urgency=UrgencyLevel.LOW,
            category=ComplaintCategory.OTHER,
            severity_score=0.2,
            credibility_score=0.3,
        )
        
        priority = metrics.calculate_priority()
        assert priority == CasePriority.LOW
    
    def test_metrics_with_entities(self):
        """Тест метрик с извлечёнными сущностями."""
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.5,
            urgency=UrgencyLevel.HIGH,
            category=ComplaintCategory.MEDICAL,
            keywords=["врач", "приём", "жалоба"],
            mentioned_persons=["Петров", "Иванова"],
            mentioned_locations=["кабинет 215"],
            mentioned_dates=["15 января"],
        )
        
        assert len(metrics.keywords) == 3
        assert len(metrics.mentioned_persons) == 2
        assert "Петров" in metrics.mentioned_persons


class TestTranscriptionData:
    """Тесты для TranscriptionData."""
    
    def test_create_transcription_data(self):
        """Тест создания данных транскрипции."""
        data = TranscriptionData(
            text="Транскрибированный текст",
            language="ru",
            confidence=0.95,
            duration=10.5,
        )
        
        assert data.text == "Транскрибированный текст"
        assert data.language == "ru"
        assert data.confidence == 0.95
        assert data.duration == 10.5
    
    def test_transcription_with_segments(self):
        """Тест транскрипции с сегментами."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Первый сегмент"},
            {"start": 5.0, "end": 10.0, "text": "Второй сегмент"},
        ]
        
        data = TranscriptionData(
            text="Полный текст",
            language="ru",
            confidence=0.9,
            duration=10.0,
            segments=segments,
        )
        
        assert len(data.segments) == 2


class TestRoutingDecision:
    """Тесты для RoutingDecision."""
    
    def test_create_routing_decision(self):
        """Тест создания решения маршрутизации."""
        decision = RoutingDecision(
            department="medical_quality",
            responsible_role="quality_manager",
        )
        
        assert decision.department == "medical_quality"
        assert decision.responsible_role == "quality_manager"
        assert decision.escalation_required is False
        assert decision.sla_hours == 24
    
    def test_routing_with_escalation(self):
        """Тест маршрутизации с эскалацией."""
        decision = RoutingDecision(
            department="medical_quality",
            responsible_role="quality_manager",
            escalation_required=True,
            escalation_reason="Critical urgency",
            sla_hours=2,
            priority_boost=True,
            additional_departments=["legal", "hr_department"],
        )
        
        assert decision.escalation_required is True
        assert decision.escalation_reason == "Critical urgency"
        assert len(decision.additional_departments) == 2


class TestFraudScore:
    """Тесты для FraudScore."""
    
    def test_create_fraud_score(self):
        """Тест создания скора антифрода."""
        score = FraudScore(
            credibility_score=0.7,
            spam_probability=0.2,
            is_suspicious=False,
        )
        
        assert score.credibility_score == 0.7
        assert score.spam_probability == 0.2
        assert score.is_suspicious is False
    
    def test_suspicious_fraud_score(self):
        """Тест подозрительного скора."""
        score = FraudScore(
            credibility_score=0.2,
            spam_probability=0.8,
            is_suspicious=True,
            flags=["RATE_LIMIT_EXCEEDED", "SUSPICIOUS_PATTERN"],
            recommendations=["Рекомендуется верификация"],
        )
        
        assert score.is_suspicious is True
        assert len(score.flags) == 2
        assert "RATE_LIMIT_EXCEEDED" in score.flags


class TestProcessingResult:
    """Тесты для ProcessingResult."""
    
    def test_create_processing_result(self):
        """Тест создания результата обработки."""
        artifacts = TextArtifacts(
            original="Текст",
            normalized="Текст",
            neutral="Текст",
        )
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.3,
            urgency=UrgencyLevel.MEDIUM,
            category=ComplaintCategory.MEDICAL,
        )
        
        result = ProcessingResult(
            text_artifacts=artifacts,
            metrics=metrics,
            total_time=5.5,
            intake_channel=IntakeChannel.API,
        )
        
        assert result.text_artifacts == artifacts
        assert result.metrics == metrics
        assert result.total_time == 5.5
        assert result.intake_channel == IntakeChannel.API

