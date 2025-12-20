"""
Domain models for the agent system.

Модели данных для мультиагентной системы.
Независимы от основного приложения.
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


# ==================== Enums ====================

class ComplaintCategory(str, Enum):
    """Категория жалобы по типу учреждения."""
    MEDICAL = "medical"
    SCHOOL = "school"
    HOUSING = "housing"
    SERVICE = "service"
    HOTEL = "hotel"
    RETAIL = "retail"
    GOVERNMENT = "government"
    OTHER = "other"
    
    @property
    def display_name(self) -> str:
        names = {
            "medical": "Медицинская",
            "school": "Образовательная",
            "housing": "ЖКХ",
            "service": "Обслуживание",
            "hotel": "Гостиничная",
            "retail": "Торговля",
            "government": "Государственные услуги",
            "other": "Прочее",
        }
        return names.get(self.value, self.value)


class SentimentLevel(str, Enum):
    """Уровень эмоциональной окраски."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"
    
    @property
    def display_name(self) -> str:
        names = {
            "positive": "Положительная",
            "neutral": "Нейтральная",
            "negative": "Негативная",
            "very_negative": "Очень негативная",
        }
        return names.get(self.value, self.value)


class UrgencyLevel(str, Enum):
    """Уровень срочности."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @property
    def sla_hours(self) -> int:
        """SLA в часах для каждого уровня срочности."""
        sla_map = {
            "low": 72,
            "medium": 24,
            "high": 8,
            "critical": 2,
        }
        return sla_map.get(self.value, 24)


class VerificationLevel(str, Enum):
    """Уровень верификации источника жалобы."""
    ANONYMOUS = "anonymous"
    PSEUDO_ANONYMOUS = "pseudo"
    VISIT_CONFIRMED = "visit"
    IDENTIFIED = "identified"
    EMPLOYEE = "employee"
    
    @property
    def trust_weight(self) -> float:
        weights = {
            "anonymous": 0.3,
            "pseudo": 0.5,
            "visit": 0.7,
            "identified": 0.9,
            "employee": 1.0,
        }
        return weights.get(self.value, 0.3)


class IntakeChannel(str, Enum):
    """Канал поступления жалобы."""
    WEB_FORM = "web_form"
    MOBILE_APP = "mobile_app"
    IVR_PHONE = "ivr_phone"
    KIOSK = "kiosk"
    ROBOT = "robot"
    QR_CODE = "qr_code"
    MESSENGER = "messenger"
    EMAIL = "email"
    API = "api"


class CasePriority(str, Enum):
    """Приоритет обработки кейса."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# ==================== Data Models ====================

@dataclass
class TextArtifacts:
    """
    Три артефакта текста:
    1. original - исходная транскрипция
    2. normalized - дословная нормализация
    3. neutral - управленческая выжимка
    """
    original: str
    normalized: str
    neutral: str
    
    language: str = "ru"
    audio_duration: Optional[float] = None
    transcription_time: Optional[float] = None


@dataclass
class ComplaintMetrics:
    """Метрики анализа жалобы."""
    sentiment: SentimentLevel
    toxicity_score: float  # 0.0 - 1.0
    urgency: UrgencyLevel
    category: ComplaintCategory
    
    # Извлеченные сущности
    keywords: List[str] = field(default_factory=list)
    mentioned_persons: List[str] = field(default_factory=list)
    mentioned_locations: List[str] = field(default_factory=list)
    mentioned_dates: List[str] = field(default_factory=list)
    
    # Скоринг
    credibility_score: float = 0.5
    severity_score: float = 0.5
    
    # Флаги
    contains_pii: bool = False
    contains_accusations: bool = False
    is_repeat_complaint: bool = False
    
    def calculate_priority(self) -> CasePriority:
        """Вычислить приоритет на основе метрик."""
        score = (self.severity_score * 0.4 + 
                 (1.0 if self.urgency == UrgencyLevel.CRITICAL else 0.0) * 0.3 +
                 self.credibility_score * 0.3)
        
        if score >= 0.8 or self.urgency == UrgencyLevel.CRITICAL:
            return CasePriority.URGENT
        elif score >= 0.6:
            return CasePriority.HIGH
        elif score >= 0.4:
            return CasePriority.NORMAL
        return CasePriority.LOW


@dataclass
class TranscriptionData:
    """Результат транскрипции."""
    text: str
    language: str
    confidence: float
    duration: float
    segments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingDecision:
    """Решение о маршрутизации."""
    department: str
    responsible_role: str
    escalation_required: bool = False
    escalation_reason: Optional[str] = None
    sla_hours: int = 24
    priority_boost: bool = False
    additional_departments: List[str] = field(default_factory=list)
    routing_notes: str = ""


@dataclass
class FraudScore:
    """Результат антифрод проверки."""
    credibility_score: float
    spam_probability: float
    is_suspicious: bool
    flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    rate_limit_ok: bool = True
    content_check_ok: bool = True
    pattern_check_ok: bool = True
    verification_weight: float = 0.5


@dataclass
class ProcessingResult:
    """Результат полной обработки жалобы."""
    text_artifacts: TextArtifacts
    metrics: ComplaintMetrics
    routing_decision: Optional[RoutingDecision] = None
    fraud_score: Optional[FraudScore] = None
    
    # Времена обработки
    transcription_time: float = 0.0
    analysis_time: float = 0.0
    summarization_time: float = 0.0
    total_time: float = 0.0
    
    # Метаданные
    audio_file_path: Optional[str] = None
    intake_channel: IntakeChannel = IntakeChannel.API
    warnings: List[str] = field(default_factory=list)

