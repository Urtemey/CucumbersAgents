"""Antifraud Agent - Скоринг достоверности и защита от спама."""

import hashlib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from collections import defaultdict

from agents.base import BaseAgent, AgentResult
from agents.config import get_agent_config
from agents.models import (
    ComplaintMetrics,
    VerificationLevel,
    FraudScore,
)


class SourceFingerprint:
    """Отпечаток источника для псевдо-анонимной идентификации."""
    
    def __init__(self, hash_value: str):
        self.hash = hash_value
        self.created_at = datetime.utcnow()
        self.complaints_count = 0
        self.last_complaint_at: Optional[datetime] = None
        self.verified_visits = 0
        self.spam_reports = 0


class AntifraudAgent(BaseAgent):
    """
    Агент антифрод защиты.
    
    Выполняет:
    - Скоринг достоверности на основе верификации
    - Rate limiting по источнику
    - Детекция спама и вбросов
    - Корреляция с историей жалоб
    """
    
    def __init__(
        self,
        rate_limit_per_hour: int = None,
        rate_limit_per_day: int = None,
        min_text_length: int = None,
    ):
        super().__init__("AntifraudAgent")
        
        config = get_agent_config()
        self.rate_limit_per_hour = rate_limit_per_hour or config.antifraud.rate_limit_per_hour
        self.rate_limit_per_day = rate_limit_per_day or config.antifraud.rate_limit_per_day
        self.min_text_length = min_text_length or config.antifraud.min_text_length
        self.suspicious_patterns = config.antifraud.suspicious_patterns
        
        # In-memory хранилище
        self._fingerprints: Dict[str, SourceFingerprint] = {}
        self._rate_limits: Dict[str, List[datetime]] = defaultdict(list)
        self._complaint_history: List[Dict] = []
    
    async def initialize(self) -> bool:
        """Инициализация агента."""
        self.log_start("Initializing antifraud rules")
        return True
    
    async def process(
        self,
        text: str,
        metrics: ComplaintMetrics,
        verification_level: VerificationLevel = VerificationLevel.ANONYMOUS,
        source_id: str = None,
        ip_hash: str = None,
        device_hash: str = None,
    ) -> AgentResult[FraudScore]:
        """
        Выполнить антифрод проверку.
        """
        start_time = time.time()
        
        try:
            self.log_start("Antifraud check")
            
            flags = []
            recommendations = []
            
            # Создаем fingerprint
            fingerprint_hash = self._create_fingerprint(source_id, ip_hash, device_hash)
            fingerprint = self._get_or_create_fingerprint(fingerprint_hash)
            
            # 1. Проверка rate limit
            rate_limit_ok = self._check_rate_limit(fingerprint_hash)
            if not rate_limit_ok:
                flags.append("RATE_LIMIT_EXCEEDED")
                recommendations.append("Подождите перед отправкой следующей жалобы")
            
            # 2. Проверка контента
            content_check_ok, content_flags = self._check_content(text)
            if not content_check_ok:
                flags.extend(content_flags)
            
            # 3. Проверка паттернов
            pattern_check_ok, pattern_flags = self._check_patterns(text, metrics)
            if not pattern_check_ok:
                flags.extend(pattern_flags)
            
            # 4. Проверка истории источника
            history_score = self._check_source_history(fingerprint)
            if history_score < 0.3:
                flags.append("LOW_HISTORY_TRUST")
                recommendations.append("Рекомендуется верификация источника")
            
            # 5. Корреляция с другими жалобами
            correlation_flags = self._check_correlation(text, metrics)
            flags.extend(correlation_flags)
            
            # Вычисление финального скора
            base_credibility = verification_level.trust_weight
            
            penalty = 0.0
            if not rate_limit_ok:
                penalty += 0.3
            if not content_check_ok:
                penalty += 0.2
            if not pattern_check_ok:
                penalty += 0.2
            if "POTENTIAL_COORDINATED_ATTACK" in flags:
                penalty += 0.4
            
            credibility_score = max(0.0, min(1.0, base_credibility - penalty + history_score * 0.2))
            
            # Спам вероятность
            spam_signals = len([f for f in flags if f in [
                "RATE_LIMIT_EXCEEDED", "SHORT_TEXT", "SUSPICIOUS_PATTERN",
                "GIBBERISH_DETECTED", "REPEATED_CONTENT"
            ]])
            spam_probability = min(1.0, spam_signals * 0.25)
            
            is_suspicious = spam_probability > 0.5 or credibility_score < 0.3
            
            # Обновляем fingerprint
            fingerprint.complaints_count += 1
            fingerprint.last_complaint_at = datetime.utcnow()
            
            self._record_complaint(fingerprint_hash)
            self._add_to_history(text, metrics, fingerprint_hash)
            
            fraud_score = FraudScore(
                credibility_score=credibility_score,
                spam_probability=spam_probability,
                is_suspicious=is_suspicious,
                flags=flags,
                recommendations=recommendations,
                rate_limit_ok=rate_limit_ok,
                content_check_ok=content_check_ok,
                pattern_check_ok=pattern_check_ok,
                verification_weight=verification_level.trust_weight,
            )
            
            processing_time = time.time() - start_time
            self.log_complete("antifraud", processing_time)
            
            result = AgentResult.ok(
                data=fraud_score,
                processing_time=processing_time,
            )
            
            if is_suspicious:
                result.add_warning(f"Suspicious complaint detected: {', '.join(flags)}")
            
            return result
            
        except Exception as e:
            self.log_error("antifraud", e)
            
            fallback = FraudScore(
                credibility_score=0.5,
                spam_probability=0.0,
                is_suspicious=False,
            )
            return AgentResult.ok(data=fallback, processing_time=time.time() - start_time)
    
    def _create_fingerprint(self, source_id: str = None, ip_hash: str = None, device_hash: str = None) -> str:
        components = [source_id or "unknown", ip_hash or "no_ip", device_hash or "no_device"]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _get_or_create_fingerprint(self, hash_value: str) -> SourceFingerprint:
        if hash_value not in self._fingerprints:
            self._fingerprints[hash_value] = SourceFingerprint(hash_value)
        return self._fingerprints[hash_value]
    
    def _check_rate_limit(self, fingerprint_hash: str) -> bool:
        now = datetime.utcnow()
        history = self._rate_limits.get(fingerprint_hash, [])
        history = [t for t in history if now - t < timedelta(days=1)]
        self._rate_limits[fingerprint_hash] = history
        
        hour_ago = now - timedelta(hours=1)
        hour_count = sum(1 for t in history if t > hour_ago)
        
        if hour_count >= self.rate_limit_per_hour:
            return False
        if len(history) >= self.rate_limit_per_day:
            return False
        return True
    
    def _record_complaint(self, fingerprint_hash: str):
        self._rate_limits[fingerprint_hash].append(datetime.utcnow())
    
    def _check_content(self, text: str) -> tuple:
        flags = []
        if len(text) < self.min_text_length:
            flags.append("SHORT_TEXT")
        unique_chars = len(set(text.lower()))
        if len(text) > 0 and unique_chars / len(text) < 0.1:
            flags.append("GIBBERISH_DETECTED")
        if text.isupper() and len(text) > 50:
            flags.append("ALL_CAPS")
        return len(flags) == 0, flags
    
    def _check_patterns(self, text: str, metrics: ComplaintMetrics) -> tuple:
        import re
        flags = []
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text):
                flags.append("SUSPICIOUS_PATTERN")
                break
        template_phrases = ["ваш сервис ужасен", "требую компенсацию", "буду жаловаться везде"]
        matches = sum(1 for phrase in template_phrases if phrase.lower() in text.lower())
        if matches >= 2:
            flags.append("TEMPLATE_LIKE_TEXT")
        return len(flags) == 0, flags
    
    def _check_source_history(self, fingerprint: SourceFingerprint) -> float:
        if fingerprint.complaints_count == 0:
            return 0.5
        if fingerprint.verified_visits > 0:
            return min(1.0, 0.6 + fingerprint.verified_visits * 0.1)
        if fingerprint.complaints_count > 10 and fingerprint.verified_visits == 0:
            return 0.3
        if fingerprint.spam_reports > 0:
            return max(0.1, 0.5 - fingerprint.spam_reports * 0.2)
        return 0.5
    
    def _check_correlation(self, text: str, metrics: ComplaintMetrics) -> List[str]:
        flags = []
        now = datetime.utcnow()
        recent_window = timedelta(hours=1)
        recent_similar = 0
        for complaint in self._complaint_history:
            if now - complaint["timestamp"] > recent_window:
                continue
            if complaint["category"] == metrics.category:
                if set(complaint["mentioned_persons"]) & set(metrics.mentioned_persons):
                    recent_similar += 1
        if recent_similar >= 3:
            flags.append("POTENTIAL_COORDINATED_ATTACK")
        return flags
    
    def _add_to_history(self, text: str, metrics: ComplaintMetrics, fingerprint: str):
        self._complaint_history.append({
            "timestamp": datetime.utcnow(),
            "category": metrics.category,
            "mentioned_persons": metrics.mentioned_persons,
            "fingerprint": fingerprint,
            "text_hash": hashlib.md5(text.encode()).hexdigest(),
        })
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self._complaint_history = [c for c in self._complaint_history if c["timestamp"] > cutoff]
    
    async def health_check(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": "ok",
            "tracked_fingerprints": len(self._fingerprints),
            "history_size": len(self._complaint_history),
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "initialized": self._initialized,
        }
    
    def report_spam(self, fingerprint_hash: str):
        if fingerprint_hash in self._fingerprints:
            self._fingerprints[fingerprint_hash].spam_reports += 1
    
    def verify_source(self, fingerprint_hash: str):
        if fingerprint_hash in self._fingerprints:
            self._fingerprints[fingerprint_hash].verified_visits += 1

