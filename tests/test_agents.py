"""
Tests for LangChain agents.

Тесты для LangChain агентов.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from complaintagents.base import BaseAgent, AgentResult
from complaintagents.analyzer import AnalyzerAgent
from complaintagents.summarizer import SummarizerAgent
from complaintagents.router import RouterAgent
from complaintagents.antifraud import AntifraudAgent
from complaintagents.transcription import TranscriptionAgent
from complaintagents.orchestrator import AgentOrchestrator
from complaintagents.models import (
    ComplaintMetrics,
    ComplaintCategory,
    SentimentLevel,
    UrgencyLevel,
    VerificationLevel,
    IntakeChannel,
)


class TestAgentResult:
    """Тесты для AgentResult."""
    
    def test_ok_result(self):
        """Тест успешного результата."""
        result = AgentResult.ok(data="test_data", processing_time=1.5)
        
        assert result.success is True
        assert result.data == "test_data"
        assert result.processing_time == 1.5
        assert result.error is None
    
    def test_fail_result(self):
        """Тест неуспешного результата."""
        result = AgentResult.fail(error="Test error")
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.data is None
    
    def test_add_warning(self):
        """Тест добавления предупреждения."""
        result = AgentResult.ok(data="test")
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        
        assert len(result.warnings) == 2
        assert "Warning 1" in result.warnings


class TestRouterAgent:
    """Тесты для RouterAgent."""
    
    @pytest.fixture
    def router_agent(self):
        """Создать RouterAgent для тестов."""
        return RouterAgent()
    
    @pytest.mark.asyncio
    async def test_initialize(self, router_agent):
        """Тест инициализации."""
        result = await router_agent.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_route_medical_complaint(self, router_agent):
        """Тест маршрутизации медицинской жалобы."""
        await router_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.3,
            urgency=UrgencyLevel.MEDIUM,
            category=ComplaintCategory.MEDICAL,
        )
        
        result = await router_agent.process(metrics)
        
        assert result.success is True
        assert result.data.department == "medical_quality"
        assert result.data.responsible_role == "quality_manager"
    
    @pytest.mark.asyncio
    async def test_route_with_escalation(self, router_agent):
        """Тест маршрутизации с эскалацией."""
        await router_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.VERY_NEGATIVE,
            toxicity_score=0.9,
            urgency=UrgencyLevel.CRITICAL,
            category=ComplaintCategory.MEDICAL,
            severity_score=0.95,
        )
        
        result = await router_agent.process(metrics)
        
        assert result.success is True
        assert result.data.escalation_required is True
        assert result.data.sla_hours == 2
    
    @pytest.mark.asyncio
    async def test_route_with_accusations(self, router_agent):
        """Тест маршрутизации с обвинениями."""
        await router_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.5,
            urgency=UrgencyLevel.HIGH,
            category=ComplaintCategory.MEDICAL,
            contains_accusations=True,
            mentioned_persons=["Врач Петров"],
        )
        
        result = await router_agent.process(metrics)
        
        assert result.success is True
        assert "legal" in result.data.additional_departments
        assert "hr_department" in result.data.additional_departments
    
    @pytest.mark.asyncio
    async def test_health_check(self, router_agent):
        """Тест проверки состояния."""
        await router_agent.initialize()
        
        health = await router_agent.health_check()
        
        assert health["status"] == "ok"
        assert health["departments_configured"] == 8
        assert health["escalation_rules"] == 5


class TestAntifraudAgent:
    """Тесты для AntifraudAgent."""
    
    @pytest.fixture
    def antifraud_agent(self):
        """Создать AntifraudAgent для тестов."""
        agent = AntifraudAgent(
            rate_limit_per_hour=5,
            rate_limit_per_day=20,
            min_text_length=10,
        )
        return agent
    
    @pytest.mark.asyncio
    async def test_initialize(self, antifraud_agent):
        """Тест инициализации."""
        result = await antifraud_agent.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_process_normal_complaint(self, antifraud_agent):
        """Тест обработки нормальной жалобы."""
        await antifraud_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.3,
            urgency=UrgencyLevel.MEDIUM,
            category=ComplaintCategory.MEDICAL,
        )
        
        result = await antifraud_agent.process(
            text="Нормальная жалоба на качество обслуживания в поликлинике.",
            metrics=metrics,
            verification_level=VerificationLevel.ANONYMOUS,
        )
        
        assert result.success is True
        assert result.data.is_suspicious is False
        assert result.data.rate_limit_ok is True
    
    @pytest.mark.asyncio
    async def test_short_text_flag(self, antifraud_agent):
        """Тест флага короткого текста."""
        await antifraud_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.0,
            urgency=UrgencyLevel.LOW,
            category=ComplaintCategory.OTHER,
        )
        
        result = await antifraud_agent.process(
            text="Плохо",  # Слишком коротко
            metrics=metrics,
        )
        
        assert result.success is True
        assert "SHORT_TEXT" in result.data.flags
    
    @pytest.mark.asyncio
    async def test_rate_limit(self, antifraud_agent):
        """Тест rate limit."""
        await antifraud_agent.initialize()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.0,
            urgency=UrgencyLevel.LOW,
            category=ComplaintCategory.OTHER,
        )
        
        # Отправляем больше жалоб чем лимит
        for i in range(6):
            result = await antifraud_agent.process(
                text=f"Жалоба номер {i} на качество обслуживания",
                metrics=metrics,
                source_id="test_source",
            )
        
        # Последняя должна превысить лимит
        assert "RATE_LIMIT_EXCEEDED" in result.data.flags
    
    @pytest.mark.asyncio
    async def test_verification_weight(self, antifraud_agent):
        """Тест веса верификации."""
        await antifraud_agent.initialize()
        antifraud_agent.clear_history()
        
        metrics = ComplaintMetrics(
            sentiment=SentimentLevel.NEGATIVE,
            toxicity_score=0.0,
            urgency=UrgencyLevel.LOW,
            category=ComplaintCategory.OTHER,
        )
        
        # Анонимная жалоба
        result_anon = await antifraud_agent.process(
            text="Анонимная жалоба на качество обслуживания",
            metrics=metrics,
            verification_level=VerificationLevel.ANONYMOUS,
            source_id="anon_test",
        )
        
        antifraud_agent.clear_history()
        
        # Верифицированная жалоба
        result_verified = await antifraud_agent.process(
            text="Верифицированная жалоба на качество обслуживания",
            metrics=metrics,
            verification_level=VerificationLevel.IDENTIFIED,
            source_id="verified_test",
        )
        
        # Верифицированная должна иметь выше credibility
        assert result_verified.data.credibility_score > result_anon.data.credibility_score


class TestTranscriptionAgent:
    """Тесты для TranscriptionAgent."""
    
    @pytest.fixture
    def transcription_agent(self, mock_whisper_model):
        """Создать TranscriptionAgent с mock."""
        agent = TranscriptionAgent(model_size="tiny")
        return agent
    
    @pytest.mark.asyncio
    async def test_initialize(self, transcription_agent, mock_whisper_model):
        """Тест инициализации."""
        result = await transcription_agent.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_process_missing_file(self, transcription_agent):
        """Тест обработки несуществующего файла."""
        await transcription_agent.initialize()
        
        result = await transcription_agent.process(Path("/nonexistent/file.wav"))
        
        assert result.success is False
        assert "not found" in result.error
    
    @pytest.mark.asyncio
    async def test_health_check(self, transcription_agent, mock_whisper_model):
        """Тест проверки состояния."""
        await transcription_agent.initialize()
        
        health = await transcription_agent.health_check()
        
        assert health["model_size"] == "tiny"


class TestAnalyzerAgent:
    """Тесты для AnalyzerAgent."""
    
    @pytest.mark.asyncio
    async def test_short_text_rejection(self):
        """Тест отклонения слишком короткого текста."""
        with patch('complaintagents.llm_provider.get_ollama_provider') as mock:
            mock_provider = MagicMock()
            mock.return_value = mock_provider
            
            agent = AnalyzerAgent()
            agent._initialized = True
            agent._chain = MagicMock()
            
            result = await agent.process("Ок")
            
            assert result.success is False
            assert "too short" in result.error


class TestOrchestratorAgent:
    """Тесты для AgentOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Создать оркестратор для тестов."""
        with patch('complaintagents.llm_provider.get_ollama_provider'):
            return AgentOrchestrator()
    
    def test_agents_initialization(self, orchestrator):
        """Тест что все агенты инициализированы."""
        assert orchestrator.transcription_agent is not None
        assert orchestrator.analyzer_agent is not None
        assert orchestrator.summarizer_agent is not None
        assert orchestrator.router_agent is not None
        assert orchestrator.antifraud_agent is not None
    
    def test_get_all_tools(self, orchestrator):
        """Тест получения всех инструментов."""
        # Добавляем инструменты в analyzer
        orchestrator.analyzer_agent.tools = [MagicMock(), MagicMock()]
        
        tools = orchestrator.get_all_tools()
        
        assert len(tools) == 2
    
    @pytest.mark.asyncio
    async def test_process_without_args(self, orchestrator):
        """Тест вызова process без аргументов."""
        result = await orchestrator.process()
        
        assert result.success is False
        assert "audio_path or text must be provided" in result.error

