"""
Orchestrator - Координация агентов с LangChain.

Все текстовые агенты используют: qwen3-vl:8b
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field

from langchain.memory import ConversationBufferMemory

from complaintagents.base import BaseAgent, AgentResult
from complaintagents.llm_provider import get_ollama_provider
from complaintagents.transcription import TranscriptionAgent
from complaintagents.analyzer import AnalyzerAgent
from complaintagents.summarizer import SummarizerAgent
from complaintagents.router import RouterAgent
from complaintagents.antifraud import AntifraudAgent
from complaintagents.models import (
    VerificationLevel,
    IntakeChannel,
    TextArtifacts,
    ComplaintMetrics,
    RoutingDecision,
    FraudScore,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


class AgentOrchestrator(BaseAgent):
    """
    Оркестратор агентов с LangChain.
    
    Все текстовые агенты используют: qwen3-vl:8b
    
    Координирует работу агентов:
    1. Транскрипция (Whisper)
    2. Анализ (qwen3-vl:8b)
    3. Суммаризация (qwen3-vl:8b)
    4. Антифрод (правила + скоринг)
    5. Маршрутизация (правила)
    """
    
    # Единая модель для всех LLM агентов
    LLM_MODEL = "qwen3-vl:8b"
    
    def __init__(
        self,
        model_name: str = None,
        whisper_model: str = "base",
    ):
        super().__init__("AgentOrchestrator")
        
        self.model_name = model_name or self.LLM_MODEL
        self.whisper_model = whisper_model
        
        # Агенты (все LLM агенты используют qwen3-vl:8b)
        self.transcription_agent = TranscriptionAgent(model_size=whisper_model)
        self.analyzer_agent = AnalyzerAgent(model_name=self.LLM_MODEL)
        self.summarizer_agent = SummarizerAgent(model_name=self.LLM_MODEL)
        self.router_agent = RouterAgent()
        self.antifraud_agent = AntifraudAgent()
        
        self._agents = [
            self.transcription_agent,
            self.analyzer_agent,
            self.summarizer_agent,
            self.router_agent,
            self.antifraud_agent,
        ]
        
        # LangChain components
        self._provider = None
        self.session_memory = ConversationBufferMemory(
            memory_key="session_history",
            return_messages=True,
        )
    
    async def initialize(self) -> bool:
        """Инициализация всех агентов."""
        try:
            self.log_start("Initializing all agents")
            
            self._provider = get_ollama_provider()
            
            # Проверяем доступность Ollama
            health = await self._provider.check_health()
            if health["status"] != "ok":
                self.logger.warning(f"Ollama not available: {health}")
            
            # Инициализируем агентов
            for agent in self._agents:
                try:
                    success = await agent.initialize()
                    if not success:
                        self.logger.warning(f"Agent {agent.name} failed to initialize")
                except Exception as e:
                    self.logger.warning(f"Agent {agent.name} error: {e}")
            
            self.logger.info("Orchestrator initialization complete")
            return True
            
        except Exception as e:
            self.log_error("initialize", e)
            return False
    
    async def process_audio(
        self,
        audio_path: Path,
        intake_channel: IntakeChannel = IntakeChannel.API,
        verification_level: VerificationLevel = VerificationLevel.ANONYMOUS,
        source_id: str = None,
    ) -> AgentResult[ProcessingResult]:
        """
        Полная обработка аудио жалобы.
        """
        start_time = time.time()
        warnings = []
        
        try:
            self.log_start(f"Processing audio: {audio_path}")
            
            # 1. Транскрипция
            transcription_result = await self.transcription_agent.process(audio_path)
            
            if not transcription_result.success:
                return AgentResult.fail(f"Transcription failed: {transcription_result.error}")
            
            transcription_data = transcription_result.data
            warnings.extend(transcription_result.warnings)
            
            # 2. Обработка текста
            return await self._process_text_pipeline(
                original_text=transcription_data.text,
                language=transcription_data.language,
                audio_duration=transcription_data.duration,
                transcription_time=transcription_result.processing_time,
                intake_channel=intake_channel,
                verification_level=verification_level,
                source_id=source_id,
                audio_file_path=str(audio_path),
                existing_warnings=warnings,
                start_time=start_time,
            )
            
        except Exception as e:
            self.log_error("process_audio", e)
            return AgentResult.fail(str(e))
    
    async def process_text(
        self,
        text: str,
        intake_channel: IntakeChannel = IntakeChannel.WEB_FORM,
        verification_level: VerificationLevel = VerificationLevel.ANONYMOUS,
        source_id: str = None,
    ) -> AgentResult[ProcessingResult]:
        """
        Обработка текстовой жалобы.
        """
        start_time = time.time()
        
        try:
            self.log_start(f"Processing text ({len(text)} chars)")
            
            return await self._process_text_pipeline(
                original_text=text,
                language="ru",
                audio_duration=None,
                transcription_time=0.0,
                intake_channel=intake_channel,
                verification_level=verification_level,
                source_id=source_id,
                audio_file_path=None,
                existing_warnings=[],
                start_time=start_time,
            )
            
        except Exception as e:
            self.log_error("process_text", e)
            return AgentResult.fail(str(e))
    
    async def _process_text_pipeline(
        self,
        original_text: str,
        language: str,
        audio_duration: Optional[float],
        transcription_time: float,
        intake_channel: IntakeChannel,
        verification_level: VerificationLevel,
        source_id: str,
        audio_file_path: Optional[str],
        existing_warnings: List[str],
        start_time: float,
    ) -> AgentResult[ProcessingResult]:
        """Внутренний пайплайн обработки текста."""
        warnings = existing_warnings.copy()
        
        # 2. Анализ
        analysis_result = await self.analyzer_agent.process(original_text)
        
        if not analysis_result.success:
            return AgentResult.fail(f"Analysis failed: {analysis_result.error}")
        
        metrics = analysis_result.data
        warnings.extend(analysis_result.warnings)
        
        # 3. Суммаризация
        summarization_result = await self.summarizer_agent.process(
            original_text=original_text,
            language=language,
            audio_duration=audio_duration,
            transcription_time=transcription_time,
        )
        
        if not summarization_result.success:
            text_artifacts = TextArtifacts(
                original=original_text,
                normalized=original_text,
                neutral=original_text,
                language=language,
            )
            warnings.append("Summarization failed, using original text")
        else:
            text_artifacts = summarization_result.data
            warnings.extend(summarization_result.warnings)
        
        # 4. Антифрод
        antifraud_result = await self.antifraud_agent.process(
            text=original_text,
            metrics=metrics,
            verification_level=verification_level,
            source_id=source_id,
        )
        
        fraud_score = antifraud_result.data if antifraud_result.success else None
        warnings.extend(antifraud_result.warnings)
        
        if fraud_score:
            metrics.credibility_score = fraud_score.credibility_score
        
        # 5. Маршрутизация
        routing_result = await self.router_agent.process(metrics)
        routing_decision = routing_result.data if routing_result.success else None
        warnings.extend(routing_result.warnings)
        
        total_time = time.time() - start_time
        
        result = ProcessingResult(
            text_artifacts=text_artifacts,
            metrics=metrics,
            routing_decision=routing_decision,
            fraud_score=fraud_score,
            transcription_time=transcription_time,
            analysis_time=analysis_result.processing_time,
            summarization_time=summarization_result.processing_time if summarization_result.success else 0.0,
            total_time=total_time,
            audio_file_path=audio_file_path,
            intake_channel=intake_channel,
            warnings=warnings,
        )
        
        self.log_complete("pipeline", total_time)
        
        # Сохраняем в память сессии
        self.session_memory.save_context(
            {"input": f"Processed complaint: {original_text[:100]}..."},
            {"output": f"Category: {metrics.category}, Urgency: {metrics.urgency}"}
        )
        
        return AgentResult.ok(data=result, processing_time=total_time)
    
    async def process(self, *args, **kwargs) -> AgentResult:
        """Общий метод обработки."""
        if "audio_path" in kwargs:
            return await self.process_audio(**kwargs)
        elif "text" in kwargs:
            return await self.process_text(**kwargs)
        else:
            return AgentResult.fail("Either audio_path or text must be provided")
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния всех агентов."""
        agent_statuses = {}
        overall_status = "ok"
        
        if self._provider:
            ollama_health = await self._provider.check_health()
            agent_statuses["ollama"] = ollama_health
            if ollama_health["status"] != "ok":
                overall_status = "degraded"
        
        for agent in self._agents:
            try:
                status = await agent.health_check()
                agent_statuses[agent.name] = status
                
                if status.get("status") == "error":
                    overall_status = "error"
                elif status.get("status") == "degraded" and overall_status == "ok":
                    overall_status = "degraded"
            except Exception as e:
                agent_statuses[agent.name] = {"status": "error", "error": str(e)}
                overall_status = "error"
        
        return {
            "name": self.name,
            "status": overall_status,
            "model": self.model_name,
            "agents": agent_statuses,
            "initialized": self._initialized,
        }
    
    def get_all_tools(self) -> List[Any]:
        """Получить все инструменты от всех агентов."""
        tools = []
        for agent in self._agents:
            if hasattr(agent, 'get_tools'):
                tools.extend(agent.get_tools())
        return tools
    
    def clear_session(self):
        """Очистить память сессии."""
        self.session_memory.clear()
        for agent in self._agents:
            agent.clear_memory()

