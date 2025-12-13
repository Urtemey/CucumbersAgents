"""Transcription Agent - ASR через Whisper."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

from agents.base import BaseAgent, AgentResult
from agents.config import get_agent_config
from agents.models import TranscriptionData


class TranscriptionAgent(BaseAgent):
    """
    Агент транскрипции аудио.
    
    Использует faster-whisper для быстрой и точной транскрипции
    с поддержкой кастомного словаря и шумоподавления.
    """
    
    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        compute_type: str = None,
        language: str = None,
    ):
        super().__init__("TranscriptionAgent")
        
        config = get_agent_config()
        self.model_size = model_size or config.whisper.model_size
        self.device = device or config.whisper.device
        self.compute_type = compute_type or config.whisper.compute_type
        self.default_language = language or config.whisper.language
        self._model = None
        
        # Кастомный словарь для улучшения распознавания
        self.custom_vocabulary = []
    
    async def initialize(self) -> bool:
        """Загрузка модели Whisper."""
        try:
            self.log_start(f"Loading Whisper model: {self.model_size}")
            
            from faster_whisper import WhisperModel
            
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            
            self.logger.info(f"Whisper model loaded: {self.model_size}")
            return True
            
        except Exception as e:
            self.log_error("initialize", e)
            return False
    
    async def process(
        self,
        audio_path: Path,
        language: str = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> AgentResult[TranscriptionData]:
        """
        Транскрибировать аудио файл.
        
        Args:
            audio_path: Путь к аудио файлу
            language: Язык (по умолчанию из конфига)
            beam_size: Размер beam search
            vad_filter: Использовать VAD фильтр
            
        Returns:
            AgentResult с TranscriptionData
        """
        start_time = time.time()
        
        try:
            await self.ensure_initialized()
            
            self.log_start(f"Transcribing: {audio_path}")
            
            if not audio_path.exists():
                return AgentResult.fail(f"Audio file not found: {audio_path}")
            
            # Транскрипция
            segments, info = self._model.transcribe(
                str(audio_path),
                language=language or self.default_language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                word_timestamps=True,
            )
            
            # Собираем результат
            segment_list = []
            text_parts = []
            
            for segment in segments:
                segment_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                        for w in (segment.words or [])
                    ]
                })
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts)
            
            # Вычисляем среднюю уверенность
            avg_confidence = 0.0
            if segment_list:
                word_probs = []
                for seg in segment_list:
                    word_probs.extend([w["probability"] for w in seg["words"]])
                if word_probs:
                    avg_confidence = sum(word_probs) / len(word_probs)
            
            processing_time = time.time() - start_time
            
            data = TranscriptionData(
                text=full_text,
                language=info.language,
                confidence=avg_confidence,
                duration=info.duration,
                segments=segment_list
            )
            
            self.log_complete("transcription", processing_time)
            
            result = AgentResult.ok(
                data=data,
                processing_time=processing_time,
                audio_duration=info.duration,
                language_probability=info.language_probability
            )
            
            # Предупреждения
            if avg_confidence < 0.7:
                result.add_warning(f"Low confidence transcription: {avg_confidence:.2f}")
            
            if info.duration < 1.0:
                result.add_warning("Very short audio (< 1 second)")
            
            return result
            
        except Exception as e:
            self.log_error("transcription", e)
            return AgentResult.fail(str(e))
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка состояния агента."""
        return {
            "name": self.name,
            "status": "ok" if self._model else "not_initialized",
            "model_size": self.model_size,
            "device": self.device,
            "initialized": self._initialized,
        }
    
    def add_custom_words(self, words: list):
        """Добавить слова в кастомный словарь."""
        self.custom_vocabulary.extend(words)
        self.logger.info(f"Added {len(words)} words to custom vocabulary")

