"""
WhisperX-based forced alignment for accurate word timestamps.
This provides true forced alignment (not just ASR) for word-level timing.
"""

import numpy as np
from typing import List, Optional, Tuple, Any, Dict
from dataclasses import dataclass
from loguru import logger

try:
    import torch
except ImportError:
    torch = None
    
try:
    import whisperx
except ImportError:
    whisperx = None
    logger.warning("WhisperX not available, alignment will not work")


@dataclass
class WordTiming:
    """Word-level timing information with confidence"""
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    

class WhisperXAligner:
    """
    Use WhisperX for true forced alignment.
    This gives accurate word-level timestamps using phoneme alignment.
    """
    
    def __init__(self, model_size: str = "large-v2", device: str = None, language: str = "en"):
        """
        Initialize WhisperX model for alignment.
        
        Args:
            model_size: Whisper model size (base, small, medium, large, large-v2, large-v3)
            device: Device to run on (cuda/cpu)
            language: Language code for alignment model
        """
        if whisperx is None:
            logger.error("WhisperX not available")
            self.model = None
            self.align_model = None
            self.device = "cpu"
            return
            
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.language = language
        
        logger.info(f"Loading WhisperX {model_size} model on {device}...")
        
        # Load transcription model
        compute_type = "float16" if device == "cuda" else "int8"
        self.model = whisperx.load_model(
            model_size, 
            device=device,
            compute_type=compute_type,
            language=language
        )
        
        # Load alignment model
        logger.info(f"Loading alignment model for language: {language}")
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=language,
            device=device
        )
        
        logger.info("WhisperX models loaded successfully")
        
    def align(
        self, 
        audio: np.ndarray, 
        text: str, 
        sampling_rate: int = 24000,
        batch_size: int = 16  # WhisperX batch size for faster processing
    ) -> List[WordTiming]:
        """
        Perform forced alignment to get accurate word timestamps.
        
        Args:
            audio: Audio waveform as numpy array
            text: Text that was spoken (for forced alignment)
            sampling_rate: Audio sampling rate (Higgs uses 24000)
            batch_size: Batch size for WhisperX processing
            
        Returns:
            List of WordTiming objects with accurate timestamps
        """
        logger.info(f"Starting WhisperX alignment for text: '{text[:50]}...'")
        
        # Check if WhisperX is available
        if self.model is None or self.align_model is None:
            logger.error("WhisperX models not loaded")
            raise RuntimeError("WhisperX is required for alignment but is not available")
        
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio to [-1, 1] range if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
            
        # WhisperX expects 16kHz audio
        if sampling_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            logger.debug(f"Resampled audio from {sampling_rate}Hz to 16000Hz")
        
        try:
            # Step 1: Transcribe with Whisper (this gets rough timestamps)
            logger.info("Step 1: Transcribing with WhisperX...")
            result = self.model.transcribe(
                audio,
                batch_size=batch_size,
                language=self.language,
                # Provide the text as initial prompt to guide transcription
                initial_prompt=text
            )
            
            # Check if transcription matches expected text
            transcribed_text = " ".join([seg["text"].strip() for seg in result["segments"]])
            logger.info(f"Transcribed: '{transcribed_text[:100]}...'")
            
            # Step 2: Forced alignment with phoneme model
            logger.info("Step 2: Performing forced alignment...")
            
            # If transcription doesn't match well, we can optionally replace segments
            # with our known text for better alignment
            if self._calculate_similarity(text, transcribed_text) < 0.8:
                logger.warning("Transcription doesn't match well, using provided text for alignment")
                # Replace transcribed text with our known text
                result["segments"] = self._create_segments_from_text(text, len(audio) / 16000)
            
            # Align with phoneme model for accurate word timestamps
            result_aligned = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False  # We only need word alignments
            )
            
            # Step 3: Extract word timings
            word_timings = []
            for segment in result_aligned["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timings.append(WordTiming(
                            word=word_info["word"].strip(),
                            start_ms=int(word_info["start"] * 1000),
                            end_ms=int(word_info["end"] * 1000),
                            confidence=word_info.get("score", 0.9)  # WhisperX uses 'score'
                        ))
            
            logger.info(f"Extracted {len(word_timings)} word timings with WhisperX")
            
            # Verify alignment quality
            if word_timings:
                extracted_text = " ".join([wt.word for wt in word_timings])
                similarity = self._calculate_similarity(text, extracted_text)
                
                if similarity < 0.5:
                    logger.error(f"Poor alignment quality: {similarity:.1%} similarity")
                    logger.error(f"Expected: {text[:100]}...")
                    logger.error(f"Got: {extracted_text[:100]}...")
                    raise ValueError(f"WhisperX alignment failed with only {similarity:.1%} accuracy")
                else:
                    logger.info(f"Alignment successful with {similarity:.1%} similarity")
            
            return word_timings
            
        except Exception as e:
            logger.error(f"WhisperX alignment failed: {e}")
            raise RuntimeError(f"WhisperX alignment error: {e}")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-level similarity between two texts"""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Simple word overlap similarity
        matching = sum(1 for w1, w2 in zip(words1, words2) 
                      if w1.strip('.,!?;:') == w2.strip('.,!?;:'))
        
        max_len = max(len(words1), len(words2))
        return matching / max_len if max_len > 0 else 0.0
    
    def _create_segments_from_text(self, text: str, duration: float) -> List[Dict]:
        """Create segment structure from known text for forced alignment"""
        # Split text into chunks of ~50 words for segments
        words = text.split()
        chunk_size = 50
        segments = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Estimate timing (will be corrected by alignment)
            start_time = (i / len(words)) * duration
            end_time = ((i + len(chunk_words)) / len(words)) * duration
            
            segments.append({
                "text": chunk_text,
                "start": start_time,
                "end": end_time
            })
        
        return segments


def create_whisperx_aligned_engine(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    whisper_model_size: str = "large-v2",
    language: str = "en",
    **kwargs
) -> Tuple[Any, Any]:
    """
    Create a Higgs Audio engine with WhisperX-based alignment.
    
    Returns:
        Tuple of (engine, aligner)
    """
    from .serve_engine import HiggsAudioServeEngine
    
    # Create standard engine
    engine = HiggsAudioServeEngine(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_name_or_path=audio_tokenizer_name_or_path,
        **kwargs
    )
    
    # Create WhisperX aligner
    aligner = WhisperXAligner(
        model_size=whisper_model_size, 
        device=kwargs.get('device', 'cuda'),
        language=language
    )
    
    # Wrap the engine's generate method
    original_generate = engine.generate
    
    def generate_with_whisperx_alignment(*args, return_timestamps=False, input_text=None, **kwargs):
        # Call original generate
        result = original_generate(*args, **kwargs)
        
        # Extract timestamps using WhisperX if requested
        if return_timestamps and hasattr(result, 'audio') and result.audio is not None:
            # Use input text if provided, otherwise extract from args
            if not input_text and len(args) > 0:
                chat_ml_sample = args[0]
                if hasattr(chat_ml_sample, 'messages'):
                    for msg in reversed(chat_ml_sample.messages):
                        if hasattr(msg, 'role') and msg.role == 'user' and hasattr(msg, 'content'):
                            if isinstance(msg.content, str):
                                input_text = msg.content
                                break
            
            if input_text:
                logger.info(f"Running WhisperX alignment for: '{input_text[:50]}...'")
                
                try:
                    # Get word timings from WhisperX
                    word_timings = aligner.align(
                        audio=result.audio,
                        text=input_text,
                        sampling_rate=result.sampling_rate
                    )
                    
                    # Add to result
                    result.word_timings = word_timings
                    result.alignment_method = "whisperx_forced_alignment"
                    
                    logger.info(f"Added {len(word_timings)} WhisperX-aligned word timings")
                    if word_timings:
                        logger.info(f"Sample: {word_timings[0]}")
                except Exception as e:
                    logger.error(f"WhisperX alignment failed: {e}")
                    # Don't fail the whole request, just skip timestamps
                    result.word_timings = []
                    result.alignment_method = "failed"
                    result.alignment_error = str(e)
        
        return result
    
    engine.generate = generate_with_whisperx_alignment
    
    return engine, aligner