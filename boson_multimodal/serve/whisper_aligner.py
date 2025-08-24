"""
Whisper-based forced alignment for accurate word timestamps.
This provides real word-level timing based on audio analysis.
"""

import numpy as np
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

try:
    import torch
except ImportError:
    torch = None
    
try:
    import whisper
except ImportError:
    whisper = None
    logger.warning("Whisper not available, using fallback alignment")


@dataclass
class WordTiming:
    """Word-level timing information with Whisper confidence"""
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    

class WhisperAligner:
    """
    Use OpenAI's Whisper model for forced alignment.
    This gives us accurate word-level timestamps based on actual audio.
    """
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper model for alignment.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda/cpu)
        """
        if whisper is None:
            logger.warning("Whisper not available, will use fallback alignment")
            self.model = None
            self.device = "cpu"
            return
            
        if device is None:
            device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            
        self.device = device
        logger.info(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        logger.info("Whisper model loaded successfully")
        
    def align(
        self, 
        audio: np.ndarray, 
        text: str, 
        sampling_rate: int = 24000
    ) -> List[WordTiming]:
        """
        Perform forced alignment to get accurate word timestamps.
        
        Args:
            audio: Audio waveform as numpy array
            text: Text that was spoken (for conditioning/prompting)
            sampling_rate: Audio sampling rate (Higgs uses 24000)
            
        Returns:
            List of WordTiming objects with accurate timestamps
        """
        logger.info(f"Starting Whisper alignment for text: '{text[:50]}...'")
        
        # If Whisper not available, use fallback
        if self.model is None:
            logger.warning("Whisper not available, using simple fallback alignment")
            words = text.strip().split()
            if not words:
                return []
            
            duration_ms = (len(audio) / sampling_rate) * 1000
            ms_per_word = duration_ms / len(words)
            
            word_timings = []
            for i, word in enumerate(words):
                word_timings.append(WordTiming(
                    word=word,
                    start_ms=int(i * ms_per_word),
                    end_ms=int((i + 1) * ms_per_word),
                    confidence=0.5  # Low confidence for fallback
                ))
            return word_timings
        
        # Ensure audio is float32 and normalized
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize audio to [-1, 1] range if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
            
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sampling_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            logger.debug(f"Resampled audio from {sampling_rate}Hz to 16000Hz")
        
        # Run Whisper with word_timestamps enabled
        # We use the text as an initial prompt to improve accuracy
        result = self.model.transcribe(
            audio,
            word_timestamps=True,
            initial_prompt=text,  # Help Whisper with expected text
            language="en",  # Assuming English, could be detected
            temperature=0.0,  # Deterministic
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        
        # Extract word timings
        word_timings = []
        
        if "segments" in result:
            for segment in result["segments"]:
                if "words" in segment:
                    for word_info in segment["words"]:
                        word_timings.append(WordTiming(
                            word=word_info["word"].strip(),
                            start_ms=int(word_info["start"] * 1000),
                            end_ms=int(word_info["end"] * 1000),
                            confidence=word_info.get("probability", 0.9)
                        ))
        
        logger.info(f"Extracted {len(word_timings)} word timings from Whisper")
        
        # If Whisper didn't find words, fall back to simple distribution
        if not word_timings:
            logger.warning("Whisper didn't extract words, using fallback")
            words = text.strip().split()
            if words:
                # Calculate duration based on original sampling rate
                duration_ms = (len(audio) / 16000) * 1000  # Audio was resampled to 16kHz
                ms_per_word = duration_ms / len(words)
                
                for i, word in enumerate(words):
                    word_timings.append(WordTiming(
                        word=word,
                        start_ms=int(i * ms_per_word),
                        end_ms=int((i + 1) * ms_per_word),
                        confidence=0.5  # Low confidence for fallback
                    ))
        
        return word_timings


class CrossAttentionAligner:
    """
    Extract alignment from Higgs Audio's cross-attention weights.
    This would require modifying the model to expose attention weights.
    """
    
    @staticmethod
    def align_from_attention(
        attention_weights: np.ndarray,
        text_tokens: List[int],
        audio_duration_ms: float,
        tokenizer
    ) -> List[WordTiming]:
        """
        Extract word timings from cross-attention weights.
        
        Args:
            attention_weights: Shape [n_layers, n_heads, audio_frames, text_tokens]
            text_tokens: List of text token IDs
            audio_duration_ms: Total audio duration
            tokenizer: Tokenizer to decode tokens to words
            
        Returns:
            List of WordTiming objects
        """
        # Average attention across layers and heads
        avg_attention = attention_weights.mean(axis=(0, 1))  # [audio_frames, text_tokens]
        
        # For each text token, find where it has maximum attention
        n_frames = avg_attention.shape[0]
        frame_duration_ms = audio_duration_ms / n_frames
        
        word_timings = []
        current_word = ""
        word_start_frame = 0
        word_attention_sum = 0
        
        for token_idx, token_id in enumerate(text_tokens):
            token_text = tokenizer.decode([token_id])
            
            # Get attention for this token across all frames
            token_attention = avg_attention[:, token_idx]
            
            # Find peak attention frame
            peak_frame = np.argmax(token_attention)
            
            # Check if this starts a new word
            if token_text.startswith(' ') or not current_word:
                if current_word:
                    # Save previous word
                    word_timings.append(WordTiming(
                        word=current_word.strip(),
                        start_ms=int(word_start_frame * frame_duration_ms),
                        end_ms=int(peak_frame * frame_duration_ms),
                        confidence=float(word_attention_sum)
                    ))
                
                current_word = token_text.lstrip()
                word_start_frame = peak_frame
                word_attention_sum = token_attention.max()
            else:
                current_word += token_text
                word_attention_sum += token_attention.max()
        
        # Add last word
        if current_word:
            word_timings.append(WordTiming(
                word=current_word.strip(),
                start_ms=int(word_start_frame * frame_duration_ms),
                end_ms=int(audio_duration_ms),
                confidence=float(word_attention_sum)
            ))
        
        return word_timings


def create_whisper_aligned_engine(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    whisper_model_size: str = "base",
    **kwargs
) -> Tuple[Any, Any]:
    """
    Create a Higgs Audio engine with Whisper-based alignment.
    
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
    
    # Create Whisper aligner
    aligner = WhisperAligner(model_size=whisper_model_size, device=kwargs.get('device', 'cuda'))
    
    # Wrap the engine's generate method
    original_generate = engine.generate
    
    def generate_with_whisper_alignment(*args, return_timestamps=False, input_text=None, **kwargs):
        # Call original generate
        result = original_generate(*args, **kwargs)
        
        # Extract timestamps using Whisper if requested
        if return_timestamps and hasattr(result, 'audio') and result.audio is not None:
            # Use input text if provided, otherwise try to extract from args
            if not input_text and len(args) > 0:
                chat_ml_sample = args[0]
                if hasattr(chat_ml_sample, 'messages'):
                    for msg in reversed(chat_ml_sample.messages):
                        if hasattr(msg, 'role') and msg.role == 'user' and hasattr(msg, 'content'):
                            if isinstance(msg.content, str):
                                input_text = msg.content
                                break
            
            if input_text:
                logger.info(f"Running Whisper alignment for: '{input_text[:50]}...'")
                
                # Get word timings from Whisper
                word_timings = aligner.align(
                    audio=result.audio,
                    text=input_text,
                    sampling_rate=result.sampling_rate
                )
                
                # Add to result
                result.word_timings = word_timings
                result.alignment_method = "whisper_forced_alignment"
                
                logger.info(f"Added {len(word_timings)} Whisper-aligned word timings")
                if word_timings:
                    logger.info(f"Sample: {word_timings[0]}")
        
        return result
    
    engine.generate = generate_with_whisper_alignment
    
    return engine, aligner