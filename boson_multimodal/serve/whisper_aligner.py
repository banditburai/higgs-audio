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
        sampling_rate: int = 24000,
        chunk_size_words: int = 30  # Process in chunks of 30 words
    ) -> List[WordTiming]:
        """
        Perform forced alignment to get accurate word timestamps.
        Uses chunked processing for long text to improve Whisper accuracy.
        
        Args:
            audio: Audio waveform as numpy array
            text: Text that was spoken (for conditioning/prompting)
            sampling_rate: Audio sampling rate (Higgs uses 24000)
            chunk_size_words: Number of words to process at once
            
        Returns:
            List of WordTiming objects with accurate timestamps
        """
        logger.info(f"Starting Whisper alignment for text: '{text[:50]}...'")
        
        # If Whisper not available, use fallback
        if self.model is None:
            logger.warning("Whisper not available, using simple fallback alignment")
            return self._simple_alignment_fallback(text, len(audio) / sampling_rate * 1000)
        
        # Prepare audio for Whisper
        audio_16k = self._prepare_audio_for_whisper(audio, sampling_rate)
        
        # Split text into chunks for better accuracy
        words = text.strip().split()
        total_duration_ms = (len(audio_16k) / 16000) * 1000
        
        # If text is short enough, process all at once
        if len(words) <= chunk_size_words:
            return self._process_single_chunk(audio_16k, text, total_duration_ms)
        
        # For long text, process in chunks
        logger.info(f"Processing {len(words)} words in chunks of {chunk_size_words}")
        all_word_timings = []
        
        # Calculate approximate duration per word for chunking audio
        ms_per_word_estimate = total_duration_ms / len(words)
        samples_per_word = int(ms_per_word_estimate * 16)  # 16 samples per ms at 16kHz
        
        for chunk_start in range(0, len(words), chunk_size_words):
            chunk_end = min(chunk_start + chunk_size_words, len(words))
            chunk_words = words[chunk_start:chunk_end]
            chunk_text = ' '.join(chunk_words)
            
            # Extract corresponding audio chunk with some overlap
            audio_start_sample = max(0, int(chunk_start * samples_per_word * 0.95))  # 5% overlap
            audio_end_sample = min(len(audio_16k), int(chunk_end * samples_per_word * 1.05))  # 5% overlap
            audio_chunk = audio_16k[audio_start_sample:audio_end_sample]
            
            # Calculate offset for this chunk
            chunk_offset_ms = (audio_start_sample / 16000) * 1000
            
            logger.debug(f"Processing chunk {chunk_start//chunk_size_words + 1}: words {chunk_start}-{chunk_end}")
            
            # Process this chunk
            chunk_timings = self._process_single_chunk(
                audio_chunk, 
                chunk_text,
                (len(audio_chunk) / 16000) * 1000,
                offset_ms=chunk_offset_ms
            )
            
            # Check if chunk processing succeeded
            if chunk_timings and len(chunk_timings) > 0:
                # Verify the chunk matches expected words
                chunk_match = self._verify_chunk_match(chunk_words, chunk_timings)
                if chunk_match > 0.5:  # More than 50% match
                    all_word_timings.extend(chunk_timings)
                else:
                    # Fallback for this chunk
                    logger.warning(f"Chunk {chunk_start//chunk_size_words + 1} failed, using fallback")
                    chunk_duration = (audio_end_sample - audio_start_sample) / 16000 * 1000
                    fallback_timings = self._simple_alignment_fallback(
                        chunk_text, 
                        chunk_duration,
                        offset_ms=chunk_offset_ms
                    )
                    all_word_timings.extend(fallback_timings)
            else:
                # Use fallback for this chunk
                chunk_duration = (audio_end_sample - audio_start_sample) / 16000 * 1000
                fallback_timings = self._simple_alignment_fallback(
                    chunk_text,
                    chunk_duration, 
                    offset_ms=chunk_offset_ms
                )
                all_word_timings.extend(fallback_timings)
        
        logger.info(f"Completed alignment: {len(all_word_timings)} word timings extracted")
        return all_word_timings
    
    def _prepare_audio_for_whisper(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Prepare audio for Whisper processing (float32, normalized, 16kHz)"""
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
            
        return audio
    
    def _process_single_chunk(
        self, 
        audio_16k: np.ndarray, 
        text: str, 
        duration_ms: float,
        offset_ms: float = 0
    ) -> List[WordTiming]:
        """Process a single chunk of audio/text through Whisper"""
        try:
            # Run Whisper with word_timestamps enabled
            result = self.model.transcribe(
                audio_16k,
                word_timestamps=True,
                initial_prompt=text,  # Help Whisper with expected text
                language="en",
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
                                start_ms=int(word_info["start"] * 1000 + offset_ms),
                                end_ms=int(word_info["end"] * 1000 + offset_ms),
                                confidence=word_info.get("probability", 0.9)
                            ))
            
            # Check if Whisper's transcription matches the input text
            whisper_text = result.get("text", "").strip()
            input_words = text.strip().split()
            whisper_words = whisper_text.split()
            
            # Calculate similarity
            matching_words = sum(1 for w1, w2 in zip(input_words, whisper_words) 
                               if w1.lower().strip('.,!?;:') == w2.lower().strip('.,!?;:'))
            match_ratio = matching_words / len(input_words) if input_words else 0
            
            logger.debug(f"Chunk match ratio: {match_ratio:.2%}")
            
            # If poor match, return empty to trigger fallback
            if match_ratio < 0.5:
                logger.warning(f"Poor Whisper match ({match_ratio:.2%}), will use fallback")
                return []
                
            return word_timings
            
        except Exception as e:
            logger.error(f"Whisper processing failed: {e}")
            return []
    
    def _verify_chunk_match(self, expected_words: List[str], timings: List[WordTiming]) -> float:
        """Verify that extracted timings match expected words"""
        if not timings:
            return 0.0
            
        timing_words = [t.word for t in timings]
        matches = sum(1 for e, t in zip(expected_words, timing_words)
                     if e.lower().strip('.,!?;:') == t.lower().strip('.,!?;:'))
        return matches / len(expected_words) if expected_words else 0.0
    
    def _simple_alignment_fallback(
        self, 
        text: str, 
        duration_ms: float,
        offset_ms: float = 0
    ) -> List[WordTiming]:
        """Simple word distribution fallback when Whisper fails"""
        words = text.strip().split()
        if not words:
            return []
        
        ms_per_word = duration_ms / len(words)
        
        word_timings = []
        for i, word in enumerate(words):
            word_timings.append(WordTiming(
                word=word,
                start_ms=int(i * ms_per_word + offset_ms),
                end_ms=int((i + 1) * ms_per_word + offset_ms),
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