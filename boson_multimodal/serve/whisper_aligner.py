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
        chunk_size_words: int = 30,  # Process in chunks of 30 words
        use_silence_detection: bool = True  # Use silence to find natural boundaries
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
        
        # If Whisper not available, fail explicitly
        if self.model is None:
            logger.error("Whisper model not loaded")
            raise RuntimeError("Whisper is required for alignment but is not available")
        
        # Prepare audio for Whisper
        audio_16k = self._prepare_audio_for_whisper(audio, sampling_rate)
        
        # Split text into chunks for better accuracy
        words = text.strip().split()
        total_duration_ms = (len(audio_16k) / 16000) * 1000
        
        # If text is short enough, process all at once
        if len(words) <= chunk_size_words:
            result = self._process_single_chunk(audio_16k, text, total_duration_ms)
            if result and len(result) > 0:
                match_ratio = self._verify_chunk_match(words, result)
                if match_ratio > 0.5:
                    logger.info(f"Whisper succeeded with {match_ratio:.1%} match")
                    return result
                else:
                    logger.error(f"Whisper alignment failed - only {match_ratio:.1%} word match")
                    raise ValueError(f"Whisper alignment failed with only {match_ratio:.1%} accuracy.")
            else:
                logger.error("Whisper failed to extract any word timings")
                raise ValueError("Whisper alignment failed. Check audio quality and text accuracy.")
        
        # For long text, try different strategies
        logger.warning(f"Text has {len(words)} words, which may be challenging for Whisper")
        
        if use_silence_detection and len(words) > chunk_size_words:
            # Try to use silence detection to find natural boundaries
            logger.info("Attempting silence-based chunking for better accuracy")
            segments = self._segment_audio_by_silence(audio_16k)
            
            if len(segments) > 1:
                logger.info(f"Found {len(segments)} audio segments based on silence")
                return self._process_with_silence_segments(segments, words, text)
        
        # Fall back to processing full audio
        logger.info("Processing full audio with Whisper (no chunking)")
        result = self._process_single_chunk(audio_16k, text, total_duration_ms)
        
        if result and len(result) > 0:
            # Verify alignment quality
            match_ratio = self._verify_chunk_match(words, result)
            if match_ratio > 0.5:
                logger.info(f"Whisper succeeded with {match_ratio:.1%} match")
                return result
            else:
                logger.error(f"Whisper alignment failed - only {match_ratio:.1%} word match")
                logger.error(f"Expected: {' '.join(words[:10])}...")
                logger.error(f"Got: {' '.join([t.word for t in result[:10]])}...")
                # Don't use fallback - fail explicitly
                raise ValueError(f"Whisper alignment failed with only {match_ratio:.1%} accuracy. "
                               f"Text may be too long or voice may be unclear.")
        else:
            logger.error("Whisper failed to extract any word timings")
            raise ValueError("Whisper alignment failed completely. Text may be too long or audio quality issues.")
    
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
            
            # If poor match, return empty to indicate failure
            if match_ratio < 0.5:
                logger.warning(f"Poor Whisper match ({match_ratio:.2%})")
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
    
    def _segment_audio_by_silence(self, audio_16k: np.ndarray, silence_thresh_db: float = -40, min_silence_ms: int = 500) -> List[Tuple[int, int, np.ndarray]]:
        """
        Segment audio by detecting silence gaps.
        
        Returns:
            List of (start_sample, end_sample, audio_segment) tuples
        """
        import librosa
        
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio_16k), ref=np.max)
        
        # Find silent regions
        silence_mask = audio_db < silence_thresh_db
        min_silence_samples = int(min_silence_ms * 16)  # 16 samples per ms at 16kHz
        
        segments = []
        in_silence = False
        silence_start = 0
        segment_start = 0
        
        for i in range(len(silence_mask)):
            if silence_mask[i] and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = i
            elif not silence_mask[i] and in_silence:
                # End of silence
                silence_duration = i - silence_start
                if silence_duration >= min_silence_samples:
                    # This is a significant silence gap - split here
                    if silence_start > segment_start:
                        segments.append((segment_start, silence_start, audio_16k[segment_start:silence_start]))
                    segment_start = i
                in_silence = False
        
        # Add final segment
        if segment_start < len(audio_16k):
            segments.append((segment_start, len(audio_16k), audio_16k[segment_start:]))
        
        return segments
    
    def _process_with_silence_segments(self, segments: List[Tuple[int, int, np.ndarray]], words: List[str], full_text: str) -> List[WordTiming]:
        """
        Process audio segments with Whisper, distributing words appropriately.
        """
        all_timings = []
        words_per_segment = len(words) // len(segments)
        word_index = 0
        
        for seg_idx, (start_sample, end_sample, audio_segment) in enumerate(segments):
            # Calculate timing offset for this segment
            offset_ms = (start_sample / 16000) * 1000
            segment_duration_ms = (len(audio_segment) / 16000) * 1000
            
            # Estimate how many words should be in this segment
            if seg_idx < len(segments) - 1:
                segment_words = words[word_index:word_index + words_per_segment]
            else:
                # Last segment gets remaining words
                segment_words = words[word_index:]
            
            segment_text = ' '.join(segment_words)
            
            logger.debug(f"Processing segment {seg_idx + 1}/{len(segments)}: {len(segment_words)} words")
            
            # Process this segment
            segment_timings = self._process_single_chunk(
                audio_segment,
                segment_text,
                segment_duration_ms,
                offset_ms=offset_ms
            )
            
            if segment_timings and len(segment_timings) > 0:
                # Verify match
                match_ratio = self._verify_chunk_match(segment_words, segment_timings)
                if match_ratio > 0.5:
                    all_timings.extend(segment_timings)
                    word_index += len(segment_words)
                else:
                    logger.warning(f"Segment {seg_idx + 1} had poor match ({match_ratio:.1%})")
                    # Still fail rather than fallback
                    raise ValueError(f"Segment {seg_idx + 1} alignment failed with {match_ratio:.1%} accuracy")
            else:
                raise ValueError(f"Segment {seg_idx + 1} failed to produce any timings")
        
        return all_timings
    
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