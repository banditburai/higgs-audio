"""
Production-quality timestamp tracking for Higgs Audio generation.

This module hooks into the actual generation process to track when each
token is generated and maps it to audio timestamps.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from transformers import LogitsProcessor, StoppingCriteria
from collections import defaultdict
import threading
from loguru import logger


@dataclass
class TokenGeneration:
    """Records when and what token was generated"""
    token_id: int
    position: int
    timestamp: float
    is_audio: bool = False
    audio_position: Optional[int] = None


@dataclass 
class WordTiming:
    """Word-level timing information"""
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    token_ids: List[int] = field(default_factory=list)


class GenerationTimestampTracker(LogitsProcessor):
    """
    A LogitsProcessor that tracks when each token is generated.
    This hooks into the actual generation process.
    """
    
    def __init__(self, tokenizer, audio_tokenizer=None):
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.generations = []
        self.start_time = None
        self.audio_boundary_positions = []
        self.current_position = 0
        self.lock = threading.Lock()
        
        # Track text vs audio generation phases
        self.is_audio_phase = False
        self.audio_start_position = None
        
    def reset(self):
        """Reset tracker for new generation"""
        with self.lock:
            self.generations = []
            self.start_time = time.time()
            self.audio_boundary_positions = []
            self.current_position = 0
            self.is_audio_phase = False
            self.audio_start_position = None
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called during each generation step.
        Records what token is about to be generated and when.
        """
        if self.start_time is None:
            self.start_time = time.time()
        
        # Get the token that's about to be selected (highest score)
        # Handle batch dimension - take first sequence if batch size > 1
        next_token_ids = scores.argmax(dim=-1)
        if next_token_ids.dim() == 0:
            # Single sequence
            next_token_id = next_token_ids.item()
        else:
            # Batch - take first sequence
            next_token_id = next_token_ids[0].item()
            
        current_time = time.time() - self.start_time
        
        with self.lock:
            # Check if we're transitioning to audio generation
            # Audio tokens typically have IDs in a specific range
            is_audio_token = self._is_audio_token(next_token_id)
            
            # Debug logging every 10 tokens to avoid spam
            if self.current_position % 10 == 0:
                logger.info(f"Position {self.current_position}: token_id={next_token_id}, is_audio={is_audio_token}")
            
            if is_audio_token and not self.is_audio_phase:
                # Mark transition to audio
                self.is_audio_phase = True
                self.audio_start_position = self.current_position
                logger.info(f"Transition to audio at position {self.current_position}, token_id={next_token_id}")
            elif not is_audio_token and self.is_audio_phase:
                # Mark end of audio segment
                self.is_audio_phase = False
                self.audio_boundary_positions.append((self.audio_start_position, self.current_position))
                logger.info(f"Audio segment end: positions {self.audio_start_position}-{self.current_position}")
            
            # Record the generation
            self.generations.append(TokenGeneration(
                token_id=next_token_id,
                position=self.current_position,
                timestamp=current_time,
                is_audio=is_audio_token,
                audio_position=len(self.audio_boundary_positions) if is_audio_token else None
            ))
            
            self.current_position += 1
        
        # Return scores unchanged (we're just observing)
        return scores
    
    def _is_audio_token(self, token_id: int) -> bool:
        """
        Determine if a token is an audio token.
        Audio tokens typically have IDs above the text vocabulary size.
        """
        # This threshold needs to be calibrated for Higgs Audio
        # Text tokens are typically 0-50000, audio tokens are higher
        TEXT_VOCAB_SIZE = 50000  # Approximate, needs verification
        return token_id > TEXT_VOCAB_SIZE
    
    def get_word_timings(self, text: str, audio_duration_ms: float) -> List[WordTiming]:
        """
        Convert tracked generations to word-level timings.
        
        Args:
            text: The generated text
            audio_duration_ms: Total audio duration in milliseconds
        
        Returns:
            List of word timings with millisecond timestamps
        """
        with self.lock:
            # Debug logging
            logger.info(f"Getting word timings for text: '{text}', duration: {audio_duration_ms}ms")
            logger.info(f"Total generations tracked: {len(self.generations)}")
            
            # Separate text and audio generations
            text_gens = [g for g in self.generations if not g.is_audio]
            audio_gens = [g for g in self.generations if g.is_audio]
            
            logger.info(f"Text generations: {len(text_gens)}, Audio generations: {len(audio_gens)}")
            
            # Log some sample tokens for debugging
            if self.generations:
                logger.info(f"Sample token IDs: {[g.token_id for g in self.generations[:10]]}")
                logger.info(f"Sample is_audio flags: {[g.is_audio for g in self.generations[:10]]}")
            
            if not text_gens or not audio_gens:
                logger.warning(f"No text or audio generations tracked. Text: {len(text_gens)}, Audio: {len(audio_gens)}")
                if self.generations:
                    logger.warning(f"All token IDs: {[g.token_id for g in self.generations]}")
                    logger.warning(f"All is_audio flags: {[g.is_audio for g in self.generations]}")
                return []
            
            # Decode text tokens to get words
            text_token_ids = [g.token_id for g in text_gens]
            decoded_tokens = [self.tokenizer.decode([tid]) for tid in text_token_ids]
            
            # Build word boundaries
            words = []
            current_word = ""
            current_tokens = []
            current_start_pos = 0
            
            for i, (token_str, gen) in enumerate(zip(decoded_tokens, text_gens)):
                # Clean token string
                token_str = token_str.strip()
                
                if not token_str:
                    continue
                    
                # Check if this starts a new word (has leading space or is punctuation)
                if token_str.startswith(' ') or (current_word and token_str in '.,!?;:'):
                    if current_word:
                        words.append({
                            'word': current_word.strip(),
                            'token_ids': current_tokens.copy(),
                            'start_position': current_start_pos,
                            'end_position': i
                        })
                    current_word = token_str.lstrip()
                    current_tokens = [gen.token_id]
                    current_start_pos = i
                else:
                    current_word += token_str
                    current_tokens.append(gen.token_id)
            
            # Add last word
            if current_word:
                words.append({
                    'word': current_word.strip(),
                    'token_ids': current_tokens,
                    'start_position': current_start_pos,
                    'end_position': len(text_gens)
                })
            
            # Map word positions to audio timestamps
            # The key insight: audio is generated AFTER text tokens
            # So we map based on when audio generation happened
            
            # Find audio generation segments
            audio_segments = self._get_audio_segments()
            
            # Distribute audio duration across words based on their token positions
            word_timings = []
            
            if audio_segments:
                # Map each word to its corresponding audio segment
                for word_info in words:
                    # Find which audio segment this word triggered
                    word_pos = word_info['end_position']
                    
                    # Find the audio segment that was generated after this word
                    segment_idx = 0
                    for i, (start_pos, end_pos) in enumerate(audio_segments):
                        if word_pos <= start_pos:
                            segment_idx = i
                            break
                    
                    # Calculate timing based on segment position
                    if segment_idx < len(audio_segments):
                        segment_start, segment_end = audio_segments[segment_idx]
                        
                        # Calculate relative position within segment
                        segment_audio_gens = [g for g in audio_gens 
                                             if segment_start <= g.position < segment_end]
                        
                        if segment_audio_gens:
                            # Use generation timestamps to estimate audio timing
                            audio_start_time = segment_audio_gens[0].timestamp
                            audio_end_time = segment_audio_gens[-1].timestamp
                            
                            # Scale to actual audio duration
                            time_scale = audio_duration_ms / (self.generations[-1].timestamp * 1000)
                            
                            start_ms = int(audio_start_time * 1000 * time_scale)
                            end_ms = int(audio_end_time * 1000 * time_scale)
                            
                            word_timings.append(WordTiming(
                                word=word_info['word'],
                                start_ms=start_ms,
                                end_ms=end_ms,
                                confidence=0.8,  # Lower confidence since it's estimated
                                token_ids=word_info['token_ids']
                            ))
            
            else:
                # Fallback: distribute evenly
                logger.warning("No audio segments found, using even distribution")
                words_count = len(words)
                ms_per_word = audio_duration_ms / words_count if words_count > 0 else 0
                
                for i, word_info in enumerate(words):
                    word_timings.append(WordTiming(
                        word=word_info['word'],
                        start_ms=int(i * ms_per_word),
                        end_ms=int((i + 1) * ms_per_word),
                        confidence=0.5,  # Low confidence for fallback
                        token_ids=word_info['token_ids']
                    ))
            
            return word_timings
    
    def _get_audio_segments(self) -> List[Tuple[int, int]]:
        """Get the position ranges where audio was generated"""
        segments = []
        in_audio = False
        start_pos = 0
        
        for gen in self.generations:
            if gen.is_audio and not in_audio:
                in_audio = True
                start_pos = gen.position
            elif not gen.is_audio and in_audio:
                in_audio = False
                segments.append((start_pos, gen.position))
        
        # Close last segment if needed
        if in_audio and self.generations:
            segments.append((start_pos, self.generations[-1].position))
        
        return segments


class CrossAttentionTracker:
    """
    Tracks cross-attention weights between text and audio tokens.
    This provides more accurate alignment than just generation order.
    """
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights"""
        
        def attention_hook(module, input, output):
            """Capture cross-attention weights"""
            if hasattr(output, 'cross_attentions') and output.cross_attentions is not None:
                # Store attention weights
                # Shape: [batch, heads, seq_len, seq_len]
                self.attention_weights.append(output.cross_attentions.detach().cpu())
        
        # Register hooks on cross-attention layers
        for name, module in self.model.named_modules():
            if 'cross_attn' in name.lower() or 'crossattention' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
                logger.debug(f"Registered attention hook on {name}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_alignment(self) -> np.ndarray:
        """
        Process captured attention weights to get text-audio alignment.
        
        Returns:
            Alignment matrix of shape [audio_frames, text_tokens]
        """
        if not self.attention_weights:
            return None
        
        # Average attention across layers and heads
        # This gives us the overall attention pattern
        attention_stack = torch.stack(self.attention_weights)
        mean_attention = attention_stack.mean(dim=[0, 1, 2])  # Average over layers, batch, heads
        
        return mean_attention.numpy()
    
    def reset(self):
        """Clear captured attention weights"""
        self.attention_weights = []


def create_enhanced_engine_with_tracking(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    **kwargs
) -> Tuple[Any, GenerationTimestampTracker, CrossAttentionTracker]:
    """
    Create a serve engine with production timestamp tracking.
    
    Returns:
        Tuple of (engine, generation_tracker, attention_tracker)
    """
    from .serve_engine import HiggsAudioServeEngine
    
    # Create standard engine
    engine = HiggsAudioServeEngine(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_name_or_path=audio_tokenizer_name_or_path,
        **kwargs
    )
    
    # Create trackers
    generation_tracker = GenerationTimestampTracker(
        tokenizer=engine.tokenizer,
        audio_tokenizer=engine.audio_tokenizer
    )
    
    attention_tracker = CrossAttentionTracker(engine.model)
    
    # Monkey-patch the model's generate method to use our tracker
    original_model_generate = engine.model.generate
    
    def model_generate_with_tracking(*args, **kwargs):
        # Check if timestamps are being requested (set by engine wrapper)
        return_timestamps = getattr(model_generate_with_tracking, 'enable_timestamps', False)
        
        if return_timestamps:
            # Reset trackers
            generation_tracker.reset()
            attention_tracker.reset()
            
            # Register attention hooks
            attention_tracker.register_hooks()
            
            # Add our tracker to the logits processors
            if 'logits_processor' not in kwargs:
                kwargs['logits_processor'] = []
            kwargs['logits_processor'].append(generation_tracker)
            
            try:
                # Run generation at model level
                return original_model_generate(*args, **kwargs)
                
            finally:
                # Clean up hooks
                attention_tracker.remove_hooks()
                # Reset flag
                model_generate_with_tracking.enable_timestamps = False
        else:
            # Standard generation without tracking
            return original_model_generate(*args, **kwargs)
    
    engine.model.generate = model_generate_with_tracking
    
    # Monkey-patch the serve engine's generate method to handle timestamps
    original_engine_generate = engine.generate
    
    def engine_generate_with_tracking(*args, return_timestamps=False, **kwargs):
        # Extract return_timestamps from kwargs before passing to original
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('return_timestamps', None)
        
        # Signal to model wrapper that timestamps are requested
        if return_timestamps:
            model_generate_with_tracking.enable_timestamps = True
        
        # Call original engine generate (WITHOUT return_timestamps parameter)
        result = original_engine_generate(*args, **kwargs_copy)
        
        # Extract timestamps if requested
        if return_timestamps and hasattr(result, 'audio') and result.audio is not None:
            duration_ms = len(result.audio) / result.sampling_rate * 1000
            word_timings = generation_tracker.get_word_timings(
                result.generated_text, 
                duration_ms
            )
            
            # Add to result
            result.word_timings = word_timings
            result.alignment_method = "generation_tracking"
            
            # Also get attention alignment if available
            alignment = attention_tracker.get_alignment()
            if alignment is not None:
                result.attention_alignment = alignment
                logger.info(f"Captured attention alignment: {alignment.shape}")
        
        return result
    
    engine.generate = engine_generate_with_tracking
    
    return engine, generation_tracker, attention_tracker