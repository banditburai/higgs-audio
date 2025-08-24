"""
Production-quality timestamp tracking for Higgs Audio generation - Version 2.

This version properly handles Higgs Audio's dual-phase generation:
1. Text generation phase (generates text tokens) 
2. Audio generation phase (generates audio tokens)

Instead of using a LogitsProcessor, we track at a higher level to capture both phases.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from loguru import logger


@dataclass
class WordTiming:
    """Word-level timing information"""
    word: str
    start_ms: int
    end_ms: int
    confidence: float = 1.0
    token_ids: List[int] = field(default_factory=list)


class SimpleTimestampTracker:
    """
    Simplified timestamp tracker that works with Higgs Audio's generation pattern.
    
    Since Higgs Audio generates text first, then audio, we can create a simple
    mapping based on the text structure and audio duration.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def extract_word_timings(
        self, 
        text: str, 
        text_tokens: np.ndarray,
        audio_tokens: np.ndarray,
        audio_duration_ms: float
    ) -> List[WordTiming]:
        """
        Extract word timings based on text and audio generation patterns.
        
        Args:
            text: The generated text
            text_tokens: Array of text token IDs
            audio_tokens: Array of audio token IDs  
            audio_duration_ms: Total audio duration in milliseconds
            
        Returns:
            List of word timings
        """
        logger.info(f"Extracting word timings for: '{text}'")
        logger.info(f"Text tokens shape: {text_tokens.shape if text_tokens is not None else 'None'}")
        logger.info(f"Audio tokens shape: {audio_tokens.shape if audio_tokens is not None else 'None'}")
        logger.info(f"Audio duration: {audio_duration_ms}ms")
        
        # Parse the text into words
        words = []
        current_word = ""
        word_token_ids = []
        current_word_tokens = []
        
        # Decode each token to build words
        for token_id in text_tokens:
            token_str = self.tokenizer.decode([token_id])
            
            # Check if this starts a new word (has leading space or is punctuation)
            if token_str.startswith(' ') or (current_word and token_str in '.,!?;:'):
                if current_word:
                    words.append({
                        'word': current_word.strip(),
                        'token_ids': current_word_tokens.copy()
                    })
                current_word = token_str.lstrip()
                current_word_tokens = [token_id]
            else:
                current_word += token_str
                current_word_tokens.append(token_id)
        
        # Add last word
        if current_word:
            words.append({
                'word': current_word.strip(),
                'token_ids': current_word_tokens
            })
        
        # Create word timings with even distribution
        # This is a simple approach - in production you might want more sophisticated alignment
        word_timings = []
        if words:
            ms_per_word = audio_duration_ms / len(words)
            
            for i, word_info in enumerate(words):
                # Skip empty words
                if not word_info['word']:
                    continue
                    
                word_timings.append(WordTiming(
                    word=word_info['word'],
                    start_ms=int(i * ms_per_word),
                    end_ms=int((i + 1) * ms_per_word),
                    confidence=0.8,  # Medium confidence for simple distribution
                    token_ids=word_info['token_ids']
                ))
        
        logger.info(f"Extracted {len(word_timings)} word timings")
        return word_timings


def create_simple_tracking_engine(
    model_name_or_path: str,
    audio_tokenizer_name_or_path: str,
    **kwargs
) -> Tuple[Any, SimpleTimestampTracker]:
    """
    Create a serve engine with simple timestamp tracking.
    
    Returns:
        Tuple of (engine, timestamp_tracker)
    """
    from .serve_engine import HiggsAudioServeEngine
    
    # Create standard engine
    engine = HiggsAudioServeEngine(
        model_name_or_path=model_name_or_path,
        audio_tokenizer_name_or_path=audio_tokenizer_name_or_path,
        **kwargs
    )
    
    # Create simple tracker
    tracker = SimpleTimestampTracker(tokenizer=engine.tokenizer)
    
    # Wrap the engine's generate method to extract timestamps
    original_generate = engine.generate
    
    def generate_with_tracking(*args, return_timestamps=False, **kwargs):
        # Call original generate
        result = original_generate(*args, **kwargs)
        
        # Extract timestamps if requested
        if return_timestamps and hasattr(result, 'audio') and result.audio is not None:
            # Calculate audio duration
            duration_ms = len(result.audio) / result.sampling_rate * 1000
            
            # Extract word timings using our simple approach
            word_timings = tracker.extract_word_timings(
                text=result.generated_text,
                text_tokens=result.generated_text_tokens,
                audio_tokens=result.generated_audio_tokens,
                audio_duration_ms=duration_ms
            )
            
            # Add to result
            result.word_timings = word_timings
            result.alignment_method = "simple_distribution"
            
            logger.info(f"Added {len(word_timings)} word timings to result")
        
        return result
    
    engine.generate = generate_with_tracking
    
    return engine, tracker