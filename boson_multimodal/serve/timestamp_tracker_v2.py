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
        
        # First, let's clean the text by removing special tokens
        # Common special tokens in Higgs Audio
        special_tokens = ['<|audio_out_bos|>', '<|AUDIO_OUT|>', '<|audio_eos|>', '<|eot_id|>', 
                         '<|audio_in_bos|>', '<|AUDIO_IN|>', '<|audio_in_eos|>']
        
        clean_text = text
        for token in special_tokens:
            clean_text = clean_text.replace(token, '')
        clean_text = clean_text.strip()
        
        # If no clean text, we might be looking at the wrong field
        if not clean_text:
            logger.warning(f"No clean text found after removing special tokens from: '{text}'")
            # Return empty list - the actual text might be elsewhere
            return []
        
        # Parse the clean text into words
        words = []
        for word in clean_text.split():
            if word.strip():
                words.append({'word': word.strip(), 'token_ids': []})
        
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
    
    def generate_with_tracking(*args, return_timestamps=False, input_text=None, **kwargs):
        # Log the input
        logger.info(f"=== TIMESTAMP TRACKING DEBUG ===")
        logger.info(f"return_timestamps: {return_timestamps}")
        logger.info(f"input_text provided: {input_text}")
        
        # Extract input text from ChatMLSample if available
        if not input_text and len(args) > 0:
            # First arg should be chat_ml_sample
            chat_ml_sample = args[0]
            if hasattr(chat_ml_sample, 'messages'):
                # Find the last user message
                for msg in reversed(chat_ml_sample.messages):
                    if hasattr(msg, 'role') and msg.role == 'user' and hasattr(msg, 'content'):
                        if isinstance(msg.content, str):
                            input_text = msg.content
                            logger.info(f"Extracted input text from messages: '{input_text}'")
                            break
        
        # Call original generate
        result = original_generate(*args, **kwargs)
        
        # Log what we got back
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Result attributes: {dir(result)}")
        logger.info(f"Has audio: {hasattr(result, 'audio') and result.audio is not None}")
        if hasattr(result, 'generated_text'):
            logger.info(f"Generated text: '{result.generated_text}'")
        if hasattr(result, 'generated_text_tokens'):
            logger.info(f"Generated text tokens shape: {result.generated_text_tokens.shape if hasattr(result.generated_text_tokens, 'shape') else 'no shape'}")
            logger.info(f"Generated text tokens sample: {result.generated_text_tokens[:10] if hasattr(result.generated_text_tokens, '__getitem__') else 'cannot index'}")
        if hasattr(result, 'generated_audio_tokens'):
            logger.info(f"Generated audio tokens shape: {result.generated_audio_tokens.shape if hasattr(result.generated_audio_tokens, 'shape') else 'no shape'}")
        
        # Extract timestamps if requested
        if return_timestamps and hasattr(result, 'audio') and result.audio is not None:
            # Calculate audio duration
            duration_ms = len(result.audio) / result.sampling_rate * 1000
            logger.info(f"Audio duration: {duration_ms}ms")
            
            # Determine which text to use
            text_to_use = result.generated_text if hasattr(result, 'generated_text') else ""
            
            # Check if generated text is only special tokens
            special_tokens = ['<|audio_out_bos|>', '<|AUDIO_OUT|>', '<|audio_eos|>', '<|eot_id|>']
            is_only_special = all(token in text_to_use for token in special_tokens) or not text_to_use.replace(''.join(special_tokens), '').strip()
            
            if is_only_special and input_text:
                logger.info(f"Generated text has only special tokens, using input text: '{input_text}'")
                text_to_use = input_text
            
            # Extract word timings using our simple approach
            word_timings = tracker.extract_word_timings(
                text=text_to_use,
                text_tokens=result.generated_text_tokens if hasattr(result, 'generated_text_tokens') else None,
                audio_tokens=result.generated_audio_tokens if hasattr(result, 'generated_audio_tokens') else None,
                audio_duration_ms=duration_ms
            )
            
            # Add to result
            result.word_timings = word_timings
            result.alignment_method = "simple_distribution"
            
            logger.info(f"Added {len(word_timings) if word_timings else 0} word timings to result")
            if word_timings:
                logger.info(f"Sample word timing: {word_timings[0]}")
        else:
            logger.info(f"Not adding timestamps. return_timestamps={return_timestamps}, has_audio={hasattr(result, 'audio')}")
        
        return result
    
    engine.generate = generate_with_tracking
    
    return engine, tracker