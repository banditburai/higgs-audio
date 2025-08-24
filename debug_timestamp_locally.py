#!/usr/bin/env python3
"""
Local debugging script to understand the Higgs Audio generation flow.
This helps us debug without needing to redeploy to Modal.
"""

import sys
import json
from dataclasses import dataclass
from typing import List, Optional

# Mock the basic structures we need
@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatMLSample:
    messages: List[Message]

@dataclass
class MockAudioResponse:
    """Mock response that mimics HiggsAudioResponse"""
    audio: object  # numpy array in real case
    generated_text: str
    generated_text_tokens: object  # numpy array
    generated_audio_tokens: object  # numpy array
    sampling_rate: int
    
    def __init__(self):
        # Simulate what we're seeing in production
        self.audio = MockArray(shape=(24000,))  # 1 second at 24kHz
        self.generated_text = "<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|><|eot_id|>"
        self.generated_text_tokens = MockArray(data=[50001, 50002, 50003, 50004])  # Mock token IDs
        self.generated_audio_tokens = MockArray(shape=(8, 100))  # Mock audio tokens
        self.sampling_rate = 24000

class MockArray:
    """Mock numpy array for testing"""
    def __init__(self, shape=None, data=None):
        self.shape = shape
        self.data = data if data else list(range(10))  # Sample data
        
    def __len__(self):
        return self.shape[0] if self.shape else len(self.data)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.data[idx] if self.data else []
        return self.data[idx] if self.data and idx < len(self.data) else 0

class MockTokenizer:
    """Mock tokenizer for testing"""
    def decode(self, token_ids):
        # Map token IDs to text
        token_map = {
            50001: "<|audio_out_bos|>",
            50002: "<|AUDIO_OUT|>", 
            50003: "<|audio_eos|>",
            50004: "<|eot_id|>",
        }
        if isinstance(token_ids, list) and len(token_ids) == 1:
            return token_map.get(token_ids[0], f"[TOKEN_{token_ids[0]}]")
        return "".join([token_map.get(tid, f"[TOKEN_{tid}]") for tid in token_ids])

def test_timestamp_extraction():
    """Test our timestamp extraction logic locally"""
    
    # Import just the parts we need, avoiding torch dependency
    sys.path.insert(0, '/Users/firefly/Code/sandbox/higgs-audio')
    
    # Inline the SimpleTimestampTracker logic to avoid import issues
    class SimpleTimestampTracker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def extract_word_timings(self, text, text_tokens, audio_tokens, audio_duration_ms):
            print(f"[Tracker] Extracting word timings for: '{text}'")
            print(f"[Tracker] Text tokens: {text_tokens}")
            print(f"[Tracker] Audio duration: {audio_duration_ms}ms")
            
            # Clean special tokens
            special_tokens = ['<|audio_out_bos|>', '<|AUDIO_OUT|>', '<|audio_eos|>', '<|eot_id|>', 
                             '<|audio_in_bos|>', '<|AUDIO_IN|>', '<|audio_in_eos|>']
            
            clean_text = text
            for token in special_tokens:
                clean_text = clean_text.replace(token, '')
            clean_text = clean_text.strip()
            
            print(f"[Tracker] Clean text: '{clean_text}'")
            
            if not clean_text:
                print(f"[Tracker] No clean text found after removing special tokens")
                return []
            
            # Parse into words
            words = []
            for word in clean_text.split():
                if word.strip():
                    words.append({'word': word.strip(), 'token_ids': []})
            
            print(f"[Tracker] Found {len(words)} words")
            
            # Create word timings
            from dataclasses import dataclass
            
            @dataclass
            class WordTiming:
                word: str
                start_ms: int
                end_ms: int
                confidence: float = 0.8
                token_ids: list = None
            
            word_timings = []
            if words:
                ms_per_word = audio_duration_ms / len(words)
                
                for i, word_info in enumerate(words):
                    word_timings.append(WordTiming(
                        word=word_info['word'],
                        start_ms=int(i * ms_per_word),
                        end_ms=int((i + 1) * ms_per_word),
                        confidence=0.8,
                        token_ids=word_info.get('token_ids', [])
                    ))
            
            return word_timings
    
    # Create mock objects
    tokenizer = MockTokenizer()
    tracker = SimpleTimestampTracker(tokenizer)
    
    # Create mock response
    response = MockAudioResponse()
    
    print("=== LOCAL DEBUG TEST ===")
    print(f"Generated text: '{response.generated_text}'")
    print(f"Text tokens: {response.generated_text_tokens.data}")
    print(f"Audio shape: {response.generated_audio_tokens.shape}")
    
    # Test extraction
    duration_ms = len(response.audio) / response.sampling_rate * 1000
    print(f"Duration: {duration_ms}ms")
    
    # Test with the problematic text
    word_timings = tracker.extract_word_timings(
        text=response.generated_text,
        text_tokens=response.generated_text_tokens,
        audio_tokens=response.generated_audio_tokens,
        audio_duration_ms=duration_ms
    )
    
    print(f"Extracted {len(word_timings)} word timings")
    for wt in word_timings:
        print(f"  - {wt}")
    
    # Now test with actual text
    print("\n=== Testing with real text ===")
    response.generated_text = "Hello world"
    response.generated_text_tokens = MockArray(data=[1001, 1002])  # Mock normal text tokens
    
    word_timings = tracker.extract_word_timings(
        text=response.generated_text,
        text_tokens=response.generated_text_tokens,
        audio_tokens=response.generated_audio_tokens,
        audio_duration_ms=duration_ms
    )
    
    print(f"Extracted {len(word_timings)} word timings")
    for wt in word_timings:
        print(f"  - {wt}")

def test_with_input_text():
    """Test what happens when we pass the input text instead"""
    from dataclasses import dataclass
    
    @dataclass
    class WordTiming:
        word: str
        start_ms: int
        end_ms: int
        confidence: float = 0.8
        token_ids: list = None
    
    class SimpleTimestampTracker:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            
        def extract_word_timings(self, text, text_tokens, audio_tokens, audio_duration_ms):
            # For input text, we don't need to clean special tokens
            words = text.strip().split()
            
            word_timings = []
            if words:
                ms_per_word = audio_duration_ms / len(words)
                
                for i, word in enumerate(words):
                    word_timings.append(WordTiming(
                        word=word,
                        start_ms=int(i * ms_per_word),
                        end_ms=int((i + 1) * ms_per_word),
                        confidence=0.8,
                        token_ids=[]
                    ))
            
            return word_timings
    
    tokenizer = MockTokenizer()
    tracker = SimpleTimestampTracker(tokenizer)
    
    # Simulate what should happen
    input_text = "Hello world, this is a test"
    duration_ms = 2000  # 2 seconds
    
    print("\n=== Testing with input text directly ===")
    print(f"Input text: '{input_text}'")
    
    # Create mock tokens (in real case, these would be from the model)
    mock_text_tokens = MockArray(data=list(range(10)))
    mock_audio_tokens = MockArray(shape=(8, 100))
    
    word_timings = tracker.extract_word_timings(
        text=input_text,  # Pass input text instead of generated
        text_tokens=mock_text_tokens,
        audio_tokens=mock_audio_tokens,
        audio_duration_ms=duration_ms
    )
    
    print(f"Extracted {len(word_timings)} word timings")
    for wt in word_timings:
        print(f"  - Word: '{wt.word}', Start: {wt.start_ms}ms, End: {wt.end_ms}ms")

if __name__ == "__main__":
    test_timestamp_extraction()
    test_with_input_text()