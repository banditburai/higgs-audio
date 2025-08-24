"""
Modal deployment with timestamp tracking support.
This version mounts our local modifications for testing.
"""

import modal
from modal import App, Image, Volume, asgi_app
from pathlib import Path
import os

# App configuration
app = App("higgs-audio-timestamps")

# Configuration
config = {
    'model_name': 'bosonai/higgs-audio-v2-generation-3B-base',
    'default_voice': 'en_woman',
    'max_chunk_size': 800,
    'temperature': 0.3,
    'max_new_tokens': 2048
}

# Define the container image with dependencies
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "curl", 
        "wget",
        "build-essential",
        "ffmpeg",
    )
    .pip_install(
        # Web framework
        "fastapi",
        "uvicorn[standard]",
        
        # Core Higgs Audio dependencies (matching working version)
        "descript-audio-codec",
        "torch",
        "transformers>=4.45.1,<4.47.0",
        "librosa",
        "dacite",
        "boto3==1.35.36",
        "s3fs",
        "torchvision",
        "torchaudio",
        "json_repair",
        "pandas",
        "pydantic",
        "vector_quantize_pytorch",
        "loguru",
        "pydub",
        "omegaconf",
        "click",
        "langid",
        "jieba",
        "accelerate>=0.26.0",
        "soundfile",
        "datasets",
        "scipy",
        
        # Audio processing
        "audioread",
        "ffmpeg-python",
        "numpy",
        
        # For timestamp testing
        "openai-whisper",
        
        # Force cache invalidation (2025-08-24-17:45)
        "requests==2.32.5",
    )
    .run_commands(
        # Clone YOUR FORK with timestamp tracking (force rebuild 2025-08-24-17:45)  
        "git clone https://github.com/banditburai/higgs-audio.git /app/higgs-audio",
        "cd /app/higgs-audio && git checkout timestamp-tracking || true",  # Use timestamp branch if it exists
        "cd /app/higgs-audio && git pull origin timestamp-tracking",  # Force pull latest changes
        "cd /app/higgs-audio && pip install -e .",
        force_build=True
    )
    .env({"HF_HOME": "/cache/huggingface", "TRANSFORMERS_CACHE": "/cache/transformers"})
)

# Volume for model caching
model_cache = Volume.from_name("higgs-audio-models", create_if_missing=True)

# Secrets for HuggingFace
secrets = []
if os.getenv('HUGGINGFACE_TOKEN'):
    secrets.append(modal.Secret.from_name("huggingface-secret"))

@app.function(
    image=image,
    gpu="A10G",
    memory=32768,
    timeout=900,
    volumes={"/cache": model_cache},
    secrets=secrets,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=10)
@asgi_app()
def higgs_tts_api_timestamps():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    from typing import Optional, List
    import base64
    import numpy as np
    import soundfile as sf
    import io
    import torch
    import sys
    import os
    
    # Add Higgs Audio to path
    sys.path.insert(0, '/app/higgs-audio')
    
    # Set up environment
    if os.getenv('HUGGINGFACE_TOKEN'):
        os.environ['HF_TOKEN'] = os.getenv('HUGGINGFACE_TOKEN')
    
    # Import our simplified timestamp tracking (v2)
    from boson_multimodal.serve.timestamp_tracker_v2 import (
        create_simple_tracking_engine,
        WordTiming
    )
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    
    # Create FastAPI app
    web_app = FastAPI(
        title="Higgs Audio TTS API with Timestamps",
        version="2.0.0"
    )
    
    # Function to initialize engine (cached)
    def get_serve_engine():
        if not hasattr(get_serve_engine, 'engine'):
            print("Initializing Production Higgs Audio Engine with Tracking...")
            
            # Check device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Initialize simplified engine with timestamp tracking v2
            engine, tracker = create_simple_tracking_engine(
                model_name_or_path=config['model_name'],
                audio_tokenizer_name_or_path='bosonai/higgs-audio-v2-tokenizer',  # Use separate tokenizer like working version
                device=device
            )
            get_serve_engine.engine = engine
            get_serve_engine.tracker = tracker
            print("Engine initialized with simplified timestamp tracking (v2)!")
        
        return get_serve_engine.engine
    
    # Request/Response models
    class TTSRequest(BaseModel):
        text: str
        voice: Optional[str] = config['default_voice']
        temperature: Optional[float] = 0.3
        max_new_tokens: Optional[int] = 2048
        format: Optional[str] = "wav"
        return_timestamps: Optional[bool] = False  # New field
        timestamp_granularity: Optional[str] = "word"  # word|sentence
    
    class TTSResponse(BaseModel):
        audio: str  # base64 encoded
        format: str
        sample_rate: int
        duration_seconds: float
        character_count: int
        voice: str
        model: str
        word_timings: Optional[List[dict]] = None
        segments: Optional[List[dict]] = None
        alignment_method: Optional[str] = None
    
    @web_app.get("/")
    async def root():
        return {
            "service": "Higgs Audio TTS with Timestamps",
            "version": "2.0.0",
            "features": ["voice_cloning", "timestamps", "long_form_audio"],
            "model": config['model_name']
        }
    
    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp_support": True}
    
    @web_app.post("/generate", response_model=TTSResponse)
    async def generate_tts(request: TTSRequest):
        """
        Generate TTS audio with optional word-level timestamps
        """
        
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="No text provided")
        
        if len(request.text) > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"Text too long ({len(request.text)} chars, max 10000)"
            )
        
        try:
            print(f"Generating TTS for {len(request.text)} characters...")
            print(f"  Return timestamps: {request.return_timestamps}")
            
            # Get or initialize the serve engine
            engine = get_serve_engine()
            
            # Create system prompt
            system_prompt = (
                "Generate audio following instruction.\n\n"
                "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
            )
            
            # Build messages
            messages = [Message(role="system", content=system_prompt)]
            
            # Add voice prompt if specified
            if request.voice and request.voice != "smart":
                voice_audio_path = f"/app/higgs-audio/examples/voice_prompts/{request.voice}.wav"
                voice_text_path = f"/app/higgs-audio/examples/voice_prompts/{request.voice}.txt"
                
                if os.path.exists(voice_audio_path) and os.path.exists(voice_text_path):
                    with open(voice_text_path, 'r') as f:
                        voice_text = f.read().strip()
                    
                    messages.append(Message(role="user", content=voice_text))
                    messages.append(Message(
                        role="assistant", 
                        content=AudioContent(audio_url=voice_audio_path)
                    ))
            
            # Add the main text
            messages.append(Message(role="user", content=request.text))
            
            # Generate audio with timestamps if requested
            if request.return_timestamps:
                print(f"=== MODAL ENDPOINT DEBUG ===")
                print(f"Input text: '{request.text}'")
                print(f"Voice: {request.voice}")
                print("Using production tracking for timestamps...")
                
                output = engine.generate(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                    return_timestamps=True,  # This triggers our tracking
                    input_text=request.text  # Pass the input text for timestamp extraction
                )
                
                print(f"Output type: {type(output)}")
                print(f"Has word_timings: {hasattr(output, 'word_timings')}")
                if hasattr(output, 'word_timings'):
                    print(f"Word timings: {output.word_timings}")
                print(f"Generated text: '{output.generated_text if hasattr(output, 'generated_text') else 'N/A'}'")
                print(f"Audio shape: {output.audio.shape if hasattr(output, 'audio') and hasattr(output.audio, 'shape') else 'N/A'}")
                
                # Extract timestamp data
                word_timings = None
                segments = None
                alignment_method = "direct_tracking"
                
                if hasattr(output, 'word_timings') and output.word_timings:
                    word_timings = [
                        {
                            "word": wt.word,
                            "start_ms": wt.start_ms,
                            "end_ms": wt.end_ms,
                            "confidence": wt.confidence
                        }
                        for wt in output.word_timings
                    ]
                    # If no valid word timings, try to generate from input text
                    if not word_timings or (len(word_timings) == 1 and '<|' in word_timings[0]['word']):
                        print(f"Regenerating word timings from input text: '{request.text}'")
                        words = request.text.strip().split()
                        if words:
                            duration = len(output.audio) / output.sampling_rate * 1000
                            ms_per_word = duration / len(words)
                            word_timings = [
                                {
                                    "word": word,
                                    "start_ms": int(i * ms_per_word),
                                    "end_ms": int((i + 1) * ms_per_word),
                                    "confidence": 0.7
                                }
                                for i, word in enumerate(words)
                            ]
                
                if hasattr(output, 'segments') and output.segments:
                    segments = [
                        {
                            "id": seg.id,
                            "text": seg.text,
                            "start_ms": seg.start_ms,
                            "end_ms": seg.end_ms,
                            "words": [
                                {
                                    "word": w.word,
                                    "start_ms": w.start_ms,
                                    "end_ms": w.end_ms,
                                    "confidence": w.confidence
                                }
                                for w in seg.words
                            ]
                        }
                        for seg in output.segments
                    ]
                
                if hasattr(output, 'alignment_method'):
                    alignment_method = output.alignment_method
                
            else:
                # Use regular generation
                print("Using standard generation without timestamps...")
                output = engine.generate(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=0.95,
                    top_k=50,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )
                word_timings = None
                segments = None
                alignment_method = None
            
            # Convert audio to WAV format
            audio_array = torch.from_numpy(output.audio)
            buffer = io.BytesIO()
            import torchaudio
            torchaudio.save(
                buffer, 
                audio_array[None, :],
                output.sampling_rate,
                format="WAV"
            )
            buffer.seek(0)
            
            # Calculate duration
            duration = len(output.audio) / output.sampling_rate
            
            # Encode as base64
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            print(f"Successfully generated {duration:.2f} seconds of audio")
            if word_timings:
                print(f"  With {len(word_timings)} word timings")
            
            return TTSResponse(
                audio=audio_base64,
                format="wav",
                sample_rate=output.sampling_rate,
                duration_seconds=duration,
                character_count=len(request.text),
                voice=request.voice,
                model=config['model_name'],
                word_timings=word_timings,
                segments=segments,
                alignment_method=alignment_method
            )
            
        except Exception as e:
            print(f"Error generating TTS: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(e)}"
            )
    
    return web_app

# Test function to verify installation
@app.function(image=image, volumes={"/cache": model_cache}, secrets=secrets)
def test_timestamp_system():
    import sys
    sys.path.insert(0, '/app/higgs-audio')
    
    try:
        from boson_multimodal.serve.timestamp_tracker_v2 import (
            create_simple_tracking_engine,
            SimpleTimestampTracker,
            WordTiming
        )
        print("✅ Simplified timestamp system (v2) imported successfully!")
        print(f"  - create_simple_tracking_engine: {create_simple_tracking_engine}")
        print(f"  - SimpleTimestampTracker: {SimpleTimestampTracker}")
        print(f"  - WordTiming: {WordTiming}")
        return True
    except Exception as e:
        print(f"❌ Failed to import timestamp system: {e}")
        return False

if __name__ == "__main__":
    # For local testing
    print("Deploy with: modal deploy modal_higgs_timestamps.py")