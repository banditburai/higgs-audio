# WhisperX vs Wav2Vec2 for Forced Alignment

## Executive Summary

For Higgs Audio TTS word-level timestamp alignment, **WhisperX is the better choice** due to its balance of accuracy, speed, and ease of integration.

## Detailed Comparison

### WhisperX

**Architecture**: Built on top of Whisper with additional phoneme alignment
**Process**: 
1. Transcribes with Whisper
2. Uses phoneme recognition model for forced alignment
3. Produces accurate word-level timestamps

**Pros**:
- ✅ **High accuracy**: 94.6% word recognition on TIMIT dataset
- ✅ **Fast**: 60-70x real-time speed with large-v2 model
- ✅ **Easy integration**: Works directly with Whisper models we already have
- ✅ **Production ready**: Widely used, well-maintained
- ✅ **Handles long audio**: Specifically designed for long-form audio
- ✅ **Good with various voices**: Works with character voices better

**Cons**:
- ❌ Requires additional phoneme model
- ❌ Slightly less accurate than traditional MFA
- ❌ Needs GPU for best performance

**Accuracy metrics**:
- TIMIT: 37,685/39,834 words correct (94.6%)
- Buckeye: 278,480/285,347 words correct (97.6%)

### Wav2Vec2

**Architecture**: Facebook's self-supervised speech model with CTC alignment
**Process**:
1. Encodes audio with CNN layers
2. Uses CTC (Connectionist Temporal Classification) for alignment
3. Can be fine-tuned for specific languages/accents

**Pros**:
- ✅ **Memory efficient**: 5x less memory than TorchAudio's implementation
- ✅ **Multilingual**: Supports 1,126+ languages
- ✅ **Flexible**: Can be fine-tuned for specific voices/accents
- ✅ **Good phoneme accuracy**: 82% F1 score on phoneme recognition

**Cons**:
- ❌ **More complex setup**: Requires CTC decoding implementation
- ❌ **Slower**: Not optimized for speed like WhisperX
- ❌ **Less accurate on conversational speech**: Variable performance
- ❌ **Requires fine-tuning**: Best results need dataset-specific training

**Accuracy metrics**:
- Phoneme Error Rate: 14.8% on disordered speech
- Boundary accuracy: Within 20ms tolerance for most phonemes

## Key Differentiators

### 1. **Purpose-Built Design**
- **WhisperX**: Specifically designed for word-level timestamp alignment
- **Wav2Vec2**: General ASR model adapted for alignment

### 2. **Speed**
- **WhisperX**: 60-70x real-time (extremely fast)
- **Wav2Vec2**: ~5-10x real-time (moderate)

### 3. **Integration Complexity**
- **WhisperX**: Drop-in replacement for Whisper, minimal code changes
- **Wav2Vec2**: Requires CTC implementation, more complex pipeline

### 4. **Handling Character Voices**
- **WhisperX**: Better at handling non-standard voices (like Sheldon)
- **Wav2Vec2**: May struggle without fine-tuning

## Recommendation for Higgs Audio

**Use WhisperX** for the following reasons:

1. **Immediate compatibility**: We already use Whisper, so WhisperX is a natural extension
2. **Speed matters**: 60-70x real-time means minimal latency for users
3. **Character voice handling**: Better performance with voices like "bigbang_sheldon"
4. **Production stability**: WhisperX is battle-tested in production environments
5. **Simpler implementation**: Less code changes required

## Implementation Strategy

```python
# Simple WhisperX integration
import whisperx

model = whisperx.load_model("large-v2", device="cuda")
audio = whisperx.load_audio("audio.wav")

# Transcribe with Whisper
result = model.transcribe(audio)

# Align with phoneme model for accurate timestamps
model_a, metadata = whisperx.load_align_model(language_code="en", device="cuda")
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device="cuda")

# Extract word-level timestamps
for segment in result_aligned["segments"]:
    for word in segment["words"]:
        print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")
```

## Fallback Strategy

If WhisperX fails on certain content:
1. Try with smaller chunks (as we've implemented)
2. Use silence detection to find natural boundaries
3. As last resort, fail explicitly rather than using inaccurate simple distribution

## Conclusion

WhisperX provides the best balance of:
- Accuracy (94-97% word recognition)
- Speed (60-70x real-time)
- Ease of integration
- Production readiness

This makes it the optimal choice for Higgs Audio's word-level timestamp needs.