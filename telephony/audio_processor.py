"""
Audio processing for Waybeo telephony ↔ Gemini Live.

- Waybeo: 8kHz int16 PCM frames (JSON sample arrays)
- Gemini input: 16kHz int16 PCM (base64)
- Gemini output: typically 24kHz int16 PCM (base64) → downsample back to 8kHz for telephony

We use librosa for high-quality resampling (same approach as the singleinterface telephony release).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np
import audioop

@dataclass(frozen=True)
class AudioRates:
    telephony_sr: int = 8000
    gemini_input_sr: int = 16000
    gemini_output_sr: int = 24000


class AudioProcessor:
    def __init__(self, rates: AudioRates):
        self.rates = rates

    @staticmethod
    def int16_to_float32(samples: np.ndarray) -> np.ndarray:
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def float32_to_int16(samples: np.ndarray) -> np.ndarray:
        # gentle gain reduction to reduce clipping artifacts
        samples = samples * 0.90
        samples = np.clip(samples, -1.0, 1.0)
        return np.round(samples * 32767.0).astype(np.int16)

    def resample_int16(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if samples.size == 0 or orig_sr == target_sr:
            return samples.astype(np.int16, copy=False)
        samples_f = self.int16_to_float32(samples)
        out_f = librosa.resample(
            samples_f, orig_sr=orig_sr, target_sr=target_sr, res_type="linear"
        )
        return self.float32_to_int16(out_f)

    @staticmethod
    def apply_fade(samples: np.ndarray, fade_samples: int = 16) -> np.ndarray:
        if samples.size < fade_samples * 2:
            return samples
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        out = samples.copy()
        out[:fade_samples] = (out[:fade_samples] * fade_in).astype(np.int16)
        out[-fade_samples:] = (out[-fade_samples:] * fade_out).astype(np.int16)
        return out

    def waybeo_samples_to_np(self, samples: List[int]) -> np.ndarray:
        return np.array(samples, dtype=np.int16)

    def np_to_waybeo_samples(self, samples: np.ndarray) -> List[int]:
        return samples.astype(np.int16, copy=False).tolist()

    # ---- Input (Waybeo -> Gemini) ----
    def process_input_8k_to_gemini_16k_b64(self, samples_8k: np.ndarray) -> str:
        samples_16k = self.resample_int16(
            samples_8k, orig_sr=self.rates.telephony_sr, target_sr=self.rates.gemini_input_sr
        )
        return base64.b64encode(samples_16k.tobytes()).decode("utf-8")

    # ---- Output (Gemini -> Waybeo) ----
    def process_output_gemini_b64_to_8k_samples(self, audio_b64: str) -> List[int]:
        raw = base64.b64decode(audio_b64)
        # Gemini audio output is int16 PCM
        samples_out = np.frombuffer(raw, dtype=np.int16)
        samples_8k = self.resample_int16(
            samples_out,
            orig_sr=self.rates.gemini_output_sr,
            target_sr=self.rates.telephony_sr,
        )
        samples_8k = self.apply_fade(samples_8k)
        return self.np_to_waybeo_samples(samples_8k)


    def process_output_gemini_b64_to_8k_elision(self, audio_b64: str) -> List[int]:
        """Process Gemini audio for Elision format - returns A-law bytes as list of ints"""
        pcm16_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32767.0

        audio_np = librosa.resample(
            audio_np,
            orig_sr=self.rates.gemini_output_sr,
            target_sr=self.rates.telephony_sr,
            res_type="linear"  # Use scipy-based resampling (doesn't require samplerate)
        )

        #audio_np = np.clip(audio_np, -1.0, 1.0)
        pcm16_8k = (audio_np * 32767).astype(np.int16).tobytes()
        alaw_bytes = audioop.lin2alaw(pcm16_8k, 2)
        # Convert bytes to list of ints for buffer
        return list(alaw_bytes)

    # ========= MAIN FUNCTION ============
    def alaw_to_openai_format(self, alaw_bytes: bytes, output_rate=16000) -> List[int]:
        """Convert A-law bytes to PCM16 samples as list of ints"""
        pcm16 = alaw_to_pcm16(alaw_bytes)                 # Step1: decode A-LAW to PCM16
        #pcm16_resampled = resample_pcm16(pcm16, 8000, output_rate)   # Step2: resample to 16k/24k
        return pcm16.tolist()  # Convert numpy array to list of ints

# ========= A-LAW DECODE LOOKUP TABLE =========
# Generate full 256-value A-Law decode table
def create_alaw_decode_table():
    alaw_table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        a = i ^ 0x55
        t = (a & 0x0F) << 4
        seg = (a & 0x70) >> 4
        if seg >= 1:
            t = (t + 0x100) << (seg - 1)
        if a & 0x80:
            alaw_table[i] = t
        else:
            alaw_table[i] = -t
    return alaw_table

ALAW_DECODE_TABLE = create_alaw_decode_table()


# ========= A-LAW → PCM16 =============
def alaw_to_pcm16(alaw_bytes: bytes) -> np.ndarray:
    alaw_np = np.frombuffer(alaw_bytes, dtype=np.uint8)
    pcm16 = ALAW_DECODE_TABLE[alaw_np]
    return pcm16.astype(np.int16)


# ========= RESAMPLE PCM16 ============
def resample_pcm16(pcm16_data: np.ndarray, input_rate=8000, output_rate=16000) -> bytes:
    # Convert to float32 [-1,1]
    audio_float = pcm16_data.astype(np.float32) / 32768.0

    # Librosa resample
    resampled = librosa.resample(audio_float, orig_sr=input_rate, target_sr=output_rate)

    # Convert back to PCM16 bytes
    resampled_pcm16 = (resampled * 32767.0).astype(np.int16)
    return resampled_pcm16.tobytes()

