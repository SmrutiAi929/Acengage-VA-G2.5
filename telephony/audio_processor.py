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


