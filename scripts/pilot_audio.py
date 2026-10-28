"""Deterministic PCM16 clip conversion with polyphase anti-alias filtering."""
import math


def read_clip(wav, start, end):
    rate, channels = wav.getframerate(), wav.getnchannels()
    if wav.getsampwidth() != 2:
        raise ValueError("Pilot converter currently requires PCM16")
    target_start, target_end = round(start * 16000), round(end * 16000)
    if rate == 16000 and channels == 1:
        wav.setpos(target_start)
        data = wav.readframes(target_end - target_start)
    else:
        import numpy as np
        from scipy.signal import resample_poly
        divisor = math.gcd(rate, 16000)
        up, down = 16000 // divisor, rate // divisor
        # Align the local window to the global polyphase grid and keep 100 ms context.
        begin = max(0, math.floor(start * rate) - math.ceil(rate * 0.1))
        begin = begin // down * down
        finish = min(wav.getnframes(), math.ceil(end * rate) + math.ceil(rate * 0.1))
        wav.setpos(begin)
        native = np.frombuffer(wav.readframes(finish - begin), dtype="<i2").astype(np.float64)
        native = native.reshape(-1, channels).mean(axis=1)
        converted = resample_poly(native, up, down, window=("kaiser", 5.0),
                                  padtype="constant", cval=0)
        offset = target_start - begin * up // down
        converted = converted[offset:offset + target_end - target_start]
        data = np.clip(np.rint(converted), -32768, 32767).astype("<i2").tobytes()
    if len(data) != (target_end - target_start) * 2:
        raise ValueError("Truncated converted clip")
    return data
