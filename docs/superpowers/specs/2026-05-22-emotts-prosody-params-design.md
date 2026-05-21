# EmoTTS Prosody Parameter Shaping — Design Spec

**Date:** 2026-05-22  
**Status:** Approved  

---

## Problem

The current `EmoTTS` implementation steers emotions by injecting learned vectors into transformer layer 5 (latent space). This shifts *what* the model predicts, but does not influence *how* the audio sounds acoustically. Emotions like sad (monotone, slow) and angry (sharp, energetic) have well-documented acoustic signatures that map directly to existing generation parameters. These go unused.

---

## Goal

Add per-emotion acoustic shaping by overwriting `self.temp`, `self.noise_clamp`, and `self.lsd_decode_steps` on the model when `set_emotion()` is called. No post-processing. No core files touched.

---

## Design Decisions

### D1: Which parameters to control
- `temp` — stochastic variance of latent noise → expressiveness vs. monotony
- `noise_clamp` — bounds of the noise distribution → controlled vs. erratic
- `lsd_decode_steps` — flow decode quality → smooth vs. gritty

Rationale: These three are orthogonal axes that together cover pace/energy/clarity without post-processing.

### D2: Configurable at `load_model()` (Option C)
The prosody profile is set once at model load time via an optional `prosody_overrides` dict. `set_emotion()` reads from the merged profile — callers don't pass params per call.

Rationale: Fits the scripted testing use case; keeps `set_emotion()` call-site identical to today.

### D3: Absolute override values (Option B)
Emotion params are absolute values, not multipliers of the base. Setting "angry" always means `temp=1.3`, regardless of what `load_model(temp=X)` received.

Rationale: Predictable, consistent, no hidden scaling math.

---

## Prosody Profile Defaults

Grounded in empirical speech prosody research (CREMA-D / acoustic analysis):

| Emotion   | `temp` | `noise_clamp` | `lsd_decode_steps` | Acoustic Target                  |
|-----------|--------|---------------|--------------------|----------------------------------|
| `angry`   | 1.3    | 3.0           | 1                  | High variance, sharp, energetic  |
| `disgust` | 0.5    | 1.5           | 2                  | Controlled, deliberate, smooth   |
| `fear`    | 1.6    | None          | 1                  | Erratic, unclamped, raw          |
| `happy`   | 0.9    | 2.5           | 2                  | Bright, clear, moderate energy   |
| `neutral` | 0.7    | None          | 1                  | Model defaults                   |
| `sad`     | 0.4    | 0.8           | 3                  | Monotone, tight, very smooth     |

---

## Type Contract

```python
from typing import TypedDict

class EmoProsodyParams(TypedDict):
    temp: float
    noise_clamp: float | None
    lsd_decode_steps: int
```

---

## API

### `EmoTTS.load_model(..., prosody_overrides=None)`

```python
tts = EmoTTS.load_model(
    prosody_overrides={
        "angry": {"temp": 1.5, "lsd_decode_steps": 1}  # partial override OK
    }
)
```

- `prosody_overrides`: `dict[str, dict]` — keys must be valid emotion names; values are **shallow-merged** per-emotion (only provided keys override, rest keep defaults)
- Raises `ValueError` for unknown emotion keys

### `EmoTTS.set_emotion(emotion, intensity)`

No API change. Internally now also sets `self.temp`, `self.noise_clamp`, `self.lsd_decode_steps` from the merged profile.

---

## State Management

- `self._base_params: EmoProsodyParams` — saved at `__init__` from the original `temp/noise_clamp/lsd_decode_steps` passed by `TTSModel`
- `self._prosody_profile: dict[str, EmoProsodyParams]` — the fully merged table
- `set_emotion("neutral", ...)` → restores from `_base_params` (not from the profile neutral entry, so user's custom load params are respected for neutral)

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `pocket_tts/emotts.py` | Modify | Add `EmoProsodyParams`, `EMOTION_PROSODY_DEFAULTS`, update `__init__`, `set_emotion`, `load_model` |
| `tests/test_emotts.py` | Modify | Add tests for prosody param application per emotion |

`scripts/test_emotts_cloning.py` requires **no change** — `set_emotion()` API is unchanged.

---

## Error Handling

- Unknown key in `prosody_overrides` → `ValueError("Unknown emotion 'X'. Valid: [...]")`
- Unknown key in `set_emotion` → already handled by `EmoShiftLayer.set_emotion` assertion

---

## Out of Scope

- Post-processing (pitch shift, time stretch) — explicitly excluded
- Per-call prosody override at `set_emotion()` time — deferred (Option B at call-site)
- Changes to any file in `pocket_tts/models/`, `pocket_tts/modules/` — prohibited
