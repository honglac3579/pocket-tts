# EmoTTS Prosody Parameter Shaping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `EmoTTS.set_emotion()` to also overwrite `self.temp`, `self.noise_clamp`, and `self.lsd_decode_steps` with emotion-specific absolute values, shaping acoustic prosody without post-processing.

**Architecture:** A module-level `EMOTION_PROSODY_DEFAULTS` dict maps each emotion to an `EmoProsodyParams` TypedDict. At `load_model()` time, this is shallow-merged with an optional `prosody_overrides` dict and stored on the instance. `set_emotion()` reads from the merged profile and overwrites the three generation params in place.

**Tech Stack:** Python 3.10+, PyTorch, existing `pocket_tts/emotts.py`. No new dependencies.

---

## File Map

| File | Action |
|------|--------|
| `pocket_tts/emotts.py` | Modify — add TypedDict, defaults table, update `__init__`, `set_emotion`, `load_model` |
| `tests/test_emotts.py` | Modify — add prosody param tests |

---

## Task 1: Add `EmoProsodyParams` TypedDict and `EMOTION_PROSODY_DEFAULTS`

**Files:**
- Modify: `pocket_tts/emotts.py`
- Test: `tests/test_emotts.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to tests/test_emotts.py:
from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS, EmoProsodyParams

def test_emotion_prosody_defaults_covers_all_emotions():
    from pocket_tts.emotts import EmoShiftLayer
    for emotion in EmoShiftLayer.EMOTIONS:
        assert emotion in EMOTION_PROSODY_DEFAULTS, f"Missing prosody params for {emotion}"

def test_emotion_prosody_defaults_schema():
    for emotion, params in EMOTION_PROSODY_DEFAULTS.items():
        assert "temp" in params
        assert "lsd_decode_steps" in params
        assert "noise_clamp" in params
        assert isinstance(params["temp"], float)
        assert isinstance(params["lsd_decode_steps"], int)
        assert params["noise_clamp"] is None or isinstance(params["noise_clamp"], float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_emotts.py::test_emotion_prosody_defaults_covers_all_emotions -v
```

Expected: `ImportError: cannot import name 'EMOTION_PROSODY_DEFAULTS'`

- [ ] **Step 3: Add TypedDict and defaults table to `pocket_tts/emotts.py`**

After the existing imports block, add:

```python
from typing import TypedDict


class EmoProsodyParams(TypedDict):
    temp: float
    noise_clamp: float | None
    lsd_decode_steps: int


EMOTION_PROSODY_DEFAULTS: dict[str, EmoProsodyParams] = {
    "angry":   {"temp": 1.3, "noise_clamp": 3.0,  "lsd_decode_steps": 1},
    "disgust": {"temp": 0.5, "noise_clamp": 1.5,  "lsd_decode_steps": 2},
    "fear":    {"temp": 1.6, "noise_clamp": None,  "lsd_decode_steps": 1},
    "happy":   {"temp": 0.9, "noise_clamp": 2.5,  "lsd_decode_steps": 2},
    "neutral": {"temp": 0.7, "noise_clamp": None,  "lsd_decode_steps": 1},
    "sad":     {"temp": 0.4, "noise_clamp": 0.8,  "lsd_decode_steps": 3},
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_emotts.py::test_emotion_prosody_defaults_covers_all_emotions tests/test_emotts.py::test_emotion_prosody_defaults_schema -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pocket_tts/emotts.py tests/test_emotts.py
git commit -m "feat(emotts): add EmoProsodyParams TypedDict and EMOTION_PROSODY_DEFAULTS table"
```

---

## Task 2: Store prosody profile and base params in `EmoTTS.__init__`

**Files:**
- Modify: `pocket_tts/emotts.py` — `EmoTTS.__init__`
- Test: `tests/test_emotts.py`

- [ ] **Step 1: Write the failing tests**

Add pytest fixture and tests to `tests/test_emotts.py`:

```python
import pytest
from unittest.mock import MagicMock
import torch

@pytest.fixture
def mock_emo_tts(tmp_path):
    from pocket_tts.emotts import EmoShiftLayer, EmoTTS
    weights_path = tmp_path / "SouraTTS.pt"
    torch.save(EmoShiftLayer().state_dict(), weights_path)

    base = MagicMock()
    base.parameters.return_value = iter([torch.zeros(1)])
    base.temp = 0.7
    base.noise_clamp = None
    base.lsd_decode_steps = 1
    base.flow_lm.transformer.layers = [MagicMock() for _ in range(10)]
    base.__dict__ = {
        "temp": 0.7, "noise_clamp": None, "lsd_decode_steps": 1,
        "flow_lm": base.flow_lm,
    }
    return EmoTTS(base, weights_path)

def test_emotts_stores_base_params(mock_emo_tts):
    assert hasattr(mock_emo_tts, "_base_params")
    assert mock_emo_tts._base_params["temp"] == 0.7
    assert mock_emo_tts._base_params["noise_clamp"] is None
    assert mock_emo_tts._base_params["lsd_decode_steps"] == 1

def test_emotts_stores_prosody_profile(mock_emo_tts):
    from pocket_tts.emotts import EmoShiftLayer
    assert hasattr(mock_emo_tts, "_prosody_profile")
    for emotion in EmoShiftLayer.EMOTIONS:
        assert emotion in mock_emo_tts._prosody_profile
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_emotts.py::test_emotts_stores_base_params -v
```

Expected: `AttributeError: '_base_params'`

- [ ] **Step 3: Update `EmoTTS.__init__` to store base params and profile**

In `pocket_tts/emotts.py`, add these lines at the end of `EmoTTS.__init__`, after `self._register_hook()`:

```python
        # Snapshot original generation params for neutral restoration
        self._base_params: EmoProsodyParams = {
            "temp": tts_model.temp,
            "noise_clamp": tts_model.noise_clamp,
            "lsd_decode_steps": tts_model.lsd_decode_steps,
        }

        # Prosody profile: defaults (overrides applied later in load_model or via constructor)
        self._prosody_profile: dict[str, EmoProsodyParams] = {
            emotion: dict(params)
            for emotion, params in EMOTION_PROSODY_DEFAULTS.items()
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_emotts.py::test_emotts_stores_base_params tests/test_emotts.py::test_emotts_stores_prosody_profile -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add pocket_tts/emotts.py tests/test_emotts.py
git commit -m "feat(emotts): store _base_params and _prosody_profile at init"
```

---

## Task 3: Apply prosody params in `set_emotion()`

**Files:**
- Modify: `pocket_tts/emotts.py` — `EmoTTS.set_emotion`
- Test: `tests/test_emotts.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_set_emotion_overwrites_generation_params(mock_emo_tts):
    from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS
    mock_emo_tts.set_emotion("angry", intensity=1.0)
    assert mock_emo_tts.temp == EMOTION_PROSODY_DEFAULTS["angry"]["temp"]
    assert mock_emo_tts.noise_clamp == EMOTION_PROSODY_DEFAULTS["angry"]["noise_clamp"]
    assert mock_emo_tts.lsd_decode_steps == EMOTION_PROSODY_DEFAULTS["angry"]["lsd_decode_steps"]

def test_set_emotion_sad_params(mock_emo_tts):
    from pocket_tts.emotts import EMOTION_PROSODY_DEFAULTS
    mock_emo_tts.set_emotion("sad", intensity=1.0)
    assert mock_emo_tts.temp == EMOTION_PROSODY_DEFAULTS["sad"]["temp"]
    assert mock_emo_tts.lsd_decode_steps == EMOTION_PROSODY_DEFAULTS["sad"]["lsd_decode_steps"]

def test_set_neutral_restores_base_params(mock_emo_tts):
    mock_emo_tts.set_emotion("angry", intensity=1.0)
    mock_emo_tts.set_emotion("neutral", intensity=0.0)
    assert mock_emo_tts.temp == mock_emo_tts._base_params["temp"]
    assert mock_emo_tts.noise_clamp == mock_emo_tts._base_params["noise_clamp"]
    assert mock_emo_tts.lsd_decode_steps == mock_emo_tts._base_params["lsd_decode_steps"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_emotts.py::test_set_emotion_overwrites_generation_params -v
```

Expected: FAIL — `self.temp` is not changed

- [ ] **Step 3: Replace `EmoTTS.set_emotion` in `pocket_tts/emotts.py`**

```python
    def set_emotion(self, emotion: str, intensity: float = 1.0) -> None:
        """Set the active emotion, intensity, and acoustic generation params."""
        self.emo_layer.set_emotion(emotion, intensity)

        if emotion == "neutral" or intensity == 0.0:
            # Restore original load-time params
            self.temp = self._base_params["temp"]
            self.noise_clamp = self._base_params["noise_clamp"]
            self.lsd_decode_steps = self._base_params["lsd_decode_steps"]
        else:
            profile = self._prosody_profile[emotion]
            self.temp = profile["temp"]
            self.noise_clamp = profile["noise_clamp"]
            self.lsd_decode_steps = profile["lsd_decode_steps"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_emotts.py::test_set_emotion_overwrites_generation_params tests/test_emotts.py::test_set_emotion_sad_params tests/test_emotts.py::test_set_neutral_restores_base_params -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pocket_tts/emotts.py tests/test_emotts.py
git commit -m "feat(emotts): apply acoustic prosody params in set_emotion"
```

---

## Task 4: Add `prosody_overrides` to `EmoTTS.__init__` and `load_model()`

**Files:**
- Modify: `pocket_tts/emotts.py` — `EmoTTS.__init__`, `EmoTTS.load_model`
- Test: `tests/test_emotts.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_prosody_overrides_merge(tmp_path):
    from pocket_tts.emotts import EmoShiftLayer, EmoTTS, EMOTION_PROSODY_DEFAULTS
    from unittest.mock import MagicMock
    import torch

    weights_path = tmp_path / "SouraTTS.pt"
    torch.save(EmoShiftLayer().state_dict(), weights_path)

    base = MagicMock()
    base.parameters.return_value = iter([torch.zeros(1)])
    base.temp = 0.7
    base.noise_clamp = None
    base.lsd_decode_steps = 1
    base.flow_lm.transformer.layers = [MagicMock() for _ in range(10)]
    base.__dict__ = {"temp": 0.7, "noise_clamp": None, "lsd_decode_steps": 1, "flow_lm": base.flow_lm}

    model = EmoTTS(base, weights_path, prosody_overrides={"angry": {"temp": 2.0}})
    assert model._prosody_profile["angry"]["temp"] == 2.0
    # Non-overridden keys keep defaults
    assert model._prosody_profile["angry"]["lsd_decode_steps"] == EMOTION_PROSODY_DEFAULTS["angry"]["lsd_decode_steps"]
    # Other emotions unaffected
    assert model._prosody_profile["sad"]["temp"] == EMOTION_PROSODY_DEFAULTS["sad"]["temp"]

def test_prosody_overrides_unknown_emotion_raises(tmp_path):
    from pocket_tts.emotts import EmoShiftLayer, EmoTTS
    from unittest.mock import MagicMock
    import torch

    weights_path = tmp_path / "SouraTTS.pt"
    torch.save(EmoShiftLayer().state_dict(), weights_path)

    base = MagicMock()
    base.parameters.return_value = iter([torch.zeros(1)])
    base.temp = 0.7
    base.noise_clamp = None
    base.lsd_decode_steps = 1
    base.flow_lm.transformer.layers = [MagicMock() for _ in range(10)]
    base.__dict__ = {"temp": 0.7, "noise_clamp": None, "lsd_decode_steps": 1, "flow_lm": base.flow_lm}

    with pytest.raises(ValueError, match="Unknown emotion"):
        EmoTTS(base, weights_path, prosody_overrides={"excited": {"temp": 1.5}})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_emotts.py::test_prosody_overrides_merge -v
```

Expected: `TypeError: __init__() got unexpected keyword argument 'prosody_overrides'`

- [ ] **Step 3: Update `EmoTTS.__init__` signature and apply overrides**

Replace the `__init__` signature and the `_prosody_profile` construction block:

```python
    def __init__(
        self,
        tts_model: TTSModel,
        weights_path: Path,
        meta: dict | None = None,
        prosody_overrides: dict[str, dict] | None = None,
    ):
        nn.Module.__init__(self)
        self.__dict__.update(tts_model.__dict__)

        device = next(tts_model.parameters()).device
        self.emo_layer = EmoShiftLayer(hidden_size=1024, injection_layer=5)

        state = torch.load(weights_path, map_location="cpu")
        self.emo_layer.load_state_dict(state)
        self.emo_layer.to(device)
        self.emo_layer.eval()

        self.meta = meta or {}
        self._hook_handle = None
        self._register_hook()

        self._base_params: EmoProsodyParams = {
            "temp": tts_model.temp,
            "noise_clamp": tts_model.noise_clamp,
            "lsd_decode_steps": tts_model.lsd_decode_steps,
        }

        self._prosody_profile: dict[str, EmoProsodyParams] = {
            emotion: dict(params)
            for emotion, params in EMOTION_PROSODY_DEFAULTS.items()
        }
        if prosody_overrides:
            unknown = set(prosody_overrides) - set(EMOTION_PROSODY_DEFAULTS)
            if unknown:
                valid = sorted(EMOTION_PROSODY_DEFAULTS.keys())
                raise ValueError(f"Unknown emotion(s) in prosody_overrides: {sorted(unknown)}. Valid: {valid}")
            for emotion, overrides in prosody_overrides.items():
                self._prosody_profile[emotion].update(overrides)
```

Update `EmoTTS.load_model` to accept and forward `prosody_overrides`:

```python
    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: bool = False,
        weights_url: str = "hf://Sourajit123/SouraTTS/SouraTTS.pt",
        meta_url: str = "hf://Sourajit123/SouraTTS/SouraTTS.json",
        prosody_overrides: dict[str, dict] | None = None,
    ) -> "EmoTTS":
        """Load the base model and wrap it with EmoShift steering."""
        logger.info("Loading base TTSModel...")
        tts_model = TTSModel.load_model(
            language=language,
            config=config,
            temp=temp,
            lsd_decode_steps=lsd_decode_steps,
            noise_clamp=noise_clamp,
            eos_threshold=eos_threshold,
            quantize=quantize,
        )

        logger.info(f"Downloading EmoShift weights from {weights_url}...")
        weights_path = download_if_necessary(weights_url)
        meta_path = download_if_necessary(meta_url)

        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata json: {e}")

        return cls(tts_model, weights_path, meta, prosody_overrides=prosody_overrides)
```

- [ ] **Step 4: Run all prosody tests**

```bash
uv run pytest tests/test_emotts.py -v -k "prosody or override or base_params"
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add pocket_tts/emotts.py tests/test_emotts.py
git commit -m "feat(emotts): add prosody_overrides support to __init__ and load_model"
```

---

## Task 5: Full regression check and audio verification

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -n 3 -v
```

Expected: all tests pass

- [ ] **Step 2: Run cloning script to verify audible output with prosody params active**

```bash
uv run python scripts/test_emotts_cloning.py
```

Expected: 5 files `output_*_alba.wav` generated successfully

- [ ] **Step 3: Verify all 5 outputs are distinct (different prosody = different audio)**

```bash
uv run python -c "
import hashlib, glob
for f in sorted(glob.glob('output_*_alba.wav')):
    print(f, hashlib.md5(open(f,'rb').read()).hexdigest())
"
```

Expected: 5 different hashes

- [ ] **Step 4: Update aidex index**

```bash
# Run in project root — no shell command, use aidex_update tool on emotts.py and test_emotts.py
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "test(emotts): prosody param shaping verified — 5 distinct emotional audio outputs"
```
