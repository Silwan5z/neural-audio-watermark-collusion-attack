"""Shared payload registry, trial schedule, and marked-audio cache."""
from __future__ import annotations

import csv
import hashlib
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermarks import embed, load_audio  # noqa: E402

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CLIP_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache" / "marked"
NBITS = {"audioseal": 16, "timbrewm": 10, "wavmark": 16, "voicemark": 16, "wmcodec": 16}
CAP = 0.5

MANIFEST = DATASET_DIR / "collusion_300" / "manifest.csv"

_MANIFEST_BY_SPEAKER = None


def _manifest_by_speaker():
    """Load and validate the three-utterance-per-speaker manifest once."""
    global _MANIFEST_BY_SPEAKER
    if _MANIFEST_BY_SPEAKER is not None:
        return _MANIFEST_BY_SPEAKER
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing dataset manifest: {MANIFEST}")
    grouped = {}
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            speaker = f"{row['language']}:{row['speaker_id']}"
            item = dict(row)
            item["clip_index"] = int(item["clip_index"])
            relative_path = Path("dataset") / "collusion_300" / item["path"]
            item["path"] = relative_path.as_posix()
            item["_absolute_path"] = str((MANIFEST.parent / row["path"]).resolve())
            grouped.setdefault(speaker, []).append(item)
    for speaker, rows in grouped.items():
        rows.sort(key=lambda r: r["clip_index"])
        indices = [r["clip_index"] for r in rows]
        if indices != [1, 2, 3]:
            raise ValueError(f"{speaker}: expected clip indices [1, 2, 3], got {indices}")
        for row in rows:
            if not Path(row["_absolute_path"]).is_file():
                raise FileNotFoundError(row["_absolute_path"])
    if len(grouped) != 100:
        raise ValueError(f"expected 100 speakers, found {len(grouped)}")
    _MANIFEST_BY_SPEAKER = grouped
    return grouped


def speakers():
    """Return the 100 bilingual speakers from the generated manifest."""
    return sorted(_manifest_by_speaker())


def trial_counts(n_total: int = 300, speaker_count: int = 100):
    """Distribute trials evenly across speakers."""
    quotient, remainder = divmod(n_total, speaker_count)
    return [quotient + 1 if index < remainder else quotient
            for index in range(speaker_count)]


def trial_schedule(n_total: int = 300, speaker_count: int | None = None):
    """Return ``(speaker, clip_slot)`` pairs for the fixed schedule."""
    speaker_ids = speakers()
    speaker_count = len(speaker_ids) if speaker_count is None else speaker_count
    if speaker_count != len(speaker_ids):
        raise ValueError(
            f"requested {speaker_count} speakers but dataset has {len(speaker_ids)}")
    counts = trial_counts(n_total, speaker_count)
    schedule = []
    for speaker, count in zip(speaker_ids, counts):
        for clip_slot in range(count):
            schedule.append((speaker, clip_slot))
    assert len(schedule) == n_total
    return schedule


def coalition_seed(speaker, k, clip_slot):
    """Return the deterministic coalition seed for one trial."""
    h = int.from_bytes(hashlib.sha256(speaker.encode("utf-8")).digest()[:4], "little")
    return (h * 100000 + k * 1000 + clip_slot + 42) % (2 ** 31)


def source_record(speaker, clip_slot=0):
    """Return the manifest row selected by a speaker-local trial index.

    In the 300-trial paper schedule, clip_slot=0,1,2 maps to the speaker's three
    distinct utterances. Larger schedules cycle over those utterances while
    retaining a distinct clip_slot for payload seeding.
    """
    rows = _manifest_by_speaker().get(speaker)
    if rows is None:
        raise KeyError(f"unknown speaker: {speaker}")
    return rows[int(clip_slot) % len(rows)]


def clean_path(speaker, clip_slot=0):
    """Return the scheduled clean utterance."""
    return Path(source_record(speaker, clip_slot)["_absolute_path"])


def load_clean(speaker, clip_slot=0, sr=16000):
    return load_audio(clean_path(speaker, clip_slot), sr)


def cache_path_for(model, speaker, clip_slot, payload):
    """Return the cache path for one marked utterance and payload."""
    clip_index = int(source_record(speaker, clip_slot)["clip_index"])
    return CLIP_CACHE_DIR / model / speaker / f"clip_{clip_index:02d}" / f"{payload}.wav"


def full_registry_size(model):
    return 2 ** NBITS[model]


def random_codeword_int(rng, model):
    return int(rng.integers(0, full_registry_size(model)))


def int_to_bits(value, length):
    """Convert an integer payload to an LSB-first bit vector."""
    return np.array(
        [(value >> index) & 1 for index in range(length)], dtype=np.int8)


def get_or_embed(model, speaker, payload, clip_slot=0):
    """Load or create one 16 kHz marked copy and cache it atomically."""
    payload_length = NBITS[model]
    cache_path = cache_path_for(model, speaker, clip_slot, payload)
    if cache_path.exists():
        # Several GPU workers can request a shared payload simultaneously.
        # Only accept a complete, finite cache file; otherwise regenerate it.
        try:
            cached, sr = sf.read(cache_path, dtype="float32")
            if sr == 16000 and len(cached) >= 16000 and np.isfinite(cached).all():
                return cached
        except RuntimeError:
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    clean = load_clean(speaker, clip_slot)
    bits = int_to_bits(payload, payload_length).tolist()
    wm = embed(model, clean, bits)
    # Publish atomically: readers observe either an existing valid WAV or the
    # fully written replacement, never a partial header/body.
    tmp_path = cache_path.parent / f".{payload}.{uuid.uuid4().hex}.wav"
    sf.write(tmp_path, wm, 16000, subtype="FLOAT")
    os.replace(tmp_path, cache_path)
    return wm


def sample_coalition(rng, model, k):
    """Sample k distinct payloads from a model's full payload space."""
    n = full_registry_size(model)
    return sorted(rng.choice(n, size=k, replace=False).tolist())


_REGISTRY_CACHE = {}


def full_registry_bits(model):
    """Return every payload in a model's registry as LSB-first bits."""
    if model in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[model]
    d = NBITS[model]
    n = 2 ** d
    ints = np.arange(n)
    bits = np.zeros((n, d), dtype=np.int8)
    for i in range(d):
        bits[:, i] = (ints >> i) & 1
    _REGISTRY_CACHE[model] = bits
    return bits
