"""Shared payload registry, trial schedule, and marked-audio cache."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
import uuid
from functools import lru_cache
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
CACHE_SCHEMA = 2

MANIFEST = DATASET_DIR / "collusion_300" / "manifest.csv"

_MANIFEST_BY_SPEAKER = None


@lru_cache(maxsize=None)
def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _backend_fingerprint(model: str) -> dict[str, str]:
    packages = {"audioseal": "audioseal", "wavmark": "wavmark"}
    package = packages.get(model)
    version = "vendored"
    if package is not None:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
    root = Path(__file__).resolve().parent.parent
    checkpoints = {
        "timbrewm": root / "third_party" / "timbrewm" / "results" / "ckpt" /
        "pth" / "compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar",
        "voicemark": root / "third_party" / "voicemark" / "voicemark.pth",
        "wmcodec": root / "third_party" / "wmcodec" / "save_model" /
        "g_00150000",
    }
    checkpoint = checkpoints.get(model)
    checkpoint_sha256 = (
        _sha256(str(checkpoint)) if checkpoint is not None and checkpoint.is_file()
        else "package-managed")
    implementation = root / "src" / "watermarks.py"
    native_implementation = root / "scripts" / "native_audio.py"
    return {
        "package_version": version,
        "checkpoint_sha256": checkpoint_sha256,
        "implementation_sha256": _sha256(str(implementation)),
        "native_implementation_sha256": _sha256(str(native_implementation)),
    }


def expected_cache_metadata(model: str, speaker: str, clip_slot: int,
                            payload: int, sample_rate: int) -> dict:
    source = Path(source_record(speaker, clip_slot)["_absolute_path"])
    return {
        "cache_schema": CACHE_SCHEMA,
        "model": model,
        "speaker": speaker,
        "clip_index": int(source_record(speaker, clip_slot)["clip_index"]),
        "payload": int(payload),
        "sample_rate": int(sample_rate),
        "source_audio_sha256": _sha256(str(source)),
        "backend": _backend_fingerprint(model),
    }


def cache_metadata_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(audio_path.suffix + ".json")


def cache_metadata_matches(audio_path: Path, expected: dict) -> bool:
    path = cache_metadata_path(audio_path)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")) == expected
    except (OSError, json.JSONDecodeError):
        return False


def write_cache_metadata(audio_path: Path, metadata: dict) -> None:
    path = cache_metadata_path(audio_path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _manifest_by_speaker():
    """Load and validate the three-utterance-per-speaker manifest once."""
    global _MANIFEST_BY_SPEAKER
    if _MANIFEST_BY_SPEAKER is not None:
        return _MANIFEST_BY_SPEAKER
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing dataset manifest: {MANIFEST}")
    grouped = {}
    observed_paths: set[Path] = set()
    with MANIFEST.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            speaker = f"{row['language']}:{row['speaker_id']}"
            item = dict(row)
            item["clip_index"] = int(item["clip_index"])
            if int(item["sample_rate"]) != 16000:
                raise ValueError(f"manifest sample rate must be 16000: {row}")
            if float(item["duration_seconds"]) != 10.0:
                raise ValueError(f"manifest duration must be 10.0 seconds: {row}")
            relative_path = Path("dataset") / "collusion_300" / item["path"]
            item["path"] = relative_path.as_posix()
            absolute_path = (MANIFEST.parent / row["path"]).resolve()
            if absolute_path in observed_paths:
                raise ValueError(f"duplicate audio path in manifest: {absolute_path}")
            observed_paths.add(absolute_path)
            item["_absolute_path"] = str(absolute_path)
            grouped.setdefault(speaker, []).append(item)
    for speaker, rows in grouped.items():
        rows.sort(key=lambda r: r["clip_index"])
        indices = [r["clip_index"] for r in rows]
        if indices != [1, 2, 3]:
            raise ValueError(f"{speaker}: expected clip indices [1, 2, 3], got {indices}")
        for row in rows:
            path = Path(row["_absolute_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            info = sf.info(str(path))
            if info.samplerate != 16000 or info.channels != 1:
                raise ValueError(
                    f"{path}: expected mono 16 kHz WAV, got "
                    f"{info.channels} channel(s) at {info.samplerate} Hz")
            if info.frames != 160000:
                raise ValueError(
                    f"{path}: expected exactly 160000 frames, got {info.frames}")
    if len(grouped) != 100:
        raise ValueError(f"expected 100 speakers, found {len(grouped)}")
    if len(observed_paths) != 300:
        raise ValueError(f"expected 300 unique audio paths, found {len(observed_paths)}")
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
    metadata = expected_cache_metadata(
        model, speaker, clip_slot, payload, sample_rate=16000)
    if cache_path.exists() and cache_metadata_matches(cache_path, metadata):
        # Several GPU workers can request a shared payload simultaneously.
        # Only accept a complete, finite cache file; otherwise regenerate it.
        try:
            cached, sr = sf.read(str(cache_path), dtype="float32")
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
    sf.write(str(tmp_path), wm, 16000, subtype="FLOAT")
    os.replace(tmp_path, cache_path)
    write_cache_metadata(cache_path, metadata)
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
