"""
Pure (Qt-free, GPU-free) helpers for the "Lock Target Faces" feature.

Locked target faces are persisted to ``locked_faces.json`` in the project root
so the same target people are automatically restored (together with the source
faces / embeddings assigned to them) on every media load and after restarts.

Each locked face entry is a superset of the workspace ``target_faces_data``
format (``cropped_face``, ``embedding_store``, ``parameters``,
``assigned_input_faces``, ``assigned_merged_embeddings``,
``assigned_input_embedding``) plus:

- ``recognition_model``: the recognition model active when the face was locked.
- ``assigned_sources``: the assigned sources stored with their *embeddings*
  (not only their session card ids), so the pairing survives restarts even
  when the original input-face / embedding cards are not loaded.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

LOCKED_FACES_FILENAME = "locked_faces.json"
LOCKED_FACES_FORMAT_VERSION = 1

SOURCE_KIND_INPUT_FACE = "input_face"
SOURCE_KIND_EMBEDDING = "embedding"


def get_locked_faces_path(project_root_path: Path | str) -> Path:
    """Return the location of the locked faces file inside the project root."""
    return Path(project_root_path) / LOCKED_FACES_FILENAME


def empty_locked_faces_data() -> Dict[str, Any]:
    return {
        "version": LOCKED_FACES_FORMAT_VERSION,
        "lock_enabled": False,
        "locked_faces": {},
    }


# ---------------------------------------------------------------------------
# Embedding (de)serialization
# ---------------------------------------------------------------------------


def embedding_store_to_json(
    embedding_store: Optional[Mapping[str, Any]],
) -> Dict[str, list]:
    """Convert ``{model: np.ndarray}`` into a JSON serializable dict."""
    serialized: Dict[str, list] = {}
    for model, embedding in (embedding_store or {}).items():
        if embedding is None:
            continue
        array = np.asarray(embedding)
        if array.size == 0:
            continue
        serialized[str(model)] = array.tolist()
    return serialized


def embedding_store_from_json(
    data: Optional[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """Convert a JSON embedding store back into ``{model: np.ndarray}``."""
    store: Dict[str, np.ndarray] = {}
    for model, embedding in (data or {}).items():
        if embedding is None:
            continue
        array = np.asarray(embedding, dtype=np.float32)
        if array.size == 0:
            continue
        store[str(model)] = array
    return store


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_locked_faces_file(path: Path | str) -> Dict[str, Any]:
    """Load the locked faces file.

    Never raises: a missing file returns empty data. A corrupt file is copied to
    ``locked_faces.json.corrupt`` (so it is not silently lost on the next save)
    and empty data is returned.
    """
    path = Path(path)
    data = empty_locked_faces_data()
    if not path.is_file():
        return data
    try:
        with open(path, "r", encoding="utf-8") as data_file:
            loaded = json.load(data_file)
        if not isinstance(loaded, dict):
            raise ValueError("Top level JSON value is not an object")
    except (OSError, ValueError) as e:
        print(f"[WARN] Could not read locked faces file '{path}': {e}")
        try:
            shutil.copyfile(path, str(path) + ".corrupt")
            print(f"[WARN] A copy of the unreadable file was kept as '{path}.corrupt'.")
        except OSError:
            pass
        return data

    locked_faces = loaded.get("locked_faces", {})
    if not isinstance(locked_faces, dict):
        locked_faces = {}
    data["version"] = loaded.get("version", LOCKED_FACES_FORMAT_VERSION)
    data["lock_enabled"] = bool(loaded.get("lock_enabled", False))
    data["locked_faces"] = {
        str(face_id): entry
        for face_id, entry in locked_faces.items()
        if isinstance(entry, dict)
    }
    return data


def save_locked_faces_file(path: Path | str, data: Mapping[str, Any]) -> bool:
    """Atomically write the locked faces file. Returns True on success."""
    path = Path(path)
    payload = {
        "version": LOCKED_FACES_FORMAT_VERSION,
        "lock_enabled": bool(data.get("lock_enabled", False)),
        "locked_faces": dict(data.get("locked_faces", {})),
    }
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as data_file:
            json.dump(payload, data_file)
        os.replace(tmp_path, path)
        return True
    except (OSError, TypeError, ValueError) as e:
        print(f"[ERROR] Could not save locked faces file '{path}': {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Entry building
# ---------------------------------------------------------------------------


def build_input_face_source(
    face_id: str, media_path: str, embedding_store: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "kind": SOURCE_KIND_INPUT_FACE,
        "id": str(face_id),
        "media_path": str(media_path or ""),
        "embedding_store": embedding_store_to_json(embedding_store),
    }


def build_embedding_source(
    embedding_id: str, embedding_name: str, embedding_store: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "kind": SOURCE_KIND_EMBEDDING,
        "id": str(embedding_id),
        "name": str(embedding_name or ""),
        "embedding_store": embedding_store_to_json(embedding_store),
    }


def build_locked_face_entry(
    cropped_face: Any,
    embedding_store: Mapping[str, Any],
    parameters: Mapping[str, Any],
    recognition_model: str,
    assigned_sources: Iterable[Mapping[str, Any]],
    assigned_input_embedding: Optional[Mapping[str, Any]] = None,
    locked_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one JSON serializable locked face entry."""
    sources = [dict(source) for source in assigned_sources]
    return {
        "cropped_face": np.asarray(cropped_face, dtype=np.uint8).tolist(),
        "embedding_store": embedding_store_to_json(embedding_store),
        "parameters": dict(parameters or {}),
        "recognition_model": str(recognition_model),
        "assigned_input_faces": [
            s["id"] for s in sources if s.get("kind") == SOURCE_KIND_INPUT_FACE
        ],
        "assigned_merged_embeddings": [
            s["id"] for s in sources if s.get("kind") == SOURCE_KIND_EMBEDDING
        ],
        "assigned_sources": sources,
        "assigned_input_embedding": embedding_store_to_json(assigned_input_embedding),
        "locked_at": locked_at or datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Restore logic
# ---------------------------------------------------------------------------


def check_recognition_model(
    entry: Mapping[str, Any], recognition_model: str
) -> Tuple[bool, str]:
    """Check whether a locked face can be matched with the current recognition model.

    The face is compatible when its stored ``embedding_store`` contains a
    non-empty embedding for ``recognition_model``. Returns ``(ok, reason)``.
    """
    store = entry.get("embedding_store") or {}
    embedding = store.get(recognition_model)
    if embedding is not None and np.asarray(embedding).size > 0:
        return True, ""
    stored_model = entry.get("recognition_model") or ", ".join(store.keys()) or "none"
    return (
        False,
        f"locked with recognition model '{stored_model}', but the current "
        f"Face Recognition Model is '{recognition_model}'",
    )


def similarity_score(vector1: Any, vector2: Any) -> float:
    """Same scale as ``FunctionWorker.findCosineDistance`` (0-100, 100 = identical)."""
    v1 = np.asarray(vector1, dtype=np.float64).ravel()
    v2 = np.asarray(vector2, dtype=np.float64).ravel()
    if v1.size == 0 or v2.size == 0 or v1.size != v2.size:
        return 0.0
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denominator == 0:
        return 0.0
    cos_dist = 1 - float(np.dot(v1, v2)) / float(denominator)
    return 100 - cos_dist * 50


def find_matching_face_id(
    embedding: Any,
    candidates: Mapping[str, Tuple[Any, float]],
) -> Optional[str]:
    """Return the id of the best candidate whose similarity >= its threshold.

    ``candidates`` maps face_id -> (embedding, similarity_threshold).
    """
    best_id: Optional[str] = None
    best_score = -1.0
    for face_id, (candidate_embedding, threshold) in candidates.items():
        if candidate_embedding is None:
            continue
        score = similarity_score(embedding, candidate_embedding)
        if score >= float(threshold) and score > best_score:
            best_score = score
            best_id = face_id
    return best_id


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)) if path else ""


def resolve_source(
    source: Mapping[str, Any],
    input_faces: Mapping[str, str],
    embeddings: Mapping[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    """Find a card currently loaded in the session for a stored source.

    Args:
        source: a stored ``assigned_sources`` item.
        input_faces: loaded input faces, face_id -> media_path.
        embeddings: loaded merged embeddings, embedding_id -> embedding_name.

    Returns ``(kind, card_id)`` or ``(None, None)`` when no card matches.
    Matching order: same card id, then same media path (input faces) or same
    name (embeddings).
    """
    kind = source.get("kind")
    source_id = str(source.get("id", ""))
    if kind == SOURCE_KIND_INPUT_FACE:
        if source_id and source_id in input_faces:
            return SOURCE_KIND_INPUT_FACE, source_id
        wanted = _normalize_path(str(source.get("media_path", "")))
        if wanted:
            for face_id, media_path in input_faces.items():
                if _normalize_path(str(media_path)) == wanted:
                    return SOURCE_KIND_INPUT_FACE, face_id
    elif kind == SOURCE_KIND_EMBEDDING:
        if source_id and source_id in embeddings:
            return SOURCE_KIND_EMBEDDING, source_id
        wanted_name = str(source.get("name", ""))
        if wanted_name:
            for embedding_id, name in embeddings.items():
                if name == wanted_name:
                    return SOURCE_KIND_EMBEDDING, embedding_id
    return None, None


def fallback_embedding_name(source: Mapping[str, Any]) -> str:
    """Name for an embedding card created when a stored source is not loaded."""
    if source.get("kind") == SOURCE_KIND_EMBEDDING and source.get("name"):
        return str(source["name"])
    media_path = str(source.get("media_path", ""))
    if media_path:
        return f"Locked {Path(media_path).stem}"
    return "Locked Source"


def plan_restore(
    locked_faces: Mapping[str, Mapping[str, Any]],
    recognition_model: str,
    existing_faces: Mapping[str, Tuple[Any, float]],
) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """Decide what to do with each locked face.

    Args:
        locked_faces: face_id -> entry (from the locked faces file).
        recognition_model: currently selected recognition model.
        existing_faces: target faces already in the session,
            face_id -> (embedding for recognition_model, similarity threshold).

    Returns ``(to_create, to_mark_locked, skipped)``:
        - to_create: face ids whose cards must be created.
        - to_mark_locked: face ids already present in the session (same id).
        - skipped: (face_id, reason) pairs that are not restored.
    """
    to_create: List[str] = []
    to_mark: List[str] = []
    skipped: List[Tuple[str, str]] = []
    for face_id, entry in locked_faces.items():
        if face_id in existing_faces:
            to_mark.append(face_id)
            continue
        ok, reason = check_recognition_model(entry, recognition_model)
        if not ok:
            skipped.append((face_id, reason))
            continue
        if not entry.get("cropped_face"):
            skipped.append((face_id, "no face thumbnail stored"))
            continue
        embedding = entry["embedding_store"][recognition_model]
        duplicate_of = find_matching_face_id(embedding, existing_faces)
        if duplicate_of is not None:
            skipped.append(
                (
                    face_id,
                    f"same person is already in the target faces ({duplicate_of})",
                )
            )
            continue
        to_create.append(face_id)
    return to_create, to_mark, skipped
