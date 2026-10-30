"""
LF-* tests for app.helpers.locked_faces (Lock Target Faces persistence).

Pure logic tests: no Qt, no GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.helpers import locked_faces as lf


def _unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


def _entry(embedding, model="arcface_128", sources=None):
    return lf.build_locked_face_entry(
        cropped_face=np.zeros((4, 4, 3), dtype=np.uint8),
        embedding_store={model: embedding},
        parameters={"SimilarityThresholdSlider": 60},
        recognition_model=model,
        assigned_sources=sources or [],
        assigned_input_embedding={"Inswapper128ArcFace": np.ones(4)},
    )


# --- File I/O ----------------------------------------------------------------


def test_lf01_missing_file_returns_empty_data(tmp_path):
    data = lf.load_locked_faces_file(tmp_path / lf.LOCKED_FACES_FILENAME)
    assert data == lf.empty_locked_faces_data()


def test_lf02_save_and_load_roundtrip(tmp_path):
    path = lf.get_locked_faces_path(tmp_path)
    assert path.name == "locked_faces.json"
    entry = _entry(_unit([1, 0, 0, 0]))
    assert lf.save_locked_faces_file(
        path, {"lock_enabled": True, "locked_faces": {"face-1": entry}}
    )
    loaded = lf.load_locked_faces_file(path)
    assert loaded["lock_enabled"] is True
    assert loaded["version"] == lf.LOCKED_FACES_FORMAT_VERSION
    assert loaded["locked_faces"]["face-1"] == json.loads(json.dumps(entry))
    # No temp file left behind (atomic write).
    assert not path.with_name(path.name + ".tmp").exists()


def test_lf03_corrupt_file_is_kept_and_returns_empty(tmp_path):
    path = lf.get_locked_faces_path(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    data = lf.load_locked_faces_file(path)
    assert data["locked_faces"] == {}
    assert (tmp_path / "locked_faces.json.corrupt").read_text(
        encoding="utf-8"
    ) == "{not json"


def test_lf04_invalid_entries_are_dropped(tmp_path):
    path = lf.get_locked_faces_path(tmp_path)
    path.write_text(
        json.dumps({"locked_faces": {"a": {"x": 1}, "b": "nope"}}), encoding="utf-8"
    )
    data = lf.load_locked_faces_file(path)
    assert list(data["locked_faces"]) == ["a"]
    assert data["lock_enabled"] is False


# --- Entry format ----------------------------------------------------------


def test_lf05_entry_is_superset_of_workspace_target_face_format():
    sources = [
        lf.build_input_face_source("in-1", "/faces/alice.png", {"m": np.ones(2)}),
        lf.build_embedding_source("emb-1", "Bob", {"m": np.zeros(2)}),
    ]
    entry = _entry(_unit([1, 0, 0, 0]), sources=sources)
    for key in (
        "cropped_face",
        "embedding_store",
        "parameters",
        "assigned_input_faces",
        "assigned_merged_embeddings",
        "assigned_input_embedding",
    ):
        assert key in entry
    assert entry["assigned_input_faces"] == ["in-1"]
    assert entry["assigned_merged_embeddings"] == ["emb-1"]
    # Sources keep their embeddings, not only the session card ids.
    assert entry["assigned_sources"][0]["embedding_store"] == {"m": [1.0, 1.0]}
    assert entry["recognition_model"] == "arcface_128"
    json.dumps(entry)  # must be serializable


def test_lf06_embedding_store_roundtrip_skips_empty():
    store = {"a": np.arange(3, dtype=np.float32), "b": np.array([]), "c": None}
    serialized = lf.embedding_store_to_json(store)
    assert serialized == {"a": [0.0, 1.0, 2.0]}
    restored = lf.embedding_store_from_json(serialized)
    np.testing.assert_allclose(restored["a"], [0, 1, 2])
    assert restored["a"].dtype == np.float32


# --- Matching / recognition model --------------------------------------------


def test_lf07_similarity_matches_function_worker_scale():
    assert lf.similarity_score([1, 0], [1, 0]) == pytest.approx(100.0)
    assert lf.similarity_score([1, 0], [0, 1]) == pytest.approx(50.0)
    assert lf.similarity_score([1, 0], [-1, 0]) == pytest.approx(0.0)
    assert lf.similarity_score([1, 0], [1, 0, 0]) == 0.0
    assert lf.similarity_score([0, 0], [1, 0]) == 0.0


def test_lf08_recognition_model_mismatch_is_reported():
    entry = _entry(_unit([1, 0, 0, 0]), model="arcface_128")
    ok, reason = lf.check_recognition_model(entry, "arcface_128")
    assert ok and reason == ""
    ok, reason = lf.check_recognition_model(entry, "edgeface")
    assert not ok
    assert "arcface_128" in reason and "edgeface" in reason


def test_lf09_find_matching_face_id_respects_threshold():
    candidates = {
        "near": (_unit([1, 0.1, 0, 0]), 60.0),
        "far": (_unit([0, 0, 1, 0]), 60.0),
        "none": (None, 60.0),
    }
    assert lf.find_matching_face_id(_unit([1, 0, 0, 0]), candidates) == "near"
    candidates["near"] = (_unit([1, 0.1, 0, 0]), 99.9)
    assert lf.find_matching_face_id(_unit([1, 0, 0, 0]), candidates) is None


# --- Source resolution -------------------------------------------------------


def test_lf10_resolve_input_face_by_id_then_media_path():
    source = lf.build_input_face_source("old-id", "/faces/alice.png", {})
    assert lf.resolve_source(source, {"old-id": "/x.png"}, {}) == (
        lf.SOURCE_KIND_INPUT_FACE,
        "old-id",
    )
    assert lf.resolve_source(source, {"new-id": "/faces/alice.png"}, {}) == (
        lf.SOURCE_KIND_INPUT_FACE,
        "new-id",
    )
    assert lf.resolve_source(source, {"x": "/faces/bob.png"}, {}) == (None, None)


def test_lf11_resolve_embedding_by_id_then_name():
    source = lf.build_embedding_source("emb-old", "Alice", {})
    assert lf.resolve_source(source, {}, {"emb-old": "Renamed"}) == (
        lf.SOURCE_KIND_EMBEDDING,
        "emb-old",
    )
    assert lf.resolve_source(source, {}, {"emb-new": "Alice"}) == (
        lf.SOURCE_KIND_EMBEDDING,
        "emb-new",
    )
    assert lf.resolve_source(source, {}, {"emb-new": "Bob"}) == (None, None)
    assert lf.resolve_source({"kind": "unknown"}, {}, {}) == (None, None)


def test_lf12_fallback_embedding_names():
    assert lf.fallback_embedding_name({"kind": "embedding", "name": "Alice"}) == (
        "Alice"
    )
    assert (
        lf.fallback_embedding_name(
            {"kind": "input_face", "media_path": "/faces/alice.png"}
        )
        == "Locked alice"
    )
    assert lf.fallback_embedding_name({"kind": "input_face"}) == "Locked Source"


# --- Restore planning -------------------------------------------------------


def test_lf13_plan_restore_creates_marks_and_skips():
    alice = _unit([1, 0, 0, 0])
    bob = _unit([0, 1, 0, 0])
    carol = _unit([0, 0, 1, 0])
    locked = {
        "alice": _entry(alice),
        "bob": _entry(bob),
        "carol": _entry(carol),
        "dave": _entry(_unit([0, 0, 0, 1]), model="edgeface"),
    }
    existing = {
        "bob": (bob, 60.0),  # same id already loaded -> only mark locked
        "carol-dup": (_unit([0, 0.05, 1, 0]), 60.0),  # same person, other id
    }
    to_create, to_mark, skipped = lf.plan_restore(locked, "arcface_128", existing)
    assert to_create == ["alice"]
    assert to_mark == ["bob"]
    skipped_ids = dict(skipped)
    assert set(skipped_ids) == {"carol", "dave"}
    assert "carol-dup" in skipped_ids["carol"]
    assert "recognition model" in skipped_ids["dave"]


def test_lf14_plan_restore_skips_entries_without_thumbnail():
    entry = _entry(_unit([1, 0, 0, 0]))
    entry["cropped_face"] = []
    to_create, _, skipped = lf.plan_restore({"a": entry}, "arcface_128", {})
    assert to_create == []
    assert skipped and skipped[0][0] == "a"
