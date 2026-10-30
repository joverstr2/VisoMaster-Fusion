"""
LFA-* tests for app.ui.widgets.actions.locked_face_actions.

Uses lightweight stand-ins for MainWindow and the face card buttons; the
list_view_actions / common_actions helpers that create Qt widgets are
monkeypatched. No GPU required.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.helpers import locked_faces as lf
from app.ui.widgets.actions import locked_face_actions as lfa

MODEL = "arcface_128"


def _unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeTargetFace:
    def __init__(self, face_id, embedding):
        self.face_id = face_id
        self.cropped_face = np.zeros((4, 4, 3), dtype=np.uint8)
        self.embedding_store = {MODEL: embedding}
        self.assigned_input_faces = {}
        self.assigned_merged_embeddings = {}
        self.assigned_input_embedding = {}
        self.is_locked = False
        self.clicked = 0
        self.label_refreshes = 0

    def get_embedding(self, model):
        return self.embedding_store.get(model)

    def calculate_assigned_input_embedding(self):
        stores = list(self.assigned_input_faces.values()) + list(
            self.assigned_merged_embeddings.values()
        )
        self.assigned_input_embedding = (
            {"Inswapper128ArcFace": np.mean([s[MODEL] for s in stores], axis=0)}
            if stores
            else {}
        )

    def refresh_display_label(self):
        self.label_refreshes += 1

    def click(self):
        self.clicked += 1
        self.main_window.selected_target_face_id = self.face_id


class _FakeToggle:
    def __init__(self):
        self.values = []

    def set_value(self, value):
        self.values.append(value)


@pytest.fixture
def toasts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lfa.common_widget_actions,
        "create_and_show_toast_message",
        lambda mw, title, message, style_type="information": calls.append(
            (title, style_type)
        ),
    )
    return calls


@pytest.fixture
def main_window(tmp_path, monkeypatch, toasts):
    mw = SimpleNamespace(
        project_root_path=tmp_path,
        control={"RecognitionModelSelection": MODEL, lfa.LOCK_TOGGLE_NAME: False},
        target_faces={},
        input_faces={},
        merged_embeddings={},
        parameters={},
        default_parameters=SimpleNamespace(data={"SimilarityThresholdSlider": 60}),
        selected_target_face_id=None,
        is_batch_processing=False,
        parameter_widgets={lfa.LOCK_TOGGLE_NAME: _FakeToggle()},
    )

    def add_target(main_window, cropped_face, embedding_store, pixmap, face_id):
        face = _FakeTargetFace(face_id, embedding_store.get(MODEL))
        face.main_window = main_window
        main_window.target_faces[face_id] = face

    def add_embedding(main_window, name, store, embedding_id):
        main_window.merged_embeddings[embedding_id] = SimpleNamespace(
            embedding_name=name, embedding_store=store
        )

    monkeypatch.setattr(
        lfa.common_widget_actions, "get_pixmap_from_frame", lambda mw, f: None
    )
    monkeypatch.setattr(
        lfa.list_view_actions, "add_media_thumbnail_to_target_faces_list", add_target
    )
    monkeypatch.setattr(
        lfa.list_view_actions, "create_and_add_embed_button_to_list", add_embedding
    )
    from app.ui.widgets.actions import save_load_actions

    monkeypatch.setattr(
        save_load_actions,
        "convert_parameters_to_supported_type",
        lambda mw, params, cls: dict(params),
    )
    return mw


def _add_face(mw, face_id, embedding):
    face = _FakeTargetFace(face_id, embedding)
    face.main_window = mw
    mw.target_faces[face_id] = face
    mw.parameters[face_id] = {"SimilarityThresholdSlider": 60}
    return face


def _add_input_face(mw, face_id, media_path, embedding):
    mw.input_faces[face_id] = SimpleNamespace(
        media_path=media_path, embedding_store={MODEL: embedding}
    )


def _file(mw):
    return lf.load_locked_faces_file(lf.get_locked_faces_path(mw.project_root_path))


# --- Suppression ---------------------------------------------------------------


def test_lfa01_suppression_is_nested_and_after_runs_once(main_window):
    calls = []

    @lfa.suppress_locked_faces_during(lambda mw: calls.append("after"))
    def load(mw, inner=False):
        assert lfa.is_restore_suppressed(mw)
        if inner:
            load(mw)
        return "done"

    assert load(main_window, inner=True) == "done"
    assert calls == ["after"]
    assert not lfa.is_restore_suppressed(main_window)


def test_lfa02_lock_inactive_when_disabled_suppressed_or_batch(main_window):
    assert not lfa.is_lock_active(main_window)
    main_window.control[lfa.LOCK_TOGGLE_NAME] = True
    assert lfa.is_lock_active(main_window)
    main_window.is_batch_processing = True
    assert not lfa.is_lock_active(main_window)
    main_window.is_batch_processing = False
    with lfa.suppress_locked_faces_restore(main_window):
        assert not lfa.is_lock_active(main_window)


# --- Lock / unlock --------------------------------------------------------------


def test_lfa03_lock_writes_entry_with_source_embeddings(main_window, toasts):
    alice = _unit([1, 0, 0, 0])
    face = _add_face(main_window, "t1", _unit([0, 1, 0, 0]))
    _add_input_face(main_window, "in1", "/faces/alice.png", alice)
    face.assigned_input_faces["in1"] = main_window.input_faces["in1"].embedding_store

    assert lfa.lock_target_face(main_window, "t1")
    assert face.is_locked and face.label_refreshes == 1
    entry = _file(main_window)["locked_faces"]["t1"]
    assert entry["recognition_model"] == MODEL
    assert entry["assigned_sources"][0]["media_path"] == "/faces/alice.png"
    assert entry["assigned_sources"][0]["embedding_store"][MODEL] == pytest.approx(
        alice.tolist()
    )
    # Toggle is off -> hint to turn it on.
    assert toasts[-1] == ("Target Face Locked", "information")


def test_lfa04_lock_without_sources_warns(main_window, toasts):
    _add_face(main_window, "t1", _unit([1, 0, 0, 0]))
    lfa.lock_target_face(main_window, "t1")
    assert toasts[-1] == ("Target Face Locked", "warning")


def test_lfa05_toggle_lock_unlocks_and_removes_from_file(main_window, toasts):
    face = _add_face(main_window, "t1", _unit([1, 0, 0, 0]))
    lfa.toggle_target_face_lock(main_window, "t1")
    assert "t1" in _file(main_window)["locked_faces"]
    lfa.toggle_target_face_lock(main_window, "t1")
    assert not face.is_locked
    assert _file(main_window)["locked_faces"] == {}
    assert toasts[-1] == ("Target Face Unlocked", "information")


def test_lfa06_remove_locked_face_unlocks_it(main_window):
    _add_face(main_window, "t1", _unit([1, 0, 0, 0]))
    lfa.lock_target_face(main_window, "t1")
    lfa.on_target_face_removed(main_window, "t1")
    assert _file(main_window)["locked_faces"] == {}


# --- Restore --------------------------------------------------------------------


def _lock_and_clear_session(mw, face_id="t1"):
    """Lock a face with an input face source, then simulate a new session."""
    face = _add_face(mw, face_id, _unit([0, 1, 0, 0]))
    _add_input_face(mw, "in-old", "/faces/alice.png", _unit([1, 0, 0, 0]))
    face.assigned_input_faces["in-old"] = mw.input_faces["in-old"].embedding_store
    face.calculate_assigned_input_embedding()
    lfa.lock_target_face(mw, face_id)
    mw.target_faces.clear()
    mw.parameters.clear()
    mw.input_faces.clear()
    mw.selected_target_face_id = None


def test_lfa07_restore_reassigns_input_face_by_media_path(main_window):
    _lock_and_clear_session(main_window)
    # Same input image loaded again in a new session gets a new card id.
    _add_input_face(main_window, "in-new", "/faces/alice.png", _unit([1, 0, 0, 0]))

    assert lfa.restore_locked_faces(main_window) == ["t1"]
    face = main_window.target_faces["t1"]
    assert face.is_locked
    assert list(face.assigned_input_faces) == ["in-new"]
    assert face.assigned_input_embedding
    assert main_window.parameters["t1"]["SimilarityThresholdSlider"] == 60
    # Restoring again does not duplicate the face.
    assert lfa.restore_locked_faces(main_window) == []
    assert list(main_window.target_faces) == ["t1"]


def test_lfa08_restore_recreates_missing_source_as_embedding(main_window):
    _lock_and_clear_session(main_window)
    lfa.restore_locked_faces(main_window)
    face = main_window.target_faces["t1"]
    assert face.assigned_input_faces == {}
    (embedding_id,) = face.assigned_merged_embeddings
    assert main_window.merged_embeddings[embedding_id].embedding_name == "Locked alice"


def test_lfa09_restore_skips_on_recognition_model_mismatch(main_window, toasts):
    _lock_and_clear_session(main_window)
    main_window.control["RecognitionModelSelection"] = "edgeface"
    assert lfa.restore_locked_faces(main_window) == []
    assert main_window.target_faces == {}
    assert toasts[-1] == ("Locked Faces Skipped", "warning")


def test_lfa10_restore_skips_when_same_person_already_present(main_window):
    _lock_and_clear_session(main_window)
    _add_face(main_window, "found", _unit([0, 1, 0.05, 0]))
    assert lfa.restore_locked_faces(main_window) == []
    assert list(main_window.target_faces) == ["found"]


# --- Hooks / toggle -------------------------------------------------------------


def test_lfa11_toggle_persists_flag_and_restores(main_window):
    _lock_and_clear_session(main_window)
    lfa.on_lock_target_faces_toggled(main_window, True)
    assert _file(main_window)["lock_enabled"] is True
    assert "t1" in main_window.target_faces
    assert main_window.target_faces["t1"].clicked == 1

    lfa.on_lock_target_faces_toggled(main_window, False)
    assert _file(main_window)["lock_enabled"] is False
    # Locked entries are kept when the toggle is turned off.
    assert "t1" in _file(main_window)["locked_faces"]


def test_lfa12_toggle_ignored_while_suppressed(main_window):
    with lfa.suppress_locked_faces_restore(main_window):
        lfa.on_lock_target_faces_toggled(main_window, True)
    assert _file(main_window)["lock_enabled"] is False


def test_lfa13_startup_syncs_toggle_from_file_and_restores(main_window):
    _lock_and_clear_session(main_window)
    lfa.on_lock_target_faces_toggled(main_window, True)
    main_window.target_faces.clear()
    main_window.selected_target_face_id = None
    # Workspace from a previous session had the toggle off.
    main_window.control[lfa.LOCK_TOGGLE_NAME] = False

    lfa.on_startup(main_window)
    assert main_window.control[lfa.LOCK_TOGGLE_NAME] is True
    assert main_window.parameter_widgets[lfa.LOCK_TOGGLE_NAME].values[-1] is True
    assert main_window.selected_target_face_id == "t1"


def test_lfa14_job_load_does_not_inject_locked_faces(main_window):
    _lock_and_clear_session(main_window)
    lfa.on_lock_target_faces_toggled(main_window, True)
    main_window.target_faces.clear()

    @lfa.suppress_locked_faces_during(lfa.after_job_loaded)
    def load_job(mw):
        mw.control[lfa.LOCK_TOGGLE_NAME] = False  # job settings applied

    load_job(main_window)
    assert main_window.control[lfa.LOCK_TOGGLE_NAME] is True
    assert main_window.target_faces == {}


def test_lfa15_close_snapshot_keeps_locked_at_and_assignment_changes(main_window):
    face = _add_face(main_window, "t1", _unit([1, 0, 0, 0]))
    lfa.lock_target_face(main_window, "t1")
    locked_at = _file(main_window)["locked_faces"]["t1"]["locked_at"]
    _add_input_face(main_window, "in1", "/faces/bob.png", _unit([0, 0, 1, 0]))
    face.assigned_input_faces["in1"] = main_window.input_faces["in1"].embedding_store
    lfa.on_close(main_window)
    entry = _file(main_window)["locked_faces"]["t1"]
    assert entry["locked_at"] == locked_at
    assert entry["assigned_input_faces"] == ["in1"]
