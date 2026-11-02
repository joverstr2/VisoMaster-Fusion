"""
UI glue for the "Lock Target Faces" feature.

- Right-click a target face card -> "Lock Target Face" / "Unlock Target Face".
- Settings > Swap settings -> "Lock Target Faces" toggle.

When the toggle is ON, locked target faces (and the source faces / embeddings
assigned to them) are kept when a new media is loaded and restored at startup,
using ``locked_faces.json`` in the project root (see app/helpers/locked_faces.py).
Unlocked target faces behave exactly as before.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)

import numpy as np
from PySide6 import QtCore

from app.helpers import locked_faces as lf
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import list_view_actions

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

LOCK_TOGGLE_NAME = "LockTargetFacesToggle"
LOCKED_LABEL_PREFIX = "\U0001f512 "  # lock emoji
DEFAULT_RECOGNITION_MODEL = "arcface_128"
# Delay before parameter changes of a locked face are written to the file, so
# dragging a slider does not rewrite locked_faces.json on every step.
SAVE_DEBOUNCE_MS = 1500


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def get_locked_faces_path(main_window: "MainWindow"):
    return lf.get_locked_faces_path(main_window.project_root_path)


def _load(main_window: "MainWindow") -> Dict[str, Any]:
    return lf.load_locked_faces_file(get_locked_faces_path(main_window))


def _save(main_window: "MainWindow", data: Dict[str, Any]) -> bool:
    return lf.save_locked_faces_file(get_locked_faces_path(main_window), data)


def _recognition_model(main_window: "MainWindow") -> str:
    return str(
        main_window.control.get("RecognitionModelSelection", DEFAULT_RECOGNITION_MODEL)
    )


def _suppress_count(main_window: "MainWindow") -> int:
    count = getattr(main_window, "_locked_faces_restore_suppressed", 0)
    return count if isinstance(count, int) else 0


@contextmanager
def suppress_locked_faces_restore(main_window: "MainWindow") -> Iterator[None]:
    """Disable locked face handling while a workspace / job is being loaded."""
    main_window._locked_faces_restore_suppressed = _suppress_count(main_window) + 1
    try:
        yield
    finally:
        main_window._locked_faces_restore_suppressed = max(
            0, _suppress_count(main_window) - 1
        )


def is_restore_suppressed(main_window: "MainWindow") -> bool:
    return _suppress_count(main_window) > 0


def suppress_locked_faces_during(
    after: Optional[Callable[["MainWindow"], None]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: run ``func(main_window, ...)`` with locked face handling
    suppressed, then call ``after(main_window)`` once the outermost call ends."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(main_window: "MainWindow", *args: Any, **kwargs: Any) -> Any:
            outermost = not is_restore_suppressed(main_window)
            with suppress_locked_faces_restore(main_window):
                result = func(main_window, *args, **kwargs)
            if outermost and after is not None:
                try:
                    after(main_window)
                except Exception as e:
                    print(f"[WARN] Lock Target Faces: post-load step failed: {e}")
            return result

        return wrapper

    return decorator


def is_lock_enabled(main_window: "MainWindow") -> bool:
    return bool(main_window.control.get(LOCK_TOGGLE_NAME, False))


def is_lock_active(main_window: "MainWindow") -> bool:
    """True when locked faces must be kept / restored right now."""
    return (
        is_lock_enabled(main_window)
        and not is_restore_suppressed(main_window)
        and not getattr(main_window, "is_batch_processing", False)
    )


def is_face_locked(target_face_button: Any) -> bool:
    return bool(getattr(target_face_button, "is_locked", False))


def get_locked_face_ids(main_window: "MainWindow") -> List[str]:
    return [
        face_id
        for face_id, button in main_window.target_faces.items()
        if is_face_locked(button)
    ]


def _refresh_label(target_face_button: Any) -> None:
    refresh = getattr(target_face_button, "refresh_display_label", None)
    if callable(refresh):
        refresh()


def _toast(main_window: "MainWindow", title: str, message: str, style: str) -> None:
    try:
        common_widget_actions.create_and_show_toast_message(
            main_window, title, message, style_type=style
        )
    except Exception as e:  # Toasts are cosmetic, never break the workflow.
        print(f"[WARN] Could not show toast message '{title}': {e}")


# ---------------------------------------------------------------------------
# Snapshot (session -> file)
# ---------------------------------------------------------------------------


def _collect_sources(
    main_window: "MainWindow", target_face_button: Any
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for input_face_id, store in target_face_button.assigned_input_faces.items():
        input_face_button = main_window.input_faces.get(input_face_id)
        media_path = getattr(input_face_button, "media_path", "") or ""
        if input_face_button is not None:
            store = input_face_button.embedding_store
        sources.append(lf.build_input_face_source(input_face_id, media_path, store))
    for embedding_id, store in target_face_button.assigned_merged_embeddings.items():
        embed_button = main_window.merged_embeddings.get(embedding_id)
        name = getattr(embed_button, "embedding_name", "") or ""
        if embed_button is not None:
            store = embed_button.embedding_store
        sources.append(lf.build_embedding_source(embedding_id, name, store))
    return sources


def _face_parameters(main_window: "MainWindow", face_id: str) -> Dict[str, Any]:
    """Per-face parameters of a target face as a plain dict ({} if none)."""
    parameters = main_window.parameters.get(face_id)
    parameters_data = getattr(parameters, "data", parameters) or {}
    return dict(parameters_data)


def build_entry_for_target_face(
    main_window: "MainWindow", target_face_button: Any
) -> Dict[str, Any]:
    face_id = target_face_button.face_id
    recognition_model = _recognition_model(main_window)
    # Make sure the embedding for the current recognition model is stored, so
    # the face can be matched again after a restart.
    try:
        target_face_button.get_embedding(recognition_model)
    except Exception as e:
        print(f"[WARN] Could not compute '{recognition_model}' embedding: {e}")

    return lf.build_locked_face_entry(
        cropped_face=target_face_button.cropped_face,
        embedding_store=target_face_button.embedding_store,
        parameters=_face_parameters(main_window, face_id),
        recognition_model=recognition_model,
        assigned_sources=_collect_sources(main_window, target_face_button),
        assigned_input_embedding=target_face_button.assigned_input_embedding,
    )


def snapshot_locked_faces(main_window: "MainWindow") -> bool:
    """Write the current state of every locked target face card to the file.

    Locked faces stored in the file but not present in the session are kept.
    """
    locked_ids = get_locked_face_ids(main_window)
    if not locked_ids:
        return True
    data = _load(main_window)
    for face_id in locked_ids:
        button = main_window.target_faces[face_id]
        try:
            entry = build_entry_for_target_face(main_window, button)
        except Exception as e:
            print(f"[ERROR] Could not snapshot locked face {face_id}: {e}")
            continue
        previous = data["locked_faces"].get(face_id, {})
        if previous.get("locked_at"):
            entry["locked_at"] = previous["locked_at"]
        if not entry.get("parameters") and previous.get("parameters"):
            # The session has no parameters for this face (e.g. they were
            # already dropped by a media change): keep the saved ones instead
            # of overwriting them with nothing (which restores the defaults).
            entry["parameters"] = previous["parameters"]
        data["locked_faces"][face_id] = entry
    data["lock_enabled"] = is_lock_enabled(main_window)
    return _save(main_window, data)


def _flush_scheduled_save(main_window: "MainWindow") -> None:
    if is_restore_suppressed(main_window):
        return
    try:
        snapshot_locked_faces(main_window)
    except Exception as e:
        print(f"[ERROR] Could not save locked faces: {e}")


def schedule_locked_faces_save(main_window: "MainWindow") -> None:
    """Save the locked faces shortly after a per-face parameter change."""
    if is_restore_suppressed(main_window):
        return
    timer = getattr(main_window, "_locked_faces_save_timer", None)
    if not isinstance(timer, QtCore.QTimer):
        try:
            timer = QtCore.QTimer()
        except Exception:
            # No Qt event loop available: save immediately.
            _flush_scheduled_save(main_window)
            return
        timer.setSingleShot(True)
        timer.setInterval(SAVE_DEBOUNCE_MS)
        timer.timeout.connect(lambda: _flush_scheduled_save(main_window))
        main_window._locked_faces_save_timer = timer
    timer.start()


def flush_pending_locked_faces_save(main_window: "MainWindow") -> None:
    """Write a pending (debounced) save right away, e.g. on close."""
    timer = getattr(main_window, "_locked_faces_save_timer", None)
    if isinstance(timer, QtCore.QTimer) and timer.isActive():
        timer.stop()


def update_locked_face_if_needed(
    main_window: "MainWindow", target_face_button: Any
) -> None:
    """Persist assignment changes made to a locked target face."""
    if target_face_button is None or not is_face_locked(target_face_button):
        return
    if is_restore_suppressed(main_window):
        return
    snapshot_locked_faces(main_window)


# ---------------------------------------------------------------------------
# Lock / Unlock actions (context menu)
# ---------------------------------------------------------------------------


def lock_target_face(main_window: "MainWindow", face_id: str) -> bool:
    button = main_window.target_faces.get(face_id)
    if button is None:
        return False
    button.is_locked = True
    if not snapshot_locked_faces(main_window):
        button.is_locked = False
        _toast(
            main_window,
            "Lock Target Face",
            "Could not write locked_faces.json",
            "error",
        )
        return False
    _refresh_label(button)
    if not button.assigned_input_faces and not button.assigned_merged_embeddings:
        _toast(
            main_window,
            "Target Face Locked",
            "No input face / embedding is assigned yet. Select one while this "
            "face is selected and it will be remembered.",
            "warning",
        )
    elif not is_lock_enabled(main_window):
        _toast(
            main_window,
            "Target Face Locked",
            "Turn on 'Lock Target Faces' in Settings to keep it on every media load.",
            "information",
        )
    else:
        _toast(
            main_window,
            "Target Face Locked",
            "It will be kept on every media load and after restarts.",
            "success",
        )
    return True


def unlock_target_face(main_window: "MainWindow", face_id: str) -> bool:
    data = _load(main_window)
    removed = data["locked_faces"].pop(face_id, None) is not None
    if removed:
        _save(main_window, data)
    button = main_window.target_faces.get(face_id)
    if button is not None:
        button.is_locked = False
        _refresh_label(button)
    return removed or button is not None


def toggle_target_face_lock(main_window: "MainWindow", face_id: str) -> None:
    button = main_window.target_faces.get(face_id)
    if button is None:
        return
    if is_face_locked(button):
        unlock_target_face(main_window, face_id)
        _toast(
            main_window,
            "Target Face Unlocked",
            "It will be cleared on the next media load, as usual.",
            "information",
        )
    else:
        lock_target_face(main_window, face_id)


# ---------------------------------------------------------------------------
# Restore (file -> session)
# ---------------------------------------------------------------------------


def _existing_faces_for_matching(
    main_window: "MainWindow", recognition_model: str
) -> Dict[str, Tuple[Any, float]]:
    existing: Dict[str, Tuple[Any, float]] = {}
    for face_id, button in main_window.target_faces.items():
        embedding = None
        try:
            store = getattr(button, "embedding_store", {}) or {}
            embedding = store.get(recognition_model)
        except Exception:
            embedding = None
        params: Any = main_window.parameters.get(face_id, {})
        threshold = (
            params.get("SimilarityThresholdSlider", 60)
            if hasattr(params, "get")
            else 60
        )
        existing[face_id] = (embedding, float(threshold))
    return existing


def _assign_sources(
    main_window: "MainWindow", target_face_button: Any, entry: Dict[str, Any]
) -> List[str]:
    """Assign the stored sources to a restored target face. Returns warnings."""
    warnings: List[str] = []
    input_faces = {
        face_id: getattr(button, "media_path", "")
        for face_id, button in main_window.input_faces.items()
    }
    embeddings = {
        embedding_id: getattr(button, "embedding_name", "")
        for embedding_id, button in main_window.merged_embeddings.items()
    }
    for source in entry.get("assigned_sources", []):
        kind, card_id = lf.resolve_source(source, input_faces, embeddings)
        if kind == lf.SOURCE_KIND_INPUT_FACE and card_id is not None:
            target_face_button.assigned_input_faces[card_id] = main_window.input_faces[
                card_id
            ].embedding_store
            continue
        if kind == lf.SOURCE_KIND_EMBEDDING and card_id is not None:
            target_face_button.assigned_merged_embeddings[card_id] = (
                main_window.merged_embeddings[card_id].embedding_store
            )
            continue
        # Source is not loaded in this session: recreate it as an embedding card
        # from the stored embedding so the pairing still works.
        store = lf.embedding_store_from_json(source.get("embedding_store"))
        if not store:
            warnings.append("a source has no stored embedding")
            continue
        name = lf.fallback_embedding_name(source)
        embedding_id = (
            str(source.get("id"))
            if source.get("kind") == lf.SOURCE_KIND_EMBEDDING
            and source.get("id")
            and str(source.get("id")) not in main_window.merged_embeddings
            else str(uuid.uuid1().int)
        )
        list_view_actions.create_and_add_embed_button_to_list(
            main_window, name, store, embedding_id
        )
        embeddings[embedding_id] = name
        target_face_button.assigned_merged_embeddings[embedding_id] = (
            main_window.merged_embeddings[embedding_id].embedding_store
        )
    return warnings


def _create_target_face(
    main_window: "MainWindow", face_id: str, entry: Dict[str, Any]
) -> Optional[Any]:
    from app.ui.widgets.actions import save_load_actions

    cropped_face = np.array(entry["cropped_face"]).astype("uint8")
    pixmap = common_widget_actions.get_pixmap_from_frame(main_window, cropped_face)
    embedding_store = lf.embedding_store_from_json(entry.get("embedding_store"))
    list_view_actions.add_media_thumbnail_to_target_faces_list(
        main_window, cropped_face, embedding_store, pixmap, face_id
    )
    button = main_window.target_faces.get(face_id)
    if button is None:
        return None

    default_params = cast(
        misc_helpers.ParametersDict, main_window.default_parameters
    ).data
    sanitized_params = save_load_actions.sanitize_state_dictionary(
        entry.get("parameters", {}), default_params
    )
    main_window.parameters[face_id] = (
        save_load_actions.convert_parameters_to_supported_type(
            main_window, sanitized_params, misc_helpers.ParametersDict
        )
    )
    return button


def restore_locked_faces(main_window: "MainWindow") -> List[str]:
    """Add the locked faces stored in the file that are missing from the session.

    Faces already present (same face id) are only marked as locked. Faces whose
    stored embeddings do not match the current Face Recognition Model are
    skipped with a warning. Returns the ids of the created target faces.
    """
    data = _load(main_window)
    locked_faces = data.get("locked_faces", {})
    if not locked_faces:
        return []

    recognition_model = _recognition_model(main_window)
    to_create, to_mark, skipped = lf.plan_restore(
        locked_faces,
        recognition_model,
        _existing_faces_for_matching(main_window, recognition_model),
    )

    for face_id in to_mark:
        button = main_window.target_faces[face_id]
        button.is_locked = True
        _refresh_label(button)

    created: List[str] = []
    for face_id in to_create:
        entry = locked_faces[face_id]
        try:
            new_button = _create_target_face(main_window, face_id, entry)
            if new_button is None:
                skipped.append((face_id, "the target face card could not be created"))
                continue
            for warning in _assign_sources(main_window, new_button, entry):
                print(f"[WARN] Locked face {face_id}: {warning}")
            new_button.calculate_assigned_input_embedding()
            if not new_button.assigned_input_embedding and entry.get(
                "assigned_input_embedding"
            ):
                new_button.assigned_input_embedding = lf.embedding_store_from_json(
                    entry["assigned_input_embedding"]
                )
            new_button.is_locked = True
            _refresh_label(new_button)
            created.append(face_id)
        except Exception as e:
            print(f"[ERROR] Could not restore locked face {face_id}: {e}")
            skipped.append((face_id, str(e)))

    if skipped:
        for face_id, reason in skipped:
            print(f"[WARN] Locked target face {face_id} not restored: {reason}")
        model_skips = [r for _, r in skipped if "recognition model" in r]
        if model_skips:
            _toast(
                main_window,
                "Locked Faces Skipped",
                f"{len(model_skips)} locked face(s) skipped: {model_skips[0]}. "
                "Switch the Face Recognition Model back or lock the faces again.",
                "warning",
            )
    if created:
        print(f"[INFO] Restored {len(created)} locked target face(s).")
    return created


def select_first_locked_face(main_window: "MainWindow") -> bool:
    """Select the first locked face if no target face is selected."""
    if main_window.selected_target_face_id:
        return False
    for face_id in get_locked_face_ids(main_window):
        main_window.target_faces[face_id].click()
        return True
    return False


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def on_lock_target_faces_toggled(main_window: "MainWindow", value: bool) -> None:
    """exec_function of the 'Lock Target Faces' toggle."""
    if is_restore_suppressed(main_window):
        return
    data = _load(main_window)
    data["lock_enabled"] = bool(value)
    _save(main_window, data)
    if value:
        restore_locked_faces(main_window)
        select_first_locked_face(main_window)


def sync_lock_toggle_from_file(main_window: "MainWindow") -> bool:
    """Apply the lock_enabled flag stored in locked_faces.json to the UI toggle.

    The file is authoritative so the setting survives restarts even when no
    workspace is auto-loaded. Does not run the toggle's exec function.
    """
    enabled = bool(_load(main_window).get("lock_enabled", False))
    main_window.control[LOCK_TOGGLE_NAME] = enabled
    widget = getattr(main_window, "parameter_widgets", {}).get(LOCK_TOGGLE_NAME)
    if widget is not None:
        # Suppressed so the toggle's exec function does not run from here.
        with suppress_locked_faces_restore(main_window):
            widget.set_value(enabled)
    return enabled


def after_workspace_loaded(main_window: "MainWindow") -> None:
    """Called at the end of a workspace load (including the startup one)."""
    if is_restore_suppressed(main_window):
        return
    if sync_lock_toggle_from_file(main_window):
        restore_locked_faces(main_window)
    else:
        # Keep the lock markers visible on faces that are locked in the file.
        locked = _load(main_window).get("locked_faces", {})
        for face_id, button in main_window.target_faces.items():
            if face_id in locked:
                button.is_locked = True
                _refresh_label(button)


def after_job_loaded(main_window: "MainWindow") -> None:
    """Jobs keep their own saved faces; only re-apply the global toggle state."""
    if is_restore_suppressed(main_window):
        return
    sync_lock_toggle_from_file(main_window)


def on_startup(main_window: "MainWindow") -> None:
    after_workspace_loaded(main_window)
    select_first_locked_face(main_window)


def on_close(main_window: "MainWindow") -> None:
    flush_pending_locked_faces_save(main_window)
    try:
        snapshot_locked_faces(main_window)
    except Exception as e:
        print(f"[ERROR] Could not save locked faces on close: {e}")


def on_target_face_removed(main_window: "MainWindow", face_id: str) -> None:
    """'Remove from List' on a locked face also unlocks it."""
    data = _load(main_window)
    if data["locked_faces"].pop(face_id, None) is not None:
        _save(main_window, data)
        print(f"[INFO] Locked target face {face_id} removed and unlocked.")
