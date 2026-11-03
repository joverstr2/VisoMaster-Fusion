"""Browse dialogs remember their last folder (separately) across restarts."""

from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from app.helpers import browse_folders as bf
from app.ui.widgets.actions import list_view_actions


class _LineEdit:
    def __init__(self):
        self.value = ""

    def setText(self, text):
        self.value = str(text)

    def setToolTip(self, _tooltip):
        pass


class _Recorder:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


@pytest.fixture
def window(tmp_path):
    return SimpleNamespace(
        project_root_path=tmp_path,
        last_target_media_folder_path="",
        last_input_media_folder_path="",
        targetVideosPathLineEdit=_LineEdit(),
        inputFacesPathLineEdit=_LineEdit(),
        outputFolderLineEdit=_LineEdit(),
        control={"OutputMediaFolder": ""},
    )


def _patch_dialogs(monkeypatch, folder="", files=None):
    get_dir = _Recorder(folder)
    get_files = _Recorder((files or [], ""))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", get_dir)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", get_files)
    return get_dir, get_files


def _make_tree(tmp_path):
    media = tmp_path / "media"
    (media / "clips").mkdir(parents=True)
    (media / "faces").mkdir()
    (media / "out").mkdir()
    return media


def _start_dir(recorder):
    return recorder.calls[-1][1]["dir"]


def test_target_folder_is_remembered_and_reopens_at_parent(
    window, tmp_path, monkeypatch
):
    media = _make_tree(tmp_path)
    clips = str(media / "clips")
    _unblock(monkeypatch)
    get_dir, _ = _patch_dialogs(monkeypatch, folder=clips)
    with pytest.raises(_Stop):
        list_view_actions.select_target_medias(window, "folder")
    assert _start_dir(get_dir) == ""  # nothing remembered yet
    assert window.targetVideosPathLineEdit.value == clips
    assert bf.load_browse_folders(bf.get_browse_folders_path(tmp_path)) == {
        bf.KEY_TARGET_MEDIA: clips
    }

    # "Restart": session state is gone, the saved location is not
    window.last_target_media_folder_path = ""
    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_target_medias(window, "folder")
    # opens at the parent, so "clips" and its sibling folders are visible
    assert _start_dir(get_dir) == str(media)


class _Stop(Exception):
    pass


def _raise_stop(*_a, **_k):
    raise _Stop()


def _unblock(monkeypatch):
    from app.ui.widgets.actions import video_control_actions

    monkeypatch.setattr(
        video_control_actions, "block_if_issue_scan_active", lambda *_a: False
    )
    monkeypatch.setattr(
        list_view_actions, "clear_stop_loading_target_media", _raise_stop
    )
    monkeypatch.setattr(
        list_view_actions, "clear_stop_loading_input_media", _raise_stop
    )


def test_cancel_does_not_change_saved_folder(window, tmp_path, monkeypatch):
    media = _make_tree(tmp_path)
    file_path = bf.get_browse_folders_path(tmp_path)
    bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, str(media / "clips"))
    _unblock(monkeypatch)
    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_target_medias(window, "folder")
    assert _start_dir(get_dir) == str(media)
    assert bf.load_browse_folders(file_path)[bf.KEY_TARGET_MEDIA] == str(
        media / "clips"
    )


def test_target_files_dialog_opens_inside_remembered_folder(
    window, tmp_path, monkeypatch
):
    media = _make_tree(tmp_path)
    file_path = bf.get_browse_folders_path(tmp_path)
    bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, str(media / "clips"))
    _unblock(monkeypatch)
    picked = str(media / "faces" / "a.mp4")
    _, get_files = _patch_dialogs(monkeypatch, files=[picked])
    with pytest.raises(_Stop):
        list_view_actions.select_target_medias(window, "files")
    assert _start_dir(get_files) == str(media / "clips")
    # the folder of the picked files becomes the remembered target folder
    assert bf.load_browse_folders(file_path)[bf.KEY_TARGET_MEDIA] == str(
        media / "faces"
    )


def test_input_faces_folder_is_independent_of_target(window, tmp_path, monkeypatch):
    media = _make_tree(tmp_path)
    file_path = bf.get_browse_folders_path(tmp_path)
    bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, str(media / "clips"))
    _unblock(monkeypatch)
    get_dir, _ = _patch_dialogs(monkeypatch, folder=str(media / "faces"))
    with pytest.raises(_Stop):
        list_view_actions.select_input_face_images(window, "folder")
    assert _start_dir(get_dir) == ""  # input faces had nothing saved
    saved = bf.load_browse_folders(file_path)
    assert saved[bf.KEY_INPUT_FACES] == str(media / "faces")
    assert saved[bf.KEY_TARGET_MEDIA] == str(media / "clips")

    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_input_face_images(window, "folder")
    assert _start_dir(get_dir) == str(media)


def test_output_folder_remembered(window, tmp_path, monkeypatch):
    media = _make_tree(tmp_path)
    created = []
    monkeypatch.setattr(
        list_view_actions,
        "common_widget_actions",
        SimpleNamespace(create_control=lambda _w, _name, value: created.append(value)),
    )
    get_dir, _ = _patch_dialogs(monkeypatch, folder=str(media / "out"))
    list_view_actions.select_output_media_folder(window)
    assert _start_dir(get_dir) == ""
    assert get_dir.calls[-1][0] == (window,)
    assert created == [str(media / "out")]
    file_path = bf.get_browse_folders_path(tmp_path)
    assert bf.load_browse_folders(file_path) == {
        bf.KEY_OUTPUT_FOLDER: str(media / "out")
    }

    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_output_media_folder(window)
    assert _start_dir(get_dir) == str(media)


def test_output_dialog_defaults_to_current_output_folder(window, tmp_path, monkeypatch):
    media = _make_tree(tmp_path)
    window.control["OutputMediaFolder"] = str(media / "out")
    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_output_media_folder(window)
    assert _start_dir(get_dir) == str(media / "out")


def test_deleted_folder_falls_back_to_existing_parent(window, tmp_path, monkeypatch):
    media = _make_tree(tmp_path)
    file_path = bf.get_browse_folders_path(tmp_path)
    bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, str(media / "gone" / "x"))
    _unblock(monkeypatch)
    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_target_medias(window, "folder")
    assert _start_dir(get_dir) == str(media)


def test_missing_project_root_does_not_break_dialog(monkeypatch):
    _unblock(monkeypatch)
    win = SimpleNamespace(
        last_target_media_folder_path="", targetVideosPathLineEdit=_LineEdit()
    )
    get_dir, _ = _patch_dialogs(monkeypatch, folder="")
    list_view_actions.select_target_medias(win, "folder")
    assert _start_dir(get_dir) == ""
