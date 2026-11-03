import json
import ntpath

import pytest

from app.helpers import browse_folders as bf


def _fake_isdir(existing):
    existing_norm = {ntpath.normcase(p) for p in existing}

    def isdir(path):
        return ntpath.normcase(path) in existing_norm

    return isdir


# ---------------------------------------------------------------- normalize


def test_normalize_windows_paths_strip_trailing_separator_and_quotes():
    assert bf.normalize_folder('"C:/Videos/Clips/"') == "C:\\Videos\\Clips"
    assert bf.normalize_folder("C:\\Videos\\Clips\\") == "C:\\Videos\\Clips"
    assert bf.normalize_folder("  ") == ""
    assert bf.normalize_folder(None) == ""


def test_parent_folder_windows_and_roots():
    assert bf.parent_folder("C:/Videos/Clips") == "C:\\Videos"
    assert bf.parent_folder("C:\\") == "C:\\"
    assert bf.parent_folder("\\\\server\\share\\clips") == "\\\\server\\share\\"
    assert bf.parent_folder("/home/user/clips") == "/home/user"
    assert bf.parent_folder("/") == "/"


def test_nearest_existing_folder_walks_up_windows():
    isdir = _fake_isdir(["C:\\", "C:\\Videos"])
    assert bf.nearest_existing_folder("C:/Videos/Gone/Deeper", isdir) == "C:\\Videos"
    assert bf.nearest_existing_folder("D:\\Missing", _fake_isdir([])) == ""


def test_nearest_existing_folder_real_fs(tmp_path):
    missing = tmp_path / "a" / "b"
    assert bf.nearest_existing_folder(str(missing)) == str(tmp_path)


# ---------------------------------------------------------------- resolve


def test_folder_picker_opens_at_parent_of_last_folder():
    isdir = _fake_isdir(["C:\\", "C:\\Videos", "C:\\Videos\\Clips"])
    assert (
        bf.resolve_start_folder("C:/Videos/Clips", prefer_parent=True, isdir=isdir)
        == "C:\\Videos"
    )


def test_file_picker_opens_inside_last_folder():
    isdir = _fake_isdir(["C:\\", "C:\\Videos", "C:\\Videos\\Clips"])
    assert (
        bf.resolve_start_folder("C:/Videos/Clips", prefer_parent=False, isdir=isdir)
        == "C:\\Videos\\Clips"
    )


def test_drive_root_opens_at_root():
    isdir = _fake_isdir(["E:\\"])
    assert bf.resolve_start_folder("E:\\", prefer_parent=True, isdir=isdir) == "E:\\"


def test_missing_folder_falls_back_to_nearest_existing_parent():
    isdir = _fake_isdir(["C:\\", "C:\\Videos"])
    # Clips was deleted: open at the nearest existing ancestor (not its parent)
    assert (
        bf.resolve_start_folder(
            "C:\\Videos\\Clips\\Day1", prefer_parent=True, isdir=isdir
        )
        == "C:\\Videos"
    )


def test_nothing_exists_falls_back_to_default_then_empty():
    isdir = _fake_isdir(["C:\\", "C:\\Default"])
    assert (
        bf.resolve_start_folder(
            "Z:\\Unplugged\\Clips",
            prefer_parent=True,
            default="C:\\Default",
            isdir=isdir,
        )
        == "C:\\Default"
    )
    assert (
        bf.resolve_start_folder(
            "Z:\\Unplugged", prefer_parent=True, default="", isdir=isdir
        )
        == ""
    )
    assert bf.resolve_start_folder("", prefer_parent=True, isdir=isdir) == ""


# ---------------------------------------------------------------- persistence


def test_save_and_load_keys_are_independent(tmp_path):
    file_path = bf.get_browse_folders_path(tmp_path)
    assert file_path.name == "browse_folders.json"
    assert bf.load_browse_folders(file_path) == {}

    assert bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, "C:/Videos/Clips")
    assert bf.save_browse_folder(file_path, bf.KEY_INPUT_FACES, "D:\\Faces")
    assert bf.save_browse_folder(file_path, bf.KEY_OUTPUT_FOLDER, str(tmp_path))
    assert bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, "C:/Videos/Other")

    folders = bf.load_browse_folders(file_path)
    assert folders == {
        bf.KEY_TARGET_MEDIA: "C:\\Videos\\Other",
        bf.KEY_INPUT_FACES: "D:\\Faces",
        bf.KEY_OUTPUT_FOLDER: str(tmp_path),
    }
    data = json.loads(file_path.read_text(encoding="utf-8"))
    assert data["version"] == bf.BROWSE_FOLDERS_FORMAT_VERSION
    assert not list(tmp_path.glob(".browse_folders_*.tmp"))


def test_save_ignores_empty_values(tmp_path):
    file_path = bf.get_browse_folders_path(tmp_path)
    assert not bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, "")
    assert not bf.save_browse_folder(file_path, "", "C:/x")
    assert not file_path.exists()


@pytest.mark.parametrize(
    "content", ["", "not json", "[]", '{"folders": []}', '{"folders": {"k": 5}}']
)
def test_load_tolerates_corrupt_file(tmp_path, content):
    file_path = bf.get_browse_folders_path(tmp_path)
    file_path.write_text(content, encoding="utf-8")
    assert bf.load_browse_folders(file_path) == {}
    # and saving repairs it
    assert bf.save_browse_folder(file_path, bf.KEY_INPUT_FACES, str(tmp_path))
    assert bf.load_browse_folders(file_path) == {bf.KEY_INPUT_FACES: str(tmp_path)}


def test_get_start_folder_real_fs(tmp_path):
    clips = tmp_path / "Videos" / "Clips"
    clips.mkdir(parents=True)
    file_path = bf.get_browse_folders_path(tmp_path)
    bf.save_browse_folder(file_path, bf.KEY_TARGET_MEDIA, str(clips))

    assert bf.get_start_folder(
        file_path, bf.KEY_TARGET_MEDIA, prefer_parent=True
    ) == str(clips.parent)
    assert bf.get_start_folder(
        file_path, bf.KEY_TARGET_MEDIA, prefer_parent=False
    ) == str(clips)
    # other keys have nothing saved yet -> default
    assert bf.get_start_folder(
        file_path, bf.KEY_OUTPUT_FOLDER, prefer_parent=True, default=str(tmp_path)
    ) == str(tmp_path)
    clips.rmdir()
    assert bf.get_start_folder(
        file_path, bf.KEY_TARGET_MEDIA, prefer_parent=True
    ) == str(clips.parent)
