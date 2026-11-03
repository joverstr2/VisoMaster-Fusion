"""
Pure (Qt-free) helpers that remember where each folder / file browse dialog
was last pointed, so the next browse starts there again - also after restarts.

Locations are persisted to ``browse_folders.json`` in the project root, one
independent entry per dialog key (target media, input faces, output folder).

Windows paths (``C:\\Videos\\Clips``, ``C:/Videos/Clips``, UNC shares) are
handled with ``ntpath`` regardless of the host OS, so the logic is identical
(and testable) on every platform.
"""

from __future__ import annotations

import json
import ntpath
import os
import posixpath
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, Optional

BROWSE_FOLDERS_FILENAME = "browse_folders.json"
BROWSE_FOLDERS_FORMAT_VERSION = 1

KEY_TARGET_MEDIA = "target_media"
KEY_INPUT_FACES = "input_faces"
KEY_OUTPUT_FOLDER = "output_folder"

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

IsDirFn = Callable[[str], bool]


def get_browse_folders_path(project_root_path: Path | str) -> Path:
    """Return the location of the browse folders file inside the project root."""
    return Path(project_root_path) / BROWSE_FOLDERS_FILENAME


def _path_module(path: str):
    """Pick ``ntpath`` for Windows-style paths, else the host ``os.path``."""
    if os.name == "nt":
        return ntpath
    if _WINDOWS_DRIVE_RE.match(path) or path.startswith("\\\\") or "\\" in path:
        return ntpath
    return posixpath


def normalize_folder(path: Optional[str]) -> str:
    """Strip whitespace / quotes and trailing separators (but keep roots)."""
    if not path:
        return ""
    text = str(path).strip().strip('"').strip()
    if not text:
        return ""
    module = _path_module(text)
    normalized = module.normpath(text)
    return "" if normalized == "." else normalized


def parent_folder(path: Optional[str]) -> str:
    """Return the parent of ``path`` (a root such as ``C:\\`` or ``/`` is its own parent)."""
    folder = normalize_folder(path)
    if not folder:
        return ""
    return _path_module(folder).dirname(folder) or folder


def nearest_existing_folder(path: Optional[str], isdir: IsDirFn = os.path.isdir) -> str:
    """Walk up from ``path`` to the closest folder that exists ("" if none)."""
    current = normalize_folder(path)
    while current:
        try:
            if isdir(current):
                return current
        except (OSError, ValueError):
            pass
        parent = parent_folder(current)
        if not parent or parent == current:
            break
        current = parent
    return ""


def load_browse_folders(file_path: Path | str) -> Dict[str, str]:
    """Load the saved ``{dialog_key: folder}`` mapping; never raises."""
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    folders = data.get("folders") if isinstance(data, dict) else None
    if not isinstance(folders, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in folders.items()
        if isinstance(value, str) and value.strip()
    }


def save_browse_folder(file_path: Path | str, key: str, folder: Optional[str]) -> bool:
    """Remember ``folder`` for ``key`` (other keys are preserved). Atomic write."""
    normalized = normalize_folder(folder)
    if not key or not normalized:
        return False
    folders = load_browse_folders(file_path)
    folders[str(key)] = normalized
    payload = {"version": BROWSE_FOLDERS_FORMAT_VERSION, "folders": folders}
    target = Path(file_path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".browse_folders_", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise
    except OSError as exc:
        print(f"[WARN] Could not save browse folder location: {exc}")
        return False
    return True


def resolve_start_folder(
    saved_folder: Optional[str],
    *,
    prefer_parent: bool,
    default: Optional[str] = "",
    isdir: IsDirFn = os.path.isdir,
) -> str:
    """Decide where a browse dialog should open.

    - ``prefer_parent=True`` (folder pickers): open at the *parent* of the last
      chosen folder so it and its sibling folders are visible and one click
      away. A root (``C:\\``) has no parent, so it opens at the root itself.
    - ``prefer_parent=False`` (file pickers): open *inside* the last folder,
      because the files to pick live there.
    - If the saved folder no longer exists, walk up to the nearest existing
      parent (that ancestor is shown as-is). Failing that, use ``default``
      (also walked up), and finally "" (the dialog's own default).
    """
    saved = normalize_folder(saved_folder)
    if saved:
        existing = nearest_existing_folder(saved, isdir)
        if existing:
            if prefer_parent and existing == saved:
                parent = parent_folder(existing)
                if parent and parent != existing:
                    try:
                        if isdir(parent):
                            return parent
                    except (OSError, ValueError):
                        pass
            return existing
    return nearest_existing_folder(default, isdir)


def get_start_folder(
    file_path: Path | str,
    key: str,
    *,
    prefer_parent: bool,
    default: Optional[str] = "",
    isdir: IsDirFn = os.path.isdir,
) -> str:
    """Load the saved folder for ``key`` and resolve where the dialog opens."""
    saved = load_browse_folders(file_path).get(key, "")
    return resolve_start_folder(
        saved, prefer_parent=prefer_parent, default=default, isdir=isdir
    )
