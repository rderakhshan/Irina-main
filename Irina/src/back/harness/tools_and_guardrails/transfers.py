"""Explicit host <-> sandbox file transfers for skills and artifacts.

No host mount and no unrestricted filesystem access: the CLI names the input
paths to upload and the sandbox paths to export.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

MAX_TRANSFER_FILES = 200
MAX_TRANSFER_FILE_BYTES = 20 * 1024 * 1024
MAX_TRANSFER_TOTAL_BYTES = 50 * 1024 * 1024
_SKIP_DIRS = {"__pycache__", ".git"}


def _host_files(path: Path) -> list[tuple[Path, Path]]:
    """Return ``(file, relative_path)`` pairs without following symlinks."""
    if path.is_symlink():
        raise ValueError(f"symlinks are not allowed in transfers: {path}")
    if path.is_file():
        return [(path, Path(path.name))]
    if not path.is_dir():
        raise FileNotFoundError(f"input path does not exist: {path}")
    pairs: list[tuple[Path, Path]] = []
    for item in sorted(path.rglob("*")):
        if any(part in _SKIP_DIRS for part in item.relative_to(path).parts):
            continue
        if item.is_symlink() or not item.is_file():
            continue
        pairs.append((item, item.relative_to(path)))
    return pairs


def _upload_one(workspace: Any, source: Path, target: str) -> None:
    upload = getattr(workspace, "upload_file", None)
    if callable(upload):
        upload(source, target)
        return
    data = source.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{source} is binary and this workspace cannot upload binary files"
        ) from exc
    workspace.write_file(target, text)


def upload_inputs(
    paths: Iterable[str | Path],
    workspace: Any,
    *,
    destination: str = "input",
) -> list[str]:
    """Upload explicitly selected host files/directories into the workspace."""
    uploaded: list[str] = []
    total = 0
    count = 0
    for raw in paths:
        host = Path(raw).expanduser().resolve()
        base = host.name or "input"
        is_directory = host.is_dir()
        for source, relative in _host_files(host):
            size = source.stat().st_size
            if size > MAX_TRANSFER_FILE_BYTES:
                raise ValueError(f"input file is too large: {source}")
            total += size
            count += 1
            if total > MAX_TRANSFER_TOTAL_BYTES:
                raise ValueError("input transfer exceeds the total size limit")
            if count > MAX_TRANSFER_FILES:
                raise ValueError("input transfer exceeds the file-count limit")
            parts = (destination, base, relative.as_posix()) if is_directory else (
                destination,
                relative.as_posix(),
            )
            target = "/".join(part for part in parts if part)
            _upload_one(workspace, source, target)
            uploaded.append(target)
    return uploaded


def _merge_tree(source: Path, target: Path) -> None:
    """Copy ``source`` over ``target`` in place, replacing clashing files.

    Files the new export does not contain are left alone, so an existing
    company folder keeps its extras while regenerated files win.
    """
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            _merge_tree(child, target / child.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir():
        shutil.rmtree(target)
    shutil.copy2(source, target)


def _export_contents(
    workspace: Any,
    sandbox_path: str,
    destination: Path,
    *,
    force: bool,
) -> list[Path]:
    """Export one sandbox directory's children into ``destination``.

    An existing child is merged into rather than treated as an error: newly
    generated files replace the old ones in place. ``force`` replaces the
    whole child (removing files the new export does not include).
    """
    download = getattr(workspace, "download", None)
    if not callable(download):
        raise ValueError("this workspace cannot export directory contents")
    exported: list[Path] = []
    with tempfile.TemporaryDirectory() as temp:
        staged = Path(temp) / "staged"
        download(sandbox_path, staged)
        for child in sorted(staged.iterdir()):
            target = destination / child.name
            if target.exists():
                if force:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                    shutil.move(str(child), str(target))
                else:
                    _merge_tree(child, target)
                exported.append(target)
                continue
            shutil.move(str(child), str(target))
            exported.append(target)
    return exported


def export_artifacts(
    paths: Iterable[str | Path],
    workspace: Any,
    host_dir: str | Path,
    *,
    contents: bool = False,
    force: bool = False,
) -> list[Path]:
    """Download declared sandbox paths into an explicit host directory.

    ``contents=True`` merges a directory's children into ``host_dir`` instead of
    creating ``host_dir/<directory-name>``. This lets a skill emit
    ``application/<Company>/`` and land it directly under ``D:\\Application``.
    An existing child is merged in place, with newly generated files replacing
    the old ones; ``force`` replaces the whole child instead.
    """
    destination = Path(host_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for raw in paths:
        sandbox_path = str(raw)
        if contents:
            exported.extend(
                _export_contents(
                    workspace, sandbox_path, destination, force=force
                )
            )
            continue
        target = destination / Path(sandbox_path).name
        download = getattr(workspace, "download", None)
        if callable(download):
            exported.append(Path(download(sandbox_path, target)))
            continue
        data = workspace.read_file(sandbox_path)
        text = str(data.get("text", ""))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        exported.append(target)
    return exported
