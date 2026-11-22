"""Images from the system clipboard, and from paths a terminal pastes.

A terminal pastes text and nothing else, so an image on the clipboard is read
with whatever the desktop offers for it and saved to a file, which the prompt
then mentions like any other image. That only works where paimon runs on the
machine whose clipboard it is: over SSH there is nothing to read.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from .errors import PaimonError
from .tools.files import MAX_IMAGE_BYTES, image_type
from .tools.result import Image

_TIMEOUT = 5

_DRIVE_PATH = re.compile(r"([A-Za-z]):[\\/](.*)")


class ClipboardError(PaimonError):
    """The clipboard holds an image that cannot be used."""


def _output(command: list[str]) -> bytes | None:
    """What ``command`` prints, or None when it is missing or fails."""
    if shutil.which(command[0]) is None:
        return None
    try:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _mac_png() -> bytes | None:
    # AppleScript cannot print binary, so it writes the PNG to a file.
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "clipboard.png"
        script = [
            "set png to (the clipboard as «class PNGf»)",
            f'set out to open for access POSIX file "{target}" with write permission',
            "write png to out",
            "close access out",
        ]
        command = ["osascript"]
        for line in script:
            command += ["-e", line]
        if _output(command) is None:
            return None
        try:
            return target.read_bytes()
        except OSError:
            return None


# The formats that can be sent, best first.
_FORMATS = ("image/png", "image/jpeg", "image/webp", "image/gif")


def _offered(listing: bytes | None) -> str | None:
    """The best sendable format among those a clipboard tool lists."""
    names = (listing or b"").decode("utf-8", errors="replace").split()
    return next((name for name in _FORMATS if name in names), None)


def _wayland() -> bytes | None:
    kind = _offered(_output(["wl-paste", "--list-types"]))
    return _output(["wl-paste", "--no-newline", "--type", kind]) if kind else None


def _x11() -> bytes | None:
    kind = _offered(_output(["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"]))
    return _output(["xclip", "-selection", "clipboard", "-t", kind, "-o"]) if kind else None


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
    except OSError:
        return False


def _windows() -> bytes | None:
    """The Windows clipboard, from inside WSL. A screenshot taken in Windows
    never reaches the Linux clipboard there, so PowerShell saves it to a file
    this side can read."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "clipboard.png"
        there = _output(["wslpath", "-w", str(target)])
        if not there:
            return None
        quoted = there.decode("utf-8", errors="replace").strip().replace("'", "''")
        script = "; ".join([
            "Add-Type -AssemblyName System.Windows.Forms",
            "Add-Type -AssemblyName System.Drawing",
            "$image = [System.Windows.Forms.Clipboard]::GetImage()",
            f"if ($image) {{ $image.Save('{quoted}', [System.Drawing.Imaging.ImageFormat]::Png) }}",
        ])
        if _output(["powershell.exe", "-NoProfile", "-Sta", "-Command", script]) is None:
            return None
        try:
            return target.read_bytes()
        except OSError:
            return None


def _read() -> bytes | None:
    if sys.platform == "darwin":
        return _mac_png()
    data = None
    if os.environ.get("WAYLAND_DISPLAY"):
        data = _wayland()
    if not data and os.environ.get("DISPLAY"):
        data = _x11()
    if not data and is_wsl():
        data = _windows()
    return data


def read_image() -> Image | None:
    """The image on the clipboard, or None when it holds none that can be read.

    Blocking: it runs a desktop tool, so call it off the event loop.
    """
    data = _read()
    media_type = image_type(data[:12]) if data else None
    if media_type is None:
        return None
    if len(data) > MAX_IMAGE_BYTES:
        raise ClipboardError(f"The image on the clipboard is {len(data) / 1_048_576:.1f} MB; "
                             f"one over {MAX_IMAGE_BYTES // 1_048_576} MB cannot be sent.")
    return Image(data, media_type)


def save(image: Image, directory: Path) -> Path:
    """Write a pasted image under ``directory`` and return where it went."""
    directory.mkdir(parents=True, exist_ok=True)
    suffix = image.media_type.removeprefix("image/")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for count in range(1, 1000):
        path = directory / (f"paste-{stamp}.{suffix}" if count == 1
                            else f"paste-{stamp}-{count}.{suffix}")
        try:
            with path.open("xb") as target:
                target.write(image.data)
        except FileExistsError:
            continue
        return path
    raise ClipboardError("Could not find a free name for the pasted image.")


def pasted_image_path(text: str) -> Path | None:
    """The image file ``text`` names, when a paste is one path and nothing else.

    A file dragged onto a terminal arrives as its path: quoted, with its
    spaces escaped, or as a file:// URL, depending on the terminal.
    """
    text = text.strip()
    if not text or "\n" in text:
        return None
    if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1]
    elif text.startswith("file://"):
        text = unquote(urlparse(text).path)
        if _DRIVE_PATH.match(text[1:]):
            # file:///C:/Users/me/shot.png has the path /C:/Users/me/shot.png.
            text = text[1:]
    else:
        text = text.replace("\\ ", " ")
    drive = _DRIVE_PATH.match(text)
    if drive and is_wsl():
        # Dragged from Explorer: C:\\Users\\me\\shot.png is /mnt/c/Users/me/shot.png here.
        text = f"/mnt/{drive[1].lower()}/" + drive[2].replace("\\", "/")
    path = Path(text)
    try:
        if not path.is_absolute() or not path.is_file():
            return None
        with path.open("rb") as source:
            return path if image_type(source.read(12)) else None
    except (OSError, ValueError):
        return None


def mention(path: Path) -> str:
    """``path`` as the mention that attaches it; see mentions.expand_mentions."""
    return "@" + str(path).replace(" ", "\\ ")
