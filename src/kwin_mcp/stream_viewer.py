"""GTK4 window that displays a PNG file and refreshes it on every change.

Tiny side-binary intended to be spawned by Session when stream=True. It
points at a single file (e.g. /tmp/kwin-mcp-stream-<socket>.png) and
re-renders whenever that file's mtime changes, giving a near-live preview
of the isolated KWin session on the user's host compositor.

Run as:
    kwin-mcp-stream-viewer <path-to-png> [--title TITLE] [--poll-ms N]
"""

from __future__ import annotations

import argparse
import os
import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("GdkPixbuf", "2.0")
from gi.repository import GdkPixbuf, GLib, Gtk  # noqa: E402


class StreamWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, png_path: str, title: str, poll_ms: int) -> None:
        super().__init__(application=app, title=title)
        self.set_default_size(960, 540)

        self._png_path = png_path
        self._last_mtime: float = 0.0

        self._picture = Gtk.Picture()
        self._picture.set_can_shrink(True)
        self._picture.set_keep_aspect_ratio(True)
        self.set_child(self._picture)

        self._refresh()
        GLib.timeout_add(poll_ms, self._refresh)

    def _refresh(self) -> bool:
        try:
            mtime = os.path.getmtime(self._png_path)
        except OSError:
            return True  # file not there yet — keep polling
        if mtime == self._last_mtime:
            return True
        self._last_mtime = mtime
        try:
            # Load fresh each time — Gtk.Picture caches its surface otherwise.
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(self._png_path)
            self._picture.set_pixbuf(pixbuf)
        except GLib.Error:
            # Half-written file mid-refresh; the next tick will catch it.
            self._last_mtime = 0.0
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Live PNG viewer for kwin-mcp streaming")
    parser.add_argument("path", help="Path to the PNG file to watch")
    parser.add_argument("--title", default="kwin-mcp stream", help="Window title")
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=250,
        help="How often to check the file for changes (default: 250 ms)",
    )
    args = parser.parse_args()

    app = Gtk.Application(application_id="dev.kwin_mcp.StreamViewer")

    def on_activate(application: Gtk.Application) -> None:
        win = StreamWindow(application, args.path, args.title, args.poll_ms)
        win.present()

    app.connect("activate", on_activate)
    sys.exit(app.run([]))


if __name__ == "__main__":
    main()
