"""KWin Wayland session management.

Manages the lifecycle of KWin Wayland sessions:
- Virtual sessions: isolated via dbus-run-session + kwin_wayland --virtual
- Live sessions: connecting to an existing KWin compositor (real desktop or container)
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .screenshot import capture_screenshot_dbus


class SessionType(Enum):
    """Type of KWin session."""

    VIRTUAL = "virtual"
    LIVE = "live"


@dataclass
class SessionConfig:
    """Configuration for an isolated KWin session."""

    socket_name: str = ""
    screen_width: int = 1920
    screen_height: int = 1080
    enable_clipboard: bool = False
    keep_screenshots: bool = False
    isolate_home: bool = False
    keep_home: bool = False
    # When true, run KWin nested as a Wayland client of the host compositor
    # (a normal window in niri / sway / GNOME / Plasma / weston) instead of
    # using --virtual (an invisible in-memory framebuffer). Lets the user
    # watch what the AI agent is doing in real time. Requires a host Wayland
    # session ($WAYLAND_DISPLAY must be set on the parent process).
    visible: bool = False
    # When true, start a background thread that takes a screenshot of the
    # isolated session every `stream_interval_ms` ms, and spawn the bundled
    # kwin-mcp-stream-viewer as a separate process on the host compositor
    # so the user sees a live preview window of what the agent is doing.
    # Works on every compositor (the preview is just a regular host window).
    stream: bool = False
    stream_interval_ms: int = 400
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class AppInfo:
    """Tracking info for a launched application."""

    pid: int
    command: str
    log_path: Path
    process: subprocess.Popen[bytes]


@dataclass
class SessionInfo:
    """Runtime information about a running session."""

    dbus_address: str
    wayland_socket: str
    kwin_pid: int
    screenshot_dir: Path = field(default_factory=lambda: Path("/tmp"))
    home_dir: Path | None = None
    app_pid: int | None = None
    wrapper_pid: int | None = None
    apps: dict[int, AppInfo] = field(default_factory=dict)
    session_type: SessionType = SessionType.VIRTUAL


class Session:
    """An isolated KWin Wayland session.

    Uses dbus-run-session to create an isolated D-Bus session bus,
    then starts kwin_wayland --virtual inside it. Apps launched in
    this session are completely isolated from the host desktop.
    """

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._info: SessionInfo | None = None
        self._socket_name: str = ""
        self._app_counter: int = 0
        self._config: SessionConfig | None = None
        self._home_dir: Path | None = None
        # Captured stderr of the wrapper / KWin / xdg-desktop-portal warnings.
        # Goes to a file so a chatty stderr can't fill the pipe and deadlock
        # KWin's first write (see start() — we read stdout but not stderr).
        self._stderr_log_path: Path | None = None
        self._stderr_log_fp: object = None
        # Optional live preview streamer (config.stream): a background thread
        # writes screenshots to a stable PNG path, and a side-process viewer
        # window on the host compositor refreshes it on every change.
        self._stream_png_path: Path | None = None
        self._streamer_stop: threading.Event = threading.Event()
        self._streamer_thread: threading.Thread | None = None
        self._streamer_viewer_proc: subprocess.Popen[bytes] | None = None
        # Snapshot of the host process env taken at __init__ — used to launch
        # the viewer on the host compositor, before _build_env() rewrites
        # WAYLAND_DISPLAY/XDG_CURRENT_DESKTOP for the isolated session.
        self._host_env: dict[str, str] = dict(os.environ)

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    @property
    def info(self) -> SessionInfo | None:
        return self._info

    @property
    def wayland_socket(self) -> str:
        return self._socket_name

    def _xdg_isolation_env(self) -> dict[str, str]:
        """Build XDG environment overrides for home directory isolation."""
        if self._home_dir is None:
            return {}
        home = str(self._home_dir)
        return {
            "HOME": home,
            "XDG_CONFIG_HOME": str(self._home_dir / ".config"),
            "XDG_DATA_HOME": str(self._home_dir / ".local" / "share"),
            "XDG_CACHE_HOME": str(self._home_dir / ".cache"),
            "XDG_STATE_HOME": str(self._home_dir / ".local" / "state"),
        }

    def start(self, config: SessionConfig | None = None) -> SessionInfo:
        """Start an isolated KWin Wayland session.

        Returns SessionInfo with connection details.
        """
        if self.is_running:
            msg = "Session is already running"
            raise RuntimeError(msg)

        if config is None:
            config = SessionConfig()
        self._config = config

        self._socket_name = config.socket_name or f"wayland-mcp-{os.getpid()}-{int(time.time())}"

        # Create isolated home directory if requested
        if config.isolate_home:
            self._home_dir = Path(tempfile.mkdtemp(prefix="kwin-mcp-home-"))
            for subdir in (
                ".config",
                Path(".local") / "share",
                Path(".local") / "state",
                ".cache",
                ".screenshots",
            ):
                (self._home_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Clean up any stale socket files
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        for suffix in ("", ".lock"):
            path = Path(runtime_dir) / f"{self._socket_name}{suffix}"
            path.unlink(missing_ok=True)

        # Build the wrapper script that runs inside dbus-run-session
        wrapper_script = self._build_wrapper_script(config)

        # Capture stderr to a file rather than a pipe. xdg-desktop-portal
        # activation, dbus-daemon, and Qt warnings together easily exceed the
        # default 64 KiB pipe buffer; since we only read stdout (for the
        # DBUS_SESSION_BUS_ADDRESS / READY handshake) a full pipe blocks
        # KWin's first write and the socket never appears.
        fd, stderr_path = tempfile.mkstemp(prefix="kwin-mcp-stderr-", suffix=".log")
        self._stderr_log_path = Path(stderr_path)
        self._stderr_log_fp = os.fdopen(fd, "wb", buffering=0)

        # Start the isolated session in its own process group
        self._process = subprocess.Popen(
            ["dbus-run-session", "bash", "-c", wrapper_script],
            stdout=subprocess.PIPE,
            stderr=self._stderr_log_fp,
            env=self._build_env(config),
            start_new_session=True,
        )

        # Read startup output from the wrapper script.
        # Expected lines: DBUS_SESSION_BUS_ADDRESS=..., READY
        # Any other lines (e.g. from D-Bus activation) are ignored.
        dbus_address = ""
        got_ready = False
        if self._process.stdout:
            while True:
                line = self._process.stdout.readline().decode().strip()
                if not line and self._process.poll() is not None:
                    break
                if line.startswith("DBUS_SESSION_BUS_ADDRESS="):
                    dbus_address = line.split("=", 1)[1]
                elif line == "READY":
                    got_ready = True
                    break

        # Wait for kwin to be ready (socket file appears)
        socket_path = Path(runtime_dir) / self._socket_name
        if not self._wait_for_socket(socket_path, timeout=10.0):
            self.stop()
            stderr = ""
            if self._process and self._process.stderr:
                stderr = self._process.stderr.read().decode(errors="replace")
            msg = f"KWin failed to start. stderr: {stderr}"
            raise RuntimeError(msg)

        if not got_ready:
            self.stop()
            msg = "Session setup failed: did not receive READY signal"
            raise RuntimeError(msg)

        if self._home_dir is not None:
            screenshot_dir = self._home_dir / ".screenshots"
        else:
            screenshot_dir = Path(tempfile.mkdtemp(prefix="kwin-mcp-screenshots-"))

        self._info = SessionInfo(
            dbus_address=dbus_address,
            wayland_socket=self._socket_name,
            kwin_pid=self._process.pid,
            screenshot_dir=screenshot_dir,
            home_dir=self._home_dir,
        )

        if config.stream:
            self._start_streamer()

        return self._info

    def _start_streamer(self) -> None:
        """Launch the screenshot loop + viewer window on the host compositor."""
        if self._info is None or self._config is None:
            return
        self._stream_png_path = (
            Path(tempfile.gettempdir()) / f"kwin-mcp-stream-{self._socket_name}.png"
        )
        self._streamer_stop = threading.Event()
        self._streamer_thread = threading.Thread(
            target=self._stream_loop,
            name="kwin-mcp-streamer",
            daemon=True,
        )
        self._streamer_thread.start()

        # Spawn the viewer in the HOST process env (so it appears as a
        # regular window on niri/sway/GNOME/Plasma, not inside the isolated
        # virtual KWin). _build_env() rewrote our session env for KWin, but
        # we kept a snapshot of the parent env in _host_env for exactly this.
        viewer_bin = shutil.which("kwin-mcp-stream-viewer") or "kwin-mcp-stream-viewer"
        try:
            self._streamer_viewer_proc = subprocess.Popen(
                [
                    viewer_bin,
                    str(self._stream_png_path),
                    "--title",
                    f"kwin-mcp stream ({self._socket_name})",
                    "--poll-ms",
                    str(max(100, self._config.stream_interval_ms // 2)),
                ],
                env=self._host_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # Viewer not installed — keep the screenshot loop anyway so the
            # PNG is available for an external viewer the user spawns by hand.
            self._streamer_viewer_proc = None

    def _stream_loop(self) -> None:
        """Background thread: dump a fresh screenshot to the stream PNG path.

        Writes to a `.tmp` sibling and atomically renames so the viewer never
        catches a half-written file.
        """
        if self._info is None or self._stream_png_path is None or self._config is None:
            return
        target = self._stream_png_path
        tmp = target.with_suffix(".tmp.png")
        interval = max(0.1, self._config.stream_interval_ms / 1000.0)
        while not self._streamer_stop.is_set():
            try:
                capture_screenshot_dbus(
                    self._info.dbus_address,
                    tmp,
                    include_cursor=True,
                )
                os.replace(tmp, target)
            except Exception:
                # Don't kill the thread on transient D-Bus errors; the next
                # tick may succeed. Errors get muffled by design — KWin's own
                # transient hiccups during app launch are noisy.
                pass
            if self._streamer_stop.wait(interval):
                break

    def _stop_streamer(self) -> None:
        if self._streamer_thread is not None:
            self._streamer_stop.set()
            self._streamer_thread.join(timeout=2.0)
            self._streamer_thread = None
        if self._streamer_viewer_proc is not None:
            with contextlib.suppress(ProcessLookupError):
                self._streamer_viewer_proc.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._streamer_viewer_proc.wait(timeout=2.0)
            with contextlib.suppress(ProcessLookupError):
                self._streamer_viewer_proc.kill()
            self._streamer_viewer_proc = None
        if self._stream_png_path is not None:
            self._stream_png_path.unlink(missing_ok=True)
            self._stream_png_path.with_suffix(".tmp.png").unlink(missing_ok=True)
            self._stream_png_path = None

    def launch_app(self, command: list[str], extra_env: dict[str, str] | None = None) -> AppInfo:
        """Launch an application inside the isolated session.

        Returns AppInfo with pid, command, and log_path.
        """
        if not self.is_running or self._info is None:
            msg = "Session is not running"
            raise RuntimeError(msg)

        env = {
            **os.environ,
            "WAYLAND_DISPLAY": self._socket_name,
            "QT_QPA_PLATFORM": "wayland",
            "QT_LINUX_ACCESSIBILITY_ALWAYS_ON": "1",
            "QT_ACCESSIBILITY": "1",
        }
        env.update(self._xdg_isolation_env())
        if extra_env:
            env.update(extra_env)
        if self._info.dbus_address:
            env["DBUS_SESSION_BUS_ADDRESS"] = self._info.dbus_address

        # Create log file for stdout/stderr capture
        app_name = Path(command[0]).stem if command else "unknown"
        self._app_counter += 1
        log_path = self._info.screenshot_dir / f"app_{app_name}_{self._app_counter}.log"
        log_file = log_path.open("ab")

        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        # Close the fd in the parent; child has inherited it
        log_file.close()

        app_info = AppInfo(
            pid=proc.pid,
            command=" ".join(command),
            log_path=log_path,
            process=proc,
        )
        self._info.app_pid = proc.pid
        self._info.apps[proc.pid] = app_info
        return app_info

    def read_app_log(self, pid: int, last_n_lines: int = 50) -> str:
        """Read the log output of a launched app.

        Args:
            pid: PID of the app (from launch_app).
            last_n_lines: Number of trailing lines to return (0 = all).

        Returns:
            The app's stdout/stderr output.
        """
        if self._info is None:
            msg = "Session is not running"
            raise RuntimeError(msg)

        app = self._info.apps.get(pid)
        if app is None:
            available = list(self._info.apps.keys())
            msg = f"No app with PID {pid}. Available PIDs: {available}"
            raise ValueError(msg)

        if not app.log_path.exists():
            return "(no log output yet)"

        text = app.log_path.read_text(errors="replace")
        if last_n_lines > 0:
            lines = text.splitlines()
            text = "\n".join(lines[-last_n_lines:])
        return text or "(no log output yet)"

    def stop(self) -> None:
        """Stop the isolated session and clean up all processes."""
        if self._process is None:
            return

        # Tear down the streamer first — it talks to KWin over D-Bus and
        # would otherwise log a flurry of broken-pipe errors as we kill KWin.
        self._stop_streamer()

        # Send SIGTERM to the entire process group (all children)
        try:
            pgid = os.getpgid(self._process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill the entire process group
            try:
                pgid = os.getpgid(self._process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            with contextlib.suppress(ProcessLookupError):
                self._process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=3)

        # Close and remove the stderr capture file. Keep it on disk only if
        # the wrapper exited abnormally so the user can post-mortem.
        if self._stderr_log_fp is not None:
            with contextlib.suppress(Exception):
                self._stderr_log_fp.close()
            self._stderr_log_fp = None
        if self._stderr_log_path is not None and self._process.returncode in (0, -signal.SIGTERM):
            self._stderr_log_path.unlink(missing_ok=True)
            self._stderr_log_path = None

        # Clean up home directory and/or screenshot directory
        if self._home_dir is not None:
            keep_home = self._config is not None and self._config.keep_home
            keep_screenshots = self._config is not None and self._config.keep_screenshots
            if not keep_home:
                # Remove entire home dir (includes screenshots)
                shutil.rmtree(self._home_dir, ignore_errors=True)
            elif not keep_screenshots:
                # Keep home but remove screenshots subdirectory
                screenshots = self._home_dir / ".screenshots"
                if screenshots.exists():
                    shutil.rmtree(screenshots, ignore_errors=True)
        else:
            # No isolated home — use original screenshot cleanup logic
            keep = self._config is not None and self._config.keep_screenshots
            if not keep and self._info and self._info.screenshot_dir.exists():
                shutil.rmtree(self._info.screenshot_dir, ignore_errors=True)

        # Clean up socket files
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        for suffix in ("", ".lock"):
            path = Path(runtime_dir) / f"{self._socket_name}{suffix}"
            path.unlink(missing_ok=True)

        self._process = None
        self._info = None
        self._home_dir = None

    def _build_wrapper_script(self, config: SessionConfig) -> str:
        """Build the bash script that runs inside dbus-run-session."""
        return f"""\
echo "DBUS_SESSION_BUS_ADDRESS=$DBUS_SESSION_BUS_ADDRESS"

# Ensure all child processes are cleaned up on exit.
# The AT-SPI bus launcher and registryd are started via D-Bus
# auto-activation below; they are terminated automatically when our
# isolated session bus exits (dbus-run-session tears the bus down on
# parent exit), so we only need to track KWin explicitly here.
cleanup() {{
    kill $KWIN_PID 2>/dev/null
    wait $KWIN_PID 2>/dev/null
}}
trap cleanup EXIT TERM INT HUP

# Bring up the AT-SPI accessibility bus via D-Bus auto-activation.
# Calling org.a11y.Bus.GetAddress makes dbus-daemon resolve the
# service file (/usr/share/dbus-1/services/org.a11y.Bus.service) and
# exec the launcher at whatever path the current distro uses — Arch
# /usr/lib, Fedora/Debian/Ubuntu /usr/libexec, Flatpak /app/libexec,
# etc. Hardcoding a path breaks on everything except Arch.
# ATSPI_DBUS_IMPLEMENTATION=dbus-daemon (set in _build_env) prevents
# dbus-broker from sharing the host's a11y bus.
# The registryd comes up on its own when apps first touch the a11y
# bus, so no manual bootstrap is needed here.
gdbus call --session \\
    --dest=org.a11y.Bus \\
    --object-path=/org/a11y/bus \\
    --method=org.a11y.Bus.GetAddress >/dev/null 2>&1 || true
sleep 0.3

# Pre-set D-Bus activation environment BEFORE starting KWin.
# When KWin triggers portal auto-activation, portal-kde will get
# WAYLAND_DISPLAY pointing to our isolated compositor socket.
# The socket doesn't exist yet, but portal-kde will be activated
# only after KWin creates it.
dbus-update-activation-environment WAYLAND_DISPLAY={self._socket_name} QT_QPA_PLATFORM=wayland

{self._kwin_invocation(config)}
KWIN_PID=$!

# Wait for KWin socket to appear
while [ ! -e "$XDG_RUNTIME_DIR/{self._socket_name}" ]; do sleep 0.1; done
sleep 0.3

# Signal parent that setup is complete
echo "READY"

# Block until kwin exits
wait $KWIN_PID
"""

    def _kwin_invocation(self, config: SessionConfig) -> str:
        """Bash fragment that launches kwin_wayland in the background.

        - virtual mode (default): an in-memory framebuffer, invisible. KWin
          must NOT inherit $WAYLAND_DISPLAY or it tries to connect to that
          compositor as a client instead of creating its own.
        - visible mode: KWin runs as a nested Wayland client of the host
          compositor (niri, sway, GNOME, Plasma, ...) and appears as a
          regular window the user can watch. $WAYLAND_DISPLAY is inherited.
        """
        common = (
            "KWIN_WAYLAND_NO_PERMISSION_CHECKS=1 "
            "KWIN_SCREENSHOT_NO_PERMISSION_CHECKS=1 "
            f"kwin_wayland --no-lockscreen "
            f"--width {config.screen_width} --height {config.screen_height} "
            f"--socket {self._socket_name}"
        )
        if config.visible:
            # Inherit WAYLAND_DISPLAY so KWin nests inside the host compositor.
            return f"{common} &"
        return f"env -u WAYLAND_DISPLAY -u QT_QPA_PLATFORM {common} --virtual &"

    def _build_env(self, config: SessionConfig) -> dict[str, str]:
        """Build the environment for the isolated session."""
        # NOTE: do NOT set KDE_FULL_SESSION / KDE_SESSION_VERSION here. They
        # are meant to advertise a *real, complete* Plasma session, which
        # makes KWin try to talk to kded6, kglobalaccel, plasmashell, and the
        # KDE polkit agent. None of those exist in our isolated --virtual
        # session, and KWin segfaults during init on hosts that don't run
        # Plasma as the host desktop (e.g. anyone using kwin-mcp from GNOME,
        # niri, sway, hyprland, etc.).
        # XDG_CURRENT_DESKTOP=KDE is enough on its own to make the right
        # xdg-desktop-portal backend get picked.
        env = {
            **os.environ,
            "XDG_SESSION_TYPE": "wayland",
            "XDG_CURRENT_DESKTOP": "KDE",
            "QT_LINUX_ACCESSIBILITY_ALWAYS_ON": "1",
            "QT_ACCESSIBILITY": "1",
            # Force dbus-daemon for the AT-SPI bus instead of dbus-broker.
            # dbus-broker with --scope=user reuses the host's existing AT-SPI bus,
            # breaking accessibility isolation. Verified as REQUIRED.
            "ATSPI_DBUS_IMPLEMENTATION": "dbus-daemon",
            # Allow direct D-Bus screenshot capture without portal authorization.
            # Safe in isolated virtual sessions where there is no user desktop to protect.
            "KWIN_SCREENSHOT_NO_PERMISSION_CHECKS": "1",
            # Allow clients to bind restricted Wayland protocols (e.g. plasma_window_management).
            # Safe in isolated virtual sessions where there is no user desktop to protect.
            "KWIN_WAYLAND_NO_PERMISSION_CHECKS": "1",
        }
        # Remove host display references to avoid kwin connecting to host —
        # only in virtual mode. In visible mode we deliberately let KWin
        # inherit $WAYLAND_DISPLAY so it can nest as a Wayland client in
        # the host compositor.
        if not config.visible:
            env.pop("WAYLAND_DISPLAY", None)
            env.pop("DISPLAY", None)

        env.update(self._xdg_isolation_env())
        env.update(config.extra_env)
        return env

    def _wait_for_socket(self, socket_path: Path, timeout: float) -> bool:
        """Wait for the Wayland socket file to appear."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if socket_path.exists():
                return True
            # Check if process died
            if self._process and self._process.poll() is not None:
                return False
            time.sleep(0.2)
        return False

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class LiveSession:
    """Connection to an existing (non-virtual) KWin session.

    Attaches to a KWin compositor that is already running, such as
    the user's real desktop or a KWin instance inside a container.
    Does NOT manage the compositor lifecycle — stop() only disconnects.
    """

    def __init__(
        self,
        dbus_address: str,
        wayland_socket: str,
        screenshot_dir: Path,
    ) -> None:
        self._info = SessionInfo(
            dbus_address=dbus_address,
            wayland_socket=wayland_socket,
            kwin_pid=0,
            screenshot_dir=screenshot_dir,
            session_type=SessionType.LIVE,
        )
        self._running = True
        self._app_counter: int = 0
        self._keep_screenshots: bool = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def info(self) -> SessionInfo | None:
        return self._info if self._running else None

    @property
    def wayland_socket(self) -> str:
        return self._info.wayland_socket

    def launch_app(self, command: list[str], extra_env: dict[str, str] | None = None) -> AppInfo:
        """Launch an application in the live session.

        Returns AppInfo with pid, command, and log_path.
        """
        if not self._running:
            msg = "Session is not running"
            raise RuntimeError(msg)

        env = {
            **os.environ,
            "WAYLAND_DISPLAY": self._info.wayland_socket,
            "QT_QPA_PLATFORM": "wayland",
            "QT_LINUX_ACCESSIBILITY_ALWAYS_ON": "1",
            "QT_ACCESSIBILITY": "1",
        }
        if self._info.dbus_address:
            env["DBUS_SESSION_BUS_ADDRESS"] = self._info.dbus_address
        if extra_env:
            env.update(extra_env)

        app_name = Path(command[0]).stem if command else "unknown"
        self._app_counter += 1
        log_path = self._info.screenshot_dir / f"app_{app_name}_{self._app_counter}.log"
        log_file = log_path.open("ab")

        proc = subprocess.Popen(
            command,
            env=env,
            stdout=log_file,
            stderr=log_file,
        )
        log_file.close()

        app_info = AppInfo(
            pid=proc.pid,
            command=" ".join(command),
            log_path=log_path,
            process=proc,
        )
        self._info.app_pid = proc.pid
        self._info.apps[proc.pid] = app_info
        return app_info

    def read_app_log(self, pid: int, last_n_lines: int = 50) -> str:
        """Read the log output of a launched app."""
        app = self._info.apps.get(pid)
        if app is None:
            available = list(self._info.apps.keys())
            msg = f"No app with PID {pid}. Available PIDs: {available}"
            raise ValueError(msg)

        if not app.log_path.exists():
            return "(no log output yet)"

        text = app.log_path.read_text(errors="replace")
        if last_n_lines > 0:
            lines = text.splitlines()
            text = "\n".join(lines[-last_n_lines:])
        return text or "(no log output yet)"

    def stop(self, *, keep_screenshots: bool = False) -> None:
        """Disconnect from the live session.

        Only cleans up screenshot directory. Does NOT kill KWin or any apps
        that were already running before the connection.
        """
        if not self._running:
            return
        self._running = False

        # Terminate apps launched by us
        for app in self._info.apps.values():
            with contextlib.suppress(ProcessLookupError, PermissionError):
                app.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                app.process.wait(timeout=3)

        if not keep_screenshots and self._info.screenshot_dir.exists():
            shutil.rmtree(self._info.screenshot_dir, ignore_errors=True)
