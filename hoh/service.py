"""Run the `life` loop as a background service so nobody has to start it by hand.

Two layers, use either or both:
  * OS service files (start at login/boot, restart on crash, low priority): systemd (Linux), launchd (macOS),
    runit/termux-services (Android Termux), Task Scheduler (Windows). `hoh service install` WRITES the file and prints the
    exact commands to enable it (add --apply to run them). Rendering is pure text, so it is testable anywhere.
  * A tiny built-in supervisor (`hoh service start|stop|status`) for machines without an init system: detached process,
    pidfile, log file with size cap, restart with exponential back-off (reset after a healthy run), clean stop on SIGTERM.

The service only launches `python -m hoh lifelong life ...`; all the safety rules (idle gate, STOP file, budgets, anchor
check) live in that loop, so `hoh lifelong pause --state-dir ...` still stops learning without touching the service."""
import json, os, shlex, signal, subprocess, sys, time
from pathlib import Path

NAME = "hoh-life"


def life_command(state_dir, extra=(), python=None):
    return [python or sys.executable, "-m", "hoh", "lifelong", "life", "--state-dir", str(state_dir), *extra]


# ------------------------------------------------------------------ renderers (pure text)
def render_systemd(cmd, workdir, user_mode=True):
    return f"""[Unit]
Description=HOH lifelong learning loop (explore, sleep, reflect)
After=network-online.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={' '.join(shlex.quote(c) for c in cmd)}
Restart=always
RestartSec=20
Nice=10
IOSchedulingClass=idle
CPUSchedulingPolicy=idle
Environment=PYTHONUNBUFFERED=1
TimeoutStopSec=60

[Install]
WantedBy={'default.target' if user_mode else 'multi-user.target'}
"""


def render_launchd(cmd, workdir, log):
    args = "".join(f"    <string>{c}</string>\n" for c in cmd)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>ai.hoh.life</string>
  <key>ProgramArguments</key><array>
{args}  </array>
  <key>WorkingDirectory</key><string>{workdir}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>20</integer>
  <key>ProcessType</key><string>Background</string>
  <key>Nice</key><integer>10</integer>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
</dict></plist>
"""


def render_runit(cmd, workdir):
    """Termux (termux-services) / any runit tree: a `run` script. Needs `termux-wake-lock` so Android does not suspend it."""
    return f"""#!/data/data/com.termux/files/usr/bin/sh
# HOH life loop. Keep the CPU awake while it runs (Android may otherwise freeze Termux):
termux-wake-lock 2>/dev/null
cd {shlex.quote(str(workdir))}
exec nice -n 10 {' '.join(shlex.quote(c) for c in cmd)} 2>&1
"""


def render_windows(cmd, workdir):
    c = subprocess.list2cmdline(cmd)
    return (f'schtasks /Create /TN "{NAME}" /SC ONLOGON /RL LIMITED /F /TR "cmd /c cd /d {workdir} && {c}"\r\n'
            f'schtasks /Run /TN "{NAME}"\r\n')


def detect_platform():
    if os.environ.get("TERMUX_VERSION") or "com.termux" in os.environ.get("PREFIX", ""):
        return "termux"
    return {"linux": "systemd", "darwin": "launchd", "win32": "windows"}.get(sys.platform, "supervisor")


def plan(state_dir, platform="auto", extra=(), workdir=None, home=None, python=None):
    """-> {'platform', 'path', 'content', 'commands': [...], 'undo': [...]} without touching the disk."""
    platform = detect_platform() if platform == "auto" else platform
    state_dir = Path(state_dir).resolve()
    workdir = str(Path(workdir or os.getcwd()).resolve())
    home = Path(home or Path.home())
    cmd = life_command(state_dir, extra, python)
    if platform == "systemd":
        path = home / ".config/systemd/user" / f"{NAME}.service"
        return {"platform": platform, "path": path, "content": render_systemd(cmd, workdir),
                "commands": ["systemctl --user daemon-reload", f"systemctl --user enable --now {NAME}.service",
                             "loginctl enable-linger $USER   # keep running when you are logged out"],
                "undo": [f"systemctl --user disable --now {NAME}.service"]}
    if platform == "launchd":
        path = home / "Library/LaunchAgents/ai.hoh.life.plist"
        return {"platform": platform, "path": path, "content": render_launchd(cmd, workdir, str(state_dir / "service.log")),
                "commands": [f"launchctl load -w {path}"], "undo": [f"launchctl unload -w {path}"]}
    if platform == "termux":
        prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
        path = prefix / "var/service" / NAME / "run"
        return {"platform": platform, "path": path, "content": render_runit(cmd, workdir), "mode": 0o755,
                "commands": ["pkg install termux-services termux-api", "# restart Termux once, then:", f"sv-enable {NAME}"],
                "undo": [f"sv-disable {NAME}"]}
    if platform == "windows":
        path = Path(state_dir) / "install_service.cmd"
        return {"platform": platform, "path": path, "content": render_windows(cmd, workdir),
                "commands": [f'"{path}"'], "undo": [f'schtasks /Delete /TN "{NAME}" /F']}
    return {"platform": "supervisor", "path": None, "content": "", "commands": [f"python -m hoh service start --state-dir {state_dir}"],
            "undo": [f"python -m hoh service stop --state-dir {state_dir}"]}


def install(state_dir, platform="auto", extra=(), apply=False, **kw):
    p = plan(state_dir, platform, extra, **kw)
    if p["path"] is not None:
        Path(p["path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(p["path"]).write_text(p["content"], encoding="utf-8")
        if p.get("mode"):
            os.chmod(p["path"], p["mode"])
    if apply:
        p["applied"] = []
        for c in p["commands"]:
            if c.startswith("#") or "pkg install" in c or "sv-enable" in c:
                continue                                  # needs an interactive/relaunched shell; printed only
            r = subprocess.run(c.split("#")[0].strip(), shell=True, capture_output=True, text=True)
            p["applied"].append({"cmd": c, "rc": r.returncode, "err": r.stderr[-300:]})
    return p


# ------------------------------------------------------------------ built-in supervisor
class Supervisor:
    """Keeps `cmd` running. `popen`, `sleep`, `clock` are injectable for tests."""

    def __init__(self, cmd, state_dir, max_log_bytes=5_000_000, min_delay=5.0, max_delay=1800.0, healthy_after=300.0,
                 popen=subprocess.Popen, sleep=time.sleep, clock=time.time):
        self.cmd, self.dir = cmd, Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.pidfile, self.logfile = self.dir / "service.pid", self.dir / "service.log"
        self.max_log, self.min_delay, self.max_delay, self.healthy = max_log_bytes, min_delay, max_delay, healthy_after
        self.popen, self.sleep, self.clock = popen, sleep, clock
        self.child, self.stopping, self.restarts = None, False, 0

    def _rotate(self):
        if self.logfile.exists() and self.logfile.stat().st_size > self.max_log:
            self.logfile.replace(self.logfile.with_suffix(".log.1"))

    def stop(self, *_):
        self.stopping = True
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()

    def run(self, max_restarts=None):
        self.pidfile.write_text(str(os.getpid()))
        for s in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(s, self.stop)
            except ValueError:                           # not in the main thread (tests)
                pass
        delay, n = self.min_delay, 0
        try:
            while not self.stopping and (max_restarts is None or n <= max_restarts):
                self._rotate()
                started = self.clock()
                with self.logfile.open("ab") as log:
                    self.child = self.popen(self.cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    rc = self.child.wait()
                ran = self.clock() - started
                n += 1
                self.restarts = n - 1
                if self.stopping:
                    break
                delay = self.min_delay if ran >= self.healthy else min(self.max_delay, delay * 2)
                with self.logfile.open("a", encoding="utf-8") as log:
                    log.write(f"\n[supervisor] exit code {rc} after {ran:.0f}s; restarting in {delay:.0f}s\n")
                self.sleep(delay)
        finally:
            self.pidfile.unlink(missing_ok=True)
        return n


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def status(state_dir):
    pf = Path(state_dir) / "service.pid"
    pid = int(pf.read_text()) if pf.exists() and pf.read_text().strip().isdigit() else None
    return {"running": bool(pid and _alive(pid)), "pid": pid, "log": str(Path(state_dir) / "service.log")}


def start_detached(state_dir, extra=()):
    if status(state_dir)["running"]:
        return {"started": False, "reason": "already running", **status(state_dir)}
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "hoh", "service", "supervise", "--state-dir", str(state_dir), *extra]
    kw = {"start_new_session": True} if os.name != "nt" else {"creationflags": 0x00000008 | 0x00000200}
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, **kw)
    return {"started": True, "supervisor_pid": p.pid}


def stop_detached(state_dir, timeout=30.0):
    st = status(state_dir)
    if not st["running"]:
        return {"stopped": False, "reason": "not running"}
    os.kill(st["pid"], signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < timeout and _alive(st["pid"]):
        time.sleep(0.2)
    return {"stopped": not _alive(st["pid"]), "pid": st["pid"]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="hoh service", description="Run HOH's life loop in the background.")
    ap.add_argument("action", choices=["install", "print", "uninstall", "start", "stop", "status", "restart", "supervise"])
    ap.add_argument("--state-dir", default="runs/life")
    ap.add_argument("--platform", default="auto", choices=["auto", "systemd", "launchd", "termux", "windows", "supervisor"])
    ap.add_argument("--apply", action="store_true", help="install: also run the enable commands")
    a, rest = ap.parse_known_args(argv)               # everything else is passed on to `hoh lifelong life`
    rest = [r for r in rest if r != "--"]
    if a.action == "print":
        p = plan(a.state_dir, a.platform, rest)
        print(f"# {p['platform']}  ->  {p['path']}\n{p['content']}\n# then run:\n" + "\n".join(p["commands"]))
    elif a.action == "install":
        p = install(a.state_dir, a.platform, rest, apply=a.apply)
        print(json.dumps({k: str(v) for k, v in p.items() if k != "content"}, indent=2, ensure_ascii=False))
    elif a.action == "uninstall":
        p = plan(a.state_dir, a.platform, rest)
        print("run:\n" + "\n".join(p["undo"]) + (f"\nthen delete {p['path']}" if p["path"] else ""))
    elif a.action == "start":
        print(json.dumps(start_detached(a.state_dir, rest)))
    elif a.action == "stop":
        print(json.dumps(stop_detached(a.state_dir)))
    elif a.action == "restart":
        stop_detached(a.state_dir)
        print(json.dumps(start_detached(a.state_dir, rest)))
    elif a.action == "status":
        print(json.dumps(status(a.state_dir)))
    else:
        Supervisor(life_command(a.state_dir, rest), a.state_dir).run()
    return 0
