"""Process isolation for the submission: staging, unprivileged launch, IPC.

The scorer process owns the clock, the inputs' seeds, the reference and the
reward. Submitted code only runs in a worker process under UID 2000 (when the
scorer runs as root) and talks to the scorer over a socket.
"""

import json
import os
import pathlib
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import time

UID = 2000
PYTHON = "/opt/venv/bin/python"
SOURCE_SUFFIXES = {
    ".py", ".cu", ".cuh", ".h", ".hpp", ".c", ".cc", ".cpp", ".cxx", ".sh",
    ".json", ".txt", ".md", ".toml", ".yaml", ".yml", ".cmake", ".in",
}
SKIP_DIRS = {"__pycache__", "_build", "build", ".cache", ".git", ".venv"}
MAX_FILE = 16 * 1024**2
MAX_TOTAL = 64 * 1024**2
MAX_FILES = 2048
ENV_PASSTHROUGH = ("PATH", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES",
                   "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")


class InfrastructureError(RuntimeError):
  """A failure of trusted evaluation code or the machine, not the submission."""


class SubmissionError(RuntimeError):
  """The submission is missing, malformed, crashed, timed out or misbehaved."""


def isolated() -> bool:
  return os.geteuid() == 0


def stage_submission(source: pathlib.Path, dest: pathlib.Path) -> int:
  """Copies source files only (no links, binaries or caches) into `dest`."""
  if source.is_symlink() or not source.is_dir():
    raise SubmissionError(f"missing submission directory {source}")
  if not (source / "moe_ffn" / "api.py").is_file():
    raise SubmissionError("submission needs moe_ffn/api.py")
  if not (source / "summary.md").is_file():
    raise SubmissionError("submission needs summary.md")
  total = count = 0
  for parent, dirs, files in os.walk(source, followlinks=False):
    for name in dirs + files:
      entry = pathlib.Path(parent) / name
      mode = entry.lstat().st_mode
      if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
        raise SubmissionError(
            f"links and special files are not accepted: {entry.relative_to(source)}")
    dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
    for name in sorted(files):
      src = pathlib.Path(parent) / name
      if src.suffix not in SOURCE_SUFFIXES and name not in {"Makefile", "CMakeLists.txt"}:
        continue  # No prebuilt binaries, wheels or caches.
      count += 1
      size = src.stat().st_size
      total += size
      if size > MAX_FILE or total > MAX_TOTAL or count > MAX_FILES:
        raise SubmissionError("source bundle exceeds 64 MiB / 2048 files (16 MiB per file)")
      dst = dest / src.relative_to(source)
      dst.parent.mkdir(parents=True, exist_ok=True)
      shutil.copyfile(src, dst)
      if b"\0" in dst.read_bytes():
        raise SubmissionError(f"binary content is not accepted: {src.relative_to(source)}")
  return count


def own_tree(path: pathlib.Path):
  if not isolated():
    return
  for parent, dirs, files in os.walk(path):
    os.chown(parent, UID, UID)
    os.chmod(parent, 0o755)
    for name in files:
      os.chown(pathlib.Path(parent) / name, UID, UID)


def _signal_candidates(sig) -> bool:
  """Sends `sig` to every live process of the submission UID; True if any."""
  live = False
  for item in pathlib.Path("/proc").iterdir():
    if not item.name.isdigit():
      continue
    try:
      fields = dict(line.split(":", 1) for line in
                    (item / "status").read_text().splitlines() if ":" in line)
      exited = fields["State"].split()[0] in {"Z", "X"} and int(fields["Threads"]) <= 1
      if UID in map(int, fields["Uid"].split()) and not exited:
        live = True
        os.kill(int(item.name), sig)
    except (FileNotFoundError, ProcessLookupError, KeyError, ValueError):
      pass
  return live


def kill_candidates():
  """SIGKILLs every process of the submission UID, including stray children."""
  if not isolated():
    return
  deadline = time.monotonic() + 10
  while _signal_candidates(signal.SIGKILL):
    if time.monotonic() >= deadline:
      raise InfrastructureError("submission processes survived cleanup")
    time.sleep(0.01)


# Resource limits for submitted code, applied by prlimit before setpriv (a
# preexec_fn is unsafe in this multithreaded process). NPROC stays well below
# the sandbox's 2,048-task cap.
_PRLIMIT = ["prlimit", "--core=0", f"--fsize={2 * 1024**3}", "--nofile=4096",
            "--nproc=1024", "--"]


def worker_env(home: pathlib.Path, mem_fraction: float) -> dict:
  env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
  env.update(
      HOME=str(home), TMPDIR=str(home), PYTHONDONTWRITEBYTECODE="1",
      XLA_PYTHON_CLIENT_PREALLOCATE="false",
      XLA_PYTHON_CLIENT_MEM_FRACTION=f"{mem_fraction:.2f}",
      OMP_NUM_THREADS="4", TF_CPP_MIN_LOG_LEVEL="1",
  )
  return env


def run_build(pkg: pathlib.Path, home: pathlib.Path, log, timeout_s: float):
  """Runs the optional `moe_ffn/build.sh` as the submission user."""
  if not (pkg / "build.sh").is_file():
    return
  cmd = ["bash", "build.sh"]
  if isolated():
    cmd = _setpriv(cmd)
  env = worker_env(home, 0.4)
  env["PATH"] = "/opt/venv/bin:" + env.get("PATH", "/usr/bin:/bin")
  try:
    subprocess.run(cmd, cwd=pkg, env=env, stdin=subprocess.DEVNULL, stdout=log,
                   stderr=log, timeout=timeout_s, check=True,
                   start_new_session=True)
  except subprocess.TimeoutExpired as ex:
    raise SubmissionError(f"build.sh exceeded {timeout_s:.0f} s") from ex
  except subprocess.CalledProcessError as ex:
    raise SubmissionError(f"build.sh failed with exit code {ex.returncode}") from ex
  finally:
    kill_candidates()


def _setpriv(args):
  return [*_PRLIMIT, "setpriv", f"--reuid={UID}", f"--regid={UID}", "--clear-groups",
          "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all",
          "--ambient-caps=-all", *args]


class Worker:
  """A worker process running `worker.py` for one moe_ffn package."""

  def __init__(self, name, pkg_root, worker_py, home, mem_fraction, log,
               untrusted, deadline_s):
    self.name = name
    self.untrusted = untrusted
    parent_sock, child_sock = socket.socketpair()
    args = [PYTHON, "-s", "-E", str(worker_py), str(child_sock.fileno())]
    if untrusted and isolated():
      args = _setpriv(args)
    self.proc = subprocess.Popen(
        args, cwd=pkg_root, env=worker_env(home, mem_fraction),
        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        pass_fds=(child_sock.fileno(),), start_new_session=True)
    child_sock.close()
    self.sock = parent_sock
    self.call({"op": "hello"}, deadline_s)

  def _error(self, msg):
    cls = SubmissionError if self.untrusted else InfrastructureError
    return cls(f"{self.name} worker: {msg}")

  def send(self, obj, deadline_s):
    data = json.dumps(obj, allow_nan=False).encode()
    self.sock.settimeout(deadline_s)
    try:
      self.sock.sendall(struct.pack("!I", len(data)) + data)
    except OSError as ex:
      raise self._error(f"send failed ({ex})") from ex

  def receive(self, deadline_s):
    end = time.monotonic() + deadline_s

    def exact(n):
      buf = bytearray()
      while len(buf) < n:
        remaining = end - time.monotonic()
        if remaining <= 0:
          raise self._error(f"timed out after {deadline_s:.0f} s")
        self.sock.settimeout(remaining)
        try:
          chunk = self.sock.recv(n - len(buf))
        except socket.timeout as ex:
          raise self._error(f"timed out after {deadline_s:.0f} s") from ex
        if not chunk:
          raise self._error(f"exited (code {self.proc.poll()}); see the worker log")
        buf.extend(chunk)
      return bytes(buf)

    n = struct.unpack("!I", exact(4))[0]
    if n > 1_000_000:
      raise self._error("oversized response")
    try:
      obj = json.loads(exact(n))
    except ValueError as ex:
      raise self._error("malformed response") from ex
    if not isinstance(obj, dict):
      raise self._error("malformed response")
    if "error" in obj:
      raise self._error(str(obj["error"])[:2000])
    return obj

  def call(self, obj, deadline_s):
    self.send(obj, deadline_s)
    return self.receive(deadline_s)

  def timed_call(self, obj, deadline_s):
    """Returns (response, wall seconds); the scorer's clock, not the worker's."""
    data = json.dumps(obj, allow_nan=False).encode()
    self.sock.settimeout(deadline_s)
    t0 = time.perf_counter()
    self.sock.sendall(struct.pack("!I", len(data)) + data)
    resp = self.receive(deadline_s)
    return resp, time.perf_counter() - t0

  def _signal(self, sig):
    if self.untrusted and isolated():
      # Every submission-UID process, including children that left the
      # worker's process group with setsid.
      _signal_candidates(sig)
      return
    try:
      os.killpg(self.proc.pid, sig)
    except ProcessLookupError:
      pass

  def pause(self):
    self._signal(signal.SIGSTOP)

  def resume(self):
    self._signal(signal.SIGCONT)

  def close(self):
    try:
      self.resume()
      self.send({"op": "stop"}, 10)
      self.proc.wait(timeout=30)
    except (Exception, subprocess.TimeoutExpired):  # pylint: disable=broad-except
      pass
    if self.proc.poll() is None:
      try:
        os.killpg(self.proc.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      self.proc.wait()
    self.sock.close()
    if self.untrusted:
      kill_candidates()


def fail_unsupported(reason: str, logs_dir: pathlib.Path):
  """Signals an unusable sandbox before any test ran (Harbor retries it)."""
  logs_dir.mkdir(parents=True, exist_ok=True)
  (logs_dir / "environment_unsupported.json").write_text(json.dumps({"reason": reason}))
  print(f"ENVIRONMENT UNSUPPORTED: {reason}", file=sys.stderr)
  sys.exit(1)
