"""Content-checked atomic cache with process-safe, crash-released file locks."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from plotly.utils import PlotlyJSONEncoder


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      cls=PlotlyJSONEncoder).encode('utf-8')


def fingerprint(value):
    return hashlib.sha256(encoded(value)).hexdigest()


@contextmanager
def file_lock(path, timeout=30):
    """OS lock is released on process death, unlike persistent mkdir locks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            stream.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('figure.cache_busy')
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def preserve_invalid(path):
    path = Path(path)
    if path.is_file():
        identifier = uuid.uuid4().hex
        backup = path.with_name(path.name + '.invalid-' + identifier)
        # Preserve corruption evidence without turning an otherwise valid
        # Windows filename into an inaccessible MAX_PATH-sized destination.
        if len(str(backup.resolve())) >= 260:
            backup = path.with_name('invalid-' + identifier + '.bin')
            metadata = {'original_name': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            atomic_write(backup.with_suffix('.json'), encoded(metadata))
        shutil.copyfile(path, backup)
        return backup


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checked(path, identity):
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if (saved['identity'] == identity and
                saved['payload_sha256'] == fingerprint(saved['payload'])):
            return saved['payload']
    except (ValueError, KeyError, TypeError):
        pass
    return None


def save_checked(path, identity, payload):
    if path.exists() and load_checked(path, identity) is None:
        preserve_invalid(path)
    atomic_write(path, encoded({'identity': identity, 'payload': payload,
                                'payload_sha256': fingerprint(payload)}))
