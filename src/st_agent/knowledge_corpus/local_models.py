"""Optional model adapter: parent stays independent of Torch and transformers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from st_agent.contracts import ResourceRequest
from st_agent.resources import ResourceAdmission
from st_agent.trace import Trace, redact

from .contracts import CorpusError
from .indexes import EmbeddingSpec, IndexCache

MODEL_NAME = "BAAI/bge-m3"
BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
BGE_M3_WEIGHTS_SHA256 = "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38"


def _safe(value):
    value = redact(value)
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"\bhf_[A-Za-z0-9]{10,}\b", "[REDACTED]", value)
        return re.sub(
            r"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+", "[REDACTED]", value
        )
    return value


def _lock(stream):
    if stream.seek(0, os.SEEK_END) == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(stream):
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SubprocessBgeM3:
    """One finite local CPU worker per encode call; no automatic model download."""

    def __init__(
        self,
        model_dir,
        python_executable,
        *,
        worker_script=None,
        max_length=384,
        batch_size=1,
        dtype="float16",
        timeout_seconds=900,
        cpu_threads=2,
        artifact_root=None,
        resource_admission=None,
    ):
        self.model_dir = Path(model_dir).resolve()
        self.python_executable = Path(python_executable).resolve()
        repo = Path(__file__).resolve().parents[3]
        self.worker_script = Path(worker_script or repo / "evals/run_local_bge_worker.py").resolve()
        self.artifact_root = Path(artifact_root or repo / "artifacts/e2e/local-bge").resolve()
        if type(max_length) is not int or not 16 <= max_length <= 512:
            raise CorpusError("embedding.invalid_max_length")
        if (
            type(batch_size) is not int
            or not 1 <= batch_size <= 2
            or dtype not in ("float16", "float32")
        ):
            raise CorpusError("embedding.invalid_batch_or_dtype")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 86400:
            raise CorpusError("embedding.invalid_timeout")
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= 8:
            raise CorpusError("embedding.invalid_cpu_threads")
        if not self.python_executable.is_file() or not self.worker_script.is_file():
            raise CorpusError("embedding.worker_unavailable")
        self.max_length, self.batch_size, self.dtype = max_length, batch_size, dtype
        self.timeout_seconds, self.cpu_threads = timeout_seconds, cpu_threads
        self.resource_admission = resource_admission or ResourceAdmission()
        self.manifest_sha256 = self._verify_model()
        self.worker_sha256 = _digest(self.worker_script)
        try:
            probe = subprocess.run(
                [str(self.python_executable), str(self.worker_script), "--probe"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=True,
                env=self._environment(),
            )
            self.runtime_versions = json.loads(probe.stdout)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            raise CorpusError("embedding.worker_probe_failed") from error
        encoded = self._config_bytes()
        self.identity = EmbeddingSpec(
            model_name=MODEL_NAME,
            revision=BGE_M3_REVISION,
            encoding_version="cls-l2-cpu-v1-" + hashlib.sha256(encoded).hexdigest(),
            dimension=1024,
            normalized=True,
        )
        self._encoding_config_sha256 = hashlib.sha256(encoded).hexdigest()
        self._expected_identity = self.identity
        self.last_run_dir = None

    def _config_bytes(self):
        config = {
            "adapter": "local-bge-cls-l2-v1",
            "manifest": self.manifest_sha256,
            "worker": self.worker_sha256,
            "runtime": self.runtime_versions,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "device": "cpu",
            "cpu_threads": self.cpu_threads,
            "padding": "longest",
            "truncation": True,
        }
        return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def _validate_config(self):
        try:
            current = hashlib.sha256(self._config_bytes()).hexdigest()
        except (ValueError, TypeError):
            raise CorpusError("embedding.encoding_config_changed") from None
        if current != self._encoding_config_sha256 or self.identity != self._expected_identity:
            raise CorpusError("embedding.encoding_config_changed")

    @staticmethod
    def _environment():
        env = dict(os.environ)
        env.update(
            USE_TF="0",
            USE_FLAX="0",
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONIOENCODING="utf-8",
        )
        return env

    def _verify_model(self):
        path = self.model_dir / "model-manifest.json"
        if not path.is_file():
            raise CorpusError("embedding.model_not_downloaded")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("model_name") != MODEL_NAME or manifest.get("revision") != BGE_M3_REVISION:
            raise CorpusError("embedding.model_identity_mismatch")
        if not manifest.get("files") or "pytorch_model.bin" not in manifest["files"]:
            raise CorpusError("embedding.model_manifest_invalid")
        if manifest["files"]["pytorch_model.bin"]["sha256"] != BGE_M3_WEIGHTS_SHA256:
            raise CorpusError("embedding.pinned_file_mismatch")
        for name, item in manifest["files"].items():
            file = (self.model_dir / name).resolve()
            if self.model_dir not in file.parents or not file.is_file():
                raise CorpusError("embedding.model_file_invalid")
            if file.stat().st_size != item["size"] or _digest(file) != item["sha256"]:
                raise CorpusError("embedding.model_checksum_mismatch")
        return _digest(path)

    def encode(self, texts):
        if isinstance(texts, (str, bytes)):
            raise CorpusError("embedding.invalid_texts")
        texts = list(texts)
        if not texts:
            return []
        if len(texts) > 1024 or any(
            not isinstance(text, str) or not text.strip() or len(text) > 20000 for text in texts
        ):
            raise CorpusError("embedding.invalid_texts")
        if any(_safe(text) != text for text in texts):
            raise CorpusError("embedding.unsafe_text_for_artifact")
        self._validate_config()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix="encode-", dir=self.artifact_root))
        self.last_run_dir = folder
        telemetry = Trace(folder)
        telemetry.context.update(task_id="local-bge-adapter", run_id=folder.name)
        telemetry.emit(
            "embedding.encode_started",
            status="started",
            count=len(texts),
            embedding=self.identity.model_dump(mode="json"),
        )
        report = {
            "status": "started",
            "embedding": self.identity.model_dump(mode="json"),
            "count": len(texts),
            "device": "cpu",
            "dtype": self.dtype,
        }
        started = time.monotonic()
        lock_path = self.model_dir / ".encoding.lock"
        lock_stream = None
        lock_acquired = False
        try:
            lock_stream = lock_path.open("a+b")
            try:
                _lock(lock_stream)
                lock_acquired = True
            except OSError as error:
                raise CorpusError("resource.heavy_job_running") from error
            required_ram = 3_000_000_000 if self.dtype == "float16" else 4_500_000_000
            resources = ResourceRequest(
                estimated_ram_bytes=required_ram,
                minimum_free_ram_bytes=1_000_000_000,
                estimated_temp_bytes=10_000_000,
                minimum_free_disk_bytes=1_000_000_000,
                timeout_seconds=self.timeout_seconds,
                heavy=True,
            )
            decision = self.resource_admission.check(
                resources, project_path=self.model_dir, heavy_job_running=False
            )
            report["resource_decision"] = decision.model_dump(mode="json")
            telemetry.emit(
                "embedding.resource_admission",
                status="allowed" if decision.allowed else "rejected",
                decision=decision.model_dump(mode="json"),
            )
            if not decision.allowed:
                raise CorpusError(decision.error.code)
            if (
                self._verify_model() != self.manifest_sha256
                or _digest(self.worker_script) != self.worker_sha256
            ):
                raise CorpusError("embedding.identity_changed_during_encode")
            request = {
                "schema_version": 1,
                "texts": texts,
                "manifest_sha256": self.manifest_sha256,
                "worker_sha256": self.worker_sha256,
                "runtime_versions": self.runtime_versions,
                "device": "cpu",
                "dtype": self.dtype,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
                "cpu_threads": self.cpu_threads,
                "required_available_ram_bytes": required_ram + resources.minimum_free_ram_bytes,
            }
            input_path = folder / "input.json"
            input_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            worker_folder = folder / "worker"
            try:
                result = subprocess.run(
                    [
                        str(self.python_executable),
                        str(self.worker_script),
                        "--model-dir",
                        str(self.model_dir),
                        "--run-dir",
                        str(worker_folder),
                        "--input",
                        str(input_path),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.timeout_seconds,
                    env=self._environment(),
                )
            except subprocess.TimeoutExpired as error:
                for name, captured in (("stdout.txt", error.stdout), ("stderr.txt", error.stderr)):
                    if isinstance(captured, bytes):
                        captured = captured.decode("utf-8", errors="replace")
                    (folder / name).write_text(_safe(captured or ""), encoding="utf-8")
                raise CorpusError("embedding.worker_timeout") from error
            (folder / "stdout.txt").write_text(_safe(result.stdout), encoding="utf-8")
            (folder / "stderr.txt").write_text(_safe(result.stderr), encoding="utf-8")
            response_path = worker_folder / "result.json"
            if not response_path.is_file():
                raise CorpusError("embedding.worker_protocol_failure")
            response = json.loads(response_path.read_text(encoding="utf-8"))
            report["worker"] = {key: value for key, value in response.items() if key != "vectors"}
            if result.returncode or response.get("status") != "encoded":
                raise CorpusError(response.get("error_code", "embedding.worker_failed"))
            if (
                response.get("manifest_sha256") != self.manifest_sha256
                or response.get("worker_sha256") != self.worker_sha256
                or response.get("runtime_versions") != self.runtime_versions
            ):
                raise CorpusError("embedding.worker_identity_mismatch")
            vectors = IndexCache._vectors(response.get("vectors"), len(texts), self.identity)
            if self._verify_model() != self.manifest_sha256:
                raise CorpusError("embedding.identity_changed_during_encode")
            self._validate_config()
            report["status"] = "encoded"
            return vectors
        except Exception as error:
            report.update(
                status="failed",
                error_code=str(error)
                if isinstance(error, CorpusError)
                else "embedding.adapter_failed",
                exception_type=type(error).__name__,
            )
            raise
        finally:
            if lock_stream is not None:
                try:
                    if lock_acquired:
                        _unlock(lock_stream)
                finally:
                    lock_stream.close()
            report["duration_ms"] = round((time.monotonic() - started) * 1000)
            report["trace_degraded"] = telemetry.degraded
            (folder / "report.json").write_text(
                json.dumps(_safe(report), ensure_ascii=False), encoding="utf-8"
            )
            telemetry.emit(
                "embedding.encode_completed",
                status=report["status"],
                error_code=report.get("error_code"),
                duration_ms=report["duration_ms"],
                report_ref="report.json",
                report_sha256=_digest(folder / "report.json"),
            )
