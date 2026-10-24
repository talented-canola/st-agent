"""Pinned local CPU BGE-M3 worker; no project database or remote Python."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
ALLOWED_FILES = (
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "1_Pooling/config.json",
)
PINNED_FILE_HASHES = {
    "1_Pooling/config.json": ("git", "9bd85925f325e25246d94c4918dc02ab98f2a1b7"),
    "config.json": ("git", "e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad"),
    "config_sentence_transformers.json": ("git", "1fba91c78a6c8e17227058ab6d4d3acb5d8630a9"),
    "modules.json": ("git", "952a9b81c0bfd99800fabf352f69c7ccd46c5e43"),
    "pytorch_model.bin": (
        "sha256",
        "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
    ),
    "sentence_bert_config.json": ("git", "0140ba1eac83a3c9b857d64baba91969d988624b"),
    "sentencepiece.bpe.model": (
        "sha256",
        "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    ),
    "special_tokens_map.json": ("git", "b1879d702821e753ffe4245048eee415d54a9385"),
    "tokenizer.json": (
        "sha256",
        "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
    ),
    "tokenizer_config.json": ("git", "dc69ac559dcba2694012009aaa108c614541789a"),
}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def write_json(path, record):
    Path(path).write_text(
        json.dumps(redact(record), ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def redact(value):
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(
                    part in str(key).lower()
                    for part in ("secret", "api_key", "authorization", "password", "token_value")
                )
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)Bearer\s+\S+|\bsk-[\w-]+|\bhf_[A-Za-z0-9]{10,}\b", "[REDACTED]", value)
        value = re.sub(r"(?<![\w/])[A-Za-z]:[\\/][^\s\"'<>]*", "[ABS_PATH]", value)
        return re.sub(
            r"(?i)\b(?:api[_-]?key|password|secret|token)\s*[:=]\s*\S+", "[REDACTED]", value
        )
    return value


def trace(path, event, status, *, error_code=None, summary=None):
    with trace.lock:
        _trace(path, event, status, error_code=error_code, summary=summary)


def _trace(path, event, status, *, error_code=None, summary=None):
    if path is None:
        return
    record = dict(
        schema_version=1,
        event_id=str(uuid.uuid4()),
        seq=trace.seq,
        ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        trace_id=trace.run_id,
        span_id=str(uuid.uuid4()),
        parent_span_id=None,
        task_id="local-bge-worker",
        run_id=trace.run_id,
        step_id=None,
        attempt_id=None,
        turn_id=None,
        event=event,
        status=status,
        error_code=error_code,
        duration_ms=None,
        payload_ref=None,
        payload_sha256=None,
        summary=summary or {},
    )
    trace.seq += 1
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(redact(record), ensure_ascii=False) + "\n")
        stream.flush()


trace.seq = 1
trace.run_id = str(uuid.uuid4())
trace.lock = threading.RLock()


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "huggingface_hub", "safetensors")
    }


def verify_model(folder):
    manifest = json.loads((folder / "model-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model_name") != MODEL_NAME or manifest.get("revision") != REVISION:
        raise ValueError("embedding.model_identity_mismatch")
    if set(manifest["files"]) != set(ALLOWED_FILES):
        raise ValueError("embedding.model_manifest_invalid")
    for name, item in manifest["files"].items():
        file = (folder / name).resolve()
        if folder.resolve() not in file.parents or not file.is_file():
            raise ValueError("embedding.model_file_invalid")
        hashed = digest(file)
        if file.stat().st_size != item["size"] or hashed != item["sha256"]:
            raise ValueError("embedding.model_checksum_mismatch")
        mode, pinned = PINNED_FILE_HASHES[name]
        if mode == "git":
            content = file.read_bytes()
            hashed = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
        if hashed != pinned:
            raise ValueError("embedding.pinned_file_mismatch")
    pool = json.loads((folder / "1_Pooling/config.json").read_text(encoding="utf-8"))
    if (
        pool.get("word_embedding_dimension") != 1024
        or pool.get("pooling_mode_cls_token") is not True
    ):
        raise ValueError("embedding.pooling_config_mismatch")
    if any(
        pool.get(key, False)
        for key in (
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
        )
    ):
        raise ValueError("embedding.pooling_config_mismatch")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if config.get("hidden_size") != 1024 or config.get("auto_map"):
        raise ValueError("embedding.model_config_mismatch")
    return manifest


def download(folder, run_dir):
    import shutil

    from huggingface_hub import HfApi, hf_hub_download

    folder.mkdir(parents=True, exist_ok=True)
    metadata = HfApi(token=False).model_info(MODEL_NAME, revision=REVISION, files_metadata=True)
    if metadata.sha != REVISION:
        raise ValueError("embedding.remote_revision_mismatch")
    siblings = {item.rfilename: item for item in metadata.siblings}
    required = sum(siblings[name].size for name in ALLOWED_FILES)
    if shutil.disk_usage(folder).free < required * 2 + 1_000_000_000:
        raise ValueError("resource.insufficient_disk")
    files = {}
    for name in ALLOWED_FILES:
        trace(run_dir / "trace.jsonl", "model.download_file", "started", summary={"file": name})
        file = Path(
            hf_hub_download(MODEL_NAME, name, revision=REVISION, local_dir=folder, token=False)
        )
        hashed = digest(file)
        item = siblings[name]
        if file.stat().st_size != item.size or (item.lfs and hashed != item.lfs.sha256):
            raise ValueError("embedding.download_checksum_mismatch")
        files[name] = {
            "size": file.stat().st_size,
            "sha256": hashed,
            "hub_blob_id": item.blob_id,
            "hub_lfs_sha256": item.lfs.sha256 if item.lfs else None,
        }
        trace(
            run_dir / "trace.jsonl",
            "model.download_file",
            "completed",
            summary={"file": name, "size": file.stat().st_size, "sha256": hashed},
        )
    manifest = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "revision": REVISION,
        "license": "MIT",
        "canonical_uri": f"https://huggingface.co/{MODEL_NAME}/tree/{REVISION}",
        "files": files,
    }
    target = folder / "model-manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
        raise ValueError("embedding.model_manifest_conflict")
    if not target.exists():
        write_json(target, manifest)
    verify_model(folder)
    return {
        "status": "download_verified",
        "revision": REVISION,
        "file_count": len(files),
        "bytes": sum(item["size"] for item in files.values()),
        "manifest_sha256": digest(target),
    }


def encode(request, folder, trace_path):
    versions = runtime_versions()
    if tuple(int(part) for part in versions["torch"].split("+")[0].split(".")[:2]) < (2, 6):
        raise ValueError("embedding.unsafe_torch_version")
    if request.get("schema_version") != 1 or request.get("device") != "cpu":
        raise ValueError("embedding.worker_request_invalid")
    if request.get("dtype") not in ("float16", "float32"):
        raise ValueError("embedding.worker_request_invalid")
    texts = request.get("texts")
    if not isinstance(texts, list) or not texts or len(texts) > 1024:
        raise ValueError("embedding.worker_request_invalid")
    if any(not isinstance(text, str) or not text.strip() or len(text) > 20000 for text in texts):
        raise ValueError("embedding.worker_request_invalid")
    if any(redact(text) != text for text in texts):
        raise ValueError("embedding.unsafe_text_for_artifact")
    batch_size, max_length = request.get("batch_size"), request.get("max_length")
    if type(batch_size) is not int or not 1 <= batch_size <= 2:
        raise ValueError("embedding.worker_request_invalid")
    if type(max_length) is not int or not 16 <= max_length <= 512:
        raise ValueError("embedding.worker_request_invalid")
    verify_model(folder)
    if digest(folder / "model-manifest.json") != request["manifest_sha256"]:
        raise ValueError("embedding.model_identity_mismatch")
    if (
        digest(Path(__file__)) != request["worker_sha256"]
        or versions != request["runtime_versions"]
    ):
        raise ValueError("embedding.worker_identity_mismatch")
    import psutil
    import torch
    from transformers import AutoModel, AutoTokenizer

    if psutil.virtual_memory().available < request["required_available_ram_bytes"]:
        raise ValueError("resource.insufficient_ram")
    torch.set_num_threads(request["cpu_threads"])
    trace(
        trace_path,
        "embedding.model_load",
        "started",
        summary={"device": "cpu", "dtype": request["dtype"]},
    )
    tokenizer = AutoTokenizer.from_pretrained(
        folder, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        folder,
        local_files_only=True,
        trust_remote_code=False,
        weights_only=True,
        use_safetensors=False,
        dtype=getattr(torch, request["dtype"]),
    )
    model.eval()
    trace(
        trace_path,
        "embedding.model_load",
        "completed",
        summary={"device": str(next(model.parameters()).device), "runtime_versions": versions},
    )
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            result = model(**tokens).last_hidden_state[:, 0].float()
            normalized = torch.nn.functional.normalize(result, p=2, dim=1)
            if normalized.shape[1] != 1024 or not bool(torch.isfinite(normalized).all()):
                raise ValueError("embedding.invalid_vector")
            vectors.extend(normalized.cpu().tolist())
            trace(
                trace_path,
                "embedding.batch",
                "completed",
                summary={"count": len(normalized), "start": start},
            )
    if any(
        not math.isclose(sum(value * value for value in vector), 1.0, abs_tol=0.001)
        for vector in vectors
    ):
        raise ValueError("embedding.invalid_normalization")
    return {
        "status": "encoded",
        "vectors": vectors,
        "runtime_versions": versions,
        "manifest_sha256": request["manifest_sha256"],
        "worker_sha256": request["worker_sha256"],
        "device": "cpu",
        "dtype": request["dtype"],
        "count": len(vectors),
        "dimension": 1024,
        "rss_bytes": psutil.Process().memory_info().rss,
        "available_ram_after_bytes": psutil.virtual_memory().available,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--download-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    os.environ.update(
        USE_TF="0",
        USE_FLAX="0",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HUB_DISABLE_XET="1",
        HF_HUB_DOWNLOAD_TIMEOUT="30",
    )
    if args.probe:
        print(json.dumps(runtime_versions()))
        return 0
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output = args.run_dir / "result.json"
    trace_path = args.run_dir / "trace.jsonl"
    started = time.monotonic()
    watchdog = None
    if args.download:
        if not 1 <= args.download_timeout_seconds <= 1800:
            parser.error("download timeout must be between 1 and 1800 seconds")

        def timeout_download():
            result = {
                "status": "failed",
                "error_code": "embedding.download_timeout",
                "duration_ms": round((time.monotonic() - started) * 1000),
            }
            write_json(output, result)
            trace(
                trace_path,
                "worker.failed",
                "failed",
                error_code=result["error_code"],
                summary=result,
            )
            os._exit(124)

        watchdog = threading.Timer(args.download_timeout_seconds, timeout_download)
        watchdog.daemon = True
        watchdog.start()
    try:
        result = (
            download(args.model_dir, args.run_dir)
            if args.download
            else encode(
                json.loads(args.input.read_text(encoding="utf-8")), args.model_dir, trace_path
            )
        )
        result["duration_ms"] = round((time.monotonic() - started) * 1000)
        write_json(output, result)
        trace(
            trace_path,
            "worker.completed",
            "completed",
            summary={k: v for k, v in result.items() if k != "vectors"},
        )
        print(json.dumps({k: v for k, v in result.items() if k != "vectors"}))
        return 0
    except Exception as error:
        code = (
            str(error)
            if isinstance(error, ValueError) and str(error).startswith(("embedding.", "resource."))
            else "embedding.worker_failed"
        )
        result = {
            "status": "failed",
            "error_code": code,
            "exception_type": type(error).__name__,
            "duration_ms": round((time.monotonic() - started) * 1000),
        }
        write_json(output, result)
        trace(trace_path, "worker.failed", "failed", error_code=code, summary=result)
        print(json.dumps(result))
        return 1
    finally:
        if watchdog is not None:
            watchdog.cancel()


if __name__ == "__main__":
    sys.exit(main())
