"""
Crash-case deduplication filter.

Goal
----
Avoid saving *duplicate* crash cases into `bug_cases/` when the same backend crash repeats.
We only deduplicate **crash-type** cases (compile/runtime failures), NOT numerical diffs like
`Differential mismatch: ...`.

How it works
------------
- Decide whether an `error.log` looks like a crash via `is_crash_log()`.
- If it's a crash, compute a signature:
  - For some known high-volume crash classes, we use a **rule-based signature** that ignores
    varying shape numbers (e.g. `T.int64(1)` vs `T.int64(55)`), so they dedup well.
  - Otherwise, we hash a normalized error.log string.
- Keep only the first occurrence; later duplicates are dropped.
- Signatures are stored as JSONL in `bug_cases/.seen_crash_signatures.jsonl`.

How to add a new crash type to the filter
----------------------------------------
If you find a new crash pattern you want to dedup (e.g., a new backend error message),
add a regex to `CRASH_HINT_PATTERNS` below. Example:

    CRASH_HINT_PATTERNS.append(r"MyBackendError: .*something.*")

Then re-run; duplicates of that crash will be suppressed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional


DEFAULT_INDEX_NAME = ".seen_crash_signatures.jsonl"
ENV_ENABLE_DEDUP = "HIGEN_DEDUP_CRASH"  # "1" (default) or "0"

# If an error.log matches ANY of these patterns, we treat it as a crash-type case eligible for dedup.
# Keep patterns reasonably specific so we don't accidentally dedup value mismatches.
CRASH_HINT_PATTERNS: list[str] = [
    r"^Differential test failed:",
    r"Traceback \(most recent call last\)",
    r"\[ONNXRuntimeError\]",
    r"onnxruntime\.capi\..*NotImplemented",
    r"\bNOT_IMPLEMENTED\b",
    # Shape/rank constraints (often thrown as runtime/compile errors by backends)
    r"Concat expects the input tensors to have the same shape",
    r"Concat expects all input tensors to have same ndim",
    # Backend feature not implemented
    r"Pad mode.*not implemented",
    # TVM/LLVM crash signature seen in this repo
    r"LLVM module verification failed",
    r"\binference exceeded\b.*\btimeout\b",
]
_CRASH_HINT_RE = re.compile("|".join(f"(?:{p})" for p in CRASH_HINT_PATTERNS), re.IGNORECASE | re.MULTILINE)

# Known high-volume crash classes: produce a stable signature that ignores varying details.
# If a log matches one of these rules, we dedup by the rule id (not by the full text).
_SIG_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Concat expects the input tensors to have the same shape", re.IGNORECASE), "tvm_concat_shape_mismatch"),
    (re.compile(r"Concat expects all input tensors to have same ndim", re.IGNORECASE), "tvm_concat_ndim_mismatch"),
    (re.compile(r"Pad mode.*not implemented", re.IGNORECASE), "tvm_pad_mode_not_implemented"),
    (re.compile(r"LLVM module verification failed", re.IGNORECASE), "tvm_llvm_module_verification_failed"),
    (re.compile(r"\bNOT_IMPLEMENTED\b", re.IGNORECASE), "backend_not_implemented"),
]

_RE_FILE_LINE = re.compile(r'File "([^"]+)", line (\d+)')
_RE_BIG_SMALL = re.compile(r"(big_)\d+(_small_)\d+")
_RE_NUM_SUFFIX = re.compile(r"([A-Za-z]+)_(\d+)")
_RE_HEX = re.compile(r"0x[0-9a-fA-F]+")


def dedup_enabled() -> bool:
    v = os.environ.get(ENV_ENABLE_DEDUP, "1").strip().lower()
    return v not in {"0", "false", "no", "off"}


def is_crash_log(error_log: str) -> bool:
    """
    Return True if this error log should be treated as a crash-type case.

    NOTE: numerical diffs like `Differential mismatch: ...` are intentionally NOT matched.
    """
    if not error_log:
        return False
    return _CRASH_HINT_RE.search(error_log) is not None


def normalize_error_log_for_signature(msg: str) -> str:
    """
    Best-effort normalization for crash signature:
    - remove absolute file paths/line numbers in tracebacks
    - normalize big_<B>_small_<S> run names
    - normalize auto-generated suffixes like Constant_1234
    """
    s = msg or ""
    s = s.replace("\r\n", "\n")
    s = _RE_FILE_LINE.sub('File "<PATH>", line <N>', s)
    s = _RE_BIG_SMALL.sub(r"\1<B>\2<S>", s)
    s = _RE_NUM_SUFFIX.sub(r"\1_<N>", s)
    s = _RE_HEX.sub("0x<HEX>", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def crash_signature(error_log: str) -> str:
    # First, apply rule-based signatures for known high-volume classes.
    for cre, sid in _SIG_RULES:
        if cre.search(error_log or "") is not None:
            return f"rule:{sid}"
    norm = normalize_error_log_for_signature(error_log)
    if len(norm) > 20000:
        norm = norm[:20000]
    return hashlib.sha1(norm.encode("utf-8", errors="replace")).hexdigest()


def _seen_signature(index_path: Path, sig: str) -> bool:
    if not index_path.exists():
        return False
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("sig") == sig:
                    return True
    except Exception:
        # Fail open: if index is unreadable, don't dedup.
        return False
    return False


def _append_signature(index_path: Path, sig: str, case_name: str, error_log: str) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sig": sig,
        "case": case_name,
        "ts": time.time(),
        "head": (error_log.splitlines()[0] if error_log else "")[:400],
    }

    # Best-effort inter-process safety on Linux.
    try:
        import fcntl  # type: ignore

        with index_path.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
    except Exception:
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def should_keep_crash_case(
    bug_root: Path,
    *,
    case_name: str,
    error_log: Optional[str],
    index_name: str = DEFAULT_INDEX_NAME,
) -> bool:
    """
    Return True if the case should be kept (saved), False if it is a duplicate crash.

    - Only active when `dedup_enabled()` and `is_crash_log(error_log)` are True.
    - If kept, we record the signature in `bug_root/index_name`.
    """
    if not dedup_enabled():
        return True
    if not error_log:
        return True
    if not is_crash_log(error_log):
        return True

    sig = crash_signature(error_log)
    index_path = bug_root / index_name
    if _seen_signature(index_path, sig):
        return False
    _append_signature(index_path, sig, case_name, error_log)
    return True


