#!/usr/bin/env python3
"""
Triage helper for HiGen  bug cases.

Goal: among already-failed cases (bug_cases/*), quickly distinguish
  - true-positive candidates (likely real backend/TVM bugs),
  - numerical discrepancies (fp16 ULP / accumulation / reordering),
  - undefined/unstable behavior (NaN/Inf + Cast-to-int/bool, domain errors),
  - infrastructure issues (output name mismatch).

This tool is intentionally heuristic: it produces a ranked list with reasons.

Usage:
  python -m HiGen.utils.triage_bug_cases \
    --cases-root /path/to/HiGen_runs_diff_tvm_ort_ov/bug_cases_1 \
    --out-json triage.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CaseTriage:
    case_dir: str
    error_type: str
    score: float
    bucket: str
    reasons: List[str] = field(default_factory=list)
    ops: List[str] = field(default_factory=list)
    dtypes: List[str] = field(default_factory=list)
    max_abs_diff: Optional[float] = None
    max_rel_diff: Optional[float] = None
    mismatch_ratio: Optional[float] = None


_RE_MAX_ABS = re.compile(r"Max absolute difference:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_RE_MAX_REL = re.compile(r"Max relative difference:\s*([+-]?\w+(?:\.\w+)?(?:[eE][+-]?\d+)?)")
_RE_MISM = re.compile(r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)")

_CRASH_MARKERS = (
    "Traceback (most recent call last):",
    "Segmentation fault",
    "SIGSEGV",
    "SIGABRT",
    "Aborted",
    "Floating point exception",
    "core dumped",
    "TVMError",
    "InternalError",
    "Check failed:",
    "AssertionError",
    "onnxruntime.capi.onnxruntime_pybind11_state.Fail",
    "RuntimeError:",
)

# Examples:
#   Differential test failed: <message>
#   Differential test failed (backend=tvm): <message>
_RE_DIFF_TEST_FAILED = re.compile(r"Differential test failed\s*(?:\(backend=([^)]+)\))?\s*:\s*(.*)")


def _safe_float(x: str) -> Optional[float]:
    try:
        if x.lower() in {"inf", "+inf"}:
            return float("inf")
        if x.lower() == "-inf":
            return float("-inf")
        if x.lower() == "nan":
            return float("nan")
        return float(x)
    except Exception:
        return None


def _parse_error_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"raw": "", "kind": "missing_error_log"}
    raw = log_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {"raw": raw, "kind": "unknown"}
    # Differential test infra failure (often indicates a backend crash or compilation failure).
    # Example:
    #   Differential test failed (backend=tvm): LLVM module verification failed ...
    m_failed = _RE_DIFF_TEST_FAILED.search(raw)
    if m_failed:
        out["kind"] = "diff_test_failed"
        backend = m_failed.group(1)
        msg = m_failed.group(2)
        if backend:
            out["backend"] = backend
        if msg:
            out["failure_msg"] = msg.strip()
        return out
    # Crash-like logs (tracebacks, signals). These are prioritized over diff parsing.
    if any(m in raw for m in _CRASH_MARKERS):
        out["kind"] = "crash"
        return out
    if "output names mismatch" in raw:
        out["kind"] = "output_name_mismatch"
        return out
    if "nan location mismatch" in raw:
        out["kind"] = "nan_location_mismatch"
    if "+inf location mismatch" in raw or "-inf location mismatch" in raw or "inf location mismatch" in raw:
        out["kind"] = "inf_location_mismatch"

    m = _RE_MAX_ABS.search(raw)
    if m:
        out["max_abs_diff"] = _safe_float(m.group(1))
    m = _RE_MAX_REL.search(raw)
    if m:
        out["max_rel_diff"] = _safe_float(m.group(1))
    m = _RE_MISM.search(raw)
    if m:
        mism = int(m.group(1))
        total = int(m.group(2))
        out["mismatched"] = mism
        out["total"] = total
        out["mismatch_ratio"] = (mism / total) if total else None
    return out


def _load_model_meta(model_path: Path) -> Tuple[List[str], List[str]]:
    import onnx  # local dependency in project env

    m = onnx.load(model_path.as_posix())
    ops = [n.op_type for n in m.graph.node]
    dtypes: List[str] = []
    for i in list(m.graph.input) + list(m.graph.output):
        tt = i.type.tensor_type
        if not tt.HasField("elem_type"):
            continue
        dtypes.append(onnx.TensorProto.DataType.Name(tt.elem_type))
    for init in m.graph.initializer:
        dtypes.append(onnx.TensorProto.DataType.Name(init.data_type))
    # unique, stable order
    ops_u = sorted(set(ops))
    dtypes_u = sorted(set(dtypes))
    return ops_u, dtypes_u


def _load_oracle_inputs(oracle_path: Path) -> Dict[str, np.ndarray]:
    import pickle

    data = pickle.loads(oracle_path.read_bytes())
    inputs = data.get("input", {}) or {}
    return {k: np.array(v) for k, v in inputs.items()}


def _float16_ulp_like(max_abs: Optional[float]) -> bool:
    """Heuristic: diff equals common fp16 ULP steps seen in this repo."""
    if max_abs is None or not np.isfinite(max_abs):
        return False
    candidates = [
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
        0.00390625,
        0.00244140625,
        0.001953125,
        0.0009765625,
    ]
    return any(abs(max_abs - c) <= 1e-12 for c in candidates)


def _has_unstable_cast_pattern(ops: Sequence[str], dtypes: Sequence[str]) -> bool:
    # crude: float -> int/bool casts are often unstable when NaN/Inf present.
    # ONNX uses Cast op; we can't see source/target types easily without deeper graph walk.
    # This flag is combined with error log NaN/Inf mismatch to decide "unstable".
    has_cast = "Cast" in ops
    has_int = any(dt.startswith("INT") or dt.startswith("UINT") for dt in dtypes)
    has_bool = "BOOL" in dtypes
    has_float = any(dt.startswith("FLOAT") or dt == "DOUBLE" for dt in dtypes)
    return has_cast and has_float and (has_int or has_bool)


def triage_case(case_dir: Path) -> CaseTriage:
    log = _parse_error_log(case_dir / "error.log")
    model_path = case_dir / "model.onnx"
    oracle_path = case_dir / "oracle.pkl"

    ops: List[str] = []
    dtypes: List[str] = []
    if model_path.exists():
        try:
            ops, dtypes = _load_model_meta(model_path)
        except Exception as e:
            ops = []
            dtypes = []
            # still triage based on log
            log.setdefault("notes", []).append(f"model_meta_error: {e}")

    max_abs = log.get("max_abs_diff")
    max_rel = log.get("max_rel_diff")
    ratio = log.get("mismatch_ratio")

    # Base score: higher means more likely true-positive candidate.
    score = 0.0
    reasons: List[str] = []
    bucket = "unknown"

    # 1) Infrastructure mismatch
    if log["kind"] == "output_name_mismatch":
        return CaseTriage(
            case_dir=str(case_dir),
            error_type="output_name_mismatch",
            score=-10.0,
            bucket="infra",
            reasons=["Backends returned different output-name sets (infra mismatch, not a numerical bug)."],
            ops=ops,
            dtypes=dtypes,
        )

    # 1.25) Differential test failed (compilation/runtime failure).
    if log["kind"] == "diff_test_failed":
        backend = log.get("backend", "unknown")
        msg = log.get("failure_msg")
        reasons = [f"Differential test failed (backend={backend}). This is a backend failure (compile/runtime), not a value mismatch."]
        if msg:
            reasons.append(f"Failure summary: {msg}")
        return CaseTriage(
            case_dir=str(case_dir),
            error_type="diff_test_failed",
            score=7.0,  # higher than 'crash' default score bump to surface these
            bucket="diff_test_failed",
            reasons=reasons,
            ops=ops,
            dtypes=dtypes,
        )

    # 1.5) Crashes are high-priority.
    if log["kind"] == "crash":
        # We still include ops/dtypes for routing. Score higher than most numeric mismatches.
        score += 6.0
        bucket = "crash"
        reasons.append("Crash/exception detected in error.log (traceback/signal). High-priority defect candidate.")

    # 2) Known TVM llvm asin/acos/atan legalization bug candidate
    # (we saw in this repo's tvm intrin_rule_llvm.cc that asin series is inaccurate)
    if any(op in ops for op in ["Acos", "Asin", "Atan"]):
        score += 5.0
        if bucket == "unknown":
            bucket = "known_trig"
        reasons.append("Model contains Acos/Asin/Atan: TVM llvm legalization currently uses an inaccurate asin series; high priority.")

    # 3) Pure fp16 numerical discrepancies
    if _float16_ulp_like(max_abs):
        # If graph is dominated by simple ops, treat as numerical.
        simple_ops = set(ops).issubset({"Add", "Mul", "Sub", "Div", "Min", "Max", "ReduceMax", "ReduceMin", "ReduceSum",
                                        "ReduceMean", "AveragePool", "Reshape", "Transpose", "Concat", "Cast", "Clip",
                                        "Squeeze", "Unsqueeze"})
        only_fp16 = ("FLOAT16" in dtypes) and not any(dt in dtypes for dt in ["FLOAT", "DOUBLE"])
        if simple_ops and only_fp16:
            score -= 3.0
            if bucket == "unknown":
                bucket = "suspected_fp16_ulp"
            reasons.append(f"max_abs_diff={max_abs} matches fp16 ULP step; graph is simple fp16 ops → likely rounding/accumulation/reorder.")
        else:
            score -= 1.0
            if bucket == "unknown":
                bucket = "likely_numeric"
            reasons.append(f"max_abs_diff={max_abs} matches common fp16 ULP step → likely numerical discrepancy.")

    # 4) NaN/Inf based mismatches
    if log["kind"] in {"nan_location_mismatch", "inf_location_mismatch"}:
        score -= 2.0
        if bucket == "unknown":
            bucket = "suspected_unstable_nan_inf"
        reasons.append("Mismatch involves NaN/Inf locations → often backend-specific NaN propagation / fast-math / domain issues.")
        if _has_unstable_cast_pattern(ops, dtypes):
            score -= 2.0
            reasons.append("Model includes Cast + int/bool dtypes → NaN/Inf-to-int/bool is implementation-dependent (unstable).")

    # 5) Int8 overflow semantics
    if "INT8" in dtypes and "Add" in ops:
        score += 1.0
        if bucket == "unknown":
            bucket = "needs_spec_int8"
        reasons.append("Model uses INT8 Add; if outputs exceed range, backends may differ (wrap vs saturate). Needs spec decision.")

    # 6) Big diffs (not ULP-scale) without NaN/Inf hints → likely true-positive candidate
    if max_abs is not None and np.isfinite(max_abs) and max_abs > 1e-2 and log["kind"] == "unknown":
        score += 2.0
        if bucket == "unknown":
            bucket = "likely_true_positive"
        reasons.append(f"Large max_abs_diff={max_abs} without NaN/Inf mismatch hints → likely real implementation difference.")

    # Try oracle input quick sanity: NaNs in inputs often indicate unstable cases
    if oracle_path.exists():
        try:
            inputs = _load_oracle_inputs(oracle_path)
            any_nan = any(np.issubdtype(v.dtype, np.floating) and np.isnan(v).any() for v in inputs.values())
            any_inf = any(np.issubdtype(v.dtype, np.floating) and np.isinf(v).any() for v in inputs.values())
            if any_nan or any_inf:
                score -= 1.0
                if bucket == "unknown":
                    bucket = "suspected_unstable_nan_inf"
                reasons.append("Oracle inputs already contain NaN/Inf → downstream comparisons often unstable.")
        except Exception:
            pass

    # Default type if no reasons
    err_type = log.get("kind", "unknown")
    if not reasons:
        reasons.append("No specific heuristic matched; inspect intermediate tensors to locate first divergence.")
    if bucket == "unknown":
        # fall back based on error type
        if err_type == "unknown":
            bucket = "unknown"
        else:
            bucket = err_type

    return CaseTriage(
        case_dir=str(case_dir),
        error_type=err_type,
        score=float(score),
        bucket=bucket,
        reasons=reasons,
        ops=ops,
        dtypes=dtypes,
        max_abs_diff=max_abs if isinstance(max_abs, (int, float)) else None,
        max_rel_diff=max_rel if isinstance(max_rel, (int, float)) else None,
        mismatch_ratio=ratio if isinstance(ratio, (int, float)) else None,
    )


def _bucket_rank(bucket: str) -> int:
    """
    Smaller rank → earlier in output.
    Aim: keep suspected false positives at the end.
    """
    order = {
        "diff_test_failed": 0,
        "crash": 0,
        "known_trig": 1,
        "likely_true_positive": 2,
        "needs_spec_int8": 3,
        "unknown": 4,
        "likely_numeric": 7,
        "suspected_fp16_ulp": 8,
        "suspected_unstable_nan_inf": 9,
        "infra": 10,
        "output_name_mismatch": 10,
    }
    return order.get(bucket, 5)


def iter_case_dirs(cases_root: Path) -> List[Path]:
    if cases_root.is_file():
        return [cases_root.parent]
    if not cases_root.exists():
        raise FileNotFoundError(cases_root)
    out: List[Path] = []
    for p in sorted(cases_root.iterdir()):
        if not p.is_dir():
            continue
        # Include any directory that has an error.log, even if model/oracle artifacts are missing.
        # This allows triage to cover compilation failures / infra failures as well.
        if (p / "error.log").exists():
            out.append(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Triage HiGen bug cases.")
    ap.add_argument("--cases-root", type=Path, required=True, help="Directory containing big_*_small_* case folders.")
    ap.add_argument("--out-json", type=Path, help="Write triage results to JSON.")
    ap.add_argument("--top", type=int, default=30, help="Print top-N candidates.")
    ap.add_argument(
        "--exclude-ops",
        type=str,
        default="",
        help="Comma-separated op types to exclude (case is ONNX op_type, e.g. Acos,Asin,Atan).",
    )
    ap.add_argument(
        "--exclude-error-types",
        type=str,
        default="",
        help="Comma-separated error types to exclude (e.g. output_name_mismatch,nan_location_mismatch).",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=-1e9,
        help="Only keep results with score >= min-score.",
    )
    ap.add_argument(
        "--only-crashes",
        action="store_true",
        help="Only show cases whose error_type is crash (traceback/signal).",
    )
    args = ap.parse_args(argv)

    case_dirs = iter_case_dirs(args.cases_root)
    results = [triage_case(d) for d in case_dirs]

    exclude_ops = {s.strip() for s in (args.exclude_ops or "").split(",") if s.strip()}
    exclude_errs = {s.strip() for s in (args.exclude_error_types or "").split(",") if s.strip()}

    filtered: List[CaseTriage] = []
    for r in results:
        if args.only_crashes and r.error_type != "crash":
            continue
        if r.score < args.min_score:
            continue
        if exclude_errs and r.error_type in exclude_errs:
            continue
        if exclude_ops and any(op in exclude_ops for op in (r.ops or [])):
            continue
        filtered.append(r)

    # Primary: bucket rank (true-positive candidates first, suspected false positives last).
    # Secondary: score (higher first).
    results_sorted = sorted(filtered, key=lambda r: (_bucket_rank(r.bucket), -r.score))

    print(f"[+] Triaged {len(results)} cases under {args.cases_root}")
    if exclude_ops:
        print(f"[+] Excluding ops: {sorted(exclude_ops)}")
    if exclude_errs:
        print(f"[+] Excluding error types: {sorted(exclude_errs)}")
    if args.min_score != -1e9:
        print(f"[+] min-score: {args.min_score}")
    print(f"[+] Remaining after filters: {len(results_sorted)}")
    print(f"[+] Top {min(args.top, len(results_sorted))} candidates:")
    for r in results_sorted[: args.top]:
        print(f"- score={r.score:+.2f} type={r.error_type} case={r.case_dir}")
        for reason in r.reasons[:3]:
            print(f"    - {reason}")
        if len(r.reasons) > 3:
            print("    - ...")

    if args.out_json:
        payload = [asdict(r) for r in results_sorted]
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[+] Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


