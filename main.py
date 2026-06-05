"""
Hierarchical RL driver for compiler fuzzing.

This module is the entrypoint for the compiler fuzzing system and is
responsible for:
- parsing command line arguments
- initializing the reinforcement learning environment
- coordinating the dimension-selection and configuration-generation agents
- running the hierarchical training loop
- recording training logs and results

It implements a two-level RL strategy:
1. Outer loop (PPO): select combinations of test dimensions
2. Inner loop (SAC): optimize concrete parameters under fixed dimensions
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional

import numpy as np

def _bootstrap_repo_tvm(*, repo_root: Path) -> Dict[str, Any]:
    """
    Ensure this process prefers repo TVM (tvm/python) and the instrumented libtvm.so (tvm/build_gcov).
    Must run BEFORE importing modules that import TVM (rl_compiler_fuzz.utils.testing imports TVM at import time).
    """
    import sys

    info: Dict[str, Any] = {"repo_root": str(repo_root)}
    tvm_py = repo_root / "tvm" / "python"
    if tvm_py.exists():
        sys.path.insert(0, str(tvm_py))
        info["tvm_python_path"] = str(tvm_py)
    else:
        info["tvm_python_path"] = None

    build_gcov = repo_root / "tvm" / "build_gcov"
    info["tvm_build_gcov_dir"] = str(build_gcov) if build_gcov.exists() else None

    # Auto-pick build_gcov if user didn't set TVM_LIBRARY_PATH.
    if "TVM_LIBRARY_PATH" not in os.environ and (build_gcov / "libtvm.so").exists():
        os.environ["TVM_LIBRARY_PATH"] = str(build_gcov)
        info["tvm_library_path_auto"] = True
    else:
        info["tvm_library_path_auto"] = False

    # Also ensure dependent libs are resolved from the same directory.
    if "TVM_LIBRARY_PATH" in os.environ:
        ld0 = os.environ.get("LD_LIBRARY_PATH", "")
        prefix = str(os.environ["TVM_LIBRARY_PATH"])
        if prefix and (prefix not in ld0.split(os.pathsep)):
            os.environ["LD_LIBRARY_PATH"] = prefix + (os.pathsep + ld0 if ld0 else "")

    info["tvm_library_path_effective"] = os.environ.get("TVM_LIBRARY_PATH")
    info["ld_library_path_effective"] = os.environ.get("LD_LIBRARY_PATH")
    return info


def _append_cov_csv_row(
    csv_path: Path,
    *,
    step: int,
    tvm_cov: Dict[str, Any],
    ov_cov: Optional[Dict[str, Any]] = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        csv_path.write_text(
            "step,unix_ts,"
            "tvm_branches_covered,tvm_branches_total,tvm_branch_rate,tvm_lines_covered,tvm_lines_total,tvm_line_rate,"
            "ov_branches_covered,ov_branches_total,ov_branch_rate,ov_lines_covered,ov_lines_total,ov_line_rate\n",
            encoding="utf-8",
        )
    oc = ov_cov or {}
    row = (
        f"{int(step)},{int(time.time())},"
        f"{int(tvm_cov.get('branch_covered', 0) or 0)},{int(tvm_cov.get('branch_total', 0) or 0)},{float(tvm_cov.get('branch_percent', 0.0) or 0.0)},"
        f"{int(tvm_cov.get('line_covered', 0) or 0)},{int(tvm_cov.get('line_total', 0) or 0)},{float(tvm_cov.get('line_percent', 0.0) or 0.0)},"
        f"{int(oc.get('branch_covered', 0) or 0)},{int(oc.get('branch_total', 0) or 0)},{float(oc.get('branch_percent', 0.0) or 0.0)},"
        f"{int(oc.get('line_covered', 0) or 0)},{int(oc.get('line_total', 0) or 0)},{float(oc.get('line_percent', 0.0) or 0.0)}\n"
    )
    with csv_path.open("a", encoding="utf-8") as fp:
        fp.write(row)


def _run_gcovr_summary(*, tvm_src: Path, tvm_build: Path, out_json: Path) -> Dict[str, Any]:
    import subprocess
    import sys
    import shutil as _shutil

    # If nothing has been written yet, return 0s (gcovr may exit non-zero).
    try:
        any_gcda = any(tvm_build.rglob("*.gcda"))
    except Exception:
        any_gcda = False
    if not any_gcda:
        return {
            "branch_covered": 0,
            "branch_total": 0,
            "branch_percent": 0.0,
            "line_covered": 0,
            "line_total": 0,
            "line_percent": 0.0,
        }

    cmd = [
        sys.executable,
        "-m",
        "gcovr",
        "-r",
        str(tvm_src),
        "--object-directory",
        str(tvm_build),
        "--branches",
        "--json-summary",
        str(out_json),
        "--exclude",
        ".*3rdparty/.*",
        "--exclude",
        ".*build.*/.*",
    ]
    # Prefer conda toolchain gcov if available (matches how TVM was built in this env).
    gcov_exe = _shutil.which("x86_64-conda-linux-gnu-gcov") or _shutil.which("gcov")
    if gcov_exe:
        cmd += ["--gcov-executable", gcov_exe]
    # Note: gcovr may exit with 64 when there is no usable coverage data yet.
    cp = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"gcovr failed (rc={cp.returncode}):\n{cp.stderr or cp.stdout}")
    j = json.loads(out_json.read_text(encoding="utf-8"))
    return {
        "branch_covered": int(j.get("branch_covered", 0) or 0),
        "branch_total": int(j.get("branch_total", 0) or 0),
        "branch_percent": float(j.get("branch_percent", 0.0) or 0.0),
        "line_covered": int(j.get("line_covered", 0) or 0),
        "line_total": int(j.get("line_total", 0) or 0),
        "line_percent": float(j.get("line_percent", 0.0) or 0.0),
    }


def _pick_gcov_executable_for_build(build_dir: Path) -> Optional[str]:
    """Pick a gcov executable that matches how the project was compiled."""
    cache = build_dir / "CMakeCache.txt"
    try:
        if cache.exists():
            txt = cache.read_text(encoding="utf-8", errors="ignore")
            for key in ("CMAKE_C_COMPILER:FILEPATH=", "CMAKE_C_COMPILER:STRING="):
                for line in txt.splitlines():
                    if line.startswith(key):
                        cc = line.split("=", 1)[1].strip()
                        # If compiled with system gcc, use system gcov (commonly gcov-11 on Ubuntu 22.04).
                        if cc.startswith("/usr/bin/"):
                            for cand in ("/usr/bin/gcov", "/usr/bin/gcov-11", "/usr/bin/gcov-12", "/usr/bin/gcov-9"):
                                p = Path(cand)
                                if p.exists():
                                    return str(p)
                        break
    except Exception:
        pass
    # Fallback: prefer conda toolchain gcov if available.
    import shutil as _shutil

    return _shutil.which("x86_64-conda-linux-gnu-gcov") or _shutil.which("gcov")


def _run_gcovr_summary_generic(
    *,
    src_root: Path,
    obj_dir: Path,
    out_json: Path,
    gcov_exe: Optional[str] = None,
) -> Dict[str, Any]:
    import subprocess
    import sys
    import shutil as _shutil

    # If nothing has been written yet, return 0s (gcovr may exit non-zero).
    try:
        any_gcda = any(obj_dir.rglob("*.gcda"))
    except Exception:
        any_gcda = False
    if not any_gcda:
        return {
            "branch_covered": 0,
            "branch_total": 0,
            "branch_percent": 0.0,
            "line_covered": 0,
            "line_total": 0,
            "line_percent": 0.0,
        }

    cmd = [
        sys.executable,
        "-m",
        "gcovr",
        "-r",
        str(src_root),
        "--object-directory",
        str(obj_dir),
        "--branches",
        "--json-summary",
        str(out_json),
        "--exclude-directories",
        ".*/CMakeFiles/[^/]+/CompilerId.*",
        "--exclude-directories",
        ".*/CMakeFiles/CMakeTmp.*",
        "--exclude-directories",
        ".*temp/.*",
        "--gcov-ignore-errors",
        "source_not_found",
        "--gcov-ignore-errors",
        "no_working_dir_found",
        "--gcov-ignore-parse-errors",
        "negative_hits.warn_once_per_file",
        "--exclude",
        ".*3rdparty/.*",
    ]
    gcov = gcov_exe or (_shutil.which("x86_64-conda-linux-gnu-gcov") or _shutil.which("gcov"))
    if gcov:
        cmd += ["--gcov-executable", gcov]
    # IMPORTANT: run from src_root so relative paths in gcov output can be resolved.
    cp = subprocess.run(
        cmd,
        check=False,
        cwd=str(src_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"gcovr failed (rc={cp.returncode}):\n{cp.stderr or cp.stdout}")
    j = json.loads(out_json.read_text(encoding="utf-8"))
    return {
        "branch_covered": int(j.get("branch_covered", 0) or 0),
        "branch_total": int(j.get("branch_total", 0) or 0),
        "branch_percent": float(j.get("branch_percent", 0.0) or 0.0),
        "line_covered": int(j.get("line_covered", 0) or 0),
        "line_total": int(j.get("line_total", 0) or 0),
        "line_percent": float(j.get("line_percent", 0.0) or 0.0),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Defines and parses the program CLI options, including training epochs,
    output directories and compiler backend selections.

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Hierarchical RL driver for nnsmith.")
    parser.add_argument("--big-epochs", type=int, default=5, help="Number of PPO epochs.")
    parser.add_argument("--small-epochs", type=int, default=4, help="Number of SAC epochs per PPO epoch.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Override the directory where intermediate nnsmith outputs are stored.",
    )
    parser.add_argument(
        "--no-session-subdir",
        action="store_true",
        help="By default, each invocation writes into a fresh session subdir under the run root "
        "(e.g. rl_runs_diff_x/3/) so previous training.log is preserved. "
        "Pass this flag to write directly into the run root (legacy behavior).",
    )
    parser.add_argument(
        "--compiler",
        choices=["ort", "tvm", "ov"],
        default="ort",
        help="Which deep learning compiler backend to validate (affects run root and checks).",
    )
    parser.add_argument(
        "--tvm-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for TVM execution when compiler=tvm (overrides config default).",
    )
    parser.add_argument(
        "--diff-backends",
        type=str,
        default="",
        help="Comma-separated backend list for differential testing using only oracle inputs (e.g., 'ort,ov,tvm'). "
        "If provided, oracle outputs will be ignored and outputs of these backends will be compared.",
    )
    parser.add_argument(
        "--diff-rtol",
        type=float,
        default=None,
        help="Relative tolerance for differential backend comparison.",
    )
    parser.add_argument(
        "--diff-atol",
        type=float,
        default=None,
        help="Absolute tolerance for differential backend comparison.",
    )
    parser.add_argument(
        "--diff-device",
        choices=["cpu", "gpu"],
        default=None,
        help="Device for differential testing (affects ORT/TVM/OV): cpu or gpu. "
        "If omitted, defaults are chosen in code (recommended: cpu).",
    )
    parser.add_argument("--cov-every", type=int, default=0, help="Sample TVM gcovr coverage every N test cases (0 disables).")
    parser.add_argument("--cov-reset", action="store_true", help="Delete existing *.gcda under tvm/build_gcov before starting.")
    parser.add_argument("--ov-cov-every", type=int, default=0, help="Sample OpenVINO gcovr coverage every N test cases (0 disables).")
    parser.add_argument("--ov-cov-reset", action="store_true", help="Delete existing *.gcda under OpenVINO build dir before starting.")

    parser.add_argument(
    "--ov-cov-src",
    type=Path,
    default=None,
    help="OpenVINO source root for gcovr -r. Required when --ov-cov-every > 0.",)

    parser.add_argument(
    "--ov-cov-build",
    type=Path,
    default=None,
    help="OpenVINO build dir containing *.gcda files. Required when --ov-cov-every > 0.",)

    parser.add_argument("--cov-out-csv", type=Path, default=None, help="Override output CSV path. Default: <run_root>/coverage_by_steps.csv")
    return parser.parse_args()


def dump_epoch_log(log_path: Path, payload: Dict):
    """Append an epoch record to the log file.

    Writes key training information as a JSON line to `log_path`.

    Args:
        log_path: Path to the log file.
        payload: Dictionary containing data to record.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _next_session_dir(parent: Path) -> Path:
    """Pick a new numeric session directory under `parent` (1,2,3,...) and create it."""
    parent.mkdir(parents=True, exist_ok=True)
    # IMPORTANT: must be race-safe across concurrent processes.
    # We first guess a starting index, then try to create the directory atomically.
    existing: list[int] = []
    try:
        for p in parent.iterdir():
            if p.is_dir() and p.name.isdigit():
                try:
                    existing.append(int(p.name))
                except Exception:
                    pass
    except Exception:
        existing = []
    nxt = (max(existing) + 1) if existing else 1

    while True:
        session = parent / str(nxt)
        try:
            # Atomic reservation: fail if already exists.
            session.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            nxt += 1
            continue
    # Write a pointer file for convenience (avoid symlink portability issues).
    try:
        (parent / "LATEST").write_text(str(session.resolve()), encoding="utf-8")
    except Exception:
        pass
    return session


def main():
    """Main entrypoint.

    Run the full compiler fuzzing RL training flow:
    1. initialize configuration and environment
    2. create RL agents (dimension selector and config generator)
    3. run the hierarchical training loop (outer epochs + inner epochs)
    4. record logs and performance metrics
    5. save final artifacts
    """
    args = parse_args()
    # Bootstrap TVM loading BEFORE importing modules that import TVM.
    repo_root = Path(__file__).resolve().parents[1]
    bootstrap = _bootstrap_repo_tvm(repo_root=repo_root)

    from .agents import ConfigGeneratorAgent, DimensionSelectorAgent
    from .config import CompilerFuzzConfig
    from .env import CompilerFuzzEnv

    config = CompilerFuzzConfig()
    config.rl.big_epoch_num = args.big_epochs
    config.rl.small_epochs_per_big = args.small_epochs
    if args.diff_backends:
        config.diff_backends = [
            b.strip() for part in args.diff_backends.split(",") for b in part.split() if b.strip()
        ]
        config.diff_device = args.diff_device
        # Prefer running TVM first in diff mode (helps ensure TVM executes even if others fail).
        if any(b.lower() == "tvm" for b in config.diff_backends):
            config.diff_backends = [b for b in config.diff_backends if b.lower() == "tvm"] + [
                b for b in config.diff_backends if b.lower() != "tvm"
            ]
    if args.diff_rtol is not None:
        config.diff_rtol = args.diff_rtol
    if args.diff_atol is not None:
        config.diff_atol = args.diff_atol
    # Decide default run root: normal mode uses rl_runs_ort/rl_runs_tvm;
    # differential mode uses rl_runs_diff_<backend_list>
    if args.diff_backends:
        suffix = "_".join(config.diff_backends)
        default_run_root = config.paths.repo_root / f"rl_runs_diff_{suffix}"
    else:
        if args.compiler == "tvm":
            default_run_root = config.paths.repo_root / "rl_runs_tvm"
        elif args.compiler == "ov":
            default_run_root = config.paths.repo_root / "rl_runs_ov"
        else:
            default_run_root = config.paths.repo_root / "rl_runs_ort"

    # Choose run root. Default behavior: write into a fresh session subdir so previous training.log is preserved.
    base_root = args.run_root if args.run_root else default_run_root
    if args.no_session_subdir:
        config.paths.run_root = base_root
    else:
        config.paths.run_root = _next_session_dir(base_root)
    config.paths.bug_root = config.paths.run_root / "bug_cases"
    config.enable_tvm_check = args.compiler == "tvm"
    config.compiler = args.compiler  # Set compiler type for CPU/GPU/PyTorch comparison
    if args.tvm_timeout is not None:
        config.tvm_run_timeout_s = args.tvm_timeout
    config.ensure_run_root()
    print(f"[Run Root] {config.paths.run_root}")
    print(f"[TVM Bootstrap] {json.dumps(bootstrap, ensure_ascii=False)}")

    cov_every = int(args.cov_every)
    ov_cov_every = int(getattr(args, "ov_cov_every", 0))
    cov_csv = args.cov_out_csv if args.cov_out_csv else (config.paths.run_root / "coverage_by_steps.csv")
    tvm_src = repo_root / "tvm"
    tvm_build = repo_root / "tvm" / "build_gcov"
    tmp_cov_json = config.paths.run_root / ".tmp_gcovr_summary_rl.json"
    tmp_ov_cov_json = config.paths.run_root / ".tmp_gcovr_summary_openvino_rl.json"
    ov_src = args.ov_cov_src
    ov_build = args.ov_cov_build
    if bool(args.cov_reset) and tvm_build.exists():
        deleted = 0
        for p in tvm_build.rglob("*.gcda"):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
        print(f"[cov] reset: deleted_gcda={deleted} under {tvm_build}")
    if ov_cov_every > 0 or bool(getattr(args, "ov_cov_reset", False)):
        if ov_src is None or ov_build is None:
            raise ValueError(
                "OpenVINO coverage is enabled or reset is requested. "
                "Please specify both --ov-cov-src and --ov-cov-build."
            )

        if not ov_src.exists() or not ov_build.exists():
            raise FileNotFoundError(
                f"OpenVINO cov paths not found: ov_src={ov_src} ov_build={ov_build}"
            )

        if bool(getattr(args, "ov_cov_reset", False)):
            deleted = 0
            for p in ov_build.rglob("*.gcda"):
                try:
                    p.unlink()
                    deleted += 1
                except Exception:
                    pass
            print(f"[ov-cov] reset: deleted_gcda={deleted} under {ov_build}")
    if cov_every > 0 and any(b.lower() == "tvm" for b in (config.diff_backends or [])):
        # Enable isolated TVM inference to make gcda updates visible during a long-running RL process.
        os.environ.setdefault("HIGEN_TVM_ISOLATED", "1")

    env = CompilerFuzzEnv(config)
    dim_agent = DimensionSelectorAgent(config=config, obs_dim=len(env._build_observation()))
    cfg_agent = ConfigGeneratorAgent(config=config, obs_dim=len(env._build_observation()))
    obs = env.reset()

    accepted_steps = 0
    for big_epoch in range(config.rl.big_epoch_num):
        dim_action = dim_agent.select_dimensions(obs)
        dimensions = env.sample_dimension_combo(dim_action)
        print(f"[Big Epoch {big_epoch}] Selected dimensions: {dimensions}")

        small_rewards = []
        for small_epoch in range(config.rl.small_epochs_per_big):
            param_vec = cfg_agent.propose(obs)
            next_obs, reward, done, info = env.step_small_epoch(param_vec)
            cfg_agent.observe(reward, next_obs, done)
            small_rewards.append(
                {
                    "reward": reward,
                    "crashed": info["metrics"].crashed,
                    "divergence": info["metrics"].divergence,
                    "valid": info["metrics"].valid,
                    "dimensions": list(dimensions),
                    "params": info["params"],
                    # Adaptive vulnerability debug stats (prompt-based)
                    "vuln_adapt": info.get("vuln_adapt", {}),
                    # Diversity debug stats (prompt-based)
                    "diversity": info.get("diversity", {}),
                }
            )
            print(
                f"  [Small Epoch {small_epoch}] reward={reward:.2f} "
                f"crash={info['metrics'].crashed} div={info['metrics'].divergence}"
            )
            obs = next_obs
            accepted_steps += 1
            if cov_every > 0 and (accepted_steps % cov_every == 0):
                try:
                    tvm_cov = _run_gcovr_summary(tvm_src=tvm_src, tvm_build=tvm_build, out_json=tmp_cov_json)
                    ov_cov: Optional[Dict[str, Any]] = None
                    if ov_cov_every > 0 and (accepted_steps % ov_cov_every == 0):
                        ov_gcov = _pick_gcov_executable_for_build(ov_build)
                        ov_cov = _run_gcovr_summary_generic(
                            src_root=ov_src,
                            obj_dir=ov_build,
                            out_json=tmp_ov_cov_json,
                            gcov_exe=ov_gcov,
                        )
                    _append_cov_csv_row(cov_csv, step=accepted_steps, tvm_cov=tvm_cov, ov_cov=ov_cov)
                    print(
                        f"[cov] step={accepted_steps} "
                        f"tvm_lines={tvm_cov.get('line_covered')}/{tvm_cov.get('line_total')} ({tvm_cov.get('line_percent')}%) "
                        + (
                            f"ov_lines={ov_cov.get('line_covered')}/{ov_cov.get('line_total')} ({ov_cov.get('line_percent')}%)"
                            if ov_cov
                            else ""
                        )
                    )
                except Exception as e:
                    _append_cov_csv_row(
                        cov_csv,
                        step=accepted_steps,
                        tvm_cov={"branch_covered": 0, "branch_total": 0, "branch_percent": 0.0, "line_covered": 0, "line_total": 0, "line_percent": 0.0},
                        ov_cov={"branch_covered": 0, "branch_total": 0, "branch_percent": 0.0, "line_covered": 0, "line_total": 0, "line_percent": 0.0},
                    )
                    print(f"[cov] step={accepted_steps} gcovr_failed: {type(e).__name__}: {e!r}")
            if done:
                break

        obs, upper_reward, done, info = env.step_big_epoch()
        dim_agent.observe(upper_reward, obs, done)
        print(f"[Big Epoch {big_epoch}] upper reward={upper_reward:.2f}")
        dump_epoch_log(
            config.paths.run_root / "training.log",
            {
                "big_epoch": big_epoch,
                "upper_reward": upper_reward,
                "small_rewards": small_rewards,
                "upper_metrics": info["upper_metrics"].__dict__,
            },
        )
        if done:
            break


if __name__ == "__main__":
    main()

