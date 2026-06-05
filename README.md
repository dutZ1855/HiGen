# HiGen

> A hierarchical generation framework for deep learning compiler testing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)
![Backend](https://img.shields.io/badge/Backends-ORT%20%7C%20TVM%20%7C%20OpenVINO-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

HiGen is a hierarchical generation framework for deep learning compiler testing. It adopts a two-level architecture consisting of **dimension selection** and **configuration generation**, and leverages NNSmith to generate ONNX models for compiler fuzzing and differential testing.

The current prototype implements the core execution workflow, reward calculation, backend validation, bug case collection, coverage measurement, crash deduplication, and triage utilities. It is designed to support subsequent extension and replacement of reinforcement learning policies.

## Table of Contents

* [Overview](#overview)
* [External Dependencies](#external-dependencies)
* [Environment Setup](#environment-setup)
* [Quick Start](#quick-start)
* [Project Structure](#project-structure)
* [Testing Modes](#testing-modes)
* [Coverage Measurement](#coverage-measurement)
* [Output Organization](#output-organization)
* [Main Arguments](#main-arguments)
* [Environment Variables](#environment-variables)
* [Bug Case Collection](#bug-case-collection)
* [Crash Deduplication](#crash-deduplication)
* [Bug Triage](#bug-triage)
* [Version Control Notes](#version-control-notes)
* [Notes](#notes)

## Overview

HiGen follows a hierarchical testing workflow:

```text
Dimension Selection  ->  Configuration Generation  ->  ONNX Model Generation
        |                         |                          |
        v                         v                          v
   High-level Policy       Low-level Policy              NNSmith
        |                         |                          |
        +-------------------------+--------------------------+
                                  |
                                  v
                 Backend Execution and Differential Testing
                                  |
                                  v
                         Reward Feedback
```

The testing process consists of the following steps:

1. The high-level policy selects promising configuration dimensions.
2. The low-level policy generates concrete parameter configurations.
3. NNSmith generates ONNX models according to the selected configuration.
4. Multiple deep learning compiler/runtime backends execute the generated models.
5. Execution results are compared to detect crashes, inconsistencies, and differential failures.
6. Feedback signals are converted into rewards and used to guide subsequent exploration.

Supported backends:

| Backend      | Usage                                             |
| ------------ | ------------------------------------------------- |
| ONNX Runtime | Model validation and differential testing         |
| TVM          | Compiler testing and coverage measurement         |
| OpenVINO     | Compiler/runtime testing and coverage measurement |
| PyTorch      | Oracle/reference backend in normal testing mode   |

## External Dependencies

This repository does **not** include the source code or instrumented builds of NNSmith, TVM, or OpenVINO. Users need to prepare these components separately according to their own system environment.

| Component             | Required For                                       | Provided by This Repository |
| --------------------- | -------------------------------------------------- | --------------------------- |
| NNSmith               | ONNX model generation                              | No                          |
| TVM                   | TVM testing and TVM differential testing           | No                          |
| OpenVINO              | OpenVINO testing and OpenVINO differential testing | No                          |
| Instrumented TVM      | TVM coverage measurement                           | No                          |
| Instrumented OpenVINO | OpenVINO coverage measurement                      | No                          |

### NNSmith

HiGen relies on NNSmith for ONNX model generation. Please install NNSmith or place a local copy under:

```text
nnsmith-main/
```

The default configuration assumes that the NNSmith directory is located at:

```text
HiGen/nnsmith-main/
```

If your NNSmith path is different, please modify the corresponding path configuration in `config.py`.

### TVM

TVM is required when running TVM-based testing or TVM-related differential testing.

For normal TVM testing, please make sure TVM is correctly installed and can be imported in the current Python environment.

For TVM coverage measurement, an instrumented TVM build is required. The repository does not provide this build. Users should build TVM with coverage instrumentation by themselves and configure the corresponding library path, for example:

```bash
export TVM_LIBRARY_PATH=/path/to/tvm/build_gcov
```

### OpenVINO

OpenVINO is required when running OpenVINO-based testing or OpenVINO-related differential testing.

For normal OpenVINO testing, please make sure OpenVINO is correctly installed and can be imported in the current Python environment.

For OpenVINO coverage measurement, a source-built OpenVINO version with coverage instrumentation is required. The repository does not provide this build. Users must explicitly provide the OpenVINO source and build directories when enabling coverage measurement:

```text
--ov-cov-src /path/to/openvino/source
--ov-cov-build /path/to/openvino/build_gcov
```

## Environment Setup

This project provides `environment.yml` and `requirements.txt` for environment reproduction.

```bash
git clone https://github.com/dutZ1855/HiGen.git
cd HiGen

conda env create -f environment.yml
conda activate higen
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import nnsmith; print('nnsmith installed')"
python -c "import onnxruntime; print(onnxruntime.__version__)"
python -c "import tvm; print(tvm.__version__)"
python -c "import openvino; print('openvino installed')"
```

## Quick Start

Run the following commands from the repository root.

### ONNX Runtime Testing

```bash
python -m rl_compiler_fuzz.main \
  --big-epochs 20 \
  --small-epochs 100 \
  --compiler ort
```

This mode writes results to `rl_runs_ort/` by default and compares:

```text
ORT CPU  vs  ORT GPU  vs  PyTorch
```

### TVM Testing

```bash
python -m rl_compiler_fuzz.main \
  --big-epochs 20 \
  --small-epochs 100 \
  --compiler tvm \
  --tvm-timeout 300
```

This mode writes results to `rl_runs_tvm/` by default and compares:

```text
TVM CPU  vs  TVM GPU  vs  PyTorch
```

### OpenVINO Testing

```bash
python -m rl_compiler_fuzz.main \
  --big-epochs 20 \
  --small-epochs 100 \
  --compiler ov
```

This mode writes results to `rl_runs_ov/` by default and compares:

```text
OpenVINO CPU  vs  OpenVINO GPU  vs  PyTorch
```

## Project Structure

| File / Directory            | Description                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `config.py`                 | Manages dimension pools, hyperparameters, paths, and global configurations.                                        |
| `reward.py`                 | Implements high-level and low-level reward functions, including vulnerability, validity, and diversity rewards.    |
| `utils/testing.py`          | Encapsulates the workflow from parameter configuration to NNSmith model generation and backend validation.         |
| `utils/filter.py`           | Provides crash-case deduplication based on normalized error signatures.                                            |
| `utils/triage_bug_cases.py` | Provides utilities for triaging and prioritizing reported bug cases.                                               |
| `env.py`                    | Implements `CompilerFuzzEnv`, exposes `step_small_epoch` and `step_big_epoch`, and maintains diversity statistics. |
| `agents.py`                 | Provides PPO-based dimension selection and SAC-based configuration generation agents implemented with PyTorch.     |
| `main.py`                   | Main training and testing entry point.                                                                             |
| `README.md`                 | Project documentation.                                                                                             |

## Testing Modes

HiGen provides two mutually exclusive testing modes.

### Normal Testing Mode

Normal testing mode is used when `--diff-backends` is not specified.

| Compiler Option  | Compared Backends                   | Output Directory |
| ---------------- | ----------------------------------- | ---------------- |
| `--compiler ort` | ORT CPU, ORT GPU, PyTorch           | `rl_runs_ort/`   |
| `--compiler tvm` | TVM CPU, TVM GPU, PyTorch           | `rl_runs_tvm/`   |
| `--compiler ov`  | OpenVINO CPU, OpenVINO GPU, PyTorch | `rl_runs_ov/`    |

If an inconsistency is detected, the corresponding model and execution information will be saved as a bug case.

### Differential Testing Mode

Differential testing mode is enabled when `--diff-backends` is specified.

Example:

```bash
python -m rl_compiler_fuzz.main \
  --big-epochs 20 \
  --small-epochs 100 \
  --compiler ort \
  --diff-device cpu \
  --diff-backends "ort,ov,tvm"
```

Supported backend names:

```text
ort, ov, tvm
```

The output directory will be automatically created as:

```text
rl_runs_diff_ort_ov_tvm/
```

When `--diff-backends` is specified, HiGen enters differential testing mode and will not perform the normal CPU/GPU/PyTorch comparison.

## Coverage Measurement

HiGen supports runtime coverage collection for TVM and OpenVINO using `gcovr`. Coverage statistics are periodically written to:

```text
coverage_by_steps.csv
```

This file can be used to plot coverage curves during fuzzing.

### TVM Coverage

TVM coverage requires an instrumented TVM build. The repository does not provide the instrumented TVM build. Please build TVM with coverage support by yourself and specify the corresponding build directory according to your local environment.

Example:

```bash
export TVM_LIBRARY_PATH=/path/to/tvm/build_gcov
```

### OpenVINO Coverage

OpenVINO coverage requires a source-built OpenVINO version with coverage instrumentation enabled. The repository does not provide the instrumented OpenVINO build.

Users must explicitly provide the OpenVINO source directory and build directory when enabling OpenVINO coverage:

```text
--ov-cov-src /path/to/openvino/source
--ov-cov-build /path/to/openvino/build_gcov
```

### Coverage Example

```bash
python -m rl_compiler_fuzz.main \
  --big-epochs 50 \
  --small-epochs 200 \
  --compiler ort \
  --diff-device cpu \
  --diff-backends "tvm,ov,ort" \
  --cov-reset --cov-every 50 \
  --ov-cov-reset --ov-cov-every 50 \
  --ov-cov-src /path/to/openvino/source \
  --ov-cov-build /path/to/openvino/build_gcov
```

## Output Organization

By default, each run creates a new numbered subdirectory under the corresponding output root.

Example:

```text
rl_runs_diff_ort_ov_tvm/3/
```

Each run directory contains:

| File / Directory        | Description                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| `training.log`          | Records training and reward information.                                                 |
| `bug_cases/`            | Stores generated models that trigger crashes, inconsistencies, or differential failures. |
| `coverage_by_steps.csv` | Stores coverage statistics if coverage measurement is enabled.                           |

The output root also maintains a `LATEST` file, which records the absolute path of the latest run directory.

To disable the session subdirectory mechanism and write directly to the output root, use:

```bash
--no-session-subdir
```

## Main Arguments

| Argument                  | Description                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `--big-epochs`            | Number of high-level epochs.                                                                                         |
| `--small-epochs`          | Number of low-level epochs within each high-level epoch.                                                             |
| `--compiler {ort,tvm,ov}` | Specifies the main validation backend.                                                                               |
| `--tvm-timeout`           | Overrides the TVM execution timeout in seconds.                                                                      |
| `--run-root`              | Overrides the default output directory.                                                                              |
| `--diff-backends`         | Enables differential testing and specifies backend list, such as `"ort,ov,tvm"`.                                     |
| `--diff-device`           | Specifies the device used in differential testing, such as `cpu` or `gpu`.                                           |
| `--diff-rtol`             | Relative tolerance for differential comparison.                                                                      |
| `--diff-atol`             | Absolute tolerance for differential comparison.                                                                      |
| `--cov-every`             | TVM coverage sampling interval. Use `0` to disable TVM coverage.                                                     |
| `--cov-reset`             | Resets TVM coverage data before running.                                                                             |
| `--ov-cov-every`          | OpenVINO coverage sampling interval. Use `0` to disable OpenVINO coverage.                                           |
| `--ov-cov-reset`          | Resets OpenVINO coverage data before running.                                                                        |
| `--ov-cov-src`            | OpenVINO source directory for coverage collection. Required when OpenVINO coverage is enabled or reset is requested. |
| `--ov-cov-build`          | OpenVINO build directory for coverage collection. Required when OpenVINO coverage is enabled or reset is requested.  |
| `--cov-out-csv`           | Overrides the output path of `coverage_by_steps.csv`.                                                                |
| `--no-session-subdir`     | Disables numbered session directories and writes directly to the output root.                                        |

## Environment Variables

HiGen supports the following environment variables for timeout, process isolation, and crash deduplication control.

| Variable              | Description                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| `HIGEN_ORT_TIMEOUT_S` | Timeout in seconds for ONNX Runtime inference in differential testing. Default: `60`.                        |
| `HIGEN_ORT_ISOLATED`  | Whether to run ONNX Runtime inference in an isolated process. Default: `1`. Set to `0` to disable isolation. |
| `HIGEN_OV_TIMEOUT_S`  | Timeout in seconds for OpenVINO inference in differential testing. Default: `60`.                            |
| `HIGEN_TVM_ISOLATED`  | Whether to run TVM inference in an isolated process. Default: `1`. Set to `0` to disable isolation.          |
| `HIGEN_DEDUP_CRASH`   | Whether to enable crash-case deduplication. Default: `1`. Set to `0` to disable deduplication.               |

Example:

```bash
export HIGEN_ORT_TIMEOUT_S=60
export HIGEN_ORT_ISOLATED=1
export HIGEN_OV_TIMEOUT_S=60
export HIGEN_TVM_ISOLATED=1
export HIGEN_DEDUP_CRASH=1
```

## Bug Case Collection

HiGen stores bug-triggering cases under `bug_cases/`. A typical bug case directory is named as:

```text
big_<big_epoch>_small_<small_epoch>
```

Each bug case may contain:

| File         | Description                                        |
| ------------ | -------------------------------------------------- |
| `model.onnx` | The generated ONNX model.                          |
| `oracle.pkl` | Oracle input data.                                 |
| `gir.pkl`    | NNSmith graph intermediate representation.         |
| `model.pth`  | PyTorch model if available.                        |
| `error.log`  | Error message, mismatch information, or traceback. |

Successful runs and invalid configurations are removed after validation to avoid cluttering the output directory.

For cases where the number of outputs does not match, HiGen treats them as invalid configurations instead of real bugs.

## Crash Deduplication

HiGen provides a crash-case deduplication filter to avoid repeatedly saving the same crash-type failure.

Deduplication is controlled by:

```bash
export HIGEN_DEDUP_CRASH=1
```

To disable crash deduplication:

```bash
export HIGEN_DEDUP_CRASH=0
```

The deduplication index is stored under the bug case directory as:

```text
.seen_crash_signatures.jsonl
```

This file is generated automatically and should not be committed to the repository.

## Bug Triage

Not every reported mismatch represents a true compiler bug. Some differences may come from floating-point precision, FP16 rounding, NaN/Inf propagation, or backend-specific implementation details.

HiGen provides a triage utility:

```bash
python -m rl_compiler_fuzz.utils.triage_bug_cases \
  --cases-root /path/to/bug_cases \
  --out-json triage.json \
  --top 30
```

The triage tool considers:

* Error type in `error.log`
* NaN/Inf mismatch patterns
* Maximum absolute difference
* Operators and data types in `model.onnx`
* Whether oracle inputs contain NaN or Inf
* Known high-risk operators such as `Acos`, `Asin`, and `Atan`

The output is a prioritized list of bug candidates.

### Common Triage Commands

View only crash-like cases:

```bash
python -m rl_compiler_fuzz.utils.triage_bug_cases \
  --cases-root /path/to/rl_runs_diff_tvm_ort_ov/bug_cases \
  --only-crashes \
  --top 50
```

Exclude known submitted operators:

```bash
python -m rl_compiler_fuzz.utils.triage_bug_cases \
  --cases-root /path/to/rl_runs_diff_tvm_ort_ov/bug_cases \
  --exclude-ops "Acos,Asin,Atan" \
  --top 50
```

Exclude infrastructure or unstable mismatch types:

```bash
python -m rl_compiler_fuzz.utils.triage_bug_cases \
  --cases-root /path/to/rl_runs_diff_tvm_ort_ov/bug_cases \
  --exclude-error-types "output_name_mismatch,nan_location_mismatch,inf_location_mismatch" \
  --top 50
```

## Version Control Notes

Generated artifacts are not intended to be committed to the repository. Please exclude the following files and directories through `.gitignore`:

```text
__pycache__/
*.pyc
.env
.venv/
config.local.yaml

rl_runs_*/
bug_cases/
training.log
coverage_by_steps.csv
triage*.json

*.onnx
*.pkl
*.pth

*.gcda
*.gcno
*.gcov

**/.seen_crash_signatures.jsonl
.DS_Store
```

## Notes

* The project is tested mainly on Linux.
* This repository does not include NNSmith, TVM source code, or OpenVINO source code.
* Users need to install NNSmith, TVM, and OpenVINO according to their local environment and selected testing mode.
* For GPU execution, make sure CUDA, cuDNN, and related backend libraries are correctly installed.
* For TVM and OpenVINO coverage collection, source-built instrumented versions are required.
* OpenVINO coverage requires users to explicitly provide `--ov-cov-src` and `--ov-cov-build`.
* Differential testing and normal testing are mutually exclusive.
* The generated bug cases should be triaged before being reported upstream.
* Generated artifacts such as `rl_runs_*`, `bug_cases/`, `training.log`, `coverage_by_steps.csv`, `*.onnx`, `*.pkl`, and `*.pth` should be excluded from version control.
