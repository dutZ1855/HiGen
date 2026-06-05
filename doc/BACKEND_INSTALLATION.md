# Backend Installation Guide

This document describes how to install or build the backend systems used by HiGen, including TVM and OpenVINO.

HiGen supports three main backend testing modes:

| Backend      | Required For                                               |
| ------------ | ---------------------------------------------------------- |
| ONNX Runtime | Default model validation and differential testing          |
| TVM          | TVM testing and TVM-related differential testing           |
| OpenVINO     | OpenVINO testing and OpenVINO-related differential testing |

ONNX Runtime is installed through `requirements.txt`. TVM and OpenVINO need to be installed or built separately according to the experiments you want to reproduce.

## Tested Backend Versions

HiGen was mainly evaluated with the following backend versions:

| Component        | Version                |
| ---------------- | ---------------------- |
| TVM              | `0.23.0dev`            |
| OpenVINO         | `2026.0.0.dev20251223` |
| Operating System | Ubuntu 22.04.4 LTS     |
| Python           | Python 3.11            |

Both TVM and OpenVINO versions used in our experiments are development/nightly-style builds. For strict reproducibility, we recommend recording the exact commit IDs of locally built backends:

```bash
git -C /path/to/tvm rev-parse HEAD
git -C /path/to/openvino rev-parse HEAD
```

## 1. TVM Installation

TVM is required when running:

```text
--compiler tvm
```

or when using TVM in differential testing:

```text
--diff-backends "ort,ov,tvm"
```

If you only run ONNX Runtime testing, TVM is not required.

### 1.1 Install TVM from pip

If coverage measurement is not required, a compatible TVM Python package can be installed directly:

```bash
pip install apache-tvm
```

Verify TVM:

```bash
python -c "import tvm; print(tvm.__version__); print(tvm.__file__)"
```

If you use a nightly/development package, please record its version:

```bash
python -c "import tvm; print('TVM version:', tvm.__version__)"
```

### 1.2 Build TVM from Source

For more reproducible experiments or for coverage measurement, building TVM from source is recommended.

Install basic dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  git \
  cmake \
  ninja-build \
  build-essential \
  python3-dev \
  python3-pip \
  llvm-dev
```

Clone TVM with submodules:

```bash
git clone --recursive https://github.com/apache/tvm.git tvm
cd tvm
```

If TVM was cloned without `--recursive`, initialize submodules manually:

```bash
git submodule update --init --recursive
```

Create a build directory:

```bash
mkdir -p build
cp cmake/config.cmake build/config.cmake
cd build
```

Edit `build/config.cmake` and enable LLVM:

```text
set(USE_LLVM ON)
```

Build TVM:

```bash
cmake .. -G Ninja
ninja -j$(nproc)
```

Configure the Python environment:

```bash
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```

Verify TVM:

```bash
python -c "import tvm; print(tvm.__version__); print(tvm.__file__)"
```

## 2. Building TVM with Coverage Instrumentation

TVM coverage measurement requires a TVM build compiled with coverage instrumentation flags.

The following example uses GCC/gcov-style coverage flags.

```bash
git clone --recursive https://github.com/apache/tvm.git tvm
cd tvm
git submodule update --init --recursive

mkdir -p build_gcov
cp cmake/config.cmake build_gcov/config.cmake
cd build_gcov
```

Edit `build_gcov/config.cmake` and enable LLVM:

```text
set(USE_LLVM ON)
```

Configure TVM with coverage flags:

```bash
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_FLAGS="--coverage -O0 -g" \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DCMAKE_SHARED_LINKER_FLAGS="--coverage"
```

Build TVM:

```bash
ninja -j$(nproc)
```

Configure the runtime environment:

```bash
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build_gcov
```

Verify that Python loads the intended TVM build:

```bash
python -c "import tvm; print(tvm.__version__); print(tvm.__file__)"
```

After running HiGen, `.gcda` files should be generated under the TVM build directory. You can check them with:

```bash
find /path/to/tvm/build_gcov -name "*.gcda" | head
```

## 3. OpenVINO Installation

OpenVINO is required when running:

```text
--compiler ov
```

or when using OpenVINO in differential testing:

```text
--diff-backends "ort,ov,tvm"
```

If you only run ONNX Runtime testing, OpenVINO is not required.

### 3.1 Install OpenVINO from pip

If coverage measurement is not required, OpenVINO can be installed from pip:

```bash
pip install openvino
```

Verify OpenVINO:

```bash
python -c "import openvino as ov; print(ov.__version__); print(ov.__file__)"
```

If you use a nightly/development package, please record its version:

```bash
python -c "import openvino as ov; print('OpenVINO version:', ov.__version__)"
```

### 3.2 Build OpenVINO from Source

For coverage measurement or source-level reproduction, build OpenVINO from source.

Clone OpenVINO:

```bash
git clone https://github.com/openvinotoolkit/openvino.git openvino
cd openvino
git submodule update --init --recursive
```

Install build dependencies:

```bash
sudo ./install_build_dependencies.sh
```

Create a build directory:

```bash
mkdir -p build
cd build
```

Configure OpenVINO with Python support:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=ON \
  -DENABLE_WHEEL=ON
```

Build OpenVINO:

```bash
cmake --build . --parallel $(nproc)
```

If a Python wheel is generated, install it:

```bash
pip install wheel/*.whl
```

Alternatively, export the built OpenVINO runtime libraries according to your build directory:

```bash
export PYTHONPATH=/path/to/openvino/bin/intel64/Release/python:/path/to/openvino/tools/ovc:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/openvino/bin/intel64/Release:$LD_LIBRARY_PATH
```

Verify OpenVINO:

```bash
python -c "import openvino as ov; print(ov.__version__); print(ov.__file__)"
```

## 4. Building OpenVINO with Coverage Instrumentation

OpenVINO coverage measurement requires a source-built OpenVINO version compiled with coverage instrumentation flags.

```bash
git clone https://github.com/openvinotoolkit/openvino.git openvino
cd openvino
git submodule update --init --recursive
sudo ./install_build_dependencies.sh

mkdir -p build_gcov
cd build_gcov
```

Configure OpenVINO with coverage flags:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_PYTHON=ON \
  -DENABLE_WHEEL=ON \
  -DCMAKE_C_FLAGS="--coverage -O0 -g" \
  -DCMAKE_CXX_FLAGS="--coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DCMAKE_SHARED_LINKER_FLAGS="--coverage"
```

Build OpenVINO:

```bash
cmake --build . --parallel $(nproc)
```

If a Python wheel is generated, install it:

```bash
pip install wheel/*.whl
```

Alternatively, export the built OpenVINO runtime libraries:

```bash
export PYTHONPATH=/path/to/openvino/bin/intel64/Debug/python:/path/to/openvino/tools/ovc:$PYTHONPATH
export LD_LIBRARY_PATH=/path/to/openvino/bin/intel64/Debug:$LD_LIBRARY_PATH
```

Verify that Python loads the intended OpenVINO build:

```bash
python -c "import openvino as ov; print(ov.__version__); print(ov.__file__)"
```

When running HiGen with OpenVINO coverage enabled, explicitly specify both the source and build directories:

```bash
python -m higen.main \
  --big-epochs 50 \
  --small-epochs 200 \
  --compiler ort \
  --diff-device cpu \
  --diff-backends "tvm,ov,ort" \
  --cov-reset --cov-every 50 \
  --ov-cov-reset --ov-cov-every 50 \
  --ov-cov-src /path/to/openvino \
  --ov-cov-build /path/to/openvino/build_gcov
```

After running HiGen, `.gcda` files should be generated under the OpenVINO build directory. You can check them with:

```bash
find /path/to/openvino/build_gcov -name "*.gcda" | head
```

## 5. Coverage Export with gcovr

HiGen supports runtime coverage collection using `gcovr`. To reproduce coverage results, TVM and OpenVINO must be compiled with coverage instrumentation flags.

Install `gcovr`:

```bash
pip install gcovr==8.6
```

Verify `gcovr`:

```bash
gcovr --version
```

After a run, users can manually export coverage with `gcovr`.

For TVM:

```bash
gcovr \
  -r /path/to/tvm \
  --object-directory /path/to/tvm/build_gcov \
  --csv tvm_coverage.csv
```

For OpenVINO:

```bash
gcovr \
  -r /path/to/openvino \
  --object-directory /path/to/openvino/build_gcov \
  --csv openvino_coverage.csv
```

## 6. Notes

* HiGen was tested with TVM `0.23.0dev` and OpenVINO `2026.0.0.dev20251223`.
* These versions are development/nightly-style builds.
* Users should record the exact commit IDs of source-built TVM/OpenVINO for strict reproducibility.
* For coverage collection, make sure the Python environment actually loads the instrumented backend build.
* For GPU execution, make sure CUDA, cuDNN, and backend GPU plugins are correctly installed.
