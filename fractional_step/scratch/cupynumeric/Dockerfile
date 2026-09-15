# cuPyNumeric on top of the SAME image the PyTorch numbers were measured in, so
# the comparison holds machine, driver and CUDA major version fixed.
#
# A separate python is unavoidable: cuPyNumeric ships only cuda13_py313 / py314
# conda builds, and the base image is py312.  micromamba puts that env at
# /opt/cpn without touching the base interpreter, so the torch benchmark still
# runs unchanged in the same container.
#
# TWO PINS, BOTH LOAD-BEARING.  The `legate` channel carries `_cpu_` and `_gpu_`
# builds of the same version.  A docker build has no GPU visible, so conda never
# sets the `__cuda` virtual package and silently resolves to `_cpu_`.  The result
# imports fine, reports a GPU in `rt.machine`, and runs every kernel on the host:
#
#   RuntimeError: Legate was run with GPUs but was not built with GPU support
#
# -- and that error only appears because LEGATE_CONFIG asked for a GPU.  Without
# it the CPU build would have benchmarked happily and been reported as GPU
# numbers.  Exactly the failure mode of 3D_STATUS.md L15.
FROM chandc/unsloth-dgx-spark:latest

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-aarch64/latest \
      | tar -xj -C /usr/local bin/micromamba

RUN CONDA_OVERRIDE_CUDA=13.0 /usr/local/bin/micromamba create -y -p /opt/cpn \
      -c legate -c conda-forge \
      "cupynumeric=*=cuda13_py313_gpu*" \
 && /usr/local/bin/micromamba clean -ay

ENV PATH=/opt/cpn/bin:$PATH
