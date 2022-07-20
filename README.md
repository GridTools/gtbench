# GT Bench

This benchmark evaluates the performance of [GridTools](https://github.com/gridtools/gridtools), an open source C++-embedded DSL for weather and climate codes.
The benchmark implements an advection-diffusion solver using the finite difference method. The benchmark uses operator splitting for temporal integration typical of weather and climate codes, whereby the two horizontal dimensions use an explicit discretization, and the third, vertical, dimension is implicit.

## Dependencies

GT Bench depends on the GridTools library and, optionally, on the GHEX library. These libraries will
be installed automatically when building GT Bench with cmake unless specified otherwise.

Further external dependencies are listed below:
Required:
- [CMake](https://cmake.org/) (minimum version 3.18.1)
- [Boost](https://www.boost.org/) (minimum version 1.73.0)
- MPI (for example [OpenMPI](https://github.com/open-mpi/ompi))

Optional:
- [Unified Communication X (UCX)](https://github.com/openucx/ucx) (minimun version 1.8.0)
- [PMIx](https://github.com/openpmix/openpmix) (minimum version 3.1.4)
- [xpmem](https://github.com/hjelmn/xpmem) (master is recommended)

## Building

Once the external dependencies are installed, CMake can be used to configure the build as follows:
```console
$ cd /PATH/TO/GTBENCH-SOURCE
$ mkdir build && cd build
$ cmake ..
```

### Selecting the GridTools Backend

The GridTools backend specifies which hardware architecture to target.
The backend can be selected by setting the `GTBENCH_BACKEND` option when configuring with CMake:
```console
$ cmake -DGTBENCH_BACKEND=<BACKEND> ..
```
where `<BACKEND>` must be either `cpu_kfirst`, `cpu_ifirst`, or `gpu`. The `cpu_kfirst` and `cpu_ifirst` backends are two different CPU-backends of GridTools. On modern CPUs with large vector width and/or many cores, the `cpu_ifirst` backend might perform significantly better. On CPUs without vectorization or small vector width and limited parallelism, the `cpu_kfirst` backend might perform better. The `hip` backend currently supports running NVIDIA CUDA-capable GPUs and AMD HIP-capable GPUs.

### Selecting the GPU Compilation Framework

**Note** This section is only relevant for GPU targets.

For CUDA-capable GPUs, two compilation modes are available, namely compilation of GPU code with the NVIDIA NVCC compiler or alternatively with Clang. If the C++ compiler is set to Clang, the latter is preferred but can be overriden by setting the CMake variable `GT_CLANG_CUDA_MODE` to `NVCC-CUDA`.

For AMD GPUs, the CMake C++ compiler has to be set to HIPCC (using the environment variable `CXX`, as usually with CMake).

### Selecting the Runtime

The benchmark implementation brings several runtimes, implementing different scheduling and communication strategies. These can be selected using the CMake variable `GTBENCH_RUNTIME`:
```console
$ cmake -DGTBENCH_RUNTIME=<RUNTIME> ..
```
where `RUNTIME` can be `ghex_comm`, `gcl`, `simple_mpi`, `single_node`.
- The `single_node` options is useful for performing "single-node" tests to understand kernel performance.
- The `simple_mpi` implementation uses a simple MPI 2 sided communication for halo exchanges.
- The `gcl` implementation uses a optimized MPI based communication library shipped with [GridTools](https://gridtools.github.io/gridtools/latest/user_manual/user_manual.html#halo-exchanges).
- The `ghex_comm` option will use highly optimized distributed communication via the GHEX library, designed for best performance at scale.
 Additionally, this option will enable a multi-threaded version of the benchmark, where a rank may have more than one sub-domain (over-subscription), which are delegated to separate threads. **Note:** The gridtools computations use openmp threads on the CPU targets which will not be affected by this parameter.

#### Selecting the Transport Layer for GHEX

If the `ghex_comm` runtime has been selected, the underlying transport layer will be either
*UCX* or *MPI*. The behaviour can be chosen by defining the the appropriate CMake variables, see below.

To enable UCX support, pass additionally the following flags
```console
        -DGHEX_USE_UCP=ON \
        -DUCP_INCLUDE_DIR=/PATH/TO/UCX-INSTALLATION/include \
        -DUCP_LIBRARY=/PATH/TO/UCX-INSTALLATION/lib/libucp.so
```

If PMIx shall be enabled, follow the above pattern by define additionally
```console
        -DGHEX_USE_PMIx=ON \
        -DPMIX_INCLUDE_DIR=/PATH/TO/PMIX-INSTALLATION/include \
        -DPMIX_LIBRARY=/PATH/TO/PMIX-INSTALLATION/lib/libpmix.so
```

To enable xpmem support, pass additionally the following flags
```console
        -DGHEX_USE_XPMEM=ON \
        -DXPMEM_INCLUDE_DIR=/PATH/TO/XPMEM-INSTALLATION/include \
        -DXPMEM_LIBRARY=/PATH/TO/XPMEM-INSTALLATION/lib/libxpmem.so
```

## Running the Benchmark

### Benchmark

The benchmark executable requires the global horizontal domain size as a command line parameter. The simulation will then be performed on a total domain size of `NX×NY×60` grid points. To launch the benchmark use the appropriate MPI launcher (`mpirun`, `mpiexec`, `srun`, or similar):
```console
$ mpi_launcher <LAUNCHER_OPTIONS> ./benchmark --domain-size <NX> <NY>
```

Example output of a single-node benchmark run:
```
Running GTBENCH
Domain size:             100x100x60
Floating-point type:     float
GridTools backend:       cuda
Runtime:                 single_node
Median time:             0.198082s (95% confidence: 0.19754s - 0.200368s)
Columns per second:      50484.1 (95% confidence: 49908.1 - 50622.6)
```

For testing, the number of runs (and thus the run time) can be reduced as follows:
```console
$ mpi_launcher <LAUNCHER_OPTIONS> ./benchmark --domain-size <N> <NY> --runs <RUNS>
```
For example, run only once:
```console
$ mpi_launcher ./benchmark --domain-size 24000 24000 --runs 1
Running GTBENCH
Domain size:             24000x24000x60
Floating-point type:     float
GridTools backend:       cuda
Runtime:                 ghex_comm
Median time:             8.97857s
Columns per second:      6.41528e+07
```
Note that no confidence intervals are given in this case.

Note that there may be additional runtime-dependent command line options. Use `./benchmark --help` to list all available options.


### Convergence Tests

To make sure that the solver converges to the analytical solution of the advection-diffusion equation, we provide convergence tests. They might be helpful for evaluating correctness after possible code changes in the computation or runtime, or the compiler optimization level. To run them, use:
```console
$ mpi_launcher <LAUNCHER_OPTIONS> ./convergence_tests
```

The convergence tests can be run on 1, 2 or 4 MPI ranks.

Convergence orders should be close to the following theoretical numbers:

| Computation           | Spatial Order | Temporal Order |
|-----------------------|---------------|----------------|
| Horizontal Diffusion  | 6             | 1              |
| Vertical Diffusion    | 2             | 1              |
| Full Diffusion        | 2             | 1              |
| Horizontal Advection  | 5             | 1              |
| Vertical Advection    | 2             | 1              |
| Runge-Kutta Advection | 2             | 1              |

Note that the expected convergence orders of some tests do not exactly match the theoretical order. This is either due to limited numerical precision or to suboptimal range of tested spatial or temporal resolutions for a specific discretization. Deviations are expected to be larger when compiled with single-precision than with double-precision floating point numbers.
