#/usr/bin/env bash

set -e

BOOST_ROOT=$(spack location -i boost+thread)
UCX_DIR=$(spack location -i ucx)
XPMEM_DIR=/opt/cray/xpmem/default
LIBFABRIC_DIR=/opt/cray/libfabric/1.15.2.0

for backend in cpu_ifirst cpu_kfirst gpu; do
    CXX_CMAKE_ARCHITECTURE_OPTIONS=""
    if [[ $backend == "cpu_ifirst" ]] || [[ $backend == "cpu_kfirst" ]]; then
	    CMAKE_CUDA_COMPILER=""
    elif [[ $backend == "gpu" ]]; then
	    CMAKE_CUDA_COMPILER=$(which nvcc)
    fi
    for gt_runtime in single_node simple_mpi gcl ghex_comm; do
        if [[ $gt_runtime == "ghex_comm" ]]; then
            GHEX_CMAKE_OPTIONS="-DGHEX_USE_BUNDLED_LIBS=ON -DGHEX_USE_BUNDLED_GRIDTOOLS=OFF"
            for ghex_backend in MPI UCX LIBFABRIC; do
                for xpmem in ON OFF; do
                    if [[ $xpmem == "ON" ]]; then
                        XPMEM_CMAKE_FLAGS="-DGHEX_USE_XPMEM=ON -DXPMEM_DIR=${XPMEM_DIR}"
                    else
                        XPMEM_CMAKE_FLAGS=""
                    fi
                    if [[ $ghex_backend == "LIBFABRIC" ]]; then
			            GHEX_CMAKE_OPTIONS="$GHEX_CMAKE_OPTIONS -DLIBFABRIC_DIR=${LIBFABRIC_DIR}"
                    fi
                    BUILD_DIR=build_${backend}_${gt_runtime}_${ghex_backend}
                    if [[ $xpmem == "ON" ]]; then
                        BUILD_DIR="${BUILD_DIR}_xpmem"
                    fi
                    mkdir -p $BUILD_DIR
                    pushd $BUILD_DIR
                    set -x
                    cmake .. -DGTBENCH_BACKEND=$backend -DGTBENCH_RUNTIME=$gt_runtime -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -msve-vector-bits=128 -fopt-info-vec-missed -fvect-cost-model=unlimited" -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER} -DGHEX_TRANSPORT_BACKEND=$ghex_backend ${GHEX_CMAKE_OPTIONS} ${XPMEM_CMAKE_FLAGS}
                    set +x
                    cmake --build . --target install --parallel 1 --verbose 2>&1 | tee build_output.txt
                    popd
                done
            done
        else
            BUILD_DIR=build_${backend}_${gt_runtime}
	        mkdir -p $BUILD_DIR
            pushd $BUILD_DIR
            set -x
            cmake .. -DGTBENCH_BACKEND=$backend -DGTBENCH_RUNTIME=$gt_runtime -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_CXX_COMPILER=$(which g++) -DCMAKE_BUILD_TYPE=Custom -DCMAKE_CXX_FLAGS="-g -mcpu=neoverse-v2 -Ofast -msve-vector-bits=128 -fopt-info-vec-missed -fvect-cost-model=unlimited" -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            set +x
            cmake --build . --target install --parallel 1 --verbose 2>&1 | tee build_output.txt
            popd
        fi
    done
done

