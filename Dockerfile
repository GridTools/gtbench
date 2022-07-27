ARG BUILD_FROM=ubuntu:20.04
FROM ${BUILD_FROM} as base
LABEL maintainer="Felix Thaler <thaler@cscs.ch>"

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y \
    build-essential \
    file \
    git \
    libmpich-dev \
    tar \
    software-properties-common \
    wget \
    libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.18.4
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/bin cmake-${CMAKE_VERSION}-Linux-x86_64/share /usr/local/ && \
    rm -rf cmake-${CMAKE_VERSION}-Linux-x86_64*

ARG BOOST_VERSION=1.67.0
RUN export BOOST_VERSION_UNDERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
    wget -q https://boostorg.jfrog.io/artifactory/main/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERLINE}.tar.gz && \
    tar xzf boost_${BOOST_VERSION_UNDERLINE}.tar.gz && \
    cp -r boost_${BOOST_VERSION_UNDERLINE}/boost /usr/local/include/ && \
    rm -rf boost_${BOOST_VERSION_UNDERLINE}*

FROM base
ARG GTBENCH_BACKEND=cpu_ifirst
ARG GTBENCH_RUNTIME=ghex_comm
ARG GTBENCH_PYTHON_BINDINGS=OFF
COPY . /gtbench
RUN cd /gtbench && \
    mkdir -p build && \
    cd build && \
    if [ -d /opt/rocm ]; then export ROCM_PATH=/opt/rocm; export PATH=${ROCM_PATH}/bin:${PATH}; export CXX=${ROCM_PATH}/bin/hipcc; fi && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTBENCH_BACKEND=${GTBENCH_BACKEND} \
    -DGTBENCH_RUNTIME=${GTBENCH_RUNTIME} \
    -DGTBENCH_PYTHON_BINDINGS=${GTBENCH_PYTHON_BINDINGS} \
    .. && \
    make -j $(nproc) install && \
    rm -rf /gtbench/build
ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

CMD ["convergence_tests"]
