ARG BUILD_FROM=ubuntu:19.10
FROM ${BUILD_FROM}
LABEL maintainer="Felix Thaler <thaler@cscs.ch>"

RUN apt-get update -qq && \
    apt-get install -qq -y \
    build-essential \
    wget \
    git \
    tar \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.14.5
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/bin cmake-${CMAKE_VERSION}-Linux-x86_64/share /usr/local/ && \
    rm -rf cmake-${CMAKE_VERSION}-Linux-x86_64*

ARG BOOST_VERSION=1.67.0
RUN export BOOST_VERSION_UNERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
    wget -q https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    tar xzf boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    cp -r boost_${BOOST_VERSION_UNERLINE}/boost /usr/local/include/ && \
    rm -rf boost_${BOOST_VERSION_UNERLINE}*

RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz && \
    tar xf mpich-3.1.4.tar.gz && \
    cd mpich-3.1.4 && \
    ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf mpich-3.1.4*

ARG GTBENCH_BACKEND=mc
RUN git clone -b release_v1.1 https://github.com/GridTools/gridtools.git && \
    mkdir -p gridtools/build && \
    cd gridtools/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DGT_ENABLE_BACKEND_X86=$(if [ "${GTBENCH_BACKEND}" = x86 ]; then echo ON; else echo OFF; fi) \
    -DGT_ENABLE_BACKEND_MC=$(if [ "${GTBENCH_BACKEND}" = mc ]; then echo ON; else echo OFF; fi) \
    -DGT_ENABLE_BACKEND_CUDA=$(if [ "${GTBENCH_BACKEND}" = cuda ]; then echo ON; else echo OFF; fi) \
    -DGT_ENABLE_BACKEND_NAIVE=OFF \
    .. && \
    make -j $(nproc) install && \
    cd ../.. && \
    rm -rf gridtools

ARG GTBENCH_COMMUNICATION_BACKEND=ghex_comm
RUN if [ "${GTBENCH_COMMUNICATION_BACKEND}" = ghex_comm ]; then \
    git clone https://github.com/GridTools/GHEX.git && \
    mkdir -p GHEX/build && \
    cd GHEX/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGridTools_DIR=/usr/local/lib/cmake \
    .. && \
    make -j $(nproc) install && \
    cd ../.. && \
    rm -rf GHEX; \
    fi

COPY . /gtbench
RUN cd /gtbench && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGridTools_DIR=/usr/local/lib/cmake \
    -DGHEX_DIR=/usr/local/lib/cmake \
    -DGTBENCH_BACKEND=${GTBENCH_BACKEND} \
    -DGTBENCH_COMMUNICATION_BACKEND=${GTBENCH_COMMUNICATION_BACKEND} \
    .. && \
    make -j $(nproc) && \
    cp benchmark convergence_tests /usr/bin/ && \
    rm -rf /gtbench/build

CMD ["/usr/bin/convergence_tests"]