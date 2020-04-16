ARG BUILD_FROM=ubuntu:19.10
FROM ${BUILD_FROM} as base
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
    ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr/local/ && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf mpich-3.1.4*

ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake

RUN git clone -b release_v1.1 https://github.com/GridTools/gridtools.git && \
    mkdir -p gridtools/build && \
    cd gridtools/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    .. && \
    make -j $(nproc) install && \
    cd ../.. && \
    rm -rf gridtools

RUN git clone https://github.com/GridTools/GHEX.git && \
    mkdir -p GHEX/build && \
    cd GHEX/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    .. && \
    make -j $(nproc) install && \
    cd ../.. && \
    rm -rf GHEX

FROM base
ARG GTBENCH_BACKEND=mc
ARG GTBENCH_RUNTIME=ghex_comm
COPY . /gtbench
RUN cd /gtbench && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTBENCH_BACKEND=${GTBENCH_BACKEND} \
    -DGTBENCH_RUNTIME=${GTBENCH_RUNTIME} \
    .. && \
    make -j $(nproc) && \
    cp benchmark convergence_tests /usr/bin/ && \
    rm -rf /gtbench/build

CMD ["/usr/bin/convergence_tests"]