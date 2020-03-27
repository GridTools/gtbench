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

ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake

ARG BOOST_VERSION=1.67.0
RUN export BOOST_VERSION_UNERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
    wget -q https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    tar xzf boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    cp -r boost_${BOOST_VERSION_UNERLINE}/boost /usr/local/include/ && \
    rm -rf boost_${BOOST_VERSION_UNERLINE}*

ARG MPICH_VERSION=3.1.4
RUN wget -q http://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz && \
    tar xf mpich-${MPICH_VERSION}.tar.gz && \
    cd mpich-${MPICH_VERSION} && \
    ./configure --disable-fortran --enable-fast=all,O3 --prefix=/usr && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf mpich-${MPICH_VERSION}*

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
