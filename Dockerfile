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
    wget && \
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

FROM base
ARG GTBENCH_BACKEND=cpu_ifirst
ARG GTBENCH_RUNTIME=ghex_comm
COPY . /gtbench
RUN cd /gtbench && \
    mkdir -p build && \
    cd build && \
    if [ -d /opt/rocm ]; then export CXX=/opt/rocm/bin/hipcc; fi && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTBENCH_BACKEND=${GTBENCH_BACKEND} \
    -DGTBENCH_RUNTIME=${GTBENCH_RUNTIME} \
    .. && \
    make -j $(nproc) && \
    cp benchmark convergence_tests /usr/bin/ && \
    rm -rf /gtbench/build

CMD ["/usr/bin/convergence_tests"]
