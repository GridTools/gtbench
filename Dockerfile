ARG BASE=base_cpu
FROM fthaler/gtbench:base_cpu
LABEL maintainer="Felix Thaler <thaler@cscs.ch>"

COPY . /gtbench
ARG GTBENCH_BACKEND=mc
ARG GTBENCH_COMMUNICATION_BACKEND=ghex_comm
RUN cd /gtbench && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGTBENCH_BACKEND=${GTBENCH_BACKEND} \
    -DGTBENCH_COMMUNICATION_BACKEND=${GTBENCH_COMMUNICATION_BACKEND} \
    .. && \
    make -j $(nproc) && \
    cp benchmark convergence_tests /usr/bin/ && \
    rm -rf /gtbench/build

CMD ["/usr/bin/convergence_tests"]