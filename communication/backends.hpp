#pragma once

#include "./single_node.hpp"

#ifdef GTBENCH_USE_GHEX
#include "./ghex_comm.hpp"
#endif

#ifdef GTBENCH_USE_SIMPLE_MPI
#include "./simple_mpi.hpp"
#endif
