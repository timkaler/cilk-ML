#include <adept_source.h>
#include <adept_arrays.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

// Defined by Makefile_serial
#ifdef TFK_ADEPT_SERIAL
#include <cilk/cilk_stub.h>
#endif

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../activations.hpp"
#include "../cxxopts.hpp"
#include "../io_helpers.hpp"
#include "../optimization.hpp"

#include "../common/gettime.h"

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::vector;
using std::string;

#define GLOBAL_ITER_THRESH 5

#ifndef TFK_ADEPT_SERIAL
void tfk_init() {
  thread_local_worker_id = __cilkrts_get_worker_number();
  tfk_reducer.get_tls_references();
}
#endif

std::default_random_engine generator(17);

#define _max(a,b,c,d) max(max(a,b), max(c,d))

// Computes the sum of elements of arr in parallel.
aReal recursive_sum(aReal* arr, int start, int end) {
  if (end - start < 128) {
    aReal ret = arr[start];
    for (int i = start+1; i < end; ++i) {
      ret += arr[i];
    }
    return ret;
  }
  int size = end - start;
  aReal left = cilk_spawn recursive_sum(arr, start, start + size/2);
  aReal right = recursive_sum(arr, start + size/2, end);
  cilk_sync;
  return left + right;
}
