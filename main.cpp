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

#include "./activations.hpp"
#include "./cxxopts.hpp"
#include "./Graph.hpp"
#include "./io_helpers.hpp"
#include "./mnist_parser.h"
#include "./optimization.hpp"

#include "./common/gettime.h"

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::vector;
using std::string;

#include "./networks/util.cpp"
#include "./networks/mlp.cpp"
#include "./networks/gcn.cpp"
#include "./networks/cnn.cpp"
#include "./networks/lstm.cpp"

int main(int argc, char** argv) {
  int alg = -1;
  cxxopts::Options options(argv[0], "Options");
  options.add_options()("a,algorithm", "Algorithm 0,1", cxxopts::value<int>(alg));
  options.parse(argc, argv);
  std::vector<int> layer_sizes;
  switch(alg) {
    case 0:
      // (mlp1) MLP single layer
      layer_sizes.push_back(800);
      learn_mnist(layer_sizes);
      break;
    case 1:
      // (mlp2) MLP two layers
      layer_sizes.push_back(400);
      layer_sizes.push_back(100);
      learn_mnist(layer_sizes);
      break;
    case 2:
      // (gcn1) GCN Pubmed
      learn_gcn_pubmed();
      break;
    case 3:
      // (gcn2) GCN email-Eu-core
      learn_gcn();
      break;
    case 4:
      // (cnn1) CNN modernized version of lenet-5 with maxpool and ReLU
      learn_mnist_lenet5();
      break;
    case 5:
      // (cnn2) CNN original version of lenet-5 with average pool and tanh
      learn_mnist_lenet5_tanh();
      break;
    case 6:
      // LSTM without added parallelism
      learn_lstm(false);
      break;
    case 7:
      // LSTM with added parallelism
      learn_lstm(true);
      break;
    default:
      std::cout << "No algorithm specified. Usage: " 
                << argv[0] << " -a <alg_num> (0-7)" << std::endl;
  }
  return 0;
}
