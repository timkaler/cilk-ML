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
#include "./io_helpers.hpp"
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

#ifndef TFK_ADEPT_SERIAL
void tfk_init() {
  thread_local_worker_id = __cilkrts_get_worker_number();
  tfk_reducer.get_tls_references();
}
#endif

int N = 500;
int LEN = 100;
int NUM_ASCII = 128;
int BATCH_SIZE = 10;
int NUM_ITER = 100;
double LR = 0.01;
int HIDDEN_FEATURES = 300; // Same number of hidden features as cell features

std::default_random_engine generator(17);
std::uniform_int_distribution<int> batch_dis(0, N-1);

// =============================================================================

// Run the LSTM for inference. Generates and returns a string.
string inference_lstm(vector<aMatrix>& weights) {
  // Start with the letter 'T'
  string out_text = "T";
  aMatrix input = aMatrix(NUM_ASCII, 1);
  for (int i = 0; i < NUM_ASCII; ++i) {
    input(i, 0) = 0.0;
  }
  input(int('T'), 0) = 1.0;

  // Initialize/zero the input state
  aMatrix cell = aMatrix(HIDDEN_FEATURES, 1);
  aMatrix hidden = aMatrix(HIDDEN_FEATURES, 1);
  for (int i = 0; i < HIDDEN_FEATURES; ++i) {
    cell(i, 0) = 0.0;
    hidden(i, 0) = 0.0;
  }
  aMatrix output = aMatrix(NUM_ASCII, 1);

  // Generate 1000 characters
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < 1000; ++i) {
    // Compute the cell, hidden, and output layers
    aMatrix temp_f = activations::sigmoid(weights[0] ** hidden + weights[1] ** input + weights[2]);
    aMatrix temp_i = activations::sigmoid(weights[3] ** hidden + weights[4] ** input + weights[5]);
    aMatrix temp_c = tanh(weights[6] ** hidden + weights[7] ** input + weights[8]);
    aMatrix temp_o = activations::sigmoid(weights[9] ** hidden + weights[10] ** input + weights[11]);
    cell = cell * temp_f + temp_i * temp_c;
    hidden = temp_o * tanh(cell);
    output = activations::softmax(weights[12] ** hidden, 1.0);

    // Generate a character using the output distribution
    double val = distribution(generator);
    int j;
    for (j = 0; j < NUM_ASCII; ++j) {
      val -= output(j, 0).value();
      if (val <= 0) break;
    }
    out_text += char(j);

    // Assign generated output character to next input
    for (int k = 0; k < NUM_ASCII; ++k) {
      input(k, 0) = 0.0;
    }
    input(j, 0) = 1.0;
  }
  return out_text;
}

// This version of compute_lstm is for a single data point, not a batch.
// This (allows?) us to do the cilk_spawn/cilk_sync parallelism properly.
vector<aMatrix> compute_lstm(vector<aMatrix>& weights, vector<Matrix>& input) {
  // Note: the hidden/cell matrices are offset to allow us to initialize the
  // first element in the vectors to a 0 matrix
  vector<aMatrix> hidden = vector<aMatrix>(input.size() + 1);
  vector<aMatrix> cell = vector<aMatrix>(input.size() + 1);
  vector<aMatrix> output = vector<aMatrix>(input.size());
  hidden[0] = aMatrix(HIDDEN_FEATURES, 1);
  for (int j = 0; j < HIDDEN_FEATURES; ++j) {
    hidden[0](j, 0) = 0.0;
  }
  cell[0] = aMatrix(HIDDEN_FEATURES, 1);
  for (int j = 0; j < HIDDEN_FEATURES; ++j) {
    cell[0](j, 0) = 0.0;
  }

  // Weights are in the following order:
  // W_f, input_f, b_f, W_i, input_i, b_i,
  // W_c, input_c, b_c, W_o, input_o, b_o, output matrix
  aMatrix temp_f, temp_i, temp_c, temp_o;
  for (int j = 0; j < input.size(); ++j) {
    temp_f = activations::sigmoid(weights[0] ** hidden[j] + weights[1] ** input[j] + weights[2]);
    temp_i = activations::sigmoid(weights[3] ** hidden[j] + weights[4] ** input[j] + weights[5]);
    temp_c = tanh(weights[6] ** hidden[j] + weights[7] ** input[j] + weights[8]);
    temp_o = activations::sigmoid(weights[9] ** hidden[j] + weights[10] ** input[j] + weights[11]);
    cell[j+1] = cell[j] * temp_f + temp_i * temp_c;
    hidden[j+1] = temp_o * tanh(cell[j+1]);
    output[j] = activations::softmax(weights[12] ** hidden[j+1], 1.0);
  }
  return output;
}

void learn_lstm() {
  // Hyperparameters
  using adept::Stack;
  Stack stack;

  // Load the Paul Graham dataset to 500 100-char datapoints, using a one-hot
  // encoding of each character
  vector<vector<Matrix>> input = parse_paul_graham(N, LEN, NUM_ASCII);

  // Randomly initialize the weights
  vector<vector<aMatrix>*> weight_hyper_list;
  vector<aMatrix> weight_list;
  // W_f, input_f, b_f: "forget gate layer"
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, HIDDEN_FEATURES));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, NUM_ASCII));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, 1));
    // W_i, input_i, b_i: "input gate layer"
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, HIDDEN_FEATURES));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, NUM_ASCII));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, 1));
  // W_c, input_c, b_c: candidate values for cell state
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, HIDDEN_FEATURES));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, NUM_ASCII));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, 1));
  // W_o, input_o, b_o: "output gate layer"
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, HIDDEN_FEATURES));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, NUM_ASCII));
  weight_list.push_back(aMatrix(HIDDEN_FEATURES, 1));
  // Weight matrix for output layer: [HIDDEN_FEATURES, NUM_ASCII]
  weight_list.push_back(aMatrix(NUM_ASCII, HIDDEN_FEATURES));
  weight_hyper_list.push_back(&weight_list);

  for (int i = 0; i < weight_list.size(); ++i) {
    float range = sqrt(6.0 / (weight_list[i].dimensions()[0] + weight_list[i].dimensions()[1]));
    std::uniform_real_distribution<double> distribution(-range, range);
    for (int j = 0; j < weight_list[i].dimensions()[0]; ++j) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; ++k) {
        weight_list[i](j, k) = distribution(generator);
      }
    }
  }

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights_zero(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);
  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  // Train the LSTM over many iterations
  for (int iter = 0; iter < NUM_ITER; ++iter) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    aMatrix batch_loss = aMatrix(BATCH_SIZE, 1);
    aReal loss;
    vector<double> accuracies(BATCH_SIZE, 0.0);

    cilk_for (int i = 0; i < BATCH_SIZE; ++i) {
      // Randomly sample input to create batch data
      vector<Matrix> sample = input[batch_dis(generator)];
      // Run the LSTM on the batch data
      vector<aMatrix> output_softmax = compute_lstm(*weight_hyper_list[0], sample);
      // Compute the loss. TODO: use a reducer here
      batch_loss(i, 0) = 0.0;
      for (int j = 0; j < LEN-1; ++j) {
        batch_loss(i, 0) += 1.0 * activations::logitCrossEntropy(output_softmax[j], sample[j+1])
                  / (1.0 * BATCH_SIZE * (LEN - 1));
      }
      // Aggregate the accuracy. TODO: use a reducer here
      
      for (int j = 0; j < LEN-1; ++j) {
        int argmax = 0;
        double argmaxvalue = output_softmax[j](0, 0).value();
        for (int k = 0; k < NUM_ASCII; ++k) {
          if (argmaxvalue < output_softmax[j](k, 0).value()) {
            argmaxvalue = output_softmax[j](k, 0).value();
            argmax = k;
          }
        }
        if (sample[j+1](argmax, 0) == 1) {
          accuracies[i] += 1.0 / BATCH_SIZE / LEN;
        }
      }
    }
    double accuracy = accumulate(accuracies.begin(), accuracies.end(), 0.0);
    loss = sum(batch_loss);

    // Compute and apply gradient update using ADAM optimizer
    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    // Apply gradient descent and update weights
    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old,
                               gradients, momentums, velocities, 1.0, LR, iter+1);

    std::cout.precision(5);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter << ", loss: " << loss.value() << ", accuracy: " << accuracy 
              << ", time (sec): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0 << std::endl;
  }
  stack.pause_recording();

  // Now do some inference (generate text) =====================================
  
  for (int iter = 0; iter < 10; ++iter) {
    string output = inference_lstm(*weight_hyper_list[0]);
    std::cout << "\nGenerated output text:\n" << output << "\n";
  }
}

int main(int argc, char** argv) {
  learn_lstm();
  return 0;
}
