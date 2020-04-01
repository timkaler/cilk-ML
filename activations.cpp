// Copyright 2019 Tim Kaler MIT License

#include <adept.h>
#include <adept_arrays.h>
#include "./activations.hpp"

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;

using namespace activations;

// Requires a softmax input: i.e. all coordinates of yHat, y should be in (0, 1)
// For numerical stability, we compute log(x) ~ log(x + 1e-12)
aReal activations::crossEntropy(aMatrix yHat, aMatrix y) {
  aReal loss_sum = 0.0;
  for (int i = 0; i < y.dimensions()[0]; ++i) {
    for (int j = 0; j < y.dimensions()[1]; ++j) {
      loss_sum += -1.0 * y(i,j) * log(yHat(i,j)+1e-12) -
                  (1.0 - y(i,j)) * log(1-yHat(i,j)+1e-12);
    }
  }
  return loss_sum / (1.0 * y.dimensions()[0] * y.dimensions()[1]);
}

// TODO: Why do we use logitCrossEntropy sometimes and crossEntropy other times?
aReal activations::logitCrossEntropy(aMatrix yHat, aMatrix y) {
  aReal loss_sum = 0.0;
  for (int i = 0; i < y.dimensions()[0]; i++) {
    for (int j = 0; j < y.dimensions()[1]; j++) {
      loss_sum += -1.0 * y(i,j) * log(yHat(i,j)+1e-12);
    }
  }
  return loss_sum / (1.0 * y.dimensions()[0] * y.dimensions()[1]);
}

// Softmax with softening parameter p (active in case you want to learn it)
// Divide everything by the maximum value for numerical stability
aMatrix activations::softmax(aMatrix arg, aReal p) {
  aReal mval = maxval(p * arg);
  return exp(p*arg-mval) / sum(exp(p*arg-mval));
}

aMatrix activations::relu(aMatrix arg) {
  return fmax(0.0, arg);
}

aMatrix activations::sigmoid(aMatrix arg) {
  return 0.5 * (tanh(0.5 * arg) + 1);
}
