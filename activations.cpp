// Copyright 2019 Tim Kaler MIT License

#include <adept.h>
#include <adept_arrays.h>
#include "./activations.hpp"

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

// Requires a softmax input --- i.e. all coordinates of yHat, y should be in (0,1)
// For numerical stability, we compute log(x) as log(x + epsilon) where epsilon = 1e-8.
aReal crossEntropy(aMatrix yHat, aMatrix y) {
  aReal loss_sum = 0.0;
  aReal n = y.dimensions()[0]*1.0*y.dimensions()[1];
  for (int i = 0; i < y.dimensions()[0]; i++) {
    for (int j = 0; j < y.dimensions()[1]; j++) {
      loss_sum += -1.0*y[i][j]*log(yHat[i][j]+1e-8) - (1.0-y[i][j])*log(1-yHat[i][j] + 1e-8);
    }
  }
  return loss_sum/n;
}

aReal tfksig(aReal arg) {
  return fmax(0.0, arg);
}

aVector tfksig(aVector arg) {
  return fmax(0.0, arg);
}

aMatrix tfksig(aMatrix arg) {
  return fmax(0.0, arg);
}

// softmax with a softening parameter p, active in case you want to learn softening param itself.
aMatrix tfksoftmax(aMatrix arg, aReal p) {
  // max val divided out for numerical stability
  aReal mval = maxval(p*arg);

  return exp(p*arg-mval)/sum(exp(p*arg-mval));
}

aMatrix mmul(aMatrix weights, aMatrix input) {
  return weights**input;
}

// Sigmoid function ranging from 0 to 1.
aReal tfksigmoid(aReal arg) {
  // Branch is for numerical stability.
  if (arg > 0.0) {
    return 1.0/(1+exp(-1.0*arg));
  } else {
    return exp(arg)/(exp(arg) + 1);
  }
}

aMatrix tfksigmoid(aMatrix arg) {
  return tfksig(arg);
}


