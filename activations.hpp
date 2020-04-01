// Copyright 2019 Tim Kaler MIT License

#include <adept.h>
#include <adept_arrays.h>

#ifndef CILKML_ACTIVATIONS_H
#define CILKML_ACTIVATIONS_H

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;

namespace activations {
  aReal crossEntropy(aMatrix yHat, aMatrix y);
  aReal logitCrossEntropy(aMatrix yHat, aMatrix y);

  aMatrix relu(aMatrix arg);
  aMatrix sigmoid(aMatrix arg);

  aMatrix softmax(aMatrix arg, aReal p);
}

#endif  // CILKML_ACTIVATIONS_H
