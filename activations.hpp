// Copyright 2019 Tim Kaler MIT License

#include <adept.h>
#include <adept_arrays.h>

#ifndef CILKML_ACTIVATIONS_H
#define CILKML_ACTIVATIONS_H

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

namespace activations {
  aReal crossEntropy(aMatrix yHat, aMatrix y);
  aReal logitCrossEntropy(aMatrix yHat, aMatrix y);

  aReal relu(aReal arg);
  aVector relu(aVector arg);
  aMatrix relu(aMatrix arg);

  aMatrix softmax(aMatrix arg, aReal p);

  aReal sigmoid(aReal arg);
  aMatrix sigmoid(aMatrix arg);
}

#endif  // CILKML_ACTIVATIONS_H
