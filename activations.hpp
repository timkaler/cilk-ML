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

aReal crossEntropy(aMatrix yHat, aMatrix y);

aReal tfksig(aReal arg);
aVector tfksig(aVector arg);
aMatrix tfksig(aMatrix arg);

aMatrix tfksoftmax(aMatrix arg, aReal p);

aReal tfksigmoid(aReal arg);
aMatrix tfksigmoid(aMatrix arg);

aMatrix mmul(aMatrix weights, aMatrix input);

#endif  // CILKML_ACTIVATIONS_H
