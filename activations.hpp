
#include <adept.h>
#include <adept_arrays.h>

#ifndef CILKML_ACTIVATIONS_H
#define CILKML_ACTIVATIONS_H

using namespace adept;

aReal crossEntropy(aMatrix yHat, aMatrix y);

aReal tfksig(aReal arg);
aVector tfksig(aVector arg);
aMatrix tfksig(aMatrix arg);

aMatrix tfksoftmax(aMatrix arg, aReal p);

aReal tfksigmoid (aReal arg);
aMatrix tfksigmoid (aMatrix arg);

aMatrix mmul(aMatrix weights, aMatrix input);

#endif
