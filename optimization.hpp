// Copyright 2019 Tim Kaler MIT License

#include <adept_arrays.h>

#ifndef CILKML_OPTIMIZATION_H
#define CILKML_OPTIMIZATION_H

using namespace adept;

//extern double PARAM_ADAM_B1;
//extern double PARAM_ADAM_B2;
//extern double PARAM_ADAM_EPSILON;

double* allocate_weights_zero(std::vector<aMatrix>& weights);
double* allocate_weights(std::vector<aMatrix>& weights);

void set_values(std::vector<aMatrix>& weights, double* total_params);
void read_values(std::vector<aMatrix>& weights, double* total_params);
void read_gradients(std::vector<aMatrix>& weights, double* total_params);
void store_values_into_old(std::vector<aMatrix>& weights, double* current, double* old);

double compute_gradient_norm(std::vector<aMatrix>& weights, double* total_params);

void apply_gradient_update(std::vector<aMatrix>& weights, double* curr, double* old,
                           double* gradients, double mul);

void apply_gradient_update_ADAM(std::vector<aMatrix>& weights, double* curr, double* old,
                                double* gradients, double* momentums, double* velocities,
                                double mul, int t);

#endif
