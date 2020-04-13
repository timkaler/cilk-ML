// Copyright 2019 Tim Kaler MIT License

#ifndef CILKML_OPTIMIZATION_H
#define CILKML_OPTIMIZATION_H

#include <adept_arrays.h>
#include <vector>

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

extern double PARAM_ADAM_B1;
extern double PARAM_ADAM_B2;
extern double PARAM_ADAM_EPSILON;

double* allocate_weights_zero(std::vector<aMatrix>& weights);
double* allocate_weights_zero(std::vector<std::vector<aMatrix>*>& weights);
double* allocate_weights(std::vector<aMatrix>& weights);
double* allocate_weights(std::vector<std::vector<aMatrix>*>& weights);

void set_values(std::vector<aMatrix>& weights, double* total_params);
void set_values(std::vector<std::vector<aMatrix>*>& weights, double* total_params);
void read_values(std::vector<aMatrix>& weights, double* total_params);
void read_values(std::vector<std::vector<aMatrix>*>& weights, double* total_params);
void read_gradients(std::vector<aMatrix>& weights, double* total_params);
void read_gradients(std::vector<std::vector<aMatrix>*>& weights, double* total_params);
void store_values_into_old(std::vector<aMatrix>& weights, double* current, double* old);
void store_values_into_old(std::vector<std::vector<aMatrix>*>& weights, double* current,
                           double* old);

double compute_gradient_norm(std::vector<aMatrix>& weights, double* total_params);
double compute_gradient_norm(std::vector<std::vector<aMatrix>*>& weights, double* total_params);

void apply_gradient_update(std::vector<aMatrix>& weights, double* curr, double* old,
                           double* gradients, double mul);
void apply_gradient_update(std::vector<std::vector<aMatrix>*>& weights, double* curr, double* old,
                           double* gradients, double mul);

void apply_gradient_update_ADAM(std::vector<aMatrix>& weights, double* curr, double* old,
                                double* gradients, double* momentums, double* velocities,
                                double mul, double lr, int t);
void apply_gradient_update_ADAM(std::vector<std::vector<aMatrix>*>& weights, double* curr,
                                double* old, double* gradients, double* momentums,
                                double* velocities, double mul, double lr, int t);

#endif  // CILKML_OPTIMIZATION_H
