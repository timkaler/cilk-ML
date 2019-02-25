// Copyright 2019 Tim Kaler MIT License

#include <adept_arrays.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <vector>

double PARAM_ADAM_B1 = 0.9;
double PARAM_ADAM_B2 = 0.999;
double PARAM_ADAM_EPSILON = 1e-8;
#include "./optimization.hpp"



double* allocate_weights_zero(std::vector<aMatrix>& weights) {
  int _pcount = 0;
  for (int i = 0; i < weights.size(); i++) {
    _pcount += weights[i].dimensions()[0]*weights[i].dimensions()[1];
  }
  return (double*) calloc(_pcount, sizeof(double));
}

double* allocate_weights(std::vector<aMatrix>& weights) {
  int _pcount = 0;
  for (int i = 0; i < weights.size(); i++) {
    _pcount += weights[i].dimensions()[0]*weights[i].dimensions()[1];
  }
  return (double*) malloc(_pcount*sizeof(double));
}


void read_values(std::vector<aMatrix>& weights, double* total_params) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        total_params[_pcount+k] = weights[i][j][k].value();
      }
    }
  }
}

void apply_gradient_update(std::vector<aMatrix>& weights, double* curr, double* old,
                           double* gradients, double mul) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        curr[_pcount + k] = old[_pcount + k] - gradients[_pcount + k]*mul;
      }
    }
  }
}


void apply_gradient_update_ADAM(std::vector<aMatrix>& weights, double* curr, double* old,
                                double* gradients, double* momentums, double* velocities,
                                double mul, double lr, int t) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  double lr_t = lr * (sqrt(1.0-pow(PARAM_ADAM_B2, t)) / (1.0-pow(PARAM_ADAM_B1, t)));


  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        double g = gradients[_pcount + k]*mul;
        double m = momentums[_pcount + k];
        double v = velocities[_pcount + k];

        double m_t = PARAM_ADAM_B1 * m + (1.0 - PARAM_ADAM_B1) * g;
        double v_t = PARAM_ADAM_B2 * v + (1.0 - PARAM_ADAM_B2) * (g*g);

        double new_val = old[_pcount + k] - lr_t * m_t / (sqrt(v_t) + PARAM_ADAM_EPSILON);

        momentums[_pcount+k] = m_t;
        velocities[_pcount+k] = v_t;
        curr[_pcount + k] = new_val;
      }
    }
  }
}



void store_values_into_old(std::vector<aMatrix>& weights, double* current, double* old) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        old[_pcount+k] = current[_pcount+k];
      }
    }
  }
}

double compute_gradient_norm(std::vector<aMatrix>& weights, double* total_params) {
  int _pcount = 0;
  for (int i = 0; i < weights.size(); i++) {
    _pcount += weights[i].dimensions()[0]*weights[i].dimensions()[1];
  }


  double norm = 0.0;
  cilk::reducer_opadd<double> norm_reduction;

  cilk_for (int i = 0; i < _pcount; i+=1) {
    int end = i+1;
    if (end > _pcount) end = _pcount;
    double norm_tmp = 0.0;
    for (int j = i; j < end; j++) {
      double val = total_params[j];
      norm_tmp += val*val;
    }
    norm_reduction += norm_tmp;
  }
  norm = norm_reduction.get_value();
  return sqrt(norm);
}


void read_gradients(std::vector<aMatrix>& weights, double* total_params) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        total_params[_pcount+k] = weights[i][j][k].get_gradient();
      }
    }
  }
}


void set_values(std::vector<aMatrix>& weights, double* total_params) {
  std::vector<int> sums;
  sums.push_back(0);
  for (int i = 0; i < weights.size()-1; i++) {
    sums.push_back(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
  }

  for (int i = 1; i < weights.size(); i++) {
    sums[i] += sums[i-1];
  }

  cilk_for (int i = 0; i < weights.size(); i++) {
    int _pcount_outer = sums[i];
    cilk_for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      int _pcount = _pcount_outer + j*weights[i].dimensions()[1];
      cilk_for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        weights[i][j][k].set_value(total_params[_pcount+k]);
      }
    }
  }
}



