// Copyright 2019 Tim Kaler MIT License

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <random>
#include <map>
//#include "./adept-2.0.5/include/adept_source.h"
#include <adept_source.h>
#include <adept_arrays.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>

#include "./activations.hpp"
#include "./Graph.hpp"
#include "./optimization.hpp"
#include "./io_helpers.hpp"

//#define TFK_ADEPT_SERIAL

using namespace adept;
using namespace std;

std::default_random_engine generator(44);

void tfk_init() {
  #ifndef TFK_ADEPT_SERIAL
  thread_local_worker_id = __cilkrts_get_worker_number();
  tfk_reducer.get_tls_references();
  #endif
}

aReal compute_connect(std::vector<aMatrix>& weights, std::vector<Matrix>& data, std::vector<Real>& labels, double* accuracy,
  double* test_set_loss, bool recording) {
  #ifndef TFK_ADEPT_SERIAL
  tfk_reducer.sp_tree.open_S_node();
  tfk_init();
  #endif

  aReal loss = 0.0;

  aReal* losses = new aReal[data.size()];

  bool* correct = new bool[data.size()];


  #ifndef TFK_ADEPT_SERIAL
  tfk_init();
  tfk_reducer.sp_tree.open_P_node();
  #endif

  //#pragma cilk grainsize 1
  cilk_for (int j = 0; j < data.size(); j += 10) {
    int start_i = j;
    int end_i = j+10;
    if (end_i > data.size()) end_i = data.size();

    #ifndef TFK_ADEPT_SERIAL
    tfk_reducer.sp_tree.open_S_node();
    tfk_init();
    #endif
    for(int i = start_i; i < end_i; i++) {
      losses[i] = 0.0;

      std::vector<aMatrix> results = std::vector<aMatrix>(weights.size()-1);
      results[0] = tfksig(weights[0]**data[i]);
      for (int k = 1; k < weights.size()-1; k++) {
        // bias term.
        results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
        results[k] = tfksig(weights[k]**results[k-1]);
      }
      aMatrix mat_prediction = tfksoftmax(results[results.size()-1], 0.5);

      int argmax = 0;
      double argmaxvalue = mat_prediction[0][0].value();
      for (int k = 0; k < 3; k++) {
        if (argmaxvalue <= mat_prediction[0][k].value()) {
          argmaxvalue = mat_prediction[0][k].value();
          argmax = k;
        }
      }

      Matrix groundtruth(1,3);
      for (int k = 0; k < 3; k++) {
        groundtruth[0][k] = 0.0;
      }
      if (labels[i] > 0.5) groundtruth[0][0] = 1.0;
      if (labels[i] < -0.5) groundtruth[0][1] = 1.0;
      if (fabs(labels[i]) < 0.5) groundtruth[0][2] = 1.0;

      losses[i] += crossEntropy(mat_prediction, groundtruth);


      correct[i] = false;
      if ((argmax == 0 && labels[i] > 0.5) ||
          (argmax == 1 && labels[i] < -0.5) ||
          (argmax == 2 && fabs(labels[i]) < 0.5)) {
        correct[i] = true;
      }
    }
    #ifndef TFK_ADEPT_SERIAL
    tfk_reducer.sp_tree.close_S_node();
    #endif
  }
  #ifndef TFK_ADEPT_SERIAL
  tfk_reducer.sp_tree.close_P_node();
  tfk_init();
  #endif
  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || recording) {
      loss += losses[i];
    } else {
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    }
  }

  delete[] losses;
  delete[] correct;
  *accuracy = (100.0*ncorrect)/total;

  #ifndef TFK_ADEPT_SERIAL
  tfk_reducer.sp_tree.close_S_node();
  #endif
  return loss;

}


void learn_connect4() {
  using namespace adept;

  Stack stack;                           // Object to store differential statements

  std::vector<Matrix > data;
  std::vector<Real> labels;
  read_connect4("datasets/connect-4.data", data, labels);

  #ifndef TFK_ADEPT_SERIAL
  tfk_init();
  #endif

  std::vector<aMatrix> weight_list;

  weight_list.push_back(aMatrix(43*2,43)); // 43 x 1
  weight_list.push_back(aMatrix(43*2,43*2)); // 43 x 1
  weight_list.push_back(aMatrix(43*2,43*2)); // 43 x 1
  weight_list.push_back(aMatrix(43*2,43*2)); // 43 x 1
  weight_list.push_back(aMatrix(43*2,43*2)); // 43 x 1
  weight_list.push_back(aMatrix(3,43*2)); // 3 x 1


  // Initialize the weights.
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.5,8.0);
  for (int i = 0; i < weight_list.size(); i++) {
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i][j][k] = distribution(generator)/(weight_list[i].dimensions()[0] * weight_list[i].dimensions()[1]);
      }
    }
  }

  double* weights_raw = allocate_weights(weight_list);
  double* weights_raw_old = allocate_weights(weight_list);
  double* gradients = allocate_weights(weight_list);
  double* momentums = allocate_weights_zero(weight_list);
  double* velocities = allocate_weights_zero(weight_list);

  read_values(weight_list, weights_raw);
  read_values(weight_list, weights_raw_old);

  double learning_rate = 0.001;//1000*1.0/(data.size()/2);//1.0/G.num_vertices;

  int NUM_ITERS = 1001;

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    set_values(weight_list, weights_raw);
    stack.new_recording();                 // Clear any existing differential statements
    tfk_init();

    std::vector<Matrix> batch_data;
    std::vector<Real> batch_labels;
    std::uniform_int_distribution<int> dis(0, (data.size()-1)/2);
    for (int i = 0; i < 100; i++) {
      int _random = dis(generator);
      int random = 2*_random;
      batch_data.push_back(data[random]);
      batch_labels.push_back(labels[random]);
    }


    double accuracy = 0.0;
    double test_set_loss = 0.0;
    aReal loss;

    if (iter%100 == 0) {
      stack.pause_recording();
      loss = compute_connect(weight_list, data, labels, &accuracy, &test_set_loss, false);
      stack.continue_recording();
    } else {
      loss = compute_connect(weight_list, batch_data, batch_labels, &accuracy, &test_set_loss, true);
      loss.set_gradient(1.0);
      stack.reverse();
      read_gradients(weight_list, gradients);
    }

    std::cout.precision(9);
    std::cout.setf(ios::fixed, ios::floatfield);
    if (iter % 100 == 0) {
      std::cout << std::endl;
      std::cout << "loss:" << loss.value() << ",\t lr: " << learning_rate << "\t accuracy: " << accuracy << "% \t Test set loss: " << test_set_loss << "\r\r"<< std::endl<<std::endl;;
      continue;
    } else {
      std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate << "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_set_loss << "\r"<< std::flush;
    }

    double norm = compute_gradient_norm(weight_list, gradients);
    if (norm < 1.0) norm = 1.0;

    store_values_into_old(weight_list, weights_raw, weights_raw_old);



    //aReal newLoss = loss+0.01;
    //double local_lr = learning_rate * 1.1;
    //if (local_lr > 1.0/*/G.num_vertices*/) local_lr = 1.0;/*/G.num_vertices;*/

    //if (local_lr < 0.000001) local_lr = 0.000001;

    //while (newLoss.value() > loss.value()) {
    //  stack.new_recording();
    //  stack.pause_recording();

    //  apply_gradient_update(weight_list, weights_raw, weights_raw_old, gradients, local_lr*(1.0/norm));

    //  set_values(weight_list, weights_raw);

    //  double test_loss = 0.0;
    //  newLoss = compute_connect(weight_list, data, labels, &accuracy, &test_loss, false);
    //  //printf("\nnewLoss %f old loss %f\n\n", newLoss.value(), loss.value());
    //  if (newLoss.value() > loss.value()) {
    //    local_lr = local_lr*0.9;
    //  }
    //  stack.continue_recording();
    //}
    //learning_rate = local_lr;


    //if (iter < NUM_ITERS-1) {
      //apply_gradient_update(weight_list, weights_raw, weights_raw_old, gradients, learning_rate*(1.0/norm));
    apply_gradient_update_ADAM(weight_list, weights_raw, weights_raw_old, gradients, momentums, velocities, learning_rate, iter);
    //}
  }

}


int main(int argc, const char** argv) {

  learn_connect4();

  return 0;
}























