// Copyright 2019 Tim Kaler MIT License

#include <adept_source.h>
#include <adept_arrays.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <random>

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <map>
#include <vector>

#include "../activations.hpp"
#include "../cxxopts.hpp"
#include "../io_helpers.hpp"
#include "../Graph.hpp"
#include "../mnist_parser.h"
#include "../optimization.hpp"

#include "../common/gettime.h"

// Defined by Makefile_serial
#ifdef TFK_ADEPT_SERIAL
#include <cilk/cilk_stub.h>
#endif

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::ios;
using std::vector;

aReal compute_mnist(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                    std::vector<uint8_t>& labels,
                    int max_label, double* accuracy, double* test_set_loss) {
  aReal loss = 0.0;

  bool* correct = new bool[data.size()];
  aReal* losses = new aReal[data.size()];

  aMatrix& biases = weights[weights.size()-1];

  //#ifndef TFK_ADEPT_SERIAL
  //#pragma cilk grainsize 1
  //#endif
  //cilk_for (int j = 0; j < data.size(); j++) {
  cilk_for (int _j = 0; _j < data.size(); _j += 10) {
    int j_start = _j; 
    int j_end = j_start + 10;
    if (j_end > data.size()) j_end = data.size();
    for (int j = j_start; j < j_end; j++) { 

    losses[j] = 0.0;

    std::vector<aMatrix> results = std::vector<aMatrix>(weights.size()-1);

    //data[j][data[j].dimensions()[0]-1][0] = 1.0;
    //aMatrix& biases = weights[weights.size()-1];
    results[0] = activations::relu(weights[0]**data[j]);

    //results[0] += biases(0,0);

    //std::cout << data[j] << std::endl;
    //exit(0);


    for (int k = 1; k < weights.size()-1; k++) {
      //results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
      if (k != weights.size()-2) {
      results[k] = activations::relu(weights[k]**results[k-1]);
      } else {
      results[k] = weights[k]**results[k-1];
      }
      results[k] += biases(k,0);
    }
    aMatrix mat_prediction = activations::softmax(results[results.size()-1], 1.0);
    //printf("dimensions %d, %d\n", mat_prediction.dimensions()[0], mat_prediction.dimensions()[1]);

    //std::cout << mat_prediction <<std::endl;

    int argmax = 0;
    double argmaxvalue = mat_prediction(0,0).value();
    for (int k = 0; k < 10; k++) {
      if (argmaxvalue <= mat_prediction(k,0).value()) {
        argmaxvalue = mat_prediction(k,0).value();
        argmax = k;
      }
    }

    Matrix groundtruth(10,1);
    for (int k = 0; k < 10; k++) {
      groundtruth(k,0) = 0.0;
    }
    groundtruth(labels[j],0) = 1.0;

    //std::cout << std::endl;
    //std::cout << labels[j] << std::endl;
    //std::cout << groundtruth << std::endl;
    //std::cout << mat_prediction << std::endl;
    //std::cout << std::endl;
    losses[j] += activations::logitCrossEntropy(mat_prediction, groundtruth);

    correct[j] = false;
    if (argmax == labels[j]) {
      correct[j] = true;
    }
    } // note
  }

  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || true) {
      loss += losses[i];
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    } else {
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    }
  }

  delete[] losses;
  delete[] correct;
  *accuracy = (100.0*ncorrect)/total;

  return loss/(1.0*data.size());
}

void learn_mnist(std::vector<int>& layer_sizes) {
  timer s0,s1,s2,s3,s4;

  using adept::Stack;
  Stack stack;
  std::string data_dir_path = "datasets";

  // load MNIST dataset
  std::vector<uint8_t> train_labels, test_labels;
  std::vector<Matrix> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, 0.0, 1.0, 0, 0);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, 0.0, 1.0, 0, 0);

  //for (int i = 0; i < train_images.size(); i++) {
  //  train_images[i] /= 28.0*28.0;
  //}
  //for (int i = 0; i < test_images.size(); i++) {
  //  test_images[i] /= 28.0*28.0;
  //}

  int dim1 = train_images[0].dimensions()[0];
  int dim2 = train_images[0].dimensions()[1];

  int max_label = 0;
  int min_label = 100;

  for (int i = 0; i < train_labels.size(); i++) {
    if (train_labels[i] > max_label) max_label = train_labels[i];
    if (train_labels[i] < min_label) min_label = train_labels[i];
  }

  std::vector<aMatrix> weight_list;
  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_list.push_back(aMatrix(layer_sizes[0], 28*28));
  for (int i = 1; i < layer_sizes.size(); i++) {
    weight_list.push_back(aMatrix(layer_sizes[1], layer_sizes[0]));
  }
  weight_list.push_back(aMatrix(10, layer_sizes[layer_sizes.size()-1]));
  weight_list.push_back(aMatrix(weight_list.size(),1));
  //weight_list.push_back(aMatrix(256+1, 1024));
  //weight_list.push_back(aMatrix(64+1, 256+1));
  //weight_list.push_back(aMatrix(16+1, 64+1));
  //weight_list.push_back(aMatrix(10, 16+1));
  weight_hyper_list.push_back(&weight_list);

  // Initialize the weights.
  std::default_random_engine generator(1000);
  for (int i = 0; i < weight_list.size(); i++) {
    float range = sqrt(6.0 / (weight_list[i].dimensions()[0] + weight_list[i].dimensions()[1]));
    std::uniform_real_distribution<double> distribution(-range, range);
    if (i == weight_list.size()-1) {
      for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
        for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
          weight_list[i][j][k] = 0.0;//distribution(generator);// / (weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);
        }
      }
      continue;
    }
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i](j,k) = distribution(generator) / (weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);
      }
    }
  }

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights_zero(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);
  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.01;
  int TIME_THRESH = GLOBAL_ITER_THRESH;
  for (int iter = 0; iter < 60*1; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);

    std::vector<Matrix> batch_data;
    std::vector<uint8_t> batch_labels;
    std::uniform_int_distribution<int> dis(0, train_images.size()-1);

    for (int i = 0; i < 1000; i++) {
      int _random = dis(generator);
      int random = _random;
      batch_data.push_back(train_images[random]);
      batch_labels.push_back(train_labels[random]);
    }

    double accuracy = 0.0;
    double test_loss = 0.0;
    stack.new_recording();
    s2.start();
    aReal loss;
    if (iter%100 == 0 && false) {
      stack.pause_recording();
      loss = compute_mnist(*weight_hyper_list[0], test_images, test_labels, max_label, &accuracy, &test_loss);

        std::cout.precision(14);
        std::cout.setf(ios::fixed, ios::floatfield);
        std::cout << std::endl << std::endl << "loss:" << loss.value() << ",\t\t lr: " <<
            learning_rate <<
            "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
            "\n" << std::endl << std::endl;
      stack.continue_recording();
      continue;
    } else {
      //stack.pause_recording();
      if (iter > TIME_THRESH) {
        s0.start();
      }
      loss += compute_mnist(*weight_hyper_list[0], batch_data, batch_labels, max_label, &accuracy, &test_loss);
      if (iter > TIME_THRESH) {
        s0.stop();
      }
      //stack.continue_recording();
    }
    //stack.initialize_gradients();
    if (iter > TIME_THRESH) {
      s1.start();
    }

    loss.set_gradient(1.0);
    stack.reverse();
    if (iter > TIME_THRESH) {
      s1.stop();
    }
    if (iter > TIME_THRESH) {
      s2.stop();
    }
    read_gradients(weight_hyper_list, gradients);

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);

    std::cout.precision(4);
    std::cout.setf(ios::fixed, ios::floatfield);
    /*
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\n" << std::endl;
    */
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter << ", loss: " << loss.value() 
              << ", accuracy: " << accuracy
              << ", runtime (s): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0
              << std::endl;
  }
  s0.reportTotal("Forward pass");
  s1.reportTotal("Reverse pass");
  s2.reportTotal("Forward+Reverse pass");
}
