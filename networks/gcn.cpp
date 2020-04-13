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


aReal compute_gcn_pubmed(Graph& G, std::vector<Matrix>& groundtruth_labels, 
                         bool* is_train, bool* is_val, int max_labels, 
                         double* accuracy, double* test_set_loss) {
  std::vector<std::vector<aMatrix> > embeddings;
  embeddings.resize(G.embedding_dim_list.size()-1);
  aReal loss = 0;
  aReal loss_norm = 0.0;
  for (int i = 0; i < G.embedding_dim_list.size()-1; i++) {
    embeddings[i].resize(G.num_vertices);
  }

  for (int l = 0; l < G.embedding_dim_list.size()-1; l++) {

    bool last = (l == (G.embedding_dim_list.size()-2));

    cilk_for (int i = 0; i < G.num_vertices; i += 1) {
    //cilk_for (int i = 0; i < G.num_vertices; i++) {
      int end = i+1;
      if (end > G.num_vertices) end = G.num_vertices;
      if (i == end) continue;
      for (int j = i; j < end; j++) {
        if (last) {
          embeddings[l][j] = activations::softmax(G.get_embedding(j,l, embeddings), 1.0);
        } else {
          embeddings[l][j] = G.get_embedding(j,l, embeddings);
        }
      }
      //embeddings[l][i] = G.get_embedding(i,l, embeddings);
    }
  }
  int total_predictions = 0;
  int total_correct = 0;

  int num_train_items = 0;

  aReal* losses = new aReal[G.num_vertices];
  aReal* loss_norms = new aReal[G.num_vertices];


  cilk::reducer_opadd<float> red_test_set_loss(0.0);
  cilk::reducer_opadd<int> red_total_predictions(0);
  cilk::reducer_opadd<int> red_total_correct(0);

  cilk_for (int i = 0; i < G.num_vertices; i++) {
    aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
    //aMatrix yhat = activations::softmax(yhat_, 0.5);

    Matrix y(max_labels,1);
    double max_label_val = 0.0;
    int max_label = 0;

    int gt_label = 0;

    for (int j = 0; j < y.dimensions()[0]; j++) {
      y[j][0] = 0.0;
      if (yhat[j][0].value() > max_label_val) {
        max_label_val = yhat[j][0].value();
        max_label = j;
      }
      if (groundtruth_labels[i](j,0) > 0.5) gt_label = j;
    }


    y[gt_label][0] = 1.0;
    //if (!G.vertex_training[i]) {
    if (!is_train[i] && !is_val[i]) {
      if (max_label == gt_label) {
        *red_total_correct += 1;
      }
      *red_total_predictions += 1;
      //total_predictions++;
      //aReal tmp = sum((yhat-y)*(yhat-y));//activations::crossEntropy(yhat, y);
      aReal tmp = activations::crossEntropy(yhat, y);
      //*test_set_loss += tmp.value();
      *red_test_set_loss += tmp.value();
      losses[i] = 0.0;
      loss_norms[i] = 0.0;
    } else if (is_train[i]) {
      losses[i] = activations::crossEntropy(yhat,y);
      loss_norms[i] = 1.0;//activations::crossEntropy(yhat,y);
      //loss += activations::crossEntropy(yhat,y);
      //loss_norm += 1.0;
      //num_train_items++;
    }
  }

  *test_set_loss = red_test_set_loss.get_value();
  total_predictions = red_total_predictions.get_value();
  total_correct = red_total_correct.get_value();

  loss = recursive_sum(losses, 0, G.num_vertices);
  loss_norm = recursive_sum(loss_norms, 0, G.num_vertices);

  delete[] loss_norms;
  delete[] losses;

  //for (int i = 0; i < G.num_vertices; i++) {
  //  aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
  //  //aMatrix yhat = activations::softmax(yhat_, 0.5);

  //  Matrix y(max_labels,1);
  //  double max_label_val = 0.0;
  //  int max_label = 0;

  //  int gt_label = 0;

  //  for (int j = 0; j < y.dimensions()[0]; j++) {
  //    y[j][0] = 0.0;
  //    if (yhat[j][0].value() > max_label_val) {
  //      max_label_val = yhat[j][0].value();
  //      max_label = j;
  //    }
  //    if (groundtruth_labels[i](j,0) > 0.5) gt_label = j;
  //  }



  //  y[gt_label][0] = 1.0;
  //  //if (!G.vertex_training[i]) {
  //  if (!is_train[i] && !is_val[i]) {
  //    if (max_label == gt_label) total_correct++;
  //    total_predictions++;
  //    //aReal tmp = sum((yhat-y)*(yhat-y));//activations::crossEntropy(yhat, y);
  //    aReal tmp = activations::crossEntropy(yhat, y);
  //    *test_set_loss += tmp.value();
  //    losses[i] = 0.0;
  //    loss_norms[i] = 0.0;
  //  } else if (is_train[i]) {
  //    losses[i] = activations::crossEntropy(yhat,y);
  //    loss_norms[i] = 1.0;//activations::crossEntropy(yhat,y);
  //    //loss += activations::crossEntropy(yhat,y);
  //    //loss_norm += 1.0;
  //    //num_train_items++;
  //  }
  //}



  loss = loss / loss_norm;
  //loss /= (1.0*num_train_items);

  //printf("total predictions %d\n", total_predictions);

  //tfk_reducer.sp_tree.walk_tree_debug(tfk_reducer.sp_tree.get_root());
  //Real norm = 1.0*num_train_items;
  //aReal loss_normalized = loss / norm;

  *accuracy = ((100.0*total_correct)/(1.0*total_predictions));

  return loss;
}

aReal compute_gcn(Graph& G, std::map<int, int >& department_labels, 
                  int max_label, double* accuracy, double* test_set_loss) {
  std::vector<std::vector<aMatrix> > embeddings;
  embeddings.resize(G.embedding_dim_list.size()-1);
  aReal loss = 0;
  for (int i = 0; i < G.embedding_dim_list.size()-1; i++) {
    embeddings[i].resize(G.num_vertices);
  }

  for (int l = 0; l < G.embedding_dim_list.size()-1; l++) {

    bool last = (l == (G.embedding_dim_list.size()-2));

    cilk_for (int i = 0; i < G.num_vertices; i += 1) {
    //cilk_for (int i = 0; i < G.num_vertices; i++) {
      int end = i+1;
      if (end > G.num_vertices) end = G.num_vertices;
      if (i == end) continue;
      for (int j = i; j < end; j++) {
        if (last) {
          embeddings[l][j] = activations::softmax(G.get_embedding(j,l, embeddings), 0.5);
        } else {
          embeddings[l][j] = G.get_embedding(j,l, embeddings);
        }
      }
      //embeddings[l][i] = G.get_embedding(i,l, embeddings);
    }
  }
  int total_predictions = 0;
  int total_correct = 0;

  for (int i = 0; i < G.num_vertices; i++) {
    aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
    //aMatrix yhat = activations::softmax(yhat_, 0.5);

    Matrix y(max_label,1);
    double max_label_val = 0.0;
    int max_label = 0;
    for (int j = 0; j < y.dimensions()[0]; j++) {
      y[j][0] = 0.0;
      if (yhat[j][0].value() > max_label_val) {
        max_label_val = yhat[j][0].value();
        max_label = j;
      }
    }
    y[department_labels[i]][0] = 1.0;
    if (/*i%2==1*/!G.vertex_training[i]) {
      if (max_label == department_labels[i]) total_correct++;
      total_predictions++;
      //aReal tmp = sum((yhat-y)*(yhat-y));//activations::crossEntropy(yhat, y);
      aReal tmp = activations::crossEntropy(yhat, y);
      *test_set_loss += tmp.value();
    } else if (G.vertex_training_active[i]) {
      loss += activations::crossEntropy(yhat,y);
    }
  }


  //tfk_reducer.sp_tree.walk_tree_debug(tfk_reducer.sp_tree.get_root());

  *accuracy = ((100.0*total_correct)/(1.0*total_predictions));
  return loss;
}

// =============================================================================

void learn_gcn() {
  timer s0,s1,s2,s3,s4;
  using adept::Stack;

  Stack stack;

  Graph G(0);
  edge_list_to_graph("datasets/email-Eu-core.txt", G);

  std::map<int, int> department_labels;
  std::vector<std::pair<int,int> > pairs;
  read_pair_list("datasets/email-Eu-core-department-labels.txt", pairs);
  int max_label = 0;
  for (int i = 0; i < pairs.size(); i++) {
    department_labels[pairs[i].first] = pairs[i].second;
    if (pairs[i].second > max_label) max_label = pairs[i].second;
  }
  max_label = max_label + 1;
  G.max_label = max_label;
  printf("max label is %d\n", max_label);
  std::vector<int> counts(max_label+1);
  for (int i = 0; i < pairs.size(); i++) {
    counts[pairs[i].second]++;
  }


  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  std::vector<Matrix> feature_vectors;
  for (int i = 0; i < G.num_vertices; i++) {
    Matrix tmp2(1024, 1);
    for (int j = 0; j < 1024; j++) {
      tmp2(j,0) = distribution(generator);
    }
    feature_vectors.push_back(tmp2);
  }


  std::vector<int> _embedding_dim_list;
  _embedding_dim_list.push_back(1024);
  _embedding_dim_list.push_back(64);
  _embedding_dim_list.push_back(max_label);


  //G.generate_random_initial_embeddings();
  G.setup_embeddings(_embedding_dim_list);
  G.set_initial_embeddings(feature_vectors);

  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_hyper_list.push_back(&G.weights);
  weight_hyper_list.push_back(&G.skip_weights);


  G.vertex_values.resize(G.num_vertices);
  G.vertex_training.resize(G.num_vertices);
  G.vertex_training_active.resize(G.num_vertices);

  // Randomly divide into training and test set.
  for (int i = 0; i < G.num_vertices; i++) {
    G.vertex_values[i] = department_labels[i];
    if (distribution(generator) < 0.5) {
      G.vertex_training[i] = true;
      G.vertex_training_active[i] = true;
    } else {
      G.vertex_training[i] = false;
      G.vertex_training_active[i] = false;
    }
  }

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);

  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.001;

  int ITER_THRESH = GLOBAL_ITER_THRESH;
  for (int iter = 0; iter < 20; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    double accuracy = 0.0;
    double test_loss = 0.0;

    if (iter > ITER_THRESH) {
    s2.start();
    s0.start();
    }
    aReal loss = compute_gcn(G, department_labels, max_label, &accuracy, &test_loss);

    if (iter > ITER_THRESH) {
    s0.stop();
    }

    loss.set_gradient(1.0);
    if (iter > ITER_THRESH) {
    s1.start();
    }
    stack.reverse();
    if (iter > ITER_THRESH) {
    s1.stop();
    s2.stop();
    }
    read_gradients(weight_hyper_list, gradients);

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    //double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
    //apply_gradient_update(weight_hyper_list, weights_raw, weights_raw_old, gradients,
    //                      learning_rate/norm);

    std::cout.precision(4);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter
              << ", loss: " << loss.value()
              << ", accuracy: " << accuracy
              << ", time(sec): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0 << std::endl;
  }
  s0.reportTotal("Forward pass");
  s1.reportTotal("Reverse pass");
  s2.reportTotal("Forward+Reverse pass");
}

void learn_gcn_pubmed() {
  timer s0,s1,s2,s3,s4;
  using adept::Stack;

  Stack stack;

  Graph G(0);
  edge_list_to_graph("datasets/pubmed.edges", G);

  int n_vertices = G.num_vertices;
  int feature_dim = 500;

  bool* is_train = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_val = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_test = (bool*) calloc(n_vertices, sizeof(bool));

  std::vector<Matrix> groundtruth_labels;
  std::vector<Matrix> feature_vectors;
  for (int i = 0; i < n_vertices; i++) {
    Matrix tmp(3,1);
    for (int j = 0; j < 3; j++) {
      tmp(j,0) = 0.0;
    }
    groundtruth_labels.push_back(tmp);

    Matrix tmp2(feature_dim,1);
    for (int j = 0; j < feature_dim; j++) {
      tmp2(j,0) = 0.0;
    }
    feature_vectors.push_back(tmp2);
  }
  int max_label = 3;
  parse_pubmed_data("datasets/pubmed.trainlabels", "datasets/pubmed.vallabels",
                    "datasets/pubmed.testlabels", "datasets/pubmed_features", is_train, is_val, is_test, groundtruth_labels,
                    feature_vectors);

  std::vector<int> _embedding_dim_list;
  _embedding_dim_list.push_back(feature_dim);
  //_embedding_dim_list.push_back(32);
  _embedding_dim_list.push_back(32);
  _embedding_dim_list.push_back(max_label);


  G.setup_embeddings(_embedding_dim_list);
  G.set_initial_embeddings(feature_vectors);
  //G.generate_random_initial_embeddings();

  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_hyper_list.push_back(&G.weights);
  weight_hyper_list.push_back(&G.skip_weights);

  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  //G.vertex_values.resize(G.num_vertices);
  //G.vertex_training.resize(G.num_vertices);
  //G.vertex_training_active.resize(G.num_vertices);

  //// Randomly divide into training and test set.
  //for (int i = 0; i < G.num_vertices; i++) {
  //  G.vertex_values[i] = department_labels[i];
  //  if (distribution(generator) < 0.5) {
  //    G.vertex_training[i] = true;
  //    G.vertex_training_active[i] = true;
  //  } else {
  //    G.vertex_training[i] = false;
  //    G.vertex_training_active[i] = false;
  //  }
  //}

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);

  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.1;//0.01;

  int ITER_THRESH = GLOBAL_ITER_THRESH;

  for (int iter = 0; iter < 15; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    double accuracy = 0.0;
    double test_loss = 0.0;
    if (iter > ITER_THRESH) {
    s2.start();
    s0.start();
    }
    aReal loss = compute_gcn_pubmed(G, groundtruth_labels, is_train, is_val, max_label, &accuracy,
                                    &test_loss);
    if (iter > ITER_THRESH) {
    s0.stop();
    }

    loss.set_gradient(1.0);

    if (iter > ITER_THRESH) {
    s1.start();
    }
    stack.reverse();
    if (iter > ITER_THRESH) {
    s1.stop();
    s2.stop();
    }
    read_gradients(weight_hyper_list, gradients);
    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);
    //double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
    //apply_gradient_update(weight_hyper_list, weights_raw, weights_raw_old, gradients,
    //                      learning_rate/norm);

    std::cout.precision(4);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter
              << ", loss: " << loss.value()
              << ", accuracy: " << accuracy
              << ", time (s): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0
              << std::endl;
  }
  s0.reportTotal("Forward pass");
  s1.reportTotal("Reverse pass");
  s2.reportTotal("Forward+Reverse pass");
}
