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
#include <map>
#include <vector>

#include "./activations.hpp"
#include "./Graph.hpp"
#include "./optimization.hpp"
#include "./io_helpers.hpp"

// #define TFK_ADEPT_SERIAL

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::vector;

std::default_random_engine generator(44);

void tfk_init() {
  #ifndef TFK_ADEPT_SERIAL
  thread_local_worker_id = __cilkrts_get_worker_number();
  tfk_reducer.get_tls_references();
  #endif
}

aReal compute_connect(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                      std::vector<Real>& labels, double* accuracy,
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

  cilk_for (int j = 0; j < data.size(); j += 10) {
    int start_i = j;
    int end_i = j+10;
    if (end_i > data.size()) end_i = data.size();

    #ifndef TFK_ADEPT_SERIAL
    tfk_reducer.sp_tree.open_S_node();
    tfk_init();
    #endif
    for (int i = start_i; i < end_i; i++) {
      losses[i] = 0.0;

      std::vector<aMatrix> results = std::vector<aMatrix>(weights.size()-1);
      results[0] = tfksig(weights[0]**data[i]);
      for (int k = 1; k < weights.size()-1; k++) {
        // bias term.
        results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
        results[k] = tfksig(weights[k]**results[k-1]);
      }
      aMatrix mat_prediction = tfksoftmax(results[results.size()-1], 1.0);

      int argmax = 0;
      double argmaxvalue = mat_prediction[0][0].value();
      for (int k = 0; k < 3; k++) {
        if (argmaxvalue <= mat_prediction[0][k].value()) {
          argmaxvalue = mat_prediction[0][k].value();
          argmax = k;
        }
      }

      Matrix groundtruth(1, 3);
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
  using adept::Stack;

  Stack stack;                           // Object to store differential statements

  std::vector<Matrix > data;
  std::vector<Real> labels;
  read_connect4("datasets/connect-4.data", data, labels);

  #ifndef TFK_ADEPT_SERIAL
  tfk_init();
  #endif

  std::vector<aMatrix> weight_list;

  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(3, 43));  // 3 x 1


  // Initialize the weights.
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < weight_list.size(); i++) {
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i][j][k] =
            distribution(generator) /
                (weight_list[i].dimensions()[0] * weight_list[i].dimensions()[1]);
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

  double learning_rate = 0.1;

  int NUM_ITERS = 2000;

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    set_values(weight_list, weights_raw);
    stack.new_recording();                 // Clear any existing differential statements
    tfk_init();

    std::vector<Matrix> batch_data;
    std::vector<Real> batch_labels;
    std::uniform_int_distribution<int> dis(0, (data.size()-1)/2);
    for (int i = 0; i < 1000; i++) {
      int _random = dis(generator);
      int random = 2*_random;
      batch_data.push_back(data[random]);
      batch_labels.push_back(labels[random]);
    }


    double accuracy = 0.0;
    double test_set_loss = 0.0;
    aReal loss;

    if (iter%200 == 0) {
      stack.pause_recording();
      loss = compute_connect(weight_list, data, labels, &accuracy, &test_set_loss, false);
      stack.continue_recording();
    } else {
      loss = compute_connect(weight_list, batch_data, batch_labels, &accuracy, &test_set_loss,
                             true);
      loss.set_gradient(1.0);
      stack.reverse();
      read_gradients(weight_list, gradients);

      if (std::isnan(loss.value())) {
        std::cout << std::endl << std::endl << "Got nan, doing reset " << std::endl << std::endl;
        store_values_into_old(weight_list, weights_raw_old, weights_raw);  // move old into raw.
        continue;
      }
    }

    std::cout.precision(9);
    std::cout.setf(ios::fixed, ios::floatfield);
    if (iter % 200 == 0) {
      std::cout << std::endl;
      std::cout << "loss:" << loss.value() << ",\t lr: " << learning_rate <<
          "\t accuracy: " << accuracy << "% \t Test set loss: " << test_set_loss <<
          "\r\r"<< std::endl << std::endl;
      continue;
    } else {
      std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
          "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_set_loss <<
          "\r" << std::flush;
    }

    double norm = compute_gradient_norm(weight_list, gradients);
    if (norm < 1.0) norm = 1.0;

    store_values_into_old(weight_list, weights_raw, weights_raw_old);


    #ifdef LINE_SEARCH
    aReal newLoss = loss+0.01;
    double local_lr = learning_rate * 1.1;
    if (local_lr > 1.0) local_lr = 1.0;

    if (local_lr < 0.000001) local_lr = 0.000001;

    while (newLoss.value() > loss.value()) {
      stack.new_recording();
      stack.pause_recording();

      apply_gradient_update(weight_list, weights_raw, weights_raw_old, gradients,
                            local_lr*(1.0/norm));

      set_values(weight_list, weights_raw);

      double test_loss = 0.0;
      newLoss = compute_connect(weight_list, data, labels, &accuracy, &test_loss, false);
      if (newLoss.value() > loss.value()) {
        local_lr = local_lr*0.9;
      }
      stack.continue_recording();
    }
    learning_rate = local_lr;
    #endif

    apply_gradient_update_ADAM(weight_list, weights_raw, weights_raw_old, gradients, momentums,
                               velocities, 1.0, learning_rate, iter+1);
  }
}

aReal compute_gcn(Graph& G, std::map<int, int >& department_labels, int max_label,
                  double* accuracy, double* test_set_loss) {
  tfk_reducer.sp_tree.open_S_node();
  tfk_init();

  std::vector<std::vector<aMatrix> > embeddings;
  embeddings.resize(G.embedding_dim_list.size()-1);
  aReal loss = 0;
  for (int i = 0; i < G.embedding_dim_list.size()-1; i++) {
    embeddings[i].resize(G.num_vertices);
  }

  for (int l = 0; l < G.embedding_dim_list.size()-1; l++) {
    tfk_reducer.sp_tree.open_S_node();
    tfk_init();
    tfk_reducer.sp_tree.open_P_node();

    bool last = (l == (G.embedding_dim_list.size()-2));

    cilk_for (int i = 0; i < G.num_vertices; i += 10) {
    //cilk_for (int i = 0; i < G.num_vertices; i++) {
      int end = i+10;
      if (end > G.num_vertices) end = G.num_vertices;
      if (i == end) continue;
      tfk_reducer.sp_tree.open_S_node();
      tfk_init();
      for (int j = i; j < end; j++) {
        if (last) {
          embeddings[l][j] = tfksoftmax(G.get_embedding(j,l, embeddings), 0.5);
        } else {
          embeddings[l][j] = G.get_embedding(j,l, embeddings);
        }
      }
      //embeddings[l][i] = G.get_embedding(i,l, embeddings);
      //tfk_init();
      tfk_reducer.sp_tree.close_S_node();
    }
    tfk_reducer.sp_tree.close_P_node();
    tfk_reducer.sp_tree.close_S_node();
    tfk_init();
  }
  int total_predictions = 0;
  int total_correct = 0;

  for (int i = 0; i < G.num_vertices; i++) {
    aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
    //aMatrix yhat = tfksoftmax(yhat_, 0.5);

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
      //aReal tmp = sum((yhat-y)*(yhat-y));//crossEntropy(yhat, y);
      aReal tmp = crossEntropy(yhat, y);
      *test_set_loss += tmp.value();
    } else if (G.vertex_training_active[i]) {
      loss += crossEntropy(yhat,y);
    }
  }

  tfk_reducer.sp_tree.close_S_node();

  //tfk_reducer.sp_tree.walk_tree_debug(tfk_reducer.sp_tree.get_root());

  *accuracy = ((100.0*total_correct)/(1.0*total_predictions));
  return loss;
}









void learn_gcn() {
  using adept::Stack;
  tfk_init();

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

  std::vector<int> counts(max_label+1);
  for (int i = 0; i < pairs.size(); i++) {
    counts[pairs[i].second]++;
  }


  std::vector<int> _embedding_dim_list;
  _embedding_dim_list.push_back(64);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(max_label);


  G.setup_embeddings(_embedding_dim_list);
  G.generate_random_initial_embeddings();

  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_hyper_list.push_back(&G.weights);
  weight_hyper_list.push_back(&G.skip_weights);

  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

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

  for (int iter = 0; iter < 10000; iter++) {
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();
    tfk_init();

    double accuracy = 0.0;
    double test_loss = 0.0;

    aReal loss = compute_gcn(G, department_labels, max_label, &accuracy, &test_loss);

    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\r" << std::flush;

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    //double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
    //apply_gradient_update(weight_hyper_list, weights_raw, weights_raw_old, gradients,
    //                      learning_rate/norm);

  }
}



int main(int argc, const char** argv) {
  //learn_connect4();
  learn_gcn();
  return 0;
}























