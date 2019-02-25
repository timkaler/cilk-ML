// Copyright 2018 Tim Kaler MIT License

#include <random>
#include <adept.h>
#include <adept_arrays.h>

#include <map>
#include <utility>
#include <vector>

#include "./activations.hpp"

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

#ifndef GRAPH_H_
#define GRAPH_H_

class Graph {
  public:
    std::vector<std::vector<int> > adj;
    std::map<std::pair<int, int>, bool> embedding_cache;
    int num_vertices;
    int max_label;
    std::vector<aMatrix> weights;
    std::vector<aMatrix> skip_weights;
    std::vector<int> embedding_dim_list;

    std::vector<Matrix> vertex_first_embeddings;
    std::vector<int> vertex_values;
    std::vector<bool> vertex_training;
    std::vector<bool> vertex_training_active;

    std::vector<std::vector<aMatrix> > vertex_embeddings;

    explicit Graph(int num_vertices);
    Real edge_weight(int v, int u);
    void add_edge(int u, int v);
    aMatrix get_embedding(int vid, int layer, std::vector<std::vector<aMatrix> >& embeddings);
    aMatrix get_embedding(int vid, std::vector<std::vector<aMatrix> >& embeddings);
    void setup_embeddings(std::vector<int> embedding_dim_list);
    void generate_random_initial_embeddings();
};

#endif  // GRAPH_H_
