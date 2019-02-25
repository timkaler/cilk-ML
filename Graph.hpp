// Copyright 2019 Tim Kaler MIT License
#include <random>
#include <map>
#include <adept_arrays.h>

#include "./activations.hpp"

using namespace adept;

#ifndef GRAPH_H_
#define GRAPH_H_

class Graph {
  public:
    std::vector<std::vector<int> > adj;
    std::map<std::pair<int,int>, bool> embedding_cache;
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

    Graph(int num_vertices);
    Real edge_weight(int v, int u);
    void add_edge(int u, int v);
    aMatrix get_embedding(int vid, int layer, std::vector<std::vector<aMatrix> >& embeddings);
    aMatrix get_embedding(int vid, std::vector<std::vector<aMatrix> >& embeddings);
    void setup_embeddings(std::vector<int> embedding_dim_list);
    void generate_random_initial_embeddings();
};


#endif // GRAPH_H_
