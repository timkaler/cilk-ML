// Copyright 2019 Tim Kaler MIT License

#include <random>

#include <vector>

#include "./Graph.hpp"
#include "./activations.hpp"


void Graph::setup_embeddings(std::vector<int> _embedding_dim_list) {
  weights.clear();
  skip_weights.clear();
  this->embedding_dim_list = _embedding_dim_list;
  for (int i = 0; i < embedding_dim_list.size()-1; i++) {
    weights.push_back(aMatrix(embedding_dim_list[i+1], embedding_dim_list[i]));
    skip_weights.push_back(aMatrix(embedding_dim_list[i+1], embedding_dim_list[i]));

    std::cout << embedding_dim_list[i+1] << "," << embedding_dim_list[i] << std::endl;
  }
  std::cout << "randomizing the embeddings" << std::endl;

  std::default_random_engine generator(42);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  for (int i = 0; i < weights.size(); i++) {
    float w = 1.0/(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
    for (int j = 0; j < weights[i].dimensions()[0]; j++) {
      for (int k = 0; k < weights[i].dimensions()[1]; k++) {
        weights[i][j][k] = distribution(generator)*w;
        skip_weights[i][j][k] = distribution(generator)*w;
      }
    }
  }
}

Graph::Graph(int num_vertices) {
  this->num_vertices = num_vertices;
  adj.resize(num_vertices);
}



Real Graph::edge_weight(int v, int u) {
  return 1.0/sqrt(1.0*(adj[v].size())*(adj[u].size()));
}

void Graph::add_edge(int u, int v) {
  adj[u].push_back(v);
}


void Graph::generate_random_initial_embeddings() {
  std::default_random_engine gen;
  std::normal_distribution<double> distribution(1.0, 2.0);

  int d1 = embedding_dim_list[0];
  for (int i = 0; i < this->num_vertices; i++) {
    Matrix initial_embedding(d1, 1);
    for (int j = 0; j < d1; j++) {
      initial_embedding[j] = distribution(gen);
    }
    vertex_first_embeddings.push_back(initial_embedding);
  }
}

void Graph::set_initial_embeddings(std::vector<Matrix>& initial_embeddings) {
  vertex_first_embeddings = initial_embeddings;
}

aMatrix Graph::get_embedding(int vid, int layer, std::vector<std::vector<aMatrix> >& embeddings) {
  if (layer == 0) {
    Matrix initial_embedding = vertex_first_embeddings[vid];

    // don't apply the activation function on the initial embeddings.
    return (weights[0]**initial_embedding);
  } else {
    aMatrix ret(embedding_dim_list[layer+1], 1);

    for (int i = 0; i < ret.dimensions()[0]; i++) {
      for (int j = 0; j < ret.dimensions()[1]; j++) {
        ret[i][j] = 0.0;
      }
    }

    // NOTE(TFK): This is a bit of a nonsense way to implement a bias term,
    //              remnant of a hacky experiment.
    //aMatrix bias_(embedding_dim_list[layer], 1);
    //for (int i = 0; i < bias_.dimensions()[0]; i++) {
    //  for (int j = 0; j < bias_.dimensions()[1]; j++) {
    //    bias_[i][j] = 0.0;
    //  }
    //}
    //bias_[0][0] = 1.0;

    ret = mmul(skip_weights[layer], embeddings[layer-1][vid]);

    for (int i = 0; i < adj[vid].size(); i++) {
      if (adj[vid][i] == vid) continue;
      Real eweight = edge_weight(vid, adj[vid][i]);
      ret += eweight*mmul(weights[layer], embeddings[layer-1][adj[vid][i]]);
    }

    if (layer == embedding_dim_list.size()-2) {
      return tfksig(ret);
    } else {
      return tfksig(ret);
    }
  }
}

aMatrix Graph::get_embedding(int vid, std::vector<std::vector<aMatrix> >& embeddings) {
  return embeddings[embedding_dim_list.size()-2][vid];
}

