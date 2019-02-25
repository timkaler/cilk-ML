//tt Copyright 2018 Tim Kaler MIT License

#include "./Graph.hpp"

#include <random>


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
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    for (int i = 0; i < weights.size(); i++) {
      float w = 1.0/(weights[i].dimensions()[0]*weights[i].dimensions()[1]);
      for (int j = 0; j < weights[i].dimensions()[0]; j++) {
        for (int k = 0; k < weights[i].dimensions()[1]; k++) {
          weights[i][j][k] = distribution(generator)*w;
          skip_weights[i][j][k] = distribution(generator)*w;
        }
      }
    }


    //vertex_embeddings.resize(this->num_vertices);
    //for (int v = 0; v < this->num_vertices; v++ ) {
    //  vertex_embeddings[v].resize(embedding_dim_list.size());
    //  // now setup the vertex embeddings.
    //  for (int i = 0; i < embedding_dim_list.size(); i++) {
    //    vertex_embeddings[v][i] = aMatrix(embedding_dim_list[i+1],1);
    //  }
    //}


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
    std::normal_distribution<double> distribution(1.0,2.0);

    int d1 = embedding_dim_list[0];
    printf("random embeddings for %d x 1 \n", d1);
    for (int i = 0; i < this->num_vertices; i++) {
      Matrix initial_embedding(d1,1);
      for (int j = 0; j < d1; j++) {
        initial_embedding[j] = distribution(gen);
      }
      vertex_first_embeddings.push_back(initial_embedding);
    }
  }


  aMatrix Graph::get_embedding(int vid, int layer, std::vector<std::vector<aMatrix> >& embeddings) {
    //if (embedding_cache.find(std::make_pair(vid,layer)) != embedding_cache.end()) {
    //  return vertex_embeddings[vid][layer];
    //  //return tmp;
    //  //aMatrix tmp;
    //  //tmp = embedding_cache[std::make_pair(vid,layer)];
    //  //return tmp;
    //  //return embedding_cache[std::make_pair(vid, layer)];
    //}

    if (layer == 0) {
      //Matrix initial_embedding(max_label,1);
      Matrix initial_embedding = vertex_first_embeddings[vid];
      //for (int i = 0; i < initial_embedding.dimensions()[0]; i++) {
      //  //initial_embedding[i][0] = 0.0;//vid*(1.0/1005);
      //  initial_embedding[i][0] = vid*(1.0/1005);
      //}

      //for (int i = 0; i < adj[vid].size(); i++) {
      //  int u = adj[vid][i];
      //  int udept = vertex_values[u];
      //  initial_embedding[udept] += 1.0/adj[vid].size();
      //}

      //initial_embedding[(vertex_values[vid]+1)%max_label][0] = 1.0;//*vertex_values[vid];//*vid/this->num_vertices;
      //initial_embedding[(vertex_values[vid]+2)%max_label][0] = 1.0;//*vertex_values[vid];//*vid/this->num_vertices;

      ////initial_embedding[(vertex_values[vid]+2)%max_label][0] = 1.0;//*vertex_values[vid];//*vid/this->num_vertices;
      ////initial_embedding[(vertex_values[vid]+5)%max_label][0] = 1.0;//*vertex_values[vid];//*vid/this->num_vertices;
      ////initial_embedding[(vertex_values[vid]+2)%max_label][0] = 1.0;//*vertex_values[vid];//*vid/this->num_vertices;

      ////vertex_embeddings[vid][layer] = tfksig(edge_weight(vid,vid)*weights[0]**initial_embedding);
      ////embedding_cache[std::make_pair(vid, layer)] = true;

      return /*tfksig*//*(edge_weight(vid,vid)**/(weights[0]**initial_embedding);
    } else {
      aMatrix ret(embedding_dim_list[layer+1],1);

      for (int i = 0; i < ret.dimensions()[0]; i++) {
        for (int j = 0; j < ret.dimensions()[1]; j++) {
          ret[i][j] = 0.0;
        }
      }

      aMatrix bias_(embedding_dim_list[layer],1);
      for (int i = 0; i < bias_.dimensions()[0]; i++) {
        for (int j = 0; j < bias_.dimensions()[1]; j++) {
          bias_[i][j] = 0.0;
        }
      }
      bias_[0][0] = 1.0;

      //ret = mmul(skip_weights[layer],embeddings[layer-1][vid]);
      ret = mmul(skip_weights[layer], bias_);
      for (int i = 0; i < adj[vid].size(); i++) {
        if (adj[vid][i]==vid) continue;
        Real eweight = edge_weight(vid, adj[vid][i]);
        //ret += eweight*mmul(weights[layer],get_embedding(adj[vid][i], layer-1));
        ret += eweight*mmul(weights[layer],embeddings[layer-1][adj[vid][i]]);
                   //get_embedding(adj[vid][i], layer-1));
      }

      if (layer == embedding_dim_list.size()-2) {
        //vertex_embeddings[vid][layer] = tfksigmoid(ret);
        //embedding_cache[std::make_pair(vid, layer)] = true;
        return tfksig(ret);//tfksigmoid(ret);
      } else {
        //vertex_embeddings[vid][layer] = tfksig(ret);
        //embedding_cache[std::make_pair(vid, layer)] = true;
        return tfksig(ret);
      }
    }
    //aMatrix tmp;
    //tmp = vertex_embeddings[vid][layer];
    //return vertex_embeddings[vid][layer];
    exit(1);
    //return vertex_embeddings[vid][layer];
    //aMatrix tmp;
    //tmp = embedding_cache[std::make_pair(vid, layer)];
    //return tmp;
    //return embedding_cache[std::make_pair(vid, layer)];

  }

  aMatrix Graph::get_embedding(int vid, std::vector<std::vector<aMatrix> >& embeddings) {
    return embeddings[embedding_dim_list.size()-2][vid];
    //return get_embedding(vid, embedding_dim_list.size()-2, embeddings);
  }


  //aMatrix Graph::get_embedding(int vid, std::vector<aMatrix>& weights, int layer) {
  //  if (layer == 0) {
  //    Matrix initial_embedding(2,1);
  //    initial_embedding[0][0] = 1.0;
  //    initial_embedding[1][0] = 1.0;
  //    //initial_embedding << (1.0*vid);

  //    aMatrix ret(4,1);
  //    ret = tfksig(weights[0]**initial_embedding);
  //    //std::cout << "new embedding:"<< ret << std::endl;
  //    return ret;
  //  } else {
  //    aMatrix new_embedding;
  //    for (int i = 0; i < adj[vid].size(); i++) {
  //      Real weight = edge_weight(vid, adj[vid][i]);
  //      if (layer == 1) {
  //        //printf("case 1\n");
  //        if (i == 0) {
  //        new_embedding = tfksig(weight*mmul(weights[layer],get_embedding(adj[vid][i], weights, layer-1)));
  //        //std::cout << new_embedding << std::endl;
  //        } else {
  //        new_embedding += tfksig(weight*mmul(weights[layer],get_embedding(adj[vid][i], weights, layer-1)));
  //        }
  //      } else {
  //        if (i == 0) {
  //        //printf("case 2\n");
  //        new_embedding = tfksig(weight*mmul(weights[layer],get_embedding(adj[vid][i], weights, layer-1)));
  //        //std::cout << new_embedding.dimensions() << std::endl;
  //        } else {
  //        new_embedding += tfksig(weight*mmul(weights[layer],get_embedding(adj[vid][i], weights, layer-1)));

  //        }
  //      }
  //    }
  //    return new_embedding;
  //  }
  //}



