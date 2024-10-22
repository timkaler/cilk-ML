// Copyright 2019 Tim Kaler MIT License

#ifndef CILKML_IO_HELPERS_H
#define CILKML_IO_HELPERS_H

#include <utility>
#include <string>
#include <vector>

using std::string;
using std::vector;
using std::pair;
using std::ios;

std::vector<std::string> split(const std::string& str, const std::string& delim) {
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());
    return tokens;
}

void read_connect4(std::string filename, std::vector<Matrix>& data, std::vector<Real>& labels) {
  std::ifstream f(filename);
  std::string line;
  while (std::getline(f, line)) {
    std::string l(line);
    std::vector<std::string> items = split(l, ",");
    Matrix data_matrix = Matrix(43, 1);

    std::vector<int> data_vector;
    for (int i = 0; i < items.size()-1; i++) {
      if (items[i].find("b") != std::string::npos) {
        data_matrix[i][0] = 1.0;
      } else if (items[i].find("x") != std::string::npos) {
        data_matrix[i][0] = -1.0;
      } else if (items[i].find("o") != std::string::npos) {
        data_matrix[i][0] = 0.0;
      }
    }
    data_matrix[42][0] = 1.0;  // always is 1.

    data.push_back(data_matrix);

    if (items[items.size()-1].find("win") != std::string::npos) {
      labels.push_back(Real(1.0));
    } else if (items[items.size()-1].find("loss") != std::string::npos) {
      labels.push_back(Real(-1.0));
    } else if (items[items.size()-1].find("draw") != std::string::npos) {
      labels.push_back(Real(0.0));
    }
  }

  printf("total data size %zu\n", data.size());
  printf("total label size %zu\n", labels.size());
}



void read_pair_list(std::string filename, std::vector<std::pair<int, int> >& edges) {
  std::ifstream f(filename);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream l(line);
    int a, b;

    l >> a >> b;

    edges.push_back(std::make_pair(a, b));
  }
}

void parse_pubmed_data(std::string train_filename, std::string val_filename,
                       std::string test_filename, std::string feature_filename, bool* is_train,
                       bool* is_val, bool* is_test, std::vector<Matrix>& labels,
                       std::vector<Matrix>& features) {
  {
    std::ifstream f(train_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;

      if (c1+c2+c3 > 0.5) {
        is_train[id] = true;
        labels[id][0] = c1;
        labels[id][1] = c2;
        labels[id][2] = c3;
      }
    }
  }

  {
    std::ifstream f(val_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;
      //printf("%d, %f %f %f \n", id, c1, c2, c3);

      if (c1+c2+c3 > 0.5) {
        is_val[id] = true;
        labels[id][0] = c1;
        labels[id][1] = c2;
        labels[id][2] = c3;
      }
    }
  }

  {
    std::ifstream f(test_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id;
      float c1,c2,c3;
      l >> id >> c1 >> c2 >> c3;
      //printf("%d, %f %f %f \n", id, c1, c2, c3);

      if (c1+c2+c3 > 0.5) {
        is_test[id] = true;
        labels[id][0] = c1;
        labels[id][1] = c2;
        labels[id][2] = c3;
      }
    }
  }

  {
    std::ifstream f(feature_filename);
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream l(line);
      int id, fidx;
      float weight;
      l >> id >> fidx >> weight;
      //printf("%d %d %f\n", id, fidx, weight);
      features[id](fidx,0) = weight;
    }
  }
}

void edge_list_to_graph(std::string filename, Graph& G) {
  int max_vertex_id = 0;
  std::vector<std::pair<int, int> > edges;
  std::ifstream f(filename);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream l(line);
    int a, b;
    //std::cout << "line is:" << line << std::endl;

    l >> a >> b;

    //std::cout << "a:" << a << " b:" << b << std::endl;
    edges.push_back(std::make_pair(a, b));

    if (a > max_vertex_id) max_vertex_id = a;
    if (b > max_vertex_id) max_vertex_id = b;
  }

  G = Graph(max_vertex_id+1);
  // add self edges
  for (int i = 0; i < G.num_vertices; i++) {
    G.add_edge(i, i);
  }
  for (int i = 0; i < edges.size(); i++) {
    G.add_edge(edges[i].first, edges[i].second);
  }
}

#endif  // CILKML_IO_HELPERS_H

