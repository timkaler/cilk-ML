// Copyright 2019 Tim Kaler MIT License

#ifndef CILKML_IO_HELPERS_H
#define CILKML_IO_HELPERS_H

#include <utility>
#include <string>
#include <vector>

#include "./Graph.hpp"

using std::string;
using std::vector;
using std::pair;

void read_connect4(std::string filename, std::vector<Matrix>& data,
                   std::vector<Real>& labels);

void read_pair_list(std::string filename, std::vector<std::pair<int, int>>& edges);

void parse_pubmed_data(std::string train_filename, std::string val_filename,
                       std::string test_filename, std::string feature_filename, 
                       bool* is_train, bool* is_val, bool* is_test, 
                       std::vector<Matrix>& labels,
                       std::vector<Matrix>& features);

void edge_list_to_graph(std::string filename, Graph& G);

std::vector<std::vector<Matrix>> parse_paul_graham(int N, int LEN, int NUM_CHARS);

#endif  // CILKML_IO_HELPERS_H
