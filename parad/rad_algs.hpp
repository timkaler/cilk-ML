#ifndef RAD_ALGS_HPP
#define RAD_ALGS_HPP

#include "sp_tree.hpp"

namespace PARAD {
  void left_first_walk(SP_Node* n, args_for_collect_ops* args, worker_local_vector<OperationReference>& wl_ops);
  void right_first_walk(SP_Node* n, float** worker_local_grad_table, bool* appears_in_statement, float* gradient_);
  void reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient);
  void report_times();
}

#endif  // RAD_ALGS_HPP
