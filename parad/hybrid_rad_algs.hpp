#ifndef HYBRID_RAD_ALGS_HPP
#define HYBRID_RAD_ALGS_HPP

#include "sp_tree.hpp"

namespace PARAD {
  void hybrid_reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient);
  void hybrid_report_times();
}

#endif  // HYBRID_RAD_ALGS_HPP
