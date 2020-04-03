#ifndef WL_RAD_ALGS_HPP
#define WL_RAD_ALGS_HPP

#include "sp_tree.hpp"

namespace PARAD {
  void wl_report_times();
  void wl_reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient);
}

#endif  // WL_RAD_ALGS_HPP
