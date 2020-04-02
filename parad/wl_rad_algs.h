#ifndef WL_RAD_ALGS_H
#define WL_RAD_ALGS_H

#include "sp_tree.h"

namespace PARAD {
  void wl_report_times();
  void wl_reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient);
}

#endif  // WL_RAD_ALGS_H
