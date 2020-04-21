#include "./wl_rad_algs.hpp"

#include "./sp_tree.hpp"

#include <adept.h>
#include <cilk/cilk.h>

#include "../common/gettime.h"

namespace PARAD {

timer t1, t2, t3, t4;

void wl_report_times() {
  t1.reportTotal("t1: Allocate wl_grad_table");
  t2.reportTotal("t2: wl_right_first_walk");
  t3.reportTotal("t3: Export wl_grad_table to global gradient table");
  t4.reportTotal("t4: Free all memory");
}

void wl_right_first_walk(SP_Node* node, float** wl_grad_table, float* global_grad_table) {
  // ROOT or SERIAL node: recursively call wl_right_first_walk serially
  if (node->type == 0 || node->type == 1) {
    for (int i = node->children->size()-1; i >= 0; --i) {
      wl_right_first_walk((*(node->children))[i], wl_grad_table, global_grad_table);
    }
  }
  // PARALLEL node: recursively call wl_right_first_walk in parallel
  else if (node->type == 2) {
    for (int i = 0; i < node->children->size(); ++i) {
      cilk_spawn wl_right_first_walk((*(node->children))[i], wl_grad_table, global_grad_table);
    }
    cilk_sync;
  }
  // DATA node
  else if (node->type == 3) {
    // Extract relevant data about the stacks
    triple_vector_wl stack_data = node->data;
    const adept::Statement*__restrict statement_stack = 
            worker_local_stacks[stack_data.worker_id].statement_stack_arr;
    adept::uIndex*__restrict operation_stack =
            worker_local_stacks[stack_data.worker_id].operation_stack_arr;
    adept::Real*__restrict multiplier_stack = 
            worker_local_stacks[stack_data.worker_id].multiplier_stack_arr;
    int wid = __cilkrts_get_worker_number();
    int n_workers = __cilkrts_get_nworkers();

    // Loop backwards through the derivative statements
    for (adept::uIndex ist = stack_data.statement_stack_end;
         ist-- > stack_data.statement_stack_start; ) {
      const adept::Statement& statement = statement_stack[ist];
      if (statement.index == -1) continue;

      // Extract the gradient from reading all P gradient tables
      // Also read the global gradient table -- e.g. for first iteration
      adept::Real a = global_grad_table[statement.index];
      global_grad_table[statement.index] = 0;
      for (int i = 0; i < n_workers; ++i) {
        a += wl_grad_table[i][statement.index];
        wl_grad_table[i][statement.index] = 0;
      }
      // Loop over the operations and update the gradients
      if (a != 0.0) {
        for (adept::uIndex iop = statement_stack[ist-1].end_plus_one;
             iop < statement.end_plus_one; ++iop) {
          adept::uIndex index = operation_stack[iop];
          adept::Real grad_multiplier = multiplier_stack[iop];
          wl_grad_table[wid][index] += grad_multiplier * a;
        }
      }
    }
  }
}

void wl_reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient) {
  int n_workers = __cilkrts_get_nworkers();

  // Allocate space for the worker-local gradient tables
  t1.start();
  float** wl_grad_table = (float**) malloc(sizeof(float*) * n_workers);
  cilk_for (int i = 0; i < n_workers; ++i) {
    wl_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  t1.stop();

  // Perform a right-first traversal of the SP-Tree
  t2.start();
  wl_right_first_walk(sptape_root, wl_grad_table, _gradient);
  t2.stop();

  // Export the worker-local gradients to the global gradient table
  t3.start();
  cilk_for(uint64_t i = 0; i < tfk_reducer.max_gradient; ++i) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; ++wid) {
      _gradient[i] += wl_grad_table[wid][i];
    }
  }
  t3.stop();

  // Free all memory
  t4.start();
  cilk_for (int i = 0; i < n_workers; ++i) {
    free(wl_grad_table[i]);
  }
  free(wl_grad_table);
  t4.stop();
}

}
