// Copyright (c) 2019, Tim Kaler - MIT License

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

#include "./rad_algs.h"

#include <adept.h>

#include <vector>
#include <map>
#include "../common/utils.h"
#include "../common/blockRadixSort.h"
#include "../common/gettime.h"

namespace PARAD {
timer r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;

void left_first_walk(SP_Node* n, args_for_collect_ops* args,
                     worker_local_vector<OperationReference>& wl_ops) {
  if (n->type == 3) {
    triple_vector_wl stack = n->data;

    adept::uIndex*__restrict operation_stack_arr = 
            worker_local_stacks[stack.worker_id].operation_stack_arr;
    const adept::Statement*__restrict statement_stack_arr = 
            worker_local_stacks[stack.worker_id].statement_stack_arr;
    float** __restrict operation_stack_deposit_location =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location;
    bool* __restrict operation_stack_deposit_location_valid =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;
    bool*__restrict idx_in_statement = args->idx_in_statement;

    if (stack.statement_stack_end != stack.statement_stack_start) {
      int wid = __cilkrts_get_worker_number();
      for (adept::uIndex ist = stack.statement_stack_start;
           ist < stack.statement_stack_end; ist++) {
        const adept::Statement& statement = statement_stack_arr[ist];

        if (statement.index == -1) continue;

        for (adept::uIndex j = statement_stack_arr[ist-1].end_plus_one;
             j < statement.end_plus_one; j++) {
          adept::uIndex op_index = operation_stack_arr[j];
          if (idx_in_statement[op_index]) {
            if (stack.worker_id == args->last_statement_worker[op_index] &&
                stack.statement_stack_start <= args->last_statement_index[op_index] &&
                stack.statement_stack_end > args->last_statement_index[op_index]) {
              // In this case just deposit directly into the global gradient table.
              worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j] = &args->gradient_[op_index];
              worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid[j] = true;
            } else {
              OperationReference ref;
              ref.statement_wid = args->last_statement_worker[op_index];
              ref.statement_ist = args->last_statement_index[op_index];
              ref.operation_wid = stack.worker_id;
              ref.operation_j = j;
              ref.gradient_index = op_index;
              wl_ops.push_back(wid, ref);
            }
          }
        }
        args->last_statement_worker[statement.index] = stack.worker_id;
        args->last_statement_index[statement.index] = ist;
      }
    }
    return;
  }
  if (n->type == 2) {
    for (int i = 0; i < n->children->size(); i++) {
      cilk_spawn left_first_walk((*(n->children))[i], args, wl_ops);
    }
    cilk_sync;
  } else {
    for (int i = 0; i < n->children->size(); i++) {
      left_first_walk((*(n->children))[i], args, wl_ops);
    }
  }
}

void right_first_walk(SP_Node* n, float** worker_local_grad_table,
                      bool* appears_in_statement, float* gradient_) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
      return;
    }
    int wid = __cilkrts_get_worker_number();

    adept::uIndex*__restrict operation_stack_arr = 
            worker_local_stacks[stack.worker_id].operation_stack_arr;
    adept::Real*__restrict multiplier_stack_arr = 
            worker_local_stacks[stack.worker_id].multiplier_stack_arr;
    const adept::Statement*__restrict statement_stack_arr = 
            worker_local_stacks[stack.worker_id].statement_stack_arr;
    float** __restrict operation_stack_deposit_location =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location;
    bool* __restrict operation_stack_deposit_location_valid =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;
    float*__restrict wl_grad = worker_local_grad_table[wid];

    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      const adept::Statement& statement = statement_stack_arr[ist];
      if (statement.index == -1) continue;

      // Fetch the extract_arr containing gradients deposited by operations for this statement.
      float*__restrict extract_arr = worker_local_stacks[stack.worker_id].statement_stack_deposit_location[ist];
      int extract_arr_len = worker_local_stacks[stack.worker_id].statement_stack_deposit_location_len[ist];

      // First extract from global gradient table, containing strand-local 
      // contributions to the gradient, as well as the "input" gradients to
      // reverse-mode AD (e.g. d_loss = 1).
      adept::Real a = gradient_[statement.index];
      gradient_[statement.index] = 0;

      int nonzero_count = 0;
      if (extract_arr_len > 5000) {
        cilk::reducer_opadd<float> red_a(a);
        cilk_for (int i = 0; i < extract_arr_len; i++) {
          *red_a += extract_arr[i];
          extract_arr[i] = 0;
        }
        a += red_a.get_value();
      } else {
        for (int i = 0; i < extract_arr_len; i++) {
          a += extract_arr[i];
          extract_arr[i] = 0;
        }
      }

      if (a != 0.0) {
        for (adept::uIndex j = worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
             j < statement.end_plus_one; j++) {
          adept::Real multiplier_test = multiplier_stack_arr[j];
          adept::uIndex operation_stack_index = operation_stack_arr[j];

          if (appears_in_statement[operation_stack_index] &&
              worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid[j]) {
            float*__restrict dep = worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j];
            if (dep) {
              *dep += multiplier_test*a;
            } else {
              wl_grad[operation_stack_index] += multiplier_test*a;
            }
          } else {
            wl_grad[operation_stack_index] += multiplier_test*a;
          }
        }
      }
    }
    return;
  }
  // If a SERIAL node or a ROOT node, then recursively call routine serially.
  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      right_first_walk((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    }
  } else if (n->type == 2) {
    // If a PARALLEL node, then recursively call routine in-parallel.
    for (int j = 0; j < n->children->size(); j++) {
      cilk_spawn right_first_walk((*(n->children))[n->children->size()-j-1], worker_local_grad_table, appears_in_statement, gradient_);
    }
    cilk_sync;
  }
}

void report_times() {
  r0.reportTotal("r0: Initialize appears_in_statements");
  r1.reportTotal("r1: Allocate/initialize lsw, lsi, lsn");
  r2.reportTotal("r2: Reserve worker_local_vector space, misc initialization");
  r3.reportTotal("r3: Initialize deposit location lengths / valid arrays");
  r4.reportTotal("r4: (2) left_first_walk");
  r5.reportTotal("r5: wl_ops.collect()");
  r6.reportTotal("r6: (3) Create O* (map for ops)");
  r7.reportTotal("r7: (4) Semisort O* (by gradient index)");
  r8.reportTotal("r8: (4) Semisort O* (by statement index)");
  r9.reportTotal("r9: (5a) Create S_rcv (blocks)");
  r10.reportTotal("r10: (5b) Allocate / initialize deposit locations (O_snd)");
  r11.reportTotal("r11: (5b) Populate deposit locations (O_snd)");
  r12.reportTotal("r12: Allocate / initialize worker local gradient tables");
  r13.reportTotal("r13: (6) right_first_walk");
  r14.reportTotal("r14: (7) Accumulate worker-local gradients in global table");
  r15.reportTotal("r15: Free memory");
}

void reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient) {
  // Identify all gradient indices that appear in statements
  r0.start();
  bool* appears_in_statement = new bool[n_gradients];
  cilk_for (int i = 0; i < n_gradients; i++) {
    appears_in_statement[i] = false;
  }
  cilk::reducer_opadd<int> red_nstatements(0);
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    wl_stacks worker_stack = worker_local_stacks[i];
    *red_nstatements += worker_stack.statement_stack_arr_len;
    cilk_for (int j = 0; j < worker_stack.statement_stack_arr_len; j++) {
      if (worker_stack.statement_stack_arr[j].index >= 0 && !appears_in_statement[worker_stack.statement_stack_arr[j].index]) {
        appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
      }
    }
  }
  int64_t nstatements = red_nstatements.get_value()+1;
  r0.stop();

  // Allocate/initialize lsw, lsi, lsn
  r1.start();
  int8_t* last_statement_worker = new int8_t[n_gradients];
  int32_t* last_statement_index = new int32_t[n_gradients];
  SP_Node** last_statement_node = new SP_Node*[n_gradients];
  cilk_for (uint64_t i = 0; i < n_gradients; i++) {
    last_statement_worker[i] = -1;
    last_statement_index[i] = -1;
    last_statement_node[i] = NULL;
  }
  r1.stop();

  // Reserve worker_local_vector space, misc initialization
  r2.start();
  // We're using a worker_local_vector to accumulate the operations that pass a
  // filter. This is a practical optimization to PARAD that reduces the number
  // of operations that need distinct deposit locations.
  worker_local_vector<OperationReference> wl_ops;
  int64_t op_stack_len = 0;
  for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    op_stack_len += worker_local_stacks[i].operation_stack_arr_len;
  }
  // Obtain statement offsets for worker-local statement stacks. (used later)
  int* statement_offsets = new int[__cilkrts_get_nworkers()];
  statement_offsets[0] = 0;
  for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
    statement_offsets[i] = statement_offsets[i-1] + worker_local_stacks[i-1].statement_stack_arr_len;
  }
  wl_ops.reserve((op_stack_len*2) / __cilkrts_get_nworkers());
  r2.stop();

  // Init the deposit location lengths and the deposit-location-valid arrays.
  r3.start();
  cilk_for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
    cilk_for (int64_t i = 0; i < worker_local_stacks[wid].statement_stack_arr_len; i++) {
      worker_local_stacks[wid].statement_stack_deposit_location_len[i] = 0;
    }
    // Array of bools used because it's cheaper to check for validity by
    // accessing a bool than a 8-byte pointer (smaller mem access)
    cilk_for (int64_t i = 0; i < worker_local_stacks[wid].operation_stack_arr_len; i++) {
      worker_local_stacks[wid].operation_stack_deposit_location_valid[i] = 0;
    }
  }
  r3.stop();

  // 2) Left-first traversal collects ops that need distinct deposit locations
  r4.start();
  args_for_collect_ops args;
  args.idx_in_statement = appears_in_statement;
  args.last_statement_worker = last_statement_worker;
  args.last_statement_index = last_statement_index;
  args.gradient_ = _gradient;
  left_first_walk(sptape_root, &args, wl_ops);
  r4.stop();

  // Collect the wl_ops into a single contiguous array ops of length ops_size.
  r5.start();
  OperationReference* ops;
  int64_t ops_size = wl_ops.collect(ops);
  r5.stop();

  // 3) Create O*: map each operation in ops with index i to (statement_index, i)
  r6.start();
  std::pair<int, int>* mapped_ops = 
          (std::pair<int, int>*) malloc(sizeof(std::pair<int, int>) * ops_size);
  int64_t mapped_ops_size = ops_size;
  cilk_for (uint64_t i = 0; i < ops_size; i++) {
    // Here a global statement index is obtained via the statement offsets we 
    // obtained earlier for each worker's statement stack.
    mapped_ops[i] = std::make_pair(statement_offsets[ops[i].statement_wid] + 
                                   ops[i].statement_ist, (int)i);
  }
  r6.stop();

  // 4) Semisort the mapped_ops so that all operations associated with the same
  // statement are contiguous in-memory.
  r7.start();
  intSort::iSort(&mapped_ops[0], mapped_ops_size, nstatements+1, utils::firstF<int, int>());
  r7.stop();

  // 4) Collect boundaries between continguous blocks of operations with same
  // statement index.
  r8.start();
  int* boundaries;
  worker_local_vector<int> wl_boundaries;
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      wl_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t boundaries_size = wl_boundaries.collect(boundaries);
  // Due to using worker local vectors, we need to actually sort this list of
  // boundaries so that they appear in order. Theoretically, we could have done
  // this with a scan followed by a pack, but this usually isn't a bottleneck
  intSort::iSort(&boundaries[0], boundaries_size, mapped_ops_size, utils::identityF<int>());
  r8.stop();

  // 5a) For each contiguous block in mapped_ops create a block with start and end index.
  r9.start();
  std::pair<int,int>* blocks = (std::pair<int,int>*) 
          malloc(sizeof(std::pair<int,int>)*boundaries_size);
  int64_t blocks_size = boundaries_size;
  cilk_for (uint64_t i = 1; i < boundaries_size; i++) {
    blocks[i-1] = std::make_pair(boundaries[i-1], boundaries[i]);
  }
  if (boundaries_size > 0) {
    blocks[boundaries_size-1] = std::make_pair(boundaries[boundaries_size-1], 
                                               mapped_ops_size);
  }
  r9.stop();

  // 5b) Allocate / initialize deposit locations
  r10.start();
  float* deposit_locations = new float[mapped_ops_size];
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    deposit_locations[i] = 0;
  }
  r10.stop();

  // 5b) Populate deposit locations (i.e. O_snd)
  r11.start();
  // Each block is associated with a subarray inside the deposit array.
  cilk_for (uint64_t i = 0; i < blocks_size; i++) {
    // The statement associated with the block is assigned to the range of
    // indices corresponding to the deposit subarray associated with this block.
    if (blocks[i].second > blocks[i].first) {
      OperationReference& opref = ops[mapped_ops[blocks[i].first].second];
      if (opref.statement_wid != -1) {
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
      }
    }
    // Assign each operation in the block to distinct locations in the deposit
    // subarray for the block.
    cilk_for (uint64_t j = blocks[i].first; j < blocks[i].second; j++) {
      OperationReference& opref = ops[mapped_ops[j].second];
      if (opref.statement_wid != -1) {
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = deposit_locations + j;
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = true;
      } else {
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = false;
      }
    }
  }
  r11.stop();

  // Optimization: allocate worker-local gradient tables. These are used for 
  // gradients that are not associated with any statement, i.e. gradients that
  // will be nonzero after the reverse-mode AD
  r12.start();
  float** worker_local_grad_table = (float**) malloc(sizeof(float*) * __cilkrts_get_nworkers());
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    worker_local_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  r12.stop();

  // 6) Right-first traversal to compute the gradients
  r13.start();
  right_first_walk(sptape_root, worker_local_grad_table, appears_in_statement, _gradient);
  r13.stop();

  // 7) Accumulate worker-local gradients in the global gradient table
  r14.start();
  int n_workers = __cilkrts_get_nworkers();
  int64_t max_gradient = tfk_reducer.max_gradient;
  cilk_for (int64_t i = 0; i < max_gradient; i++) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; wid++) {
      _gradient[i] += worker_local_grad_table[wid][i];
    }
  }
  r14.stop();

  r15.start();
  // Free all the memory that we allocated.
  cilk_for (int i = 0; i < n_workers; i++) {
    free(worker_local_grad_table[i]);
  }
  free(worker_local_grad_table);
  delete[] deposit_locations;
  delete[] last_statement_worker;
  delete[] last_statement_index;
  delete[] last_statement_node;
  delete[] appears_in_statement;
  free(blocks);
  free(boundaries);
  free(ops);
  free(mapped_ops);
  r15.stop();

  return;
}

} // end namespace rad_algs
