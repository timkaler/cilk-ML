#include <adept.h>
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "../common/blockRadixSort.h"
#include "../common/gettime.h"
#include "../common/utils.h"

#include "./sp_tree.hpp"
#include "./hybrid_rad_algs.hpp"

namespace PARAD {

timer r0,r1,r2,r3,r4,r5,r6a,r6b,r6c,r6d,r6e,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19;

void hybrid_report_times() {
  r0.reportTotal("r0: Initialize appears_in_statements");
  r1.reportTotal("r1: Initialize lsw, lsi");
  r2.reportTotal("r2: Reserve worker_local_vector space; misc initialization");
  r3.reportTotal("r3: Initialize deposit location lengths / valid arrays");
  r4.reportTotal("r4: Initialize gradient_req_ops_map, gradient_use_wl, lfw_use_wl");
  r5.reportTotal("r5: statement_left_first_walk (compute gradient_req_ops_map)");
  r6a.reportTotal("r6a: Allocate space for sampled op gradient indices");
  r6b.reportTotal("r6b: Sample ops from worker-local operation stacks");
  r6c.reportTotal("r6c: Sort op_indices");
  r6d.reportTotal("r6d: Find boundaries in op_indices");
  r6e.reportTotal("r6e: Update gradient_req_ops_map");
  r7.reportTotal("r7: Compute gradient_use_wl");
  r8.reportTotal("r8: (2) hybrid_left_first_walk");
  r9.reportTotal("r9: wl_ops.collect()");
  r10.reportTotal("r10: (3) Create O* (map for ops)");
  r11.reportTotal("r11: (4) Semisort O* (by gradient index)");
  r12.reportTotal("r12: (4) Semisort O* (by statement index)");
  r13.reportTotal("r13: (5a) Create S_rcv (blocks");
  r14.reportTotal("r14: (5a) Allocate / initialize deposit locations (O_snd)");
  r15.reportTotal("r15: (5b) Populate deposit locations (O_snd)");
  r16.reportTotal("r16: Allocate wl_grad_table");
  r17.reportTotal("r17: (6) hybrid_right_first_walk");
  r18.reportTotal("r18: (7) Accumulate worker-local gradients in global table");
  r19.reportTotal("r19: Free memory");
}

void hybrid_left_first_walk(SP_Node* node, args_for_collect_ops* args,
                            worker_local_vector<OperationReference>& wl_ops,
                            bool* gradient_use_wl, bool* lfw_use_wl) {
  // ROOT or SERIAL node: recursively call hybrid_left_first_walk serially
  if (node->type == 0 || node->type == 1) {
    for (int i = 0; i < node->children->size(); ++i) {
      hybrid_left_first_walk((*(node->children))[i], args, wl_ops, gradient_use_wl, lfw_use_wl);
    }
  }
  // PARALLEL node: recursively call hybrid_left_first_walk in parallel
  else if (node->type == 2) {
    for (int i = 0; i < node->children->size(); ++i) {
      cilk_spawn hybrid_left_first_walk((*(node->children))[i], args, wl_ops, gradient_use_wl, lfw_use_wl);
    }
    cilk_sync;
  }
  // DATA node
  else if (node->type == 3) {
    triple_vector_wl stack = node->data;
    
    adept::uIndex*__restrict operation_stack_arr =
            worker_local_stacks[stack.worker_id].operation_stack_arr;
    const adept::Statement*__restrict statement_stack_arr =
            worker_local_stacks[stack.worker_id].statement_stack_arr;
    float**__restrict operation_stack_deposit_location =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location;
    bool*__restrict operation_stack_deposit_location_valid =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;
    bool*__restrict idx_in_statement = args->idx_in_statement;

    if (stack.statement_stack_start != stack.statement_stack_end) {
      int wid = __cilkrts_get_worker_number();
      for (adept::uIndex ist = stack.statement_stack_start;
           ist < stack.statement_stack_end; ++ist) {
        const adept::Statement& statement = statement_stack_arr[ist];
        if (statement.index == -1) continue;

        for (adept::uIndex iop = statement_stack_arr[ist-1].end_plus_one;
             iop < statement.end_plus_one; ++iop) {
          adept::uIndex op_index = operation_stack_arr[iop];
          // Optimization 1: operations whose gradient index never appears in a
          // statement accumulate gradients using worker-local sparse arrays
          if (idx_in_statement[op_index]) {
            // Optimization 2: operations whose gradient contributions are
            // accumulated by a statement in the same subtape (data node) use
            // global gradient table.
            if (stack.worker_id == args->last_statement_worker[op_index] &&
                stack.statement_stack_start <= args->last_statement_index[op_index] &&
                stack.statement_stack_end > args->last_statement_index[op_index]) {
              operation_stack_deposit_location[iop] = &args->gradient_[op_index];
              operation_stack_deposit_location_valid[iop] = true;
            }
            // Optimization "3": only accumulate in wl_ops if we don't plan to
            // use worker-local gradient table
            else if (!gradient_use_wl[op_index]) {
              OperationReference ref;
              ref.statement_wid = args->last_statement_worker[op_index];
              ref.statement_ist = args->last_statement_index[op_index];
              ref.operation_wid = stack.worker_id;
              ref.operation_j = iop;
              ref.gradient_index = op_index;
              wl_ops.push_back(wid, ref);
            }
            // lfw_use_wl keeps track of all gradient indices which use
            // worker-local gradient tables and are not optimized out. This is
            // an important optimization for right_first_walk
            else {
              lfw_use_wl[op_index] = true;
            }
          }
        }
        args->last_statement_worker[statement.index] = stack.worker_id;
        args->last_statement_index[statement.index] = ist;
      }
    }
  }
}

void hybrid_right_first_walk(SP_Node* node, float** wl_grad_table,
                             bool* appears_in_statement,
                             float* global_grad_table,
                             bool* gradient_use_wl) {
  // ROOT or SERIAL node: recursively call hybrid_right_first_walk serially
  if (node->type == 0 || node->type == 1) {
    for (int i = node->children->size()-1; i >= 0; --i) {
      hybrid_right_first_walk((*(node->children))[i], wl_grad_table,
                              appears_in_statement, global_grad_table,
                              gradient_use_wl);
    }
  }
  // PARALLEL node: recursively call hybrid_right_first_walk in parallel
  else if (node->type == 2) {
    for (int i = node->children->size()-1; i >= 0; --i) {
      cilk_spawn hybrid_right_first_walk((*(node->children))[i], wl_grad_table,
                                         appears_in_statement,
                                         global_grad_table, gradient_use_wl);
    }
    cilk_sync;
  }
  // DATA node
  else if (node->type == 3) {
    triple_vector_wl stack = node->data;
    if (stack.statement_stack_end == stack.statement_stack_start) { return; }
    int wid = __cilkrts_get_worker_number();
    int n_workers = __cilkrts_get_nworkers();

    adept::uIndex*__restrict operation_stack_arr =
            worker_local_stacks[stack.worker_id].operation_stack_arr;
    adept::Real*__restrict multiplier_stack_arr =
            worker_local_stacks[stack.worker_id].multiplier_stack_arr;
    const adept::Statement*__restrict statement_stack_arr =
            worker_local_stacks[stack.worker_id].statement_stack_arr;
    float**__restrict statement_stack_deposit_location =
            worker_local_stacks[stack.worker_id].statement_stack_deposit_location;
    int*__restrict statement_stack_deposit_location_len =
            worker_local_stacks[stack.worker_id].statement_stack_deposit_location_len;
    float**__restrict operation_stack_deposit_location =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location;
    bool*__restrict operation_stack_deposit_location_valid =
            worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;

    for (adept::uIndex ist = stack.statement_stack_end;
         ist-- > stack.statement_stack_start; ) {
      const adept::Statement& statement = statement_stack_arr[ist];
      if (statement.index == -1) continue;

      // Extract the gradient for this statement
      // Start by reading the global gradient table
      adept::Real a = global_grad_table[statement.index];
      global_grad_table[statement.index] = 0; 
      if (gradient_use_wl[statement.index]) {
        // Use the worker-local approach
        for (int i = 0; i < n_workers; ++i) {
          a += wl_grad_table[i][statement.index];
          wl_grad_table[i][statement.index] = 0;
        }
      } else {
        // Use the deposit array approach.
        // Fetch the extract_arr containing gradients deposited by operations
        float*__restrict extract_arr = statement_stack_deposit_location[ist];
        int extract_arr_len = statement_stack_deposit_location_len[ist];
        // Extract gradient contributions from deposit array
        for (int i = 0; i < extract_arr_len; ++i) {
          a += extract_arr[i];
          // Note: Not actually necessary with optimization 1, but makes little practical difference
          extract_arr[i] = 0;
        }
      }
      // Iterate over the operations and update the gradients
      if (a != 0.0) {
        for (adept::uIndex iop = statement_stack_arr[ist-1].end_plus_one;
             iop < statement.end_plus_one; ++iop) {
          adept::uIndex op_index = operation_stack_arr[iop];
          adept::Real grad_multiplier = multiplier_stack_arr[iop];
          if (!appears_in_statement[op_index] ||
              !operation_stack_deposit_location_valid[iop] ||
              gradient_use_wl[op_index]) {
            wl_grad_table[wid][op_index] += grad_multiplier * a;
          } else {
            float*__restrict dep = operation_stack_deposit_location[iop];
            if (dep) {
              *dep += grad_multiplier * a;
            } else {
              wl_grad_table[wid][op_index] += grad_multiplier * a;
            }
          }
        }
      }
    }
  }
}

void create_histogram(std::pair<int, int>* blocks, int blocks_size,
                      int nstatements, int mapped_ops_size) {
  // Only create the histogram if it doesn't exist yet
  ifstream test("histogram.txt");
  if (!test) {
    // Sort all the accumulated operation counts
    std::vector<int> accum_num_ops;
    double average = 0.0;
    for (int i = 0; i < blocks_size; ++i) {
      accum_num_ops.push_back(blocks[i].second - blocks[i].first);
      average += 1.0 * (blocks[i].second - blocks[i].first) / blocks_size;
    }
    std::sort(accum_num_ops.begin(), accum_num_ops.end());

    // Now write it to an output file, histogram.txt
    test.close();
    ofstream output_file("histogram.txt");
    output_file << "nstatements: " << nstatements << std::endl;
    output_file << "mapped_ops_size: " << mapped_ops_size << std::endl;
    output_file << "blocks_size: " << blocks_size << std::endl;
    output_file << "average number of accumulated ops: " << average << std::endl;
    int val = 1;
    int num = 0;
    for (auto it = accum_num_ops.begin(); it < accum_num_ops.end(); ++it) {
      if (*it != val) {
        if (num > 0) {
          output_file << val << ":" << num << std::endl;
        }
        val = *it;
        num = 1;
      } else {
        num++;
      }
    }
    output_file << val << ":" << num << std::endl;
    output_file.close();
  } else {
    test.close();
  }
}

void statement_left_first_walk(SP_Node* node, int* gradient_req_ops_map, int ops_per_stmt) {
  // ROOT or SERIAL node: recursively call statement_left_first_walk serially
  if (node->type == 0 || node->type == 1) {
    for (int i = 0; i < node->children->size(); ++i) {
      statement_left_first_walk((*(node->children))[i], gradient_req_ops_map, ops_per_stmt);
    }
  }
  // PARALLEL node: recursively call statement_left_first_walk in parallel
  else if (node->type == 2) {
    for (int i = 0; i < node->children->size(); ++i) {
      cilk_spawn statement_left_first_walk((*(node->children))[i], gradient_req_ops_map, ops_per_stmt);
    }
    cilk_sync;
  }
  // DATA node
  else if (node->type == 3) {
    triple_vector_wl stack = node->data;
    const adept::Statement*__restrict statement_stack_arr =
            worker_local_stacks[stack.worker_id].statement_stack_arr;
    for (adept::uIndex ist = stack.statement_stack_start;
         ist < stack.statement_stack_end; ++ist) {
      const adept::Statement& statement = statement_stack_arr[ist];
      if (statement.index == -1) continue;
      gradient_req_ops_map[statement.index] += ops_per_stmt;
    }
  }
}

uint32_t reduce(uint64_t x, uint32_t N) {
  return ((x & 0xffffffff) * ((uint64_t) N)) >> 32;
}

// TODO: Improve the runtime by looking for some technique that doesn't use
// multiplication. Currently this is slightly worse than the old xorshf
uint32_t hash(uint32_t x, uint32_t min, uint32_t max) {
  const static int y = 1103515242;
  uint64_t t = (uint64_t) x * y;
  t ^= t << 16;
  t ^= t >> 5;
  t ^= t << 1;
  return reduce(t, max-min+1) + min;
}

void hybrid_reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient) {
  int n_workers = __cilkrts_get_nworkers();

  // Identify all gradient indices that appear in statements
  // Optimization 1: operations whose gradient index never appears in a 
  // statement accumulate their gradients using worker local sparse arrays
  r0.start();
  int64_t nstatements = 1;
  for (int i = 0; i < n_workers; ++i) {
    nstatements += worker_local_stacks[i].statement_stack_arr_len;
  }
  bool* appears_in_statement = new bool[n_gradients];
  cilk_for (int i = 0; i < n_gradients; ++i) {
    appears_in_statement[i] = false;
  }
  cilk_for (int i = 0; i < n_workers; ++i) {
    wl_stacks worker_stack = worker_local_stacks[i];
    cilk_for (int j = 0; j < worker_stack.statement_stack_arr_len; ++j) {
      if (worker_stack.statement_stack_arr[j].index >= 0 &&
          !appears_in_statement[worker_stack.statement_stack_arr[j].index]) {
        appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
      }
    }
  }
  r0.stop();

  // Allocate/initialize lsw, lsi
  // Optimization 2: operations whose gradient contributions are accumulated
  // by a statement in the same subtape (data node) use global gradient table.
  r1.start();
  int8_t* last_statement_worker = new int8_t[n_gradients];
  int32_t* last_statement_index = new int32_t[n_gradients];
  cilk_for (uint64_t i = 0; i < n_gradients; ++i) {
    last_statement_worker[i] = -1;
    last_statement_index[i] = -1;
  }
  r1.stop();

  // Reserve worker_local_vector space; misc initialization
  r2.start();
  // We're using a worker_local_vector to accumulate the operations that pass a
  // filter. This is a practical optimization to PARAD that reduces the number
  // of operations that need distinct deposit locations.
  worker_local_vector<OperationReference>wl_ops;
  int64_t op_stack_len = 0;
  for (int i = 0; i < n_workers; ++i) {
    op_stack_len += worker_local_stacks[i].operation_stack_arr_len;
  }
  wl_ops.reserve((op_stack_len*2) / n_workers);
  // Obtain statement offsets for worker-local statement stacks (used later)
  int* statement_offsets = new int[n_workers];
  statement_offsets[0] = 0;
  for (int i = 1; i < n_workers; ++i) {
    statement_offsets[i] = statement_offsets[i-1] + worker_local_stacks[i-1].statement_stack_arr_len;
  }
  r2.stop();

  // Init the deposit location lengths and the deposit-location-valid arrays.
  r3.start();
  cilk_for (int wid = 0; wid < n_workers; ++wid) {
    cilk_for (int64_t i = 0; i < worker_local_stacks[wid].statement_stack_arr_len; ++i) {
      worker_local_stacks[wid].statement_stack_deposit_location_len[i] = 0;
    }
    cilk_for (int64_t i = 0; i < worker_local_stacks[wid].operation_stack_arr_len; ++i) {
      worker_local_stacks[wid].operation_stack_deposit_location_valid[i] = false;
    }
  }
  r3.stop();

  // Initialize gradient_req_ops_map, gradient_use_wl, lfw_use_wl
  r4.start();
  int* gradient_req_ops_map = new int[n_gradients];
  bool* gradient_use_wl = new bool[n_gradients];
  bool* lfw_use_wl = new bool[n_gradients];
  cilk_for (int i = 0; i < n_gradients; ++i) {
    gradient_req_ops_map[i] = 0;
    lfw_use_wl[i] = false;
  }
  const int sampling = 128;
  const int ops_per_stmt = 3;
  r4.stop();

  // Compute gradient_req_ops_map
  r5.start();
  statement_left_first_walk(sptape_root, gradient_req_ops_map, ops_per_stmt);
  r5.stop();

  // First allocate space to store random gradient indices associated with ops
  r6a.start();
  int* op_stack_offsets = new int[n_workers+1];
  op_stack_offsets[0] = 0;
  for (int i = 0; i < n_workers; ++i) {
    int op_stack_len = worker_local_stacks[i].operation_stack_arr_len;
    int n_samples = op_stack_len / sampling;
    op_stack_offsets[i+1] = op_stack_offsets[i] + n_samples;
  }
  int* op_indices = new int[op_stack_offsets[n_workers]];
  r6a.stop();

  // Sample operations from each worker's operation stack to populate op_indices
  r6b.start();
  cilk_for (int i = 0; i < n_workers; ++i) {
    const adept::uIndex*__restrict operation_stack_arr = worker_local_stacks[i].operation_stack_arr;
    int op_stack_len = worker_local_stacks[i].operation_stack_arr_len;
    int n_samples = op_stack_len / sampling;
    cilk_for (int j = 0; j < n_samples; ++j) {
      int rand_index = hash(j, 0, op_stack_len-1);
      op_indices[op_stack_offsets[i] + j] = operation_stack_arr[rand_index];
    }
  }
  r6b.stop();

  // Sort op_indices so that all ops with the same gradient index are contiguous
  r6c.start();
  intSort::iSort(op_indices, op_stack_offsets[n_workers], n_gradients, utils::identityF<int>());
  r6c.stop();

  // Compute boundaries in op_indices
  // This could be optimized further, but isn't usually a bottleneck
  r6d.start();
  int* index_boundaries;
  worker_local_vector<int> wl_index_boundaries;
  cilk_for (int i = 0; i < op_stack_offsets[n_workers]; ++i) {
    if (i == 0 || op_indices[i] != op_indices[i-1]) {
      wl_index_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t index_boundaries_size = wl_index_boundaries.collect(index_boundaries);
  intSort::iSort(index_boundaries, index_boundaries_size, n_gradients, utils::identityF<int>());
  r6d.stop();

  // Aggregate results by decrementing gradient_req_ops_map appropriately
  r6e.start();
  cilk_for (int i = 0; i < index_boundaries_size-1; ++i) {
    int grad_index = op_indices[index_boundaries[i]];
    gradient_req_ops_map[grad_index] -= (index_boundaries[i+1] - index_boundaries[i]);
  }
  gradient_req_ops_map[op_indices[index_boundaries[index_boundaries_size-1]]] -=
          (op_stack_offsets[n_workers] - index_boundaries[index_boundaries_size-1]);
  r6e.stop();

  // Compute gradient_use_wl
  // TODO: Make sure that this provides high probability bounds
  r7.start();
  cilk_for (int i = 0; i < n_gradients; ++i) {
    gradient_use_wl[i] = appears_in_statement[i] && (gradient_req_ops_map[i] < 0);
  }
  r7.stop();

  // 2) Left-first traversal collects ops that need distinct locations
  r8.start();
  args_for_collect_ops args;
  args.idx_in_statement = appears_in_statement;
  args.last_statement_worker = last_statement_worker;
  args.last_statement_index = last_statement_index;
  args.gradient_ = _gradient;
  hybrid_left_first_walk(sptape_root, &args, wl_ops, gradient_use_wl, lfw_use_wl);
  // lfw_use_wl keeps track of all gradient indices which use worker-local
  // gradient tables and are not optimized out in left_first_walk. This is an
  // optimization for right_first_walk
  r8.stop();

  // Collect the wl_ops into a single contiguous array
  r9.start();
  OperationReference* ops;
  int64_t ops_size = wl_ops.collect(ops);
  r9.stop();

  // 3) Create O*: map each operation in ops with index i to (statement_index, i)
  r10.start();
  std::pair<int, int>* mapped_ops =
          (std::pair<int, int>*) malloc(sizeof(std::pair<int, int>) * ops_size);
  int64_t mapped_ops_size = ops_size;
  cilk_for (uint64_t i = 0; i < ops_size; ++i) {
    // Here a global statement index is obtained via the statement offsets we
    // obtained earlier for each worker's statement stack.
    mapped_ops[i] = std::make_pair(statement_offsets[ops[i].statement_wid] +
                                   ops[i].statement_ist, (int) i);
  }
  r10.stop();

  // 4) Semisort the mapped_ops so that all operations associated with the same
  // statement are contiguous in memory
  r11.start();
  intSort::iSort(&mapped_ops[0], mapped_ops_size, nstatements+1, utils::firstF<int, int>());
  r11.stop();

  // 4) Collect boundaries between contiguous blocks of operations with the same
  // statement index
  r12.start();
  int* boundaries;
  worker_local_vector<int> wl_boundaries;
  cilk_for (uint64_t i = 0; i < mapped_ops_size; ++i) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      wl_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t boundaries_size = wl_boundaries.collect(boundaries);
  // Due to using worker local vectors, we need to actually sort this list of
  // boundaries so that they appear in order. Theoretically, we could have done
  // this with a scan followed by a pack, but this usually isn't a bottleneck
  intSort::iSort(&boundaries[0], boundaries_size, mapped_ops_size, utils::identityF<int>());
  r12.stop();

  // 5a) For each contiguous block in mapped_ops, create a block with start and end index
  r13.start();
  std::pair<int, int>* blocks = (std::pair<int, int>*)
          malloc(sizeof(std::pair<int, int>) * boundaries_size);
  int64_t blocks_size = boundaries_size;
  cilk_for (uint64_t i = 1; i < boundaries_size; ++i) {
    blocks[i-1] = std::make_pair(boundaries[i-1], boundaries[i]);
  }
  if (boundaries_size > 0) {
    blocks[boundaries_size-1] = std::make_pair(boundaries[boundaries_size-1],
                                               mapped_ops_size);
  }
  r13.stop();

  // 5b) Allocate / initialize deposit locations
  r14.start();
  float* deposit_locations = (float*) malloc(mapped_ops_size * sizeof(float));
  cilk_for (int64_t i = 0; i < mapped_ops_size; ++i) {
    deposit_locations[i] = 0.0;
  }
  r14.stop();

  // 5b) Populate deposit locations (i.e. O_snd)
  r15.start();
  // Each block is associated with a subarray inside the deposit array
  cilk_for (uint64_t i = 0; i < blocks_size; ++i) {
    // The statement associated with the block is assigned to the range of
    // indices corresponding to the deposit subarray associated with this block
    if (blocks[i].second > blocks[i].first) {
      OperationReference& opref = ops[mapped_ops[blocks[i].first].second];
      if (opref.statement_wid != -1) {
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
      }
    }
    // Assign each operation in the block to distinct locations in the deposit
    // subarray for the block
    cilk_for (uint64_t j = blocks[i].first; j < blocks[i].second; ++j) {
      OperationReference& opref = ops[mapped_ops[j].second];
      if (opref.statement_wid != -1) {
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = deposit_locations + j;
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = true;
      } else {
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = false;
      }
    }
  }
  r15.stop();

  // Allocate worker-local gradient tables. These are used for:
  // - optimization: gradients that are not associated with any statement, i.e.
  //   gradients that will be nonzero after the reverse-mode AD
  // - gradients we decided to accumulate in worker-local tables earlier
  r16.start();
  float** wl_grad_table = (float**) malloc(sizeof(float*) * n_workers);
  cilk_for (int i = 0; i < n_workers; ++i) {
    wl_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  r16.stop();

  // 6) Right-first traversal to compute the gradients
  r17.start();
  hybrid_right_first_walk(sptape_root, wl_grad_table, appears_in_statement,
                          _gradient, lfw_use_wl);
  r17.stop();

  // 7) Export worker-local gradients to the global gradient table
  r18.start();
  cilk_for (int64_t i = 0; i < tfk_reducer.max_gradient; ++i) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; ++wid) {
      _gradient[i] += wl_grad_table[wid][i];
    }
  }
  r18.stop();
  
  // Free all the memory we allocated
  r19.start();
  cilk_for (int i = 0; i < n_workers; ++i) {
    free(wl_grad_table[i]);
  }
  free(wl_grad_table);
  delete[] deposit_locations;
  delete[] last_statement_worker;
  delete[] last_statement_index;
  delete[] statement_offsets;
  delete[] appears_in_statement;
  free(gradient_req_ops_map);
  delete[] lfw_use_wl;
  delete[] gradient_use_wl;
  free(blocks);
  free(boundaries);
  free(ops);
  free(mapped_ops);
  r19.stop();
}

} // end namespace PARAD
