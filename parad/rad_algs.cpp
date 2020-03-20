// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

#include "./rad_algs.h"

//#include "./gradient_table.cpp"

//#include "./tfk_shadowmem.cpp"

//#include <cilk-adept-source/sp_node.cpp>

#include <adept.h>

#include <vector>
#include <map>
#include "../common/utils.h"
#include "../common/blockRadixSort.h"
#include "../common/gettime.h"
//#include "../common/semisort.h"

//extern wl_stacks* worker_local_stacks;
//extern tfkdiff tfk_reducer;


namespace PARAD {
timer r0,r1,r2,r3,r4,r5,r5_mid,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17;

void right_first_walk(SP_Node* n, float** worker_local_grad_table, bool* appears_in_statement, float* gradient_) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
       //delete n;
       return;
    }
    int wid = __cilkrts_get_worker_number();

    adept::uIndex*__restrict operation_stack_arr = worker_local_stacks[stack.worker_id].operation_stack_arr;
    adept::Real*__restrict multiplier_stack_arr = worker_local_stacks[stack.worker_id].multiplier_stack_arr;
    const adept::Statement*__restrict statement_stack_arr = worker_local_stacks[stack.worker_id].statement_stack_arr;
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

      // First extract from global gradient table, containing strand-local contributions to
      //   to the gradient, as well as the "input" gradients to reverse-mode AD (e.g. d_loss = 1).
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
        for (adept::uIndex j =
               worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
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

  if (n->type == 1 || n->type == 0) {
    // If a SERIAL node or a ROOT node, then recursively call routine serially.
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

void left_first_walk(SP_Node* n, args_for_collect_ops* args, worker_local_vector<OperationReference>& wl_ops) {
  if (n->type == 3) {
    triple_vector_wl stack = n->data;

    adept::uIndex*__restrict operation_stack_arr = worker_local_stacks[stack.worker_id].operation_stack_arr;
    const adept::Statement*__restrict statement_stack_arr = worker_local_stacks[stack.worker_id].statement_stack_arr;
    float** __restrict operation_stack_deposit_location =
       worker_local_stacks[stack.worker_id].operation_stack_deposit_location;
    bool* __restrict operation_stack_deposit_location_valid =
       worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;
    bool*__restrict idx_in_statement = args->idx_in_statement;

    if (stack.statement_stack_end != stack.statement_stack_start) {
      int wid = __cilkrts_get_worker_number();
      for (adept::uIndex ist = stack.statement_stack_start; ist < stack.statement_stack_end; ist++) {
        const adept::Statement& statement = statement_stack_arr[ist];

        if (statement.index == -1) continue;

        for (adept::uIndex j =
             statement_stack_arr[ist-1].end_plus_one;
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

void report_times() {
  r0.reportTotal("r0: Init appears_in_statement");
  r1.reportTotal("r1: Worker local stacks scan.");
  r2.reportTotal("r2: Allocation of lsw, lsi, lsn");
  r3.reportTotal("r3: Init lsw, lsi, lsn");
  r4.reportTotal("r4: Small loops and reserving space.");
  r5.reportTotal("r5: Init deposit lengths and valid flags.");
  r5_mid.reportTotal("r5 only statement stack");
  r6.reportTotal("r6: collect ops for semisort.");
  r7.reportTotal("r7: wl_ops.collect()");
  r8.reportTotal("r8: map for ops");
  r9.reportTotal("r9: semisort mapped ops.");
  r10.reportTotal("r10: get boundaries, multiple steps.");
  r11.reportTotal("r11: manage creation of blocks.");
  r12.reportTotal("r12: allocate and init deposit locations.");
  r13.reportTotal("r13: populate deposit locations");
  r14.reportTotal("r14: allocate and init worker local grad table.");
  r15.reportTotal("r15: right_first_walk");
  r16.reportTotal("r16: combine wl grad tables.");
  r17.reportTotal("r17: free memory at end.");
}

void reverse_ad(SP_Node* sptape_root, int64_t n_gradients, float* _gradient) {

  r0.start();
  // First identify all gradient indices that appear in statements.
  bool* appears_in_statement = new bool[n_gradients];
  cilk_for (int i = 0; i < n_gradients; i++) {
    appears_in_statement[i] = false;
  }
  r0.stop();

  r1.start();
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
  r1.stop();


  r2.start();
  // Now do a left first walk of the tree.
  int8_t* last_statement_worker = new int8_t[n_gradients];
  int32_t* last_statement_index = new int32_t[n_gradients];
  SP_Node** last_statement_node = new SP_Node*[n_gradients];
  r2.stop();

  r3.start();
  cilk_for (uint64_t i = 0; i < n_gradients; i++) {
    last_statement_worker[i] = -1;
    last_statement_index[i] = -1;
    last_statement_node[i] = NULL;
  }
  r3.stop();

  r4.start();

  OperationReference* ops;
  // We're using a worker_local_vector to accumulate the operations that pass a filter.
  //   This is a practical optimization to PARAD that reduces the number of operations that
  //   need distinct deposit locations.
  worker_local_vector<OperationReference> wl_ops;

  int64_t op_stack_len = 0;
  for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    op_stack_len += worker_local_stacks[i].operation_stack_arr_len;
  }


  // Obtain statement offsets for worker-local statement stacks. This will be used in an upcoming step.
  int* statement_offsets = new int[__cilkrts_get_nworkers()];
  statement_offsets[0] = 0;
  for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
    statement_offsets[i] = statement_offsets[i-1] + worker_local_stacks[i-1].statement_stack_arr_len;
  }

  wl_ops.reserve((op_stack_len*2)/__cilkrts_get_nworkers());
  r4.stop();


  r5.start();

  r5_mid.start();
  // Init the deposit location lengths and the deposit-location-valid arrays.
  cilk_for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
     cilk_for (int64_t i = 0; i < worker_local_stacks[wid].statement_stack_arr_len; i++) {
       worker_local_stacks[wid].statement_stack_deposit_location_len[i] = 0;
     }
     // This valid array of booleans is allocated and used because it is cheaper to access a
     //   a boolean to check for validity than a 8-byte pointer (smaller mem access).
     cilk_for (int64_t i = 0; i < worker_local_stacks[wid].operation_stack_arr_len; i++) {
       worker_local_stacks[wid].operation_stack_deposit_location_valid[i] = 0;
     }
  }
  r5_mid.stop();
  r5.stop();


  r6.start();
  args_for_collect_ops args;
  args.idx_in_statement = appears_in_statement;
  args.last_statement_worker = last_statement_worker;
  args.last_statement_index = last_statement_index;
  args.gradient_ = _gradient;
  // Actually perform the left-first traversal to collect operations that need distinct
  //   deposit locations.
  left_first_walk(sptape_root, &args, wl_ops);
  r6.stop();

  r7.start();
  // Collect the wl_ops into a single contiguous array ops of length ops_size.
  int64_t ops_size = wl_ops.collect(ops);
  r7.stop();

  r8.start();
  // Map each operation in 'ops' with index i to the pair (statement_index, i).
  std::pair<int, int>* mapped_ops = (std::pair<int, int>*) malloc(sizeof(std::pair<int, int>) * ops_size);
  int64_t mapped_ops_size = ops_size;

  // Map for sort.
  cilk_for (uint64_t i = 0; i < ops_size; i++) {
    // Here a global statement index is obtained via the statement offsets we obtained earlier
    //   for each worker's statement stack.
    mapped_ops[i] = std::make_pair(statement_offsets[ops[i].statement_wid] + ops[i].statement_ist,
                                   (int)i);
  }
  r8.stop();

  r9.start();
  // Semisort the mapped_ops so that all operations associated with the same statement are
  //   contiguous in-memory.
  intSort::iSort(&mapped_ops[0], mapped_ops_size, nstatements+1, utils::firstF<int, int>());
  r9.stop();

  r10.start();
  // Collect the boundaries between continguous blocks of operations with same statement index.
  int* boundaries;
  worker_local_vector<int> wl_boundaries;
  //int* wl_boundaries;
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      wl_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t boundaries_size = wl_boundaries.collect(boundaries);
  // Due to using worker local vectors, we need to actually sort this list of boundaries so that
  //   they appear in order. Note that (from theory point of view) we could have done this with a
  //   scan followed by a pack if needed, but this isn't bottleneck usually.
  intSort::iSort(&boundaries[0], boundaries_size, mapped_ops_size, utils::identityF<int>());
  r10.stop();

  r11.start();
  // For each contiguous block in mapped_ops create a block with a start and end index.
  std::pair<int,int>* blocks = (std::pair<int,int>*) malloc(sizeof(std::pair<int,int>)*boundaries_size);
  int64_t blocks_size = boundaries_size;

  cilk_for (uint64_t i = 1; i < boundaries_size; i++) {
    blocks[i-1] = (std::make_pair(boundaries[i-1], boundaries[i]));
  }
  if (boundaries_size > 0) {
    blocks[boundaries_size-1] = std::make_pair(boundaries[boundaries_size-1], mapped_ops_size);
  }
  r11.stop();

  r12.start();
  // Allocate an array of deposit locations.
  float* deposit_locations = new float[mapped_ops_size];
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    deposit_locations[i] = 0;
  }
  r12.stop();

  r13.start();
  // Each block is associated with a subarray inside the deposit array.
  cilk_for (uint64_t i = 0; i < blocks_size; i++) {
    // The statement associated with the block is assigned to the range of indices
    //   corresponding to the deposit subarray associated with this block.
    if (blocks[i].second > blocks[i].first) {
        OperationReference& opref = ops[mapped_ops[blocks[i].first].second];
        if (opref.statement_wid != -1) {
          worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
          worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
        }
    }

    // Assign each operation in the block to distinct locations in the deposit subarray for the block.
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
  r13.stop();


  r14.start();
  // Allocate worker-local gradient tables. These are used for gradients that are not associated
  //   any statement --- i.e. precisely the gradients that will be non-zero after the reverse-mode
  //   AD.
  float** worker_local_grad_table = (float**) malloc(sizeof(float*) * __cilkrts_get_nworkers());
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    worker_local_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  r14.stop();


  r15.start();
  // Perform the right-first traversal to actually compute the gradients.
  right_first_walk(sptape_root, worker_local_grad_table, appears_in_statement, _gradient);
  r15.stop();

  r16.start();
  int n_workers = __cilkrts_get_nworkers();
  int64_t max_gradient = tfk_reducer.max_gradient;
  // Accumulate the gradients. Should technically use sparse arrays here,
  //   but this is presently not a common bottleneck.
  cilk_for (int64_t i = 0; i < max_gradient; i++) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; wid++) {
      _gradient[i] += worker_local_grad_table[wid][i];
    }
  }
  r16.stop();

  r17.start();
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
  r17.stop();
  return;
}

} // end namespace rad_algs

