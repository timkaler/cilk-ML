// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>

#include "./sp_tree.hpp"

#include "./gradient_table.cpp"

//#include "./tfk_shadowmem.cpp"
//#include <cilk-adept-source/sp_node.cpp>

#include <adept.h>

#include <vector>
#include <map>
#include "../common/utils.h"
#include "../common/blockRadixSort.h"
#include "../common/gettime.h"
//#include "../common/semisort.h"

#define SPTREE_spawn cilk_spawn
#define SPTREE_parfor cilk_for
#define SPTREE_sync cilk_sync

extern wl_stacks* worker_local_stacks;
extern tfkdiff tfk_reducer;

// These timers are used to benchmark various parts of the code.
timer r0,r1,r2,r3,r4,r5,r5_mid,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17;

void SP_Tree::walk_tree_process_semisort(SP_Node* n, float** worker_local_grad_table, bool* appears_in_statement, float* gradient_) {
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
      walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    }
  } else if (n->type == 2) {
    // If a PARALLEL node, then recursively call routine in-parallel.
    for (int j = 0; j < n->children->size(); j++) {
      cilk_spawn walk_tree_process_semisort((*(n->children))[n->children->size()-j-1], worker_local_grad_table, appears_in_statement, gradient_);
    }
    cilk_sync;
  }
}

void SP_Tree::collect_ops_for_semisort(SP_Node* n, args_for_collect_ops* args, worker_local_vector<OperationReference>& wl_ops) {
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
      cilk_spawn collect_ops_for_semisort((*(n->children))[i], args, wl_ops);
    }
    cilk_sync;
  } else {
    for (int i = 0; i < n->children->size(); i++) {
      collect_ops_for_semisort((*(n->children))[i], args, wl_ops);
    }
  }
}

void SP_Tree::report_times() {
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
  r15.reportTotal("r15: walk_tree_process_semisort");
  r16.reportTotal("r16: combine wl grad tables.");
  r17.reportTotal("r17: free memory at end.");
}

void SP_Tree::reverse_ad_PARAD(int64_t n_gradients, float* _gradient) {
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
  // We're using a worker_local_vector to accumulate the operations that pass a
  // filter. This is a practical optimization to PARAD that reduces the number
  // of operations that need distinct deposit locations.
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
    // This valid array of booleans is allocated and used because it is cheaper
    // to access a boolean to check for validity than a 8-byte pointer
    // (smaller mem access).
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
  // Actually perform the left-first traversal to collect operations that need
  // distinct deposit locations.
  collect_ops_for_semisort(get_root(), &args, wl_ops);
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
    // Here a global statement index is obtained via the statement offsets we
    // obtained earlier for each worker's statement stack.
    mapped_ops[i] = std::make_pair(statement_offsets[ops[i].statement_wid] + ops[i].statement_ist, (int)i);
  }
  r8.stop();

  r9.start();
  // Semisort the mapped_ops so that all operations associated with the same
  // statement are contiguous in-memory.
  intSort::iSort(&mapped_ops[0], mapped_ops_size, nstatements+1, utils::firstF<int, int>());
  r9.stop();

  r10.start();
  // Collect the boundaries between continguous blocks of operations with same
  // statement index.
  int* boundaries;
  worker_local_vector<int> wl_boundaries;
  // int* wl_boundaries;
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      wl_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t boundaries_size = wl_boundaries.collect(boundaries);
  // Due to using worker local vectors, we need to actually sort this list of
  // boundaries so that they appear in order. Note that (from theory point of
  // view) we could have done this with a scan followed by a pack if needed,
  // but this isn't bottleneck usually.
  intSort::iSort(&boundaries[0], boundaries_size, mapped_ops_size, utils::identityF<int>());
  r10.stop();

  r11.start();
  // For each contiguous block in mapped_ops create a block with a start and end index.
  std::pair<int,int>* blocks = (std::pair<int,int>*) malloc(sizeof(std::pair<int,int>)*boundaries_size);
  int64_t blocks_size = boundaries_size;
  cilk_for (uint64_t i = 1; i < boundaries_size; i++) {
    blocks[i-1] = (std::make_pair(boundaries[i-1], boundaries[i]));
  }
  blocks[boundaries_size-1] = std::make_pair(boundaries[boundaries_size-1], mapped_ops_size);
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
  r13.stop();

  r14.start();
  // Allocate worker-local gradient tables. These are used for gradients that
  // are not associated any statement -- i.e. precisely the gradients that will
  // be non-zero after the reverse-mode AD.
  float** worker_local_grad_table = (float**) malloc(sizeof(float*) * __cilkrts_get_nworkers());
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    worker_local_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  r14.stop();

  r15.start();
  // Perform the right-first traversal to actually compute the gradients.
  walk_tree_process_semisort(get_root(), worker_local_grad_table, appears_in_statement, _gradient);
  r15.stop();

  r16.start();
  int n_workers = __cilkrts_get_nworkers();
  int64_t max_gradient = tfk_reducer.max_gradient;
  // Accumulate the gradients. Should technically use sparse arrays here,
  // but this is presently not a common bottleneck.
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

// init can happen at the root of the program, and upon a steal. Upon a steal: 
// a continuation was stolen. Upon a sync the parent node ought to be a P node.
void SP_Tree::init() {
  SP_Node*& current_node = imp_.view();
  //current_node = get_root();
  current_node->type = 1;
  current_node->parent = NULL;
  recording = false;
  if (current_node->children != NULL) {
    //#pragma cilk grainsize 1
    SPTREE_parfor (int i = 0; i < current_node->children->size(); i++) {
      delete (*(current_node->children))[i];
    }
    delete current_node->children;
  }
  current_node->children = new std::vector<SP_Node*>();
  recording = true;
}

SP_Node* SP_Tree::get_root() {
  SP_Node* current_node = imp_.view();
  while (current_node->parent != NULL) {
    current_node = current_node->parent;
  }
  return current_node;
}

// currently has a memory leak.
void SP_Tree::clear() {
  //bool saved_recording = recording; 
  //recording = false;
  this->init();
  //recording = saved_recording;
}

int SP_Tree::walk_tree_rootset_transform(SP_Node* n, int dep_count) {
  // Data node.
  if (n->type == 3) {
    n->rootset_id = dep_count;
    assert(n->children->size() == 0 && "A data node should not have any children.\n");
    //printf("D node dep_count is %d\n", dep_count);
    return dep_count;
  }
  n->rootset_id = 0;
  // Series node.
  if (n->type == 1 || n->type == 0) {
    int number_of_data_nodes = 0;
    int added_dep_count = 0;
    for (int i = n->children->size()-1; i >= 0; i--) {
      //printf("the node type is %d\n", (*n->children)[i]->type);
      dep_count = walk_tree_rootset_transform((*n->children)[i], dep_count);
      if ((*n->children)[i]->type == 3) {
        n->rootset_id = 1;
        number_of_data_nodes++;
      }

      if ((*n->children)[i]->type == 3 || (*n->children)[i]->rootset_id == 1) {
        dep_count += 1;
        added_dep_count++;
      }
    }
    if (n->children->size() > 0 && added_dep_count > 0) dep_count--;

    //for (int i = n->children->size()-1; i >= 0; i--) {
    //  printf("the node type is %d\n", (*n->children)[i]->type);
    //}
    //printf("children %d num data nodes %d\n", n->children->size(), number_of_data_nodes);
    //if (number_of_data_nodes == 1) {
    //  dep_count--; 
    //} else if (number_of_data_nodes != 0) printf();
    //printf("S node dep_count is %d\n", dep_count);
    return dep_count;
  }

  if (n->type == 2) {
    int max_child_dep_count = dep_count;
    for (int i = 0; i < n->children->size(); i++) {
      int child_dep_count = walk_tree_rootset_transform((*n->children)[i], dep_count);
      if (child_dep_count > max_child_dep_count) max_child_dep_count = child_dep_count;
      if ((*n->children)[i]->type == 3 || (*n->children)[i]->rootset_id == 1) {
        n->rootset_id = 1;
      }
    }
    //printf("P node dep_count is %d\n", max_child_dep_count);
    return max_child_dep_count; // returned dep_count is the maximum of all child dependence counts.
  }
  assert(false && "Illegal fall through.\n");
  return 0;
}

void SP_Tree::walk_tree_flatten_allnodes(SP_Node* n, std::vector<SP_Node*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {
      ret.push_back(n);
    }
    return;
  }
  ret.push_back(n);
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten_allnodes((*(n->children))[i], ret);
  }
}

void SP_Tree::walk_tree_flatten_datanodes(SP_Node* n, std::vector<SP_Node*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {
      ret.push_back(n);
    }
    return;
  }
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten_datanodes((*(n->children))[i], ret);
  }
}

void SP_Tree::make_ids_deterministic(int64_t n_gradients) {
  std::vector<SP_Node*> data_nodes;
  walk_tree_flatten_datanodes(get_root(), data_nodes);

  int64_t* remap = new int64_t[n_gradients];
  int64_t next_id = 0;
  for (int i = 0; i < n_gradients; i++) {
    remap[i] = -1;
  }
  for (int i = 0; i < data_nodes.size(); i++) {
    triple_vector_wl stack = data_nodes[i]->data;
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      const adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      if (statement.index == -1) continue;
      if (remap[statement.index] == -1) {
        remap[statement.index] = next_id++;
      }
      if (ist == stack.statement_stack_start) {
        for (adept::uIndex j = stack.operation_stack_start;
             j < statement.end_plus_one; j++) {
          adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
          if (remap[op_index] == -1) remap[op_index] = next_id++;
        }
      } else {
        for (adept::uIndex j =
               worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
               j < statement.end_plus_one; j++) {
          adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
          if (remap[op_index] == -1) remap[op_index] = next_id++;
        }
      }
    }
  }
  for (int i = 0; i < data_nodes.size(); i++) {
    triple_vector_wl stack = data_nodes[i]->data;
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      if (statement.index == -1) continue;
      statement.index = remap[statement.index];
      if (ist == stack.statement_stack_start) {
        for (adept::uIndex j = stack.operation_stack_start;
             j < statement.end_plus_one; j++) {
          worker_local_stacks[stack.worker_id].operation_stack_arr[j] = remap[worker_local_stacks[stack.worker_id].operation_stack_arr[j]];
        }
      } else {
        for (adept::uIndex j =
               worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
               j < statement.end_plus_one; j++) {
          worker_local_stacks[stack.worker_id].operation_stack_arr[j] = remap[worker_local_stacks[stack.worker_id].operation_stack_arr[j]];
        }
      }
    }
  }
  delete[] remap;
}

/*
void SP_Tree::walk_tree_count_gradients(SP_Node* n, int* counts) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {

      for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
          const adept::Statement& statement =
              worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
          if (statement.index == -1) continue;
          if (ist == stack.statement_stack_start) {
            for (adept::uIndex j = stack.operation_stack_start;
                    j < statement.end_plus_one; j++) {
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               counts[operation_stack_index]++;
             }
           } else {
             for (adept::uIndex j =
                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               counts[operation_stack_index]++;
             }
           }
       }
    }
    return;
  }
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_count_gradients((*(n->children))[i], ret);
  }
}
*/

SP_Tree* SP_Tree::transform_to_rootset_form() {
  int n_rootsets = walk_tree_rootset_transform(this->get_root(), 0);
  std::vector<SP_Node*> data_nodes;
  std::vector<SP_Node*> all_nodes;
  walk_tree_flatten_datanodes(this->get_root(), data_nodes);
  walk_tree_flatten_allnodes(this->get_root(), all_nodes);

  std::map<int, std::vector<SP_Node*> > rootset_to_nodes;

  int max_rootset_id = 0;
  for (int i = 0; i < data_nodes.size(); i++) {
    rootset_to_nodes[data_nodes[i]->rootset_id].push_back(data_nodes[i]);
    if (data_nodes[i]->rootset_id > max_rootset_id) max_rootset_id = data_nodes[i]->rootset_id;
  }

  SP_Tree* new_tree = new SP_Tree();
  new_tree->init();

  SP_Node* new_root = new_tree->get_root();

  //new_root->children->push_back(new SP_Node(1, new_root, 0));

  //new_tree->open_S_node();
  for (int i = max_rootset_id; i >= 0; i--) {
    //new_tree->open_P_node((void*)(i+1));
    SP_Node* P_node = new SP_Node(2, new_root, 0);
    for (int j = 0; j < rootset_to_nodes[i].size(); j++) {
      SP_Node* S_node = new SP_Node(1, P_node);
      P_node->children->push_back(S_node);
      S_node->children->push_back(rootset_to_nodes[i][j]);
      //new_tree->add_D_node(rootset_to_nodes[i][j]->data);
    }
    new_root->children->push_back(P_node);
    //new_tree->close_P_node();
  }
  //new_tree->close_S_node();
  return new_tree;
}

void SP_Tree::add_D_node(triple_vector_wl data) {
  if (!recording) return;
  SP_Node* data_node = new SP_Node(data);
  SP_Node* current_node = imp_.view();
  current_node->children->push_back(data_node);
}

void SP_Tree::open_P_node(void* sync_id) {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();

  SP_Node* new_node = new SP_Node(2, current_node, sync_id);
  current_node->children->push_back(new_node);
  current_node = new_node;
}

void SP_Tree::open_P_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();

  if (current_node == NULL) printf("Error current node is null in open P node\n");

  SP_Node* new_node = new SP_Node(2, current_node);
  current_node->children->push_back(new_node);

  current_node = new_node;
}

void SP_Tree::close_P_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close P node\n");

  // pop up.
  SP_Node* parent = current_node->parent;
  current_node = parent;
}

void SP_Tree::sync_P_nodes(void* sync_id) {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close P node\n");

  // we need to walk up the tree to get the outer-most-nested P node to close.
  std::vector<SP_Node*> ancestors;

  SP_Node* parent = current_node;

  int num_closes_needed = 0;
  int num_closes = 0;
  while (parent != NULL) {
    num_closes++;
    if (parent->type == 2 && parent->sync_id == sync_id) {
      parent->sync_id = NULL;
      num_closes_needed = num_closes;
      //break;
    }
    if (parent->type == 0) parent->sync_id = sync_id;
    parent = parent->parent;
  }

  for (int i = 0; i < num_closes_needed; i++) {
    //printf("Close %p num_closes_needed %d\n", sync_id, num_closes_needed);
    close_P_node();
  }
}

void SP_Tree::open_S_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in open S node\n");
  SP_Node* new_node = new SP_Node(1, current_node);
  current_node->children->push_back(new_node);
  current_node = new_node;
}

void SP_Tree::close_S_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close S node\n");

  // pop up.
  SP_Node* parent = current_node->parent;
  current_node = parent;
}

std::vector<triple_vector_wl*> SP_Tree::flatten_to_array() {
  std::vector<triple_vector_wl*> ret(0);
  this->walk_tree_flatten(this->get_root(), ret);
  return ret;
}

void SP_Tree::walk_tree_flatten(SP_Node* n, std::vector<triple_vector_wl*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    ret.push_back(&(n->data));
    return;
  }
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten((*(n->children))[i], ret);
  }
}

tfk_gradient_table* SP_Tree::merge_gradient_table_list(
    std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {
  if (gradient_table_list.size() == 1) return gradient_table_list[0];
  if (end-start >  4) {
    int mid = start + (end-start)/2;
    tfk_gradient_table* left = SPTREE_spawn merge_gradient_table_list(gradient_table_list, start,
                                                                    mid);
    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
    SPTREE_sync;

    left->merge_into_me(right);
    return left;
    //adept::uIndex* active_entries_right = right->get_active_entries();
    //int n_active_entries_right = right->get_n_active_entries();
    //for (int i = 0; i < n_active_entries_right; i++) {
    //  //left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
    //  left->accumulate(active_entries_right[i], right->gradient_table_local[active_entries_right[i]]);
    //}
    //return left;
  }
  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];
  for (int j = start+1; j < end; j++) {
    my_gradient_table->merge_into_me(gradient_table_list[j]);
  }
  return my_gradient_table;
}

/*
tfk_gradient_table* SP_Tree::merge_gradient_table_list(
    std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {

  if (end-start > 4) {
    int mid = start + (end-start)/2;
    tfk_gradient_table* left = cilk_spawn merge_gradient_table_list(gradient_table_list, start,
                                                                    mid);
    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
    cilk_sync;

    adept::uIndex* active_entries_right = right->get_active_entries();
    int n_active_entries_right = right->get_n_active_entries();
    for (int i = 0; i < n_active_entries_right; i++) {
      //left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
      left->accumulate(active_entries_right[i], right->gradient_table_local[active_entries_right[i]]);
    }
    return left;
  }

  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];

  for (int j = start+1; j < end; j++) {
    adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
    int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();

    for (int i = 0; i < n_active_entries; i++) {
      //my_gradient_table->accumulate(active_entries[i],
      //                              gradient_table_list[j]->extract_value(active_entries[i]));
      my_gradient_table->accumulate(active_entries[i],
                                    gradient_table_list[j]->gradient_table_local[active_entries[i]]);
    }
  }

  return my_gradient_table;
}
*/

// need to disable recording when walking over the tree.
void SP_Tree::set_recording(bool recording_) {
  this->recording = recording_;
}

void SP_Tree::walk_tree_process_one_worker(float* gradient_table) {
  std::vector<SP_Node*> ret;
  walk_tree_flatten_datanodes(get_root(), ret);
  //FILE* f = fopen("process.debug", "a");
  adept::uIndex count_ist = -1;
  for (int i = ret.size(); i-- > 0;) {
    triple_vector_wl stack = ret[i]->data;
    //for (adept::uIndex ist = worker_local_stacks[stack.worker_id].statement_stack_arr_len; ist-- > 0;) {
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      count_ist++;
      //fprintf(f, "ist %llu\n", count_ist);
      const adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      adept::uIndex idx = statement.index;
      if (idx == (adept::uIndex) -1) continue;
      float a = gradient_table[idx];
      //fprintf(f, "ist %llu float a=%e index %llu\n", count_ist, a, idx);
      gradient_table[idx] = 0.0;

      if (a != 0.0f) {
         adept::uIndex count = 0;
         for (adept::uIndex j =
                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               adept::Real multiplier_test =
                   worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               gradient_table[operation_stack_index] += multiplier_test*a;
               //fprintf(f, "ist %llu j=%llu mul=%e a=%f into idx %llu\n", count_ist, count++, multiplier_test, a, operation_stack_index);
          }
      }
    }
  }
  //fclose(f);
  //for (adept::uIndex ist = worker_local_stacks[0].statement_stack_arr_len; ist-- > 0;) {
  //  const adept::Statement& statement = worker_local_stacks[0].statement_stack_arr[ist];
  //  adept::uIndex idx = statement.index;
  //  if (idx == -1) continue;
  //  float a = gradient_table[idx];
  //  gradient_table[idx] = 0;

  //  if (a != 0.0) {
  //     for (adept::uIndex j =
  //                worker_local_stacks[0].statement_stack_arr[ist-1].end_plus_one;
  //                j < statement.end_plus_one; j++) {
  //           adept::Real multiplier_test =
  //               worker_local_stacks[0].multiplier_stack_arr[j];
  //           adept::uIndex operation_stack_index =
  //               worker_local_stacks[0].operation_stack_arr[j];
  //           gradient_table[operation_stack_index] += multiplier_test*a;
  //      }
  //  }
  //}
}

tfk_gradient_table* SP_Tree::walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table,
                                uint64_t n_gradients) {
  //printf("walk tree process\n");
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
       //delete n;
       return my_gradient_table;
    }
    //int wid = __cilkrts_get_worker_number();

    //if (my_gradient_table->dense_rep != NULL) {
    //  printf("Dense representation is being used for %d statements\n", stack.statement_stack_end-stack.statement_stack_start);
    //}
    //printf("statement stack end is %d, start is %d\n", stack.statement_stack_end, stack.statement_stack_start);
    //printf("new node\n");
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        //assert(statement.index != -1 && "Why is statement index -1?\n");
        if (statement.index == -1) continue;
        int op_count = 0;
        //printf("new statement for index %d\n", statement.index);
        adept::Real a = my_gradient_table->extract_value(statement.index);
        //if (stack.statement_stack_end - stack.statement_stack_start == 450758-449756) {
        //printf("ist is %d, areal is %f, statement index is %d\n", ist, a, statement.index);
        //}
        if (a != 0.0) {
         #ifdef TFK_DEBUG_PRINTS
         #endif
         if (ist == 0/*stack.statement_stack_start*/ && false) {
           for (adept::uIndex j = stack.operation_stack_start;
                  j < statement.end_plus_one; j++) {
             op_count++;
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             if (multiplier_test == 0.112143141) printf("test\n");
             adept::uIndex operation_stack_index =
                 worker_local_stacks[stack.worker_id].operation_stack_arr[j];
             my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);

             #ifdef TFK_DEBUG_PRINTS
             printf("%d,", operation_stack_index);
             #endif
           }
           #ifdef TFK_DEBUG_PRINTS
           printf("; ist %d start %d end %d\n", ist, stack.statement_stack_end,
                  stack.statement_stack_start);
           #endif
         } else {
           for (adept::uIndex j =
                  worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                  j < statement.end_plus_one; j++) {
             op_count++;
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             if (multiplier_test == 0.112143141) printf("test\n");
             adept::uIndex operation_stack_index =
                 worker_local_stacks[stack.worker_id].operation_stack_arr[j];

             my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);

             #ifdef TFK_DEBUG_PRINTS
             printf("%d,", operation_stack_index);
             #endif
           }
           #ifdef TFK_DEBUG_PRINTS
           printf(": ist %d start %d end %d\n", ist, stack.statement_stack_end,
                  stack.statement_stack_start);
           #endif
         }
       }
     }
    //delete n;
    return my_gradient_table;
  }
  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      if (n->type == 0) {
      my_gradient_table = walk_tree_process((*(n->children))[i], my_gradient_table, n_gradients);
      } else {
      my_gradient_table = walk_tree_process((*(n->children))[i], my_gradient_table, n_gradients);
      }
    }
  } else if (n->type == 2) {
    std::vector<tfk_gradient_table*> gradient_table_list;
    //for (int i = n->children.size()-1; i >= 0; i--) {
    //  gradient_table_list.push_back(NULL);//new tfk_gradient_table(n_gradients, my_gradient_table));
    //}

    //#pragma cilk grainsize 1

    int* wids = (int*) malloc(sizeof(int)*n->children->size()+8);
   
    tfk_gradient_table** tables = (tfk_gradient_table**) malloc(sizeof(tfk_gradient_table*)*n->children->size() + 8);
 
    for (int i = 0; i < n->children->size(); i++) {
      wids[i] = -1;
      tables[i] = NULL;
    }

    //#pragma cilk grainsize 1
    SPTREE_parfor (int j = 0; j < n->children->size(); j++) {
      int i = n->children->size()-j-1;
      tfk_gradient_table* table;
      if (i == 0) {
        tables[i] = my_gradient_table;
      }
      else if (i > 0 && /*&& wids[i-1] == __cilkrts_get_worker_number() wids[i-1] != -1*/ false) {
        tables[i] = tables[i-1];
      } else {
        table = new tfk_gradient_table(n_gradients, my_gradient_table);
        tables[i] = table;
      }
      tables[i] = walk_tree_process((*(n->children))[i], /*gradient_table_list[i]*/tables[i], n_gradients);
      wids[i] = __cilkrts_get_worker_number();
    }

    for (int i = 0; i < n->children->size(); i++) {
      if (i == 0 || tables[i] != tables[i-1]) {
        if (tables[i] != NULL) {
          gradient_table_list.push_back(tables[i]);
        }
      }
    }

    //if (gradient_table_list.size() == 0) return my_gradient_table;
    //printf("gradient table list len %d\n", gradient_table_list.size());
    
    tfk_gradient_table* merged_table = my_gradient_table;
    if (gradient_table_list.size() > 0) merged_table = merge_gradient_table_list(gradient_table_list, 0,
                                                                 gradient_table_list.size());

    //adept::uIndex* active_entries = merged_table->get_active_entries();
    //int64_t n_active_entries = merged_table->get_n_active_entries();

    //adept::uIndex* my_active_entries = my_gradient_table->get_active_entries();
    //int64_t my_n_active_entries = my_gradient_table->get_n_active_entries();

    //if (my_n_active_entries == 0 && my_gradient_table->raw_gradient_table != NULL && false) {
    //  //printf("my entries zero, others is %d\n", n_active_entries);
    //  //merged_table->gradient_table = my_gradient_table->gradient_table;
    //  //my_gradient_table->active_entries = NULL;
    //  //my_gradient_table->gradient_table_local = merged_table->gradient_table_local;

    //  for (int i = n->children.size()-1; i >= 0; i--) {
    //      if (gradient_table_list[i] != merged_table) {
    //        delete gradient_table_list[i];
    //      }
    //  }

    //  //merged_table->gradient_table = my_gradient_table;
    //  merged_table->n_active_entries = 0;
    //  //merged_table->raw_gradient_table = my_gradient_table->raw_gradient_table;
    //  //free(active_entries);
    //  return merged_table;
    //  //delete my_gradient_table;
    //  //my_gradient_table = merged_table;
    //  //my_gradient_table->active_entries = NULL;
    //  //return my_gradient_table;
    //} /*else {
    //  my_gradient_table->n_active_entries = 0;
    //}*/

    free(wids);// = (int*) malloc(sizeof(int)*n->children.size()+8);
    free(tables);// = (tfk_gradient_table**) malloc(sizeof(tfk_gradient_table*)*n->children.size() + 8);

    if (my_gradient_table->gradient_table_local.size() == 0 && my_gradient_table->raw_gradient_table == NULL && false) {
      //my_gradient_table->gradient_table_local = merged_table->gradient_table_local;
      merged_table->n_active_entries = 0;
      free(merged_table->active_entries);
      merged_table->gradient_table = my_gradient_table;
        for (int i = gradient_table_list.size()-1; i >= 0; i--) {
          if (gradient_table_list[i] != merged_table) {
            delete gradient_table_list[i];
          }
        }
      //delete n;
      return merged_table;
    } else {
      assert(merged_table == my_gradient_table && "merged table must be the merged gradient table.\n");
      //if (merged_table != my_gradient_table) {
      //  my_gradient_table->merge_into_me(merged_table);
      //}
      //for (int i = 0; i < n_active_entries; i++) {
      //  my_gradient_table->accumulate(active_entries[i],
      //                                merged_table->gradient_table_local[active_entries[i]]);
      //}
    }

    for (int i = gradient_table_list.size()-1; i >= 0; i--) {
      if (my_gradient_table != gradient_table_list[i] && gradient_table_list[i] != NULL) {
        delete gradient_table_list[i];
      }
    }
    //delete n;
    return my_gradient_table; 
  } else {
    printf("Odd error with node types in SP_Tree during reverse-pass processing.\n");
  }
  //delete n;
  return my_gradient_table;
}

void SP_Tree::walk_tree_debug(SP_Node* n, int nest_depth,FILE* f) {
  for (int i = 0; i < nest_depth; i++) {
    fprintf(f, "  ");
  }
  nest_depth += 1;
  if (n->type == 1) {
    fprintf(f, "(S:\n");
  } else if (n->type == 2) {
    fprintf(f, "(P:\n");
  }
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    fprintf(f, "D\n");
    return;
  }
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_debug((*(n->children))[i], nest_depth, f);
  }
  for (int i = 0; i < nest_depth-1; i++) {
    fprintf(f, "  ");
  }
  fprintf(f, ")\n");
}

void SP_Tree::walk_tree_debug(SP_Node* n) {
  FILE* f = fopen("sptree.debug", "a");
  walk_tree_debug(n, 0, f);
  fclose(f);
  return;
  //return walk_tree_debug(n, 0,f);

  //if (n->type == 1) {
  //  printf("(S:");
  //} else if (n->type == 2) {
  //  printf("(P:");
  //}

  //// If its a data node it must be a terminal node.
  //if (n->type == 3) {
  //  printf("D");
  //  return;
  //}


  //for (int i = 0; i < n->children.size(); i++) {
  //  walk_tree_debug(n->children[i]);
  //}
  //printf(")");
}

void SP_Tree::walk_tree_process_locks(SP_Node* n, float* gradient_, int64_t* locks) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
      //delete n;
      return;
    }
    int wid = __cilkrts_get_worker_number();
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      const adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      //assert(statement.index != -1 && "Why is statement index -1?\n");
      if (statement.index == -1) continue;

      // acquire lock.
      {
        bool succ = false;
        do {
          succ = __sync_bool_compare_and_swap(&locks[statement.index], 0, 1);
        } while (!succ || locks[statement.index] == 0);
      }

      adept::Real a = gradient_[statement.index];
      gradient_[statement.index] = 0;
      locks[statement.index] = 0; // release lock.

      if (a != 0.0) {
        for (adept::uIndex j =
               worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
               j < statement.end_plus_one; j++) {
          adept::Real multiplier_test =
              worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
          adept::uIndex operation_stack_index =
              worker_local_stacks[stack.worker_id].operation_stack_arr[j];
          // acquire lock.
          {
            bool succ = false;
            do {
              succ = __sync_bool_compare_and_swap(&locks[operation_stack_index], 0, 1);
            } while (!succ || locks[operation_stack_index] == 0);
          }
          gradient_[operation_stack_index] += multiplier_test*a;
          locks[operation_stack_index] = 0; // Release lock.
        }
      }
    }
    return;
  }
  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      walk_tree_process_locks((*(n->children))[i], gradient_, locks);
    }
  } else if (n->type == 2) {
    //if (n->children->size() < 8) {
    for (int j = 0; j < n->children->size(); j++) {
      cilk_spawn walk_tree_process_locks((*(n->children))[n->children->size()-j-1], gradient_, locks);
    }
    cilk_sync;
    //}
    //#pragma cilk grainsize 1
    //cilk_for (int j = 0; j < n->children->size(); j++) {
    //  int i = n->children->size()-j-1;
    //  walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    //}
  }
}

// SP_Node Class
SP_Node::SP_Node(triple_vector_wl data_) {
  data = data_;
  children = new std::vector<SP_Node*>();
  type = 3;
  sync_id = NULL;
}

// nx1 = nxm ** mx1
// assumes nxm is static weight matrix.
//SP_Node::SP_Node(aMatrix& left, aMatrix& right, aMatrix& ans) {
//  /*
//    for each of n statements.
//             for statement i
//  */
//}

SP_Node::~SP_Node() {
  if (children != NULL ) {
    //#pragma cilk
    //#pragma cilk grainsize 1
    for (int i = 0; i < children->size(); i++) {
      delete (*children)[i];
    }
    delete children;
  }
  //children = new std::vector<SP_Node*>();
}

SP_Node::SP_Node(int type_, SP_Node* parent_) {
  if (type_ == 2) printf("DEBUG: This should not be called for P nodes.\n");
  type = type_;
  parent = parent_;
  children = new std::vector<SP_Node*>();
  sync_id = NULL;
}

SP_Node::SP_Node(int type_, SP_Node* parent_, void* sync_id_) {
  if (type_ != 2) printf("DEBUG: Error this should only be called for P nodes.\n");
  type = type_;
  parent = parent_;
  children = new std::vector<SP_Node*>();
  sync_id = sync_id_;
}
