// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk-adept-headers/sp_tree.h>

#include <cilk-adept-source/gradient_table.cpp>

#include <cilk-adept-source/tfk_shadowmem.cpp>

#include <cilk-adept-source/sp_node.cpp>

#include <vector>
#include <map>
#include "../../../common/utils.h"
#include "../../../common/blockRadixSort.h"


#define SPTREE_spawn 
#define SPTREE_parfor for
#define SPTREE_sync


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
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        //assert(statement.index != -1 && "Why is statement index -1?\n");
        if (statement.index == -1) continue;

        // Perform the extraction.
        float* extract_arr = worker_local_stacks[stack.worker_id].statement_stack_deposit_location[ist];
        int extract_arr_len = worker_local_stacks[stack.worker_id].statement_stack_deposit_location_len[ist];
        adept::Real a = gradient_[statement.index];
        gradient_[statement.index] = 0;
        for (int i = 0; i < extract_arr_len; i++) {
        //for (int i = extract_arr_len; i-- > 0; ) {//< extract_arr_len; i++) {
          a += extract_arr[i];
          extract_arr[i] = 0;
        }

        if (a != 0.0) {
           for (adept::uIndex j =
                  worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                  j < statement.end_plus_one; j++) {
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             adept::uIndex operation_stack_index =
                 worker_local_stacks[stack.worker_id].operation_stack_arr[j];
             if (appears_in_statement[operation_stack_index] && worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j] != NULL) {
               *(worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j]) = multiplier_test*a;
               //gradient_[operation_stack_index] += multiplier_test*a;
             } else {
               //gradient_[operation_stack_index] += multiplier_test*a;
               worker_local_grad_table[wid][operation_stack_index] += multiplier_test*a;
             }
           }
       }
     }
    return;
  }

  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    }

  } else if (n->type == 2) {
    //#pragma cilk grainsize 1
    SPTREE_parfor (int j = 0; j < n->children->size(); j++) {
      int i = n->children->size()-j-1;
      walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    }
  }

}




void SP_Tree::collect_ops_for_semisort(SP_Node* n, bool* idx_in_statement, int64_t* last_statement_worker, int64_t* last_statement_index, std::vector<OperationReference>& ops) {
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {

      for (adept::uIndex ist = stack.statement_stack_start; ist < stack.statement_stack_end; ist++) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        if (statement.index == -1) continue;

        for (adept::uIndex j =
             worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
             j < statement.end_plus_one; j++) {
          adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
          if (idx_in_statement[op_index]) {
            OperationReference ref;
            ref.statement_wid = last_statement_worker[op_index];
            ref.statement_ist = last_statement_index[op_index];
            ref.operation_wid = stack.worker_id;
            ref.operation_j = j;
            ref.gradient_index = op_index;
            ops.push_back(ref);
          }
        }

        last_statement_worker[statement.index] = stack.worker_id;
        last_statement_index[statement.index] = ist;


      }


      //for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      //  const adept::Statement& statement =
      //      worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      //  if (statement.index == -1) continue;
      //    last_statement_worker[statement.index] = stack.worker_id;
      //    last_statement_index[statement.index] = ist;

      //}

    }
    return;
  }


  for (int i = 0; i < n->children->size(); i++) {
    //collect_ops_for_semisort((*(n->children))[i], ret);
    collect_ops_for_semisort((*(n->children))[i], idx_in_statement, last_statement_worker, last_statement_index, ops);
  }

}


void SP_Tree::test(int64_t n_gradients, float* _gradient) {




  // First identify all gradient indices that appear in statements.
  bool* appears_in_statement = new bool[n_gradients];
  SPTREE_parfor (int i = 0; i < n_gradients; i++) {
    appears_in_statement[i] = false;
  }

  SPTREE_parfor (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    wl_stacks worker_stack = worker_local_stacks[i];

    SPTREE_parfor (int j = 0; j < worker_stack.statement_stack_arr_len; j++) {
      if (worker_stack.statement_stack_arr[j].index >= 0) {
        appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
      }
    }
  }

  // now do a left first walk of the tree.
  int64_t* last_statement_worker = new int64_t[n_gradients];
  int64_t* last_statement_index = new int64_t[n_gradients];

  for (uint64_t i = 0; i < n_gradients; i++) {
    last_statement_worker[i] = -1;
    last_statement_index[i] = -1;
  }

  std::vector<OperationReference> ops;
  collect_ops_for_semisort(get_root(), appears_in_statement, last_statement_worker, last_statement_index, ops);


  std::vector<std::pair<int64_t, int64_t> > mapped_ops(ops.size());

  // Map for sort.
  for (uint64_t i = 0; i < ops.size(); i++) {
    mapped_ops[i] = std::make_pair(ops[i].gradient_index, i);
  }

  std::sort(mapped_ops.begin(), mapped_ops.end());

  // Now identify blocks.
  std::vector<uint64_t> boundaries;

  for (uint64_t i = 0; i < mapped_ops.size(); i++) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      boundaries.push_back(i);
    }
  }

  std::vector<std::pair<uint64_t, uint64_t> > blocks(boundaries.size());

  for (uint64_t i = 1; i < boundaries.size(); i++) {
    blocks[i-1] = (std::make_pair(boundaries[i-1], boundaries[i]));
  }
  blocks[boundaries.size()-1] = std::make_pair(boundaries[boundaries.size()-1], mapped_ops.size());



  // now augment each worker's statement stack with a pointer to extra data.

  float* deposit_locations = new float[mapped_ops.size()];
  for (uint64_t i = 0; i < mapped_ops.size(); i++) {
    deposit_locations[i] = 0;
  }

  for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
    worker_local_stacks[wid].statement_stack_deposit_location = (float**)
        malloc(sizeof(float*) * worker_local_stacks[wid].statement_stack_arr_len);
    worker_local_stacks[wid].statement_stack_deposit_location_len = (int*)
        calloc(worker_local_stacks[wid].statement_stack_arr_len, sizeof(int));
    worker_local_stacks[wid].operation_stack_deposit_location = (float**)
        malloc(sizeof(float*) * worker_local_stacks[wid].operation_stack_arr_len);
  }


  for (uint64_t i = 0; i < blocks.size(); i++) {
    //printf("block %llu size is %llu\n", i, blocks[i].second-blocks[i].first);
    for (uint64_t j = blocks[i].first; j < blocks[i].second; j++) {
      if (mapped_ops[j].first != mapped_ops[blocks[i].first].first) {
        printf("ERROR!!!!\n "); assert(false);
      }
      OperationReference& opref = ops[mapped_ops[j].second];



      // Map the statement.
      if (opref.statement_wid != -1) {
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = deposit_locations + j;
      } else {
        worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = NULL;
      } /*else {
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = NULL; //deposit_locations + blocks[i].first;
        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = 0; //blocks[i].second - blocks[i].first;
      }*/


      if (blocks[i].second - blocks[i].first <= 0) {
        printf("error block size zero!\n");
        assert(false);
      }
    }

  }


  float** worker_local_grad_table = (float**) malloc(sizeof(float*) * __cilkrts_get_nworkers());
  for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    worker_local_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }


  walk_tree_process_semisort(get_root(), worker_local_grad_table, appears_in_statement, _gradient);

  int n_workers = __cilkrts_get_nworkers();

  for (int64_t i = 0; i < mapped_ops.size(); i++) {
    if(deposit_locations[i] != 0.0) {
      printf("deposit location isn't zero %e, %llu\n", deposit_locations[i], i);
      //assert(false);
    }
  }

  for (int64_t i = 0; i < n_gradients; i++) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; wid++) {
      _gradient[i] += worker_local_grad_table[wid][i];
    }
  }

  for (int i = 0; i < n_workers; i++) {
    free(worker_local_grad_table[i]);
  }
  free(worker_local_grad_table);


  for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
    free(worker_local_stacks[wid].statement_stack_deposit_location);
    free(worker_local_stacks[wid].statement_stack_deposit_location_len);
    free(worker_local_stacks[wid].operation_stack_deposit_location);
  }
  delete[] deposit_locations;
  delete[] last_statement_worker;
  delete[] last_statement_index;
  delete[] appears_in_statement;

  return;
  //int64_t* offsets = new int64_t[__cilkrts_get_nworkers()]();
  //int64_t total_operations = worker_local_stacks[0].operation_stack_arr_len;

  //float* deposit_locations = new float[total_operations];



  //offsets[0] = 0;
  //for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
  //  offsets[i] = offsets[i-1] + worker_local_stacks[i-1].operation_stack_arr_len;
  //  total_operations += worker_local_stacks[i].operation_stack_arr_len;
  //  //worker_local_stacks[i].deposit_index = new float[worker_local_stacks[i].operation_stack_arr_len]();
  //}


  //// associate each operation with a statement.

  ////std::vector<SP_Node*> nodes;
  ////walk_tree_flatten_datanodes(this->get_root(), nodes);


  //std::pair<int, int>* op_pairs = (std::pair<int, int>*) malloc(sizeof(std::pair<int, int >) * total_operations);

  //bool* appears_in_statement = new bool[n_gradients];
  //SPTREE_parfor (int i = 0; i < n_gradients; i++) {
  //  appears_in_statement[i] = false;
  //}

  //SPTREE_parfor (int i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  wl_stacks worker_stack = worker_local_stacks[i];

  //  SPTREE_parfor (int j = 0; j < worker_stack.statement_stack_arr_len; j++) {
  //    if (worker_stack.statement_stack_arr[j].index >= 0) {
  //      appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
  //    }
  //  }
  //}

  //printf("before the copy total operations %llu\n", total_operations);

  //int64_t operations_after_filter = 0;

  //std::vector<std::pair<int64_t, int64_t> >* wl_op_vectors = new std::vector<std::pair<int64_t, int64_t> >[__cilkrts_get_nworkers()]();

  //SPTREE_parfor (int i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  wl_stacks worker_stack = worker_local_stacks[i];
  //  auto wl_op_pairs = op_pairs + offsets[i];

  //  for (int j = 0; j < worker_stack.operation_stack_arr_len; j++) {
  //    if (appears_in_statement[worker_stack.operation_stack_arr[j]]) {
  //      //operations_after_filter++;
  //      wl_op_vectors[__cilkrts_get_worker_number()].push_back(std::make_pair(worker_stack.operation_stack_arr[j], i*total_operations + j));
  //      //wl_op_pairs[j] = std::make_pair(worker_stack.operation_stack_arr[j], i*total_operations + j);
  //    }
  //  }
  //}

  //int64_t* offsets2 = new int64_t[__cilkrts_get_nworkers()];
  //offsets2[0] = 0;
  //int64_t total_size = wl_op_vectors[0].size();
  //for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
  //  offsets2[i] = offsets2[i-1] + wl_op_vectors[i-1].size();
  //  total_size += wl_op_vectors[i].size();
  //}

  //std::vector<std::pair<int64_t, int64_t> > op_vector(total_size);
  //SPTREE_parfor (int64_t i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  SPTREE_parfor (int64_t j = 0; j < wl_op_vectors[i].size(); j++) {
  //    op_vector[offsets2[i] + j] = wl_op_vectors[i][j];
  //  }
  //}
  //delete[] offsets2;
  //delete[] wl_op_vectors;
  //printf("operations after filter %llu\n", op_vector.size());


  ////int64_t count = 0;
  ////for (int i = 0; i < n_gradients; i++) {
  ////  if (appears_in_statement[i]) count++;
  ////}
  ////printf("number of gradients appearing in statement is %d\n", count);

  ////intSort::iSort(op_pairs, total_operations, n_gradients, utils::firstF<int, int>());

  //printf("before the sort total operations %llu, %llu\n", total_operations, op_pairs[0].first);

  //intSort::iSort(&(op_vector[0]), op_vector.size(), n_gradients, utils::firstF<int64_t, int64_t>());
  //printf("after the sort %llu,%llu\n", op_vector[0].first, op_vector[0].second);
  ////// replace with semisort.
  ////std::sort(op_pairs, op_pairs + total_operations);

  ////

  ////printf("done with the sort total operations %llu\n", total_operations);

  ////for (int i = 0; i < total_operations; i++) {
  ////  if (i==0 || op_pairs[i].first != op_pairs[i-1].first) {
  ////    // record this location for the statement.
  ////  }
  ////  worker_local_stacks[op_pairs[i].second.first].deposit_index[op_pairs[i].second.second] = offsets[i] + op_pairs[i].second.second;
  ////}
  //free(op_pairs);
  //delete[] offsets;
  //delete[] deposit_locations;
}


// init can happen at the root of the program, and upon a steal.
// Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
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


//void SP_Tree::walk_tree_count_gradients(SP_Node* n, int* counts) {
//  // If its a data node it must be a terminal node.
//  if (n->type == 3) {
//    triple_vector_wl stack = n->data;
//    if (stack.statement_stack_end != stack.statement_stack_start) {
//
//      for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
//          const adept::Statement& statement =
//              worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
//          if (statement.index == -1) continue;
//          if (ist == stack.statement_stack_start) {
//            for (adept::uIndex j = stack.operation_stack_start;
//                    j < statement.end_plus_one; j++) {
//               adept::uIndex operation_stack_index =
//                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
//               counts[operation_stack_index]++;
//             }
//           } else {
//             for (adept::uIndex j =
//                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
//                    j < statement.end_plus_one; j++) {
//               adept::uIndex operation_stack_index =
//                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
//               counts[operation_stack_index]++;
//             }
//           }
//       }
//    }
//    return;
//  }
//  for (int i = 0; i < n->children->size(); i++) {
//    walk_tree_count_gradients((*(n->children))[i], ret);
//  }
//}


SP_Tree* SP_Tree::transform_to_rootset_form() {
  int n_rootsets = walk_tree_rootset_transform(this->get_root(), 0);
  printf("Number of rootsets is %d\n", n_rootsets);
  std::vector<SP_Node*> data_nodes;
  std::vector<SP_Node*> all_nodes;
  walk_tree_flatten_datanodes(this->get_root(), data_nodes);
  walk_tree_flatten_allnodes(this->get_root(), all_nodes);
  printf("Number of data nodes is %llu, all nodes %llu\n", data_nodes.size(), all_nodes.size());

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
  printf("DEBUG: Open P node should not be called right now without sync id\n");
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


//tfk_gradient_table* SP_Tree::merge_gradient_table_list(
//    std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {
//
//  if (end-start > 4) {
//    int mid = start + (end-start)/2;
//    tfk_gradient_table* left = cilk_spawn merge_gradient_table_list(gradient_table_list, start,
//                                                                    mid);
//    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
//    cilk_sync;
//
//    adept::uIndex* active_entries_right = right->get_active_entries();
//    int n_active_entries_right = right->get_n_active_entries();
//    for (int i = 0; i < n_active_entries_right; i++) {
//      //left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
//      left->accumulate(active_entries_right[i], right->gradient_table_local[active_entries_right[i]]);
//    }
//    return left;
//  }
//
//
//  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];
//
//  for (int j = start+1; j < end; j++) {
//    adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
//    int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();
//
//    for (int i = 0; i < n_active_entries; i++) {
//      //my_gradient_table->accumulate(active_entries[i],
//      //                              gradient_table_list[j]->extract_value(active_entries[i]));
//      my_gradient_table->accumulate(active_entries[i],
//                                    gradient_table_list[j]->gradient_table_local[active_entries[i]]);
//    }
//  }
//
//  return my_gradient_table;
//}

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
         printf("statement %d edges:", statement.index);
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


