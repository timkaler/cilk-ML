// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk-adept-headers/sp_tree.h>

#include <cilk-adept-source/gradient_table.cpp>

#include <cilk-adept-source/tfk_shadowmem.cpp>

#include <cilk-adept-source/sp_node.cpp>

#include <vector>

// init can happen at the root of the program, and upon a steal.
// Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
void SP_Tree::init() {
  SP_Node*& current_node = imp_.view();
  current_node->type = 1;
  current_node->parent = NULL;
  recording = false;
  if (current_node->children != NULL) {
    //#pragma cilk grainsize 1
    cilk_for (int i = 0; i < current_node->children->size(); i++) {
      delete (*(current_node->children))[i];
    }
    delete current_node->children;
  }
  current_node->children = new std::vector<SP_Node*>();
  recording = true;
}

SP_Node* SP_Tree::get_root() {
  SP_Node* current_node = imp_.view();
  return current_node;
}

// currently has a memory leak.
void SP_Tree::clear() {
  //bool saved_recording = recording; 
  //recording = false;
  this->init();
  //recording = saved_recording;
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
    }
    parent = parent->parent;
  }

  for (int i = 0; i < num_closes_needed; i++) {
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
  if (end-start > 4) {
    int mid = start + (end-start)/2;
    tfk_gradient_table* left = cilk_spawn merge_gradient_table_list(gradient_table_list, start,
                                                                    mid);
    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
    cilk_sync;

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

tfk_gradient_table* SP_Tree::walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table,
                                uint64_t n_gradients) {

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

    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        if (statement.index == -1) continue;
        int op_count = 0;

        adept::Real a = my_gradient_table->extract_value(statement.index);

        if (a != 0.0) {
         #ifdef TFK_DEBUG_PRINTS
         printf("statement %d edges:", statement.index);
         #endif
         if (ist == stack.statement_stack_start) {
           for (adept::uIndex j = stack.operation_stack_start;
                  j < statement.end_plus_one; j++) {
             op_count++;
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
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

    cilk_for (int i = 0; i < n->children->size(); i++) {
      tfk_gradient_table* table;
      if (i == 0) {
        tables[i] = my_gradient_table;
      }
      else if (i > 0 && wids[i-1] == __cilkrts_get_worker_number()) {
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
    //printf("gradient table list len %d\n", gradient_table_list.size());
    tfk_gradient_table* merged_table = merge_gradient_table_list(gradient_table_list, 0,
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
      if (merged_table != my_gradient_table) {
        my_gradient_table->merge_into_me(merged_table);
      }
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




void SP_Tree::walk_tree_debug(SP_Node* n, int nest_depth) {

  for (int i = 0; i < nest_depth; i++) {
    printf("  ");
  }

  nest_depth += 1;

  if (n->type == 1) {
    printf("(S:\n");
  } else if (n->type == 2) {
    printf("(P:\n");
  }

  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    printf("D\n");
    return;
  }

  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_debug((*(n->children))[i], nest_depth);
  }

  for (int i = 0; i < nest_depth-1; i++) {
    printf("  ");
  }
  printf(")\n");
}


void SP_Tree::walk_tree_debug(SP_Node* n) {
  return walk_tree_debug(n, 0);

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


