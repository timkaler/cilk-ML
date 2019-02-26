// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk-adept-headers/sp_tree.h>

#include <cilk-adept-source/gradient_table.cpp>
#include <cilk-adept-source/sp_node.cpp>

#include <vector>

// init can happen at the root of the program, and upon a steal.
// Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
void SP_Tree::init() {
  SP_Node*& current_node = imp_.view();
  current_node->type = 1;
  current_node->parent = NULL;
  current_node->children = std::vector<SP_Node*>();
}

SP_Node* SP_Tree::get_root() {
  SP_Node* current_node = imp_.view();
  return current_node;
}

// currently has a memory leak.
void SP_Tree::clear() {
  this->init();
}




void SP_Tree::add_D_node(triple_vector_wl data) {
  SP_Node* data_node = new SP_Node(data);
  SP_Node* current_node = imp_.view();
  current_node->children.push_back(data_node);
}


void SP_Tree::open_P_node() {
  SP_Node*& current_node = imp_.view();

  if (current_node == NULL) printf("Error current node is null in open P node\n");

  SP_Node* new_node = new SP_Node(2, current_node);
  current_node->children.push_back(new_node);

  current_node = new_node;
}

void SP_Tree::close_P_node() {
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close P node\n");

  // pop up.
  SP_Node* parent = current_node->parent;
  current_node = parent;
}

void SP_Tree::open_S_node() {
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in open S node\n");
  SP_Node* new_node = new SP_Node(1, current_node);
  current_node->children.push_back(new_node);
  current_node = new_node;
}

void SP_Tree::close_S_node() {
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

  for (int i = 0; i < n->children.size(); i++) {
    walk_tree_flatten(n->children[i], ret);
  }
}


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
      left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
    }
    return left;
  }


  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];

  for (int j = start+1; j < end; j++) {
    adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
    int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();

    for (int i = 0; i < n_active_entries; i++) {
      my_gradient_table->accumulate(active_entries[i],
                                    gradient_table_list[j]->extract_value(active_entries[i]));
    }
  }

  return my_gradient_table;
}

void SP_Tree::walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table,
                                uint64_t n_gradients) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) return;

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
    return;
  }

  if (n->type == 1 || n->type == 0) {
    for (int i = n->children.size()-1; i >= 0; i--) {
      if (n->type == 0) {
      walk_tree_process(n->children[i], my_gradient_table, n_gradients);
      } else {
      walk_tree_process(n->children[i], my_gradient_table, n_gradients);
      }
    }

  } else if (n->type == 2) {
    std::vector<tfk_gradient_table*> gradient_table_list;
    for (int i = n->children.size()-1; i >= 0; i--) {
      gradient_table_list.push_back(new tfk_gradient_table(n_gradients, my_gradient_table));
    }

    #pragma cilk grainsize 1
    cilk_for (int i = 0; i < n->children.size(); i++) {
      walk_tree_process(n->children[i], gradient_table_list[i], n_gradients);
    }

    tfk_gradient_table* merged_table = merge_gradient_table_list(gradient_table_list, 0,
                                                                 gradient_table_list.size());

    adept::uIndex* active_entries = merged_table->get_active_entries();
    int64_t n_active_entries = merged_table->get_n_active_entries();

    for (int i = 0; i < n_active_entries; i++) {
      my_gradient_table->accumulate(active_entries[i],
                                    merged_table->extract_value(active_entries[i]));
    }

    for (int i = n->children.size()-1; i >= 0; i--) {
      delete gradient_table_list[i];
    }

  } else {
    printf("Odd error with node types in SP_Tree during reverse-pass processing.\n");
  }
}





void SP_Tree::walk_tree_debug(SP_Node* n) {
  if (n->type == 1) {
    printf("(S:");
  } else if (n->type == 2) {
    printf("(P:");
  }

  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    printf("D");
    return;
  }


  for (int i = 0; i < n->children.size(); i++) {
    walk_tree_debug(n->children[i]);
  }
  printf(")");
}


