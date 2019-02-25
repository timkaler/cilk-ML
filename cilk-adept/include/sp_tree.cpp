// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include "adept/sp_tree.h"
#include <vector>



// tfk_gradient_table class
    tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, tfk_gradient_table* gradient_table_) {
      gradient_table = gradient_table_;
      raw_gradient_table = NULL;

      //gradient_table_local = (adept::Real*) calloc(n_gradients, sizeof(adept::Real));
      //gradient_table_local_active = (bool*) calloc(n_gradients, sizeof(bool));

      //gradient_table_local_active_index = (uint64_t*) calloc(n_gradients, sizeof(uint64_t));
      //active_entries = (adept::uIndex*) calloc(n_gradients, sizeof(adept::uIndex));
      active_entries = NULL;
      n_active_entries = 0;
    }


    tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, adept::Real* gradient_table_raw_) {
      gradient_table = NULL;
      raw_gradient_table = gradient_table_raw_;

      //gradient_table_local = (adept::Real*) calloc(n_gradients, sizeof(adept::Real));
      //gradient_table_local_active = (bool*) calloc(n_gradients, sizeof(bool));

      //gradient_table_local_active_index = (uint64_t*) calloc(n_gradients, sizeof(uint64_t));
      //active_entries = (adept::uIndex*) calloc(n_gradients, sizeof(adept::uIndex));
      active_entries = NULL;
      n_active_entries = 0;
    }



    tfk_gradient_table::~tfk_gradient_table() {
      //free(gradient_table_local);
      //free(gradient_table_local_active);
      if (active_entries != NULL) {
        free(active_entries);
      }
    }

    void tfk_gradient_table::accumulate(adept::uIndex index, adept::Real val) {
      if (raw_gradient_table != NULL) {
        raw_gradient_table[index] += val;
        return;
      }

      //if (gradient_table_local.find(index) == gradient_table_local.end()) {
      //  active_entries[n_active_entries++] = index;
      //}
      gradient_table_local[index] += val;

      //if (!gradient_table_local_active[index]) {
      //  gradient_table_local_active[index] = true;
      //  active_entries[n_active_entries++] = index;
      //}
    }

    // extracts a value and sets it to zero.
    adept::Real tfk_gradient_table::extract_value(adept::uIndex index) {
      if (raw_gradient_table != NULL) {
        adept::Real a = raw_gradient_table[index];
        raw_gradient_table[index] = 0.0;
        return a;
      }
      if (gradient_table_local.find(index) == gradient_table_local.end() /*!gradient_table_local_active[index]*/ && gradient_table != NULL) {
      //if (!gradient_table_local_active[index] && gradient_table != NULL) {
        return gradient_table->extract_value(index);
        //adept::Real a = gradient_table[index];
        //gradient_table[index] = 0.0;
        //return a;
      }

      //adept::Real a = gradient_table_local[index];
      //gradient_table_local[index] = 0.0;
      adept::Real a = gradient_table_local[index];
      gradient_table_local.erase(index);
      //gradient_table_local[index] = 0.0;
      return a;
    }

    int64_t tfk_gradient_table::get_n_active_entries() {
      return n_active_entries;
    }

    adept::uIndex* tfk_gradient_table::get_active_entries() {
      active_entries = (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_table_local.size());
      for (auto iter = gradient_table_local.begin(); iter != gradient_table_local.end(); ++iter) {
        active_entries[n_active_entries++] = iter->first;
      }
      return active_entries;
    }



// SP_Node Class
  SP_Node::SP_Node(triple_vector_wl data_) {
    data = data_;
    type = 3;
  }

  SP_Node::SP_Node(int type_, SP_Node* parent_) {
    type = type_;
    parent = parent_;
    children = std::vector<SP_Node*>();
  }


// SP_Tree class.

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


  void SP_Tree::walk_tree_prepare(SP_Node* n, triple_vector_wl* last_worker_trace, bool* last_worker_trace_init) {
    return;
    //if (last_worker_trace == NULL) {
    //  last_worker_trace = (triple_vector_wl*) calloc(__cilkrts_get_nworkers(), sizeof(triple_vector_wl));
    //  last_worker_trace_init = (bool*) calloc(__cilkrts_get_nworkers(), sizeof(bool));
    //}

    //if (n->type == 3) {
    //  triple_vector_wl& data = n->data;
    //  //int wid = data.worker_id;
    //  //if (!last_worker_trace_init[wid]) {
    //  //  last_worker_trace_init[wid] = true;

    //  //  data.statement_stack_end = worker_local_stacks[wid].statement_stack_arr_len;
    //  //  data.operation_stack_end = worker_local_stacks[wid].operation_stack_arr_len;
    //  //  data.multiplier_stack_end = worker_local_stacks[wid].multiplier_stack_arr_len;
    //  //  data.gradient_registered_end = worker_local_stacks[wid].gradient_registered_arr_len;
    //  //  data.gradient_unregistered_end = worker_local_stacks[wid].gradient_unregistered_arr_len;

    //  //  last_worker_trace[wid] = data;
    //  //} else {
    //  //  data.statement_stack_end = last_worker_trace[wid].statement_stack_start;
    //  //  data.operation_stack_end = last_worker_trace[wid].operation_stack_start;
    //  //  data.multiplier_stack_end = last_worker_trace[wid].multiplier_stack_start;
    //  //  data.gradient_registered_end = last_worker_trace[wid].gradient_registered_start;
    //  //  data.gradient_unregistered_end = last_worker_trace[wid].gradient_unregistered_start;
    //  //  last_worker_trace[wid] = data;
    //  //}

    //  printf("statement len: %llu\n", data.statement_stack_end - data.statement_stack_start);

    //  return;
    //}

    //for (int i = n->children.size()-1; i >= 0; i--) {
    //  walk_tree_prepare(n->children[i], last_worker_trace, last_worker_trace_init);
    //}

  }


  tfk_gradient_table* SP_Tree::merge_gradient_table_list(std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {

    if (end-start > 4) {
      int mid = start + (end-start)/2;
      tfk_gradient_table* left = cilk_spawn merge_gradient_table_list(gradient_table_list, start, mid);
      tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
      cilk_sync;

      adept::uIndex* active_entries_right = right->get_active_entries();
      int n_active_entries_right = right->get_n_active_entries();
      for (int i = 0; i < n_active_entries_right; i++) {
        left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
      }
      //delete right;
      return left;
    }


    tfk_gradient_table* my_gradient_table =  gradient_table_list[start];

    for (int j = start+1; j < end; j++) {
      adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
      int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();

      for (int i = 0; i < n_active_entries; i++) {
        my_gradient_table->accumulate(active_entries[i], gradient_table_list[j]->extract_value(active_entries[i]));
      }
    }

    return my_gradient_table;
  }

  void SP_Tree::walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table, uint64_t n_gradients) {
    // If its a data node it must be a terminal node.
    if (n->type == 3) {
      // We are going to process one of the stacks.
      triple_vector_wl stack = n->data;
      if (stack.statement_stack_end == stack.statement_stack_start) return;

      for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
          const adept::Statement& statement = worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
          if (statement.index == -1) continue;
          int op_count = 0;


          adept::Real a = my_gradient_table->extract_value(statement.index);

          //if (gradient_table_local_active[statement.index]) {
          //  a = gradient_table_local[statement.index];
          //  gradient_table_local[statement.index] = 0.0;
          //} else {
          //  a = gradient_table[statement.index];
          //  //if (a != 0.0) {
          //  gradient_table[statement.index] = 0.0;
          //  gradient_table_local[statement.index] = 0.0;
          //  gradient_table_local_active[statement.index] = true;
          //  debug_set[statement.index] = true;
          //  //}
          //}
          //adept::Real a = gradient_table[statement.index];
          //gradient_table[statement.index] = 0.0;

          if (a != 0.0) {
           #ifdef TFK_DEBUG_PRINTS
           printf("statement %d edges:", statement.index);
           #endif
           if (ist == stack.statement_stack_start) {
             for (adept::uIndex j = stack.operation_stack_start;
                    j < statement.end_plus_one; j++) {
               op_count++;
               adept::Real multiplier_test = worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);
               //gradient_table_local[operation_stack_index] += multiplier_test*a;//multiplier_[i]*a;
               //gradient_table_local_active[operation_stack_index] = true;

               #ifdef TFK_DEBUG_PRINTS
               printf("%d,", operation_stack_index);
               #endif
             }
             #ifdef TFK_DEBUG_PRINTS
             printf("; ist %d start %d end %d\n", ist, stack.statement_stack_end, stack.statement_stack_start);
             #endif
           } else {
             for (adept::uIndex j = worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               op_count++;
               adept::Real multiplier_test = worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);
               //gradient_table_local[operation_stack_index] += multiplier_test*a;//multiplier_[i]*a;
               //gradient_table_local_active[operation_stack_index] = true;

               #ifdef TFK_DEBUG_PRINTS
               printf("%d,", operation_stack_index);
               #endif
             }
             #ifdef TFK_DEBUG_PRINTS
             printf(": ist %d start %d end %d\n", ist, stack.statement_stack_end, stack.statement_stack_start);
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

      //cilk_for (int i = 0; i < n_gradients; i++) {
      //  gradient_table[i] = gradient_table_local[i];
      //  gradient_table_local_active[i] = false;
      //}

    } else if (n->type == 2) {

      std::vector<tfk_gradient_table*> gradient_table_list;
      for (int i = n->children.size()-1; i >= 0; i--) {
        gradient_table_list.push_back(new tfk_gradient_table(n_gradients, my_gradient_table));
      }

      #pragma cilk grainsize 1
      cilk_for (int i = 0; i < n->children.size(); i++) {
        walk_tree_process(n->children[i], gradient_table_list[i], n_gradients);
      }


      

      tfk_gradient_table* merged_table = merge_gradient_table_list(gradient_table_list, 0, gradient_table_list.size());

      adept::uIndex* active_entries = merged_table->get_active_entries();
      int64_t n_active_entries = merged_table->get_n_active_entries();

      for (int i = 0; i < n_active_entries; i++) {
        my_gradient_table->accumulate(active_entries[i], merged_table->extract_value(active_entries[i]));
      }






      //std::vector<adept::uIndex*> active_entries_list(gradient_table_list.size());
      //std::vector<adept::uIndex> n_active_entries_list(gradient_table_list.size());

      //cilk_for (int j = 0; j < gradient_table_list.size(); j++) {
      //  active_entries_list[j] = gradient_table_list[j]->get_active_entries();
      //  n_active_entries_list[j] = gradient_table_list[j]->get_n_active_entries();
      //}




      //for (int j = 0; j < gradient_table_list.size(); j++) {
      //  adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
      //  int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();

      //  for (int i = 0; i < n_active_entries; i++) {
      //    my_gradient_table->accumulate(active_entries[i], gradient_table_list[j]->extract_value(active_entries[i]));
      //  }
      //}


      //cilk_for (int i = 0; i < n_gradients; i++) {
      //  for (int j = 0; j < gradient_table_local_list.size(); j++) {
      //    if (gradient_table_local_active_list[j][i]) {
      //      gradient_table_local[i] += gradient_table_local_list[j][i];
      //      gradient_table_local_active[i] |= gradient_table_local_active_list[j][i];
      //    }
      //  }
      //}

      for (int i = n->children.size()-1; i >= 0; i--) {
        delete gradient_table_list[i];
        //free(gradient_table_local_list[i]);
        //free(gradient_table_local_active_list[i]);
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


