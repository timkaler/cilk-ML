// Copyright (c) 2019, Tim Kaler - MIT License

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk-adept-headers/triple_vector_wl.h>
#include <cilk-adept-headers/flat_hash_map.hpp>

#include <vector>


#ifndef SP_TREE_H
#define SP_TREE_H
struct OperationReference {
  // Allows indexing into a parallel array for the statement.
  int64_t statement_wid;
  int64_t statement_ist;

  // Allows indexing into a parallel array for the operation.
  int64_t operation_wid;
  int64_t operation_j;

  int64_t gradient_index;
};

class tfk_gradient_table {
  public:
    tfk_gradient_table* gradient_table;

    ska::flat_hash_map<adept::uIndex, adept::Real> gradient_table_local;


    //adept::Real* gradient_table_local_dense;
    //int64_t local_dense_start;
    //int64_t local_dense_end;

    adept::uIndex* active_entries;
    int64_t n_active_entries;
    int64_t n_gradients;
    int64_t n_operations_done;
    int64_t* locks;

    adept::Real* dense_rep;

    adept::Real* raw_gradient_table;

    tfk_gradient_table(uint64_t n_gradients, tfk_gradient_table* gradient_table_);
    tfk_gradient_table(uint64_t n_gradients, adept::Real* gradient_table_);

    ~tfk_gradient_table();
    void accumulate(adept::uIndex index, adept::Real val);
    // extracts a value and sets it to zero.
    adept::Real extract_value(adept::uIndex index);
    int64_t get_n_active_entries();
    adept::uIndex* get_active_entries();
    void merge_into_me(tfk_gradient_table* other);
};




class SP_Node {
  public:
  // 0 Root, 1 Serial, 2 Parallel, 3 Data which should have no children
  int type;

  // Only initialized if it's a D node.
  triple_vector_wl data;

  // Only initialized if it's an S or P node.
  SP_Node* parent;
  std::vector<SP_Node*>* children;
  int rootset_id;
  void* sync_id;

  explicit SP_Node(triple_vector_wl data_);
  SP_Node(int type_, SP_Node* parent_);
  ~SP_Node();
  SP_Node(int type_, SP_Node* parent_, void* sync_id);
};

class SP_Tree {
  struct Monoid: cilk::monoid_base<SP_Node*> {
    static void reduce(SP_Node ** left, SP_Node ** right_) {
      SP_Node* right = *right_;
      if (right == NULL) {
         printf("The right type is NULL\n");
         return;
      }
      while (right->type != 0 && right->parent != NULL) {
        //delete right;
        right = right->parent;
      }

      //if ((*left)->type != 2) printf("error left type is not parallel... so how did this reduction happen?\n");

      // begin new.
      for (int i = 0; i < (right)->children->size(); i++) {
        (*right->children)[i]->parent = *left;
        (*left)->children->push_back((*(right->children))[i]);
      }

      //right->type = 1;
      //right->parent = *left;
      //(*left)->children->push_back(right);

      if ((*right_)->parent != NULL && (*right_)->type != 0) {
        *left = *right_;
      }

      return;
      // end new.

      if (right->type != 0) printf("right type is not 0 error! %d\n", (right)->type);



      //right->type = 1;
      //(*left)->children->push_back(right);

      for (int i = 0; i < (right)->children->size(); i++) {
        (*left)->children->push_back((*(right->children))[i]);
      }



      //if ((right)->sync_id == NULL) return;

      //printf("Right has sync id %p\n", right->sync_id);

      //SP_Node* parent = *left;
      //int num_closes_needed = 0;
      //int num_closes = 0;
      //void* sync_id = (right)->sync_id;//(*left)->sync_id;
      //while (parent != NULL) {
      //  num_closes++;
      //  if (parent->type == 2 && parent->sync_id == sync_id) {
      //    parent->sync_id = NULL;
      //    num_closes_needed = num_closes;
      //  }
      //  if (parent->type == 0) {
      //    parent->sync_id = sync_id;
      //  }
      //  parent = parent->parent;
      //}

      //SP_Node* current_node = *left;
      //for (int i = 0; i < num_closes_needed; i++) {
      //  SP_Node* parent = current_node->parent;
      //  assert(parent != NULL);
      //  current_node = parent;
      //}
      //*left = current_node;

      ////(*left)->sync_P_nodes((*left)->sync_id);
    }

    static void identity(SP_Node **p) {
      *p = new SP_Node(0, NULL);
    }
  };

 public:
  cilk::reducer<Monoid> imp_;

  bool recording;

  // init can happen at the root of the program, and upon a steal.
  // Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
  void init();
  void clear();
  void set_recording(bool recording_);
  SP_Node* get_root();
  void add_D_node(triple_vector_wl data);
  void open_S_node();
  void open_P_node();
  void open_P_node(void* sync_id);

  void close_S_node();
  void close_P_node();
  void sync_P_nodes(void* sync_id);

  void test(int64_t n_gradients);

  std::vector<triple_vector_wl*> flatten_to_array();
  void make_ids_deterministic(int64_t n_gradients);

  void collect_ops_for_semisort(SP_Node* n, bool* idx_in_statement, int64_t* last_statement_worker, int64_t* last_statement_index, std::vector<OperationReference>& ops);

  void walk_tree_debug(SP_Node* n);
  void walk_tree_debug(SP_Node* n, int nest_depth, FILE* f);
  void walk_tree_flatten(SP_Node* n, std::vector<triple_vector_wl*>& ret);
  void walk_tree_flatten_datanodes(SP_Node* n, std::vector<SP_Node*>& ret);
  void walk_tree_process_one_worker(float* gradient_table);
  void walk_tree_flatten_allnodes(SP_Node* n, std::vector<SP_Node*>& ret);
  int walk_tree_rootset_transform(SP_Node* n, int dep_count);
  tfk_gradient_table* walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table, uint64_t n_gradients);
  SP_Tree* transform_to_rootset_form(); 
  tfk_gradient_table* merge_gradient_table_list(std::vector<tfk_gradient_table*>& table_list,
                                                int start, int end);
};

#endif  // SP_TREE_H
