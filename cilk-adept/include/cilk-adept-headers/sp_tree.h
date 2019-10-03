// Copyright (c) 2019, Tim Kaler - MIT License

#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk-adept-headers/triple_vector_wl.h>
#include <cilk-adept-headers/flat_hash_map.hpp>

#include <vector>


#ifndef SP_TREE_H
#define SP_TREE_H

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
      if (right == NULL) printf("The right type is NULL\n");
      while (right->type != 0 && right->parent != NULL) {
        //delete right;
        right = right->parent;
      }

      if (right->type != 0) printf("right type is not 0 error! %d\n", (right)->type);

      for (int i = 0; i < (right)->children->size(); i++) {
        (*left)->children->push_back((*(right->children))[i]);
      }
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

  std::vector<triple_vector_wl*> flatten_to_array();
  void walk_tree_debug(SP_Node* n);
  void walk_tree_debug(SP_Node* n, int nest_depth);
  void walk_tree_flatten(SP_Node* n, std::vector<triple_vector_wl*>& ret);
  tfk_gradient_table* walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table, uint64_t n_gradients);

  tfk_gradient_table* merge_gradient_table_list(std::vector<tfk_gradient_table*>& table_list,
                                                int start, int end);
};

#endif  // SP_TREE_H
