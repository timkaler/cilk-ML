// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <vector>
#include <adept/triple_vector_wl.h>
#include <adept/flat_hash_map.hpp>

#ifndef SP_TREE_H
#define SP_TREE_H




class tfk_gradient_table {
  public:
    tfk_gradient_table* gradient_table;
    //adept::Real* gradient_table_local;
    //bool* gradient_table_local_active;

    ska::flat_hash_map<adept::uIndex, adept::Real> gradient_table_local;

    adept::uIndex* active_entries;
    int64_t n_active_entries;


    adept::Real* raw_gradient_table;


    tfk_gradient_table(uint64_t n_gradients, tfk_gradient_table* gradient_table_);
    tfk_gradient_table(uint64_t n_gradients, adept::Real* gradient_table_);

    ~tfk_gradient_table();
    void accumulate(adept::uIndex index, adept::Real val);
    // extracts a value and sets it to zero.
    adept::Real extract_value(adept::uIndex index);
    int64_t get_n_active_entries();
    adept::uIndex* get_active_entries();
};




//template <typename T>
class SP_Node {
  public:
  // 0 Root, 1 Serial, 2 Parallel, 3 Data which should have no children
  int type;

  // Only initialized if it's a D node.
  triple_vector_wl data;

  // Only initialized if it's an S or P node.
  SP_Node* parent;
  std::vector<SP_Node*> children;

  SP_Node(triple_vector_wl data_);
  SP_Node(int type_, SP_Node* parent_);
};

//template <typename T>
class SP_Tree {

   struct Monoid: cilk::monoid_base<SP_Node*>
   {
     static void reduce (SP_Node ** left, SP_Node ** right) {
       //if ((*left)->type == 0) printf("left type is 0 error!\n");
       if ((*right)->type != 0) printf("right type is not 0 error!\n");

       for (int i = 0; i < (*right)->children.size(); i++) {
         (*left)->children.push_back((*right)->children[i]);
       }
     }

     static void identity (SP_Node **p) {
       *p = new SP_Node(0, NULL);
       //p->type = 0;
       //p->parent = NULL;
       //p->children = std::vector<SP_Node*>();
     }


   };



public:
  //SP_Node* current_node = NULL;

  cilk::reducer<Monoid> imp_;

  // init can happen at the root of the program, and upon a steal.
  // Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
  void init();
  void clear();
  SP_Node* get_root();
  void add_D_node(triple_vector_wl data);
  void open_P_node();
  void close_P_node();
  void open_S_node();
  void close_S_node();

  std::vector<triple_vector_wl*> flatten_to_array();
  void walk_tree_debug(SP_Node* n);
  void walk_tree_flatten(SP_Node* n, std::vector<triple_vector_wl*>& ret);
  void walk_tree_prepare(SP_Node* n, triple_vector_wl* last_worker_trace, bool* last_worker_trace_init);
  void walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table, uint64_t n_gradients);

  tfk_gradient_table* merge_gradient_table_list(std::vector<tfk_gradient_table*>& table_list, int start, int end);

};

#endif
