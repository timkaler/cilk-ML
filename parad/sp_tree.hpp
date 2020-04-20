// Copyright (c) 2019, Tim Kaler - MIT License

#ifndef SP_TREE_HPP
#define SP_TREE_HPP

#include "./triple_vector_wl.hpp"
#include "./flat_hash_map.hpp"
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer.h>
//#include <adept.h>
#include <adept/base.h>
//#include <adept_arrays.h>

#include <vector>

struct OperationReference {
  // Allows indexing into a parallel array for the statement.
  int statement_wid;
  int statement_ist;

  // Allows indexing into a parallel array for the operation.
  int operation_wid;
  int operation_j;

  int gradient_index;
};

template<typename T>
struct WL_VECTOR_PADDED {
  int64_t padding[8];
  std::vector<T> vec;
  int64_t padding2[8];
};

template<typename T>
class worker_local_vector {
  public:
    WL_VECTOR_PADDED<T>* wl_vectors;
    worker_local_vector() {
      wl_vectors = new WL_VECTOR_PADDED<T>[__cilkrts_get_nworkers()];
    }

    __attribute__((always_inline))
    void push_back(int wid, T el) {
      wl_vectors[wid].vec.push_back(el);
    }

    void reserve(int64_t n) {
      cilk_for(int i = 0; i < __cilkrts_get_nworkers(); i++) {
        wl_vectors[i].vec.reserve(n);
      }
    }

    int64_t collect(T*& ret) {
      int64_t* offsets = new int64_t[__cilkrts_get_nworkers()];
      int64_t total_size = wl_vectors[0].vec.size();
      offsets[0] = 0;
      for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
        offsets[i] = offsets[i-1] + wl_vectors[i-1].vec.size();
        total_size += wl_vectors[i].vec.size();
      }
      ret = (T*)malloc(sizeof(T)*total_size);
      //ret.resize(total_size);

      cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
        int64_t off = offsets[i];
        cilk_for (int j = 0; j < wl_vectors[i].vec.size(); j++) {
          ret[off+j] = wl_vectors[i].vec[j];
        }
      }
      delete[] offsets;
      delete[] wl_vectors;
      return total_size;
    }

    // BX: Removes all elements which have gradient index gid, such that
    // gradient_use_wl[gid] == true
    void remove_wl_gradients(bool* gradient_use_wl) {
      cilk_for (int i = 0; i < __cilkrts_get_nworkers(); ++i) {
        std::vector<T> new_vec;
        for (int j = 0; j < wl_vectors[i].vec.size(); ++j) {
          T element = wl_vectors[i].vec[j];
          if (!gradient_use_wl[element.gradient_index]) {
            new_vec.push_back(element);
          }
        }
        wl_vectors[i].vec = new_vec;
      }
    }
};

static int64_t get_value_count = 0;
static int64_t set_value_count = 0;

/*
template <typename K, typename V>
class sparse_array {
  private:
    //int64_t log_head;
    int64_t log_start;
    int64_t log_end;
    K max_key;
    int64_t capacity;
    V null_value;

    V*__restrict log;
    K*__restrict dense_array;

  public:
   void allocate(int64_t capacity) {
     printf("allocate called with capacity %llu\n", capacity);
     log = (V*) malloc(sizeof(V) * capacity);
     dense_array = (K*) malloc(sizeof(K) * capacity);
     cilk_for(int64_t i = 0; i < capacity; i++) {
       dense_array[i] = 0;
     }
     //log_head = 1;
     log_start = 1;
     log_end = 1;
     this->capacity = capacity;
   }

   sparse_array(K max_key, V null_value) {
     allocate(max_key*1.5);
     this->null_value = null_value;
   }

   void set_value(K key, V val) {
     set_value_count++;
     if (dense_array[key] >= log_start &&
         dense_array[key] < log_end) {
       log[dense_array[key]-log_start] = val;
     } else {
       log[log_end-log_start] = val;
       dense_array[key] = log_end++;
     }
   }

   V get_value(K key) {
     get_value_count++;
     if (dense_array[key] >= log_start &&
         dense_array[key] < log_end) {
       return log[dense_array[key]-log_start];
     } else {
       return null_value;
     }
   }

   void clear() {
     //printf("Get_value_Count: \t\t %llu \nSet_Value_Count: \t\t %llu\n", get_value_count, set_value_count);
     log_start = log_end;
   }

   void ensure_space(K num_elements) {
     if (num_elements > capacity) {
       free(dense_array);
       free(log);
       allocate(num_elements*1.5);
     }
   }

   ~sparse_array() {
     free(log);
     free(dense_array);
   }
};
*/

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

    SP_Node* parent;
    // Only initialized if it's an S or P node.
    std::vector<SP_Node*>* children;
    int rootset_id;
    void* sync_id;

    explicit SP_Node(triple_vector_wl data_);
    SP_Node(int type_, SP_Node* parent_);
    ~SP_Node();
    SP_Node(int type_, SP_Node* parent_, void* sync_id);
};

struct args_for_collect_ops {
  bool* idx_in_statement;
  int8_t* last_statement_worker;
  int32_t* last_statement_index;
  SP_Node* last_statement_node;
  float* gradient_;
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

    void reverse_ad_PARAD(int64_t n_gradients, float* _gradient);

    std::vector<triple_vector_wl*> flatten_to_array();
    void make_ids_deterministic(int64_t n_gradients);

    void collect_ops_for_semisort(SP_Node* n, bool* idx_in_statement, int64_t* last_statement_worker, int64_t* last_statement_index, float* gradient_, worker_local_vector<OperationReference>& wl_ops);
    void collect_ops_for_semisort(SP_Node* n, args_for_collect_ops* args, worker_local_vector<OperationReference>& wl_ops);

    void walk_tree_process_semisort(SP_Node* n, float** worker_local_grad_table, bool* appears_in_statement, float* gradient_);
    void walk_tree_process_locks(SP_Node* n, float* gradient_, int64_t* locks);

    void report_times();

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

#endif  // SP_TREE_HPP
