// Copyright 2019 MIT License - Tim Kaler

#include <cilk-adept-headers/sp_tree.h>
#include <cilk-adept-headers/tfkparallel.h>

/*
External variables.

extern __thread int thread_local_worker_id;
extern wl_stacks* worker_local_stacks;
extern tfkdiff tfk_reducer;

*/

// thread-local worker id for lookups.
__thread int thread_local_worker_id;

// space for worker-local stacks
wl_stacks* worker_local_stacks;

// reducer for differential tracing.
tfkdiff tfk_reducer;

// wl_stacks class.
  wl_stacks::wl_stacks () {
     statement_stack_arr_len = 0;
     operation_stack_arr_len = 0;
     multiplier_stack_arr_len = 0;
     gradient_registered_arr_len = 0;
     gradient_unregistered_arr_len = 0;

     wl_gradient_index = 0;

     wl_steal_count = 0;

     statement_stack_arr_capacity = 1024;
     operation_stack_arr_capacity = 1024;
     multiplier_stack_arr_capacity = 1024;

     gradient_registered_arr_capacity = 1024;
     gradient_unregistered_arr_capacity = 1024;

     gradient_registered_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_registered_arr_capacity);
     gradient_unregistered_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_unregistered_arr_capacity);

     statement_stack_arr =
         (adept::internal::Statement*) malloc(sizeof(adept::internal::Statement)*statement_stack_arr_capacity);
     operation_stack_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*operation_stack_arr_capacity);

     multiplier_stack_arr =
         (adept::Real*) malloc(sizeof(adept::Real)*multiplier_stack_arr_capacity);
   }

   void wl_stacks::init () {
     statement_stack_arr_len = 0;
     operation_stack_arr_len = 0;
     multiplier_stack_arr_len = 0;
     gradient_registered_arr_len = 0;
     gradient_unregistered_arr_len = 0;


     statement_stack_arr_capacity = 1024;
     operation_stack_arr_capacity = 1024;
     multiplier_stack_arr_capacity = 1024;
     gradient_registered_arr_capacity = 1024;
     gradient_unregistered_arr_capacity = 1024;


     wl_steal_count = 0;
     wl_gradient_index = 0;

     gradient_registered_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_registered_arr_capacity);
     gradient_unregistered_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_unregistered_arr_capacity);

     statement_stack_arr =
         (adept::internal::Statement*) malloc(sizeof(adept::internal::Statement)*statement_stack_arr_capacity);
     operation_stack_arr =
         (adept::uIndex*) malloc(sizeof(adept::uIndex)*operation_stack_arr_capacity);

     multiplier_stack_arr =
         (adept::Real*) malloc(sizeof(adept::Real)*multiplier_stack_arr_capacity);
   }


   void wl_stacks::ensure_gradient_registered_space(uint64_t size) {
     if (size >= gradient_registered_arr_capacity) {
       gradient_registered_arr = (adept::uIndex*) realloc(gradient_registered_arr, size*2*sizeof(adept::uIndex));
       gradient_registered_arr_capacity = size*2;
     }
   }

   void wl_stacks::ensure_gradient_unregistered_space(uint64_t size) {
     if (size >= gradient_unregistered_arr_capacity) {
       gradient_unregistered_arr = (adept::uIndex*) realloc(gradient_unregistered_arr, size*2*sizeof(adept::uIndex));
       gradient_unregistered_arr_capacity = size*2;
     }
   }


   void wl_stacks::ensure_statement_space(uint64_t size) {
     if (size >= statement_stack_arr_capacity) {
       statement_stack_arr = (adept::internal::Statement*) realloc(statement_stack_arr, size*2*sizeof(adept::internal::Statement));
       statement_stack_arr_capacity = size*2;
     }
   }

   void wl_stacks::ensure_operation_space(uint64_t size) {
     if (size >= operation_stack_arr_capacity) {
       operation_stack_arr = (adept::uIndex*) realloc(operation_stack_arr, size*2*sizeof(adept::uIndex));
       operation_stack_arr_capacity = size*2;
     }
   }

   void wl_stacks::ensure_multiplier_space(uint64_t size) {
     if (size >= multiplier_stack_arr_capacity) {
       multiplier_stack_arr = (adept::Real*) realloc(multiplier_stack_arr, size*2*sizeof(adept::Real));
       multiplier_stack_arr_capacity = size*2;
     }
   }

   void wl_stacks::add_register_gradient(adept::uIndex index) {
     ensure_gradient_registered_space(gradient_registered_arr_len+1);
     gradient_registered_arr[gradient_registered_arr_len++] = index;
   }

   void wl_stacks::add_unregister_gradient(adept::uIndex index) {
     ensure_gradient_unregistered_space(gradient_unregistered_arr_len+1);
     gradient_unregistered_arr[gradient_unregistered_arr_len++] = index;
   }



   void wl_stacks::add_statement(adept::internal::Statement statement) {
     ensure_statement_space(statement_stack_arr_len+1);
     statement_stack_arr[statement_stack_arr_len++] = statement;
   }

   void wl_stacks::add_operation(adept::uIndex index) {
     ensure_operation_space(operation_stack_arr_len+1);
     operation_stack_arr[operation_stack_arr_len++] = index;
   }

   void wl_stacks::add_multiplier(adept::Real mul) {
     ensure_multiplier_space(multiplier_stack_arr_len+1);
     multiplier_stack_arr[multiplier_stack_arr_len++] = mul;
   }

   // fast versions of the operators assume that a space check has already been performed.
   void wl_stacks::add_statement_fast(adept::internal::Statement statement) {
     statement_stack_arr[statement_stack_arr_len++] = statement;
   }

   void wl_stacks::add_operation_fast(adept::uIndex index) {
     operation_stack_arr[operation_stack_arr_len++] = index;
   }

   void wl_stacks::add_multiplier_fast(adept::Real mul) {
     multiplier_stack_arr[multiplier_stack_arr_len++] = mul;
   }


// triple_vector_wl class.

  triple_vector_wl::triple_vector_wl(bool init) {
    worker_id = thread_local_worker_id;
    steal_count = worker_local_stacks[worker_id].wl_steal_count++;
    has_bounds = false;
    statement_stack_start = worker_local_stacks[worker_id].statement_stack_arr_len;
    operation_stack_start = worker_local_stacks[worker_id].operation_stack_arr_len;
    multiplier_stack_start = worker_local_stacks[worker_id].multiplier_stack_arr_len;
    gradient_registered_start = worker_local_stacks[worker_id].gradient_registered_arr_len;
    gradient_unregistered_start = worker_local_stacks[worker_id].gradient_unregistered_arr_len;

    statement_stack_end = statement_stack_start;
    operation_stack_end = operation_stack_start;
    multiplier_stack_end = multiplier_stack_start;
    gradient_registered_end = gradient_registered_start;
    gradient_unregistered_end = gradient_unregistered_start;
  }

  // this is a default initializer.
  triple_vector_wl::triple_vector_wl() {

  }


// tfkdiff class
  void tfkdiff::get_tls_references() {
    thread_local_worker_id = __cilkrts_get_worker_number();

    // add a data node.
    sp_tree.add_D_node(triple_vector_wl(true));
  }

  tfkdiff::tfkdiff() {
    thread_local_worker_id = __cilkrts_get_worker_number();
    worker_local_stacks =(wl_stacks*) malloc(sizeof(wl_stacks)*__cilkrts_get_nworkers());
    for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
      worker_local_stacks[i].init();
    }
  }

  void tfkdiff::clear() {
    //std::vector<triple_vector_wl> stacks =
    //collect();

    sp_tree.clear();  // clear the sp_tree.

    for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
      worker_local_stacks[i].statement_stack_arr_len = 0;
      worker_local_stacks[i].operation_stack_arr_len = 0;
      worker_local_stacks[i].multiplier_stack_arr_len = 0;
      worker_local_stacks[i].gradient_registered_arr_len = 0;
      worker_local_stacks[i].gradient_unregistered_arr_len = 0;
    }

    get_tls_references();
  }

  void tfkdiff::collect() {
    std::vector<triple_vector_wl*> ret = sp_tree.flatten_to_array();

    if (ret.size()  == 0) return;

    std::vector<std::vector<std::pair<uint64_t, triple_vector_wl*> > >
        _wl_ret(__cilkrts_get_nworkers());

    for (int i = 0; i < ret.size(); i++) {
      int wid = ret[i]->worker_id;
      _wl_ret[wid].push_back(std::make_pair(ret[i]->steal_count, (ret[i])));
    }

    std::vector<std::vector<triple_vector_wl*> > wl_ret(__cilkrts_get_nworkers());
    for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
      std::sort(_wl_ret[wid].begin(), _wl_ret[wid].end());
      for (int i = 0; i < _wl_ret[wid].size(); i++) {
        wl_ret[wid].push_back(_wl_ret[wid][i].second);
      }
    }


    for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
      if (wl_ret[wid].size() > 0) {
      for (int i = 0; i < wl_ret[wid].size()-1; i++) {
        wl_ret[wid][i]->statement_stack_end = wl_ret[wid][i+1]->statement_stack_start;
        wl_ret[wid][i]->operation_stack_end = wl_ret[wid][i+1]->operation_stack_start;
        wl_ret[wid][i]->multiplier_stack_end = wl_ret[wid][i+1]->multiplier_stack_start;
        wl_ret[wid][i]->gradient_registered_end = wl_ret[wid][i+1]->gradient_registered_start;
        wl_ret[wid][i]->gradient_unregistered_end = wl_ret[wid][i+1]->gradient_unregistered_start;
      }
        wl_ret[wid][wl_ret[wid].size()-1]->statement_stack_end =
            worker_local_stacks[wid].statement_stack_arr_len;
        wl_ret[wid][wl_ret[wid].size()-1]->operation_stack_end =
            worker_local_stacks[wid].operation_stack_arr_len;
        wl_ret[wid][wl_ret[wid].size()-1]->multiplier_stack_end =
            worker_local_stacks[wid].multiplier_stack_arr_len;
        wl_ret[wid][wl_ret[wid].size()-1]->gradient_registered_end =
            worker_local_stacks[wid].gradient_registered_arr_len;
        wl_ret[wid][wl_ret[wid].size()-1]->gradient_unregistered_end =
            worker_local_stacks[wid].gradient_unregistered_arr_len;
      }
    }
    return;
  }



