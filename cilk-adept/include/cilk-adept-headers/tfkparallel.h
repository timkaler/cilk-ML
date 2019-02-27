// Copyright 2019 Tim Kaler MIT License

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <cilk/reducer.h>
#include <cilk/reducer_list.h>
//#include <adept.h>
#include <cilk-adept-headers/sp_tree.h>
#include <cilk-adept-headers/triple_vector_wl.h>

#include <algorithm>
#include <vector>


#ifndef TFK_ADEPT_PARALLEL
#define TFK_ADEPT_PARALLEL

/*
External variables.

extern __thread int thread_local_worker_id;
extern wl_stacks* worker_local_stacks;
extern tfkdiff tfk_reducer;

*/

extern __thread int thread_local_worker_id;

class wl_stacks {
  public:
    adept::internal::Statement* statement_stack_arr;
    adept::uIndex* operation_stack_arr;
    adept::Real* multiplier_stack_arr;
    adept::uIndex* gradient_registered_arr;
    adept::uIndex* gradient_unregistered_arr;

    adept::uIndex* wl_gradient_index;

    // length of the total array.
    uint64_t statement_stack_arr_len;
    uint64_t operation_stack_arr_len;
    uint64_t multiplier_stack_arr_len;
    uint64_t gradient_registered_arr_len;
    uint64_t gradient_unregistered_arr_len;


    uint64_t statement_stack_arr_capacity;
    uint64_t operation_stack_arr_capacity;
    uint64_t multiplier_stack_arr_capacity;
    uint64_t gradient_registered_arr_capacity;
    uint64_t gradient_unregistered_arr_capacity;

    uint64_t wl_steal_count = 0;


    uint8_t buffer[4096];

    wl_stacks();
    void init();
    void ensure_gradient_registered_space(uint64_t size);
    void ensure_gradient_unregistered_space(uint64_t size);
    void ensure_statement_space(uint64_t size);
    void ensure_operation_space(uint64_t size);
    void ensure_multiplier_space(uint64_t size);
    void add_register_gradient(adept::uIndex index);
    void add_unregister_gradient(adept::uIndex index);
    void add_statement(adept::internal::Statement statement);
    void add_operation(adept::uIndex index);
    void add_multiplier(adept::Real mul);
    // fast versions of the operators assume that a space check has already been performed.
    void add_statement_fast(adept::internal::Statement statement);
    void add_operation_fast(adept::uIndex index);
    void add_multiplier_fast(adept::Real mul);
};

// space for worker-local stacks
extern wl_stacks* worker_local_stacks;


class tfkdiff {
  public:
    SP_Tree sp_tree;

    void get_tls_references();

    void clear();
    void collect();
    tfkdiff();
};

extern tfkdiff tfk_reducer;

#endif  // TFK_ADEPT_PARALLEL

