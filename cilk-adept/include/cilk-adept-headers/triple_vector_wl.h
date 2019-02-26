// Copyright 2019 Tim Kaler MIT License

#ifndef TRIPLE_VECTOR_WL_H
#define TRIPLE_VECTOR_WL_H

// worker local triple_vector.
class triple_vector_wl {
  public:
    int worker_id;
    bool has_bounds;
    uint64_t statement_stack_start;
    uint64_t operation_stack_start;
    uint64_t multiplier_stack_start;
    uint64_t gradient_registered_start;
    uint64_t gradient_unregistered_start;

    uint64_t statement_stack_end;
    uint64_t operation_stack_end;
    uint64_t multiplier_stack_end;
    uint64_t gradient_registered_end;
    uint64_t gradient_unregistered_end;

    uint64_t steal_count;

    triple_vector_wl();
    explicit triple_vector_wl(bool do_init);
};

#endif  // TRIPLE_VECTOR_WL_H
