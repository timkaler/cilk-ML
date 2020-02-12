  tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, tfk_gradient_table* gradient_table_) {
    gradient_table = gradient_table_;
    raw_gradient_table = NULL;

    active_entries = NULL;
    n_active_entries = 0;
    n_operations_done = 0;
    this->n_gradients = n_gradients;

    //gradient_table_local = ska::flat_hash_map<adept::uIndex, adept::Real>(size_hint);
    //gradient_table_local_dense = NULL;
    //local_dense_start = 0;
    //local_dense_end = 0;

    //dense_rep = NULL;
    this->dense_rep = gradient_table_->dense_rep;
    //this->locks = gradient_table_->locks;
  }


  tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, adept::Real* gradient_table_raw_) {
    gradient_table = NULL;
    raw_gradient_table = gradient_table_raw_;

    active_entries = NULL;
    n_active_entries = 0;
    n_operations_done = 0;
    this->n_gradients = n_gradients;
    this->dense_rep = raw_gradient_table;
    //this->locks = (int64_t*) calloc(n_gradients, sizeof(int64_t));

    //for (int64_t i = 0; i < n_gradients; i++) {
    //  //this->dense_rep[i] = 0.0;
    //  this->locks[i] = 0;
    //}

    //gradient_table_local_dense = NULL;
    //local_dense_start = 0;
    //local_dense_end = 0;
  }

  tfk_gradient_table::~tfk_gradient_table() {
    //if (active_entries != NULL) {
    //  free(active_entries);
    //}
    //if (dense_rep != NULL && raw_gradient_table == NULL) delete[] dense_rep;
    //if (raw_gradient_table != NULL) {
    //  free(this->locks);
    //}
  }

  __attribute__((always_inline)) void tfk_gradient_table::accumulate(adept::uIndex index, adept::Real val) {
    assert(index >= 0 && index < n_gradients);
    n_operations_done += 1;

    //bool succ = false;
    //do {
    //  succ = __sync_bool_compare_and_swap(&locks[index], 0, 1);
    //} while (!succ || locks[index] == 0);

    dense_rep[index] += val;
    //locks[index] = 0;
    return;
    //if (raw_gradient_table != NULL) {
    //  raw_gradient_table[index] += val;
    //  return;
    //}


    //if (gradient_table_local.find(index) == gradient_table_local.end()) {
    //  active_entries[n_active_entries++] = index;
    //}

    // switch to a dense representation.
    if (dense_rep != NULL || (n_operations_done > n_gradients/64)) {
      if (dense_rep == NULL) {
        dense_rep = new adept::Real[n_gradients]();
        for (int i = 0; i < n_gradients; i++) {
          dense_rep[i] = 0.0;
        }
        for (auto iter = gradient_table_local.begin(); iter != gradient_table_local.end(); ++iter) {
          dense_rep[iter->first] = iter->second;
        }
        gradient_table_local.clear();
      }
      dense_rep[index] += val;
    } else {
      if (gradient_table_local.find(index) == gradient_table_local.end()) {
        gradient_table_local[index] = val;
      } else {
        gradient_table_local[index] += val;
      }
    }

    //if (!gradient_table_local_active[index]) {
    //  gradient_table_local_active[index] = true;
    //  active_entries[n_active_entries++] = index;
    //}
  }

    // extracts a value and sets it to zero.
    adept::Real tfk_gradient_table::extract_value(adept::uIndex index) {
      assert(index >= 0 && index < n_gradients);
      //if (raw_gradient_table != NULL) {
      //  adept::Real a = raw_gradient_table[index];
      //  raw_gradient_table[index] = 0.0;
      //  return a;
      //}

      adept::Real _a = dense_rep[index];
      dense_rep[index] = 0;
      return _a;

      //if (index < 10000000) return 1.0;

      adept::Real a = 0;
      //if ((n_operations_done > n_gradients/64) && dense_rep != NULL) {
      if (dense_rep != NULL) {
        a = dense_rep[index];
        dense_rep[index] = 0.0;
      } else {
        if (gradient_table_local.find(index) == gradient_table_local.end()) {
          if (gradient_table != NULL) {
            return gradient_table->extract_value(index);
          } else {
            a = 0;
            return a;
          }
          //adept::Real a = gradient_table[index];
          //gradient_table[index] = 0.0;
          //return a;
        }

        //adept::Real a = gradient_table_local[index];
        //gradient_table_local[index] = 0.0;
        a = gradient_table_local[index];
        gradient_table_local[index] = 0.0;
        gradient_table_local.erase(index);
      }
      // NOTE(TFK): Technically we need to extract value from parent.
      if (gradient_table != NULL) {
        return a + gradient_table->extract_value(index);
      } else {
        return a;
      }
    }

    int64_t tfk_gradient_table::get_n_active_entries() {
      return gradient_table_local.size();// n_active_entries;
    }

   
    void tfk_gradient_table::merge_into_me(tfk_gradient_table* other) {
      return;
      if (other->dense_rep != NULL) {
        // the other one is in dense representation.
        for (adept::uIndex i = 0; i < n_gradients; i++) {
          if (other->dense_rep[i] != 0.0) this->accumulate(i, other->dense_rep[i]);
        }
      } else {
        adept::uIndex* other_active_entries = other->get_active_entries();
        int64_t n_other = other->get_n_active_entries();
        if (n_other == 0) return;
        for (int64_t i = 0; i < n_other; i++) {
          this->accumulate(other_active_entries[i], other->gradient_table_local[(other_active_entries[i])]);
        }
        assert(n_other == other->get_n_active_entries());
        //if (active_entries != NULL) {
        //printf("n other is %d ptr is %p\n", n_other, other_active_entries);
        free(other_active_entries);
          //active_entries = NULL;
        //}
      }




      //if ((other->n_operations_done > other->n_gradients/64) && other->dense_rep != NULL) {
      //  // the other one is in dense representation.
      //  for (adept::uIndex i = 0; i < n_gradients; i++) {
      //    if (other->dense_rep[i] != 0.0) this->accumulate(i, other->dense_rep[i]);
      //  }
      //} else {
      //  adept::uIndex* active_entries = other->get_active_entries();
      //  int64_t n_other = other->get_n_active_entries();
      //  for (int64_t i = 0; i < n_other; i++) {
      //    this->accumulate(active_entries[i], other->extract_value(active_entries[i]));
      //  }
      //  if (active_entries != NULL) {
      //    free(active_entries);
      //    active_entries = NULL;
      //  }
      //}

      //if (n_operations_done > n_gradients/64) {
      //  // I am in dense representation.
      //  

      //}
    }

 
    adept::uIndex* tfk_gradient_table::get_active_entries() {
      if (gradient_table_local.size() == 0) {
        n_active_entries = 0;
        return NULL;
      }
      adept::uIndex* to_ret_active_entries = (adept::uIndex*) malloc(sizeof(adept::uIndex)*gradient_table_local.size());
      n_active_entries = 0;
      for (auto iter = gradient_table_local.begin(); iter != gradient_table_local.end(); ++iter) {
        to_ret_active_entries[n_active_entries++] = iter->first;
      }
      return to_ret_active_entries;
    }


