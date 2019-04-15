
  tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, tfk_gradient_table* gradient_table_) {
    gradient_table = gradient_table_;
    raw_gradient_table = NULL;

    active_entries = NULL;
    n_active_entries = 0;

    //gradient_table_local = ska::flat_hash_map<adept::uIndex, adept::Real>(size_hint);

    //gradient_table_local_dense = NULL;
    //local_dense_start = 0;
    //local_dense_end = 0;
  }


  tfk_gradient_table::tfk_gradient_table(uint64_t n_gradients, adept::Real* gradient_table_raw_) {
    gradient_table = NULL;
    raw_gradient_table = gradient_table_raw_;

    active_entries = NULL;
    n_active_entries = 0;

    //gradient_table_local_dense = NULL;
    //local_dense_start = 0;
    //local_dense_end = 0;
  }

  tfk_gradient_table::~tfk_gradient_table() {
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
      gradient_table_local[index] = 0.0;
      gradient_table_local.erase(index);

      // NOTE(TFK): Technically we need to extract value from parent.
      return a + gradient_table->extract_value(index);
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


