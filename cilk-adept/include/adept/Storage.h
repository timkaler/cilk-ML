/* Storage.h -- store array of active or inactive data

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The Storage class manages the data underlying array objects, and
   uses a model of reference counting so that multiple objects can
   refer to the same data.  This enables arrays that are actually
   subsets of another array to be treated as normal array objects.

*/

#ifndef AdeptStorage_H
#define AdeptStorage_H 1

#include <string>
#include <sstream>

#include <adept/exception.h>
#include <adept/base.h>
#include <adept/Stack.h>
#include <adept/Packet.h>

#ifdef ADEPT_STORAGE_THREAD_SAFE
#include <atomic>
#endif


namespace adept {

  // -------------------------------------------------------------------
  // Global variables
  // -------------------------------------------------------------------
  //namespace internal {
  //  // To check for memory leaks, we keep a running total of the number
  //  // of Storage objects that are created and destroyed
  //  extern Index n_storage_objects_created_;
  //  extern Index n_storage_objects_deleted_;
  //}

  // -------------------------------------------------------------------
  // Definition of Storage class
  // -------------------------------------------------------------------
  template <typename Type>
  class Storage {
  public:
    // -------------------------------------------------------------------
    // Storage: 1. Constructors and destructor
    // -------------------------------------------------------------------

    // The only way to construct this object is by passing it an
    // integer indicating the size, and optionally for active objects,
    // an integer representing the index to the gradients stored in
    // the stack.
    Storage(Index n, bool IsActive = false)
      : n_(n), n_links_(1), gradient_index_(-1) {
      //for (int i = 0; i < 36*128; i++) {
      //  n_links_arr[i] = 0;
      //  n_links_locks[i] = 0;
      //}
      //n_links_arr[__cilkrts_get_worker_number()*64] = 1;
      data_ = internal::alloc_aligned<Type>(n);
      //internal::n_storage_objects_created_++; 
#ifndef ADEPT_NO_AUTOMATIC_DIFFERENTIATION
      if (IsActive) {
	gradient_index_ = ADEPT_ACTIVE_STACK->register_gradients(n);
      }
#endif
    }
    
  protected:
    // Only allow the class to destroy itself by putting in
    // "protected".  FIX - would be better to start valid
    // gradient_index at 1, so 0 is reserved for invalid values.
    ~Storage() {
      internal::free_aligned(data_);
// NOTE(TFK): Remove destructor to avoid segfault at end of execution.
//   Need to look into why this is happening.
#ifndef ADEPT_NO_AUTOMATIC_DIFFERENTIATION
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
	if (gradient_index_ >= 0) {
	  ADEPT_ACTIVE_STACK->unregister_gradients(gradient_index_, n_);
	}
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
#endif
      //internal::n_storage_objects_deleted_++; 
    }


    // Null initialization, copy and assignment methods that are
    // "protected" to prevent them being used
    Storage() { }
    Storage(Storage& storage) { };
    void operator=(Storage& storage) { };


    // -------------------------------------------------------------------
    // Storage: 2. Public member functions
    // -------------------------------------------------------------------  
  public:
    // Add link to an existing storage object
    void add_link()
    { n_links_++;}//__sync_fetch_and_add(&n_links_arr[128], 1);/*n_links_++;*/ } 

    // Remove link as follows; this is only safe in a multi-threaded
    // environment if ADEPT_STORAGE_THREAD_SAFE is defined, making
    // n_links_ atomic
    void remove_link() {

      //int wid = __cilkrts_get_worker_number();

      //while (!__sync_bool_compare_and_swap(&n_links_locks[64*wid], 0, 1)) {
      //  continue;
      //}

      //if (n_links_arr[64*wid] == 0) {
      //  // need to steal.
      //  bool success = false;
      //  while (!success) {
      //    for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
      //      if (n_links_arr[64*i] <= 0) continue;
      //      if (__sync_bool_compare_and_swap(&n_links_locks[64*i], 0, 1)) {
      //        if (n_links_arr[64*i] > 0) {
      //          n_links_arr[64*i]--;
      //          n_links_arr[64*wid]++;
      //          n_links_locks[64*i] = 0;
      //          success = true;
      //          break;
      //        }
      //      }
      //    }
      //  }
      //}

      //assert(n_links_arr[64*wid] > 0);
      //if (--n_links_arr[64*wid] == 0) {
      //  delete this;
      //}
      //n_links_locks[64*wid] = 0;
      //return;




      //if (n_links_arr[128] == 0) {
      ////if (n_links_arr[128] == 0) {
      //  throw invalid_operation("Attempt to remove more links to a storage object than set"
      //  			ADEPT_EXCEPTION_LOCATION);
      //} else {

      //}

      //else if (__sync_sub_and_fetch(&n_links_arr[128], 1) == 0) {
      //  delete this;
      //}
      if (--n_links_ == 0) {
        delete this;
      }
    }

    // Return the number of elements allocated
    Index n_allocated() const
    { return n_; }

    // Return the number of links to an object
    int n_links() const
    { return n_links_;}//n_links_arr[128];}//n_links_; }

    Index gradient_index() const
    { return gradient_index_; }

    // Return pointer to the start of the data
    Type*
    data()
    { return data_; }
    const Type*
    data() const
    { return data_; }

    // Return a string of information
    std::string
    info_string() const {
      std::stringstream x;
      x << n_ << " " << sizeof(Type) << "-byte elements allocated with "
	<< n_links_ << " links"; //n_links_ << " links";
      return x.str();
    }

    // -------------------------------------------------------------------
    // Storage: 3. Data
    // -------------------------------------------------------------------  
  private:
    // Pointer to the start of the data
    Type* data_;
    // Number of elements allocated
    Index n_;
    // Number of links to the storage object allowing for arrays and
    // array slices to point to the same data. If this falls to zero
    // the Storage object will destruct itself
#ifdef ADEPT_STORAGE_THREAD_SAFE
    // If multiple threads are to simultaneously read subsets of this
    // array then accesses to the reference counter must be made
    // atomic

//    int n_links_arr[36*128];
    std::atomic<int> n_links_;
//    int n_links_locks[36*128];


#else
    int n_links_;
#endif
    // For active variables, this s the gradient index of the first
    // element.  It would be better to only store this if Type is
    // floating point.
    Index gradient_index_;

  }; // End of Storage class
  

  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------
  inline Index n_storage_objects()
  { return 0;}/*return internal::n_storage_objects_created_
      - internal::n_storage_objects_deleted_; }*/

  inline Index n_storage_objects_created()
  { return 0;}//internal::n_storage_objects_created_; }
  
  inline Index n_storage_objects_deleted()
  { return 0;}//internal::n_storage_objects_deleted_; }
  
} // End namespace adept

#endif
