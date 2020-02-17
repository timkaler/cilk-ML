// Copyright (c) 2019, Tim Kaler - MIT License
#include <cilk/cilk.h>
#include <cilk/reducer.h>
#include <cilk/reducer_opadd.h>
#include <cilk-adept-headers/sp_tree.h>

#include <cilk-adept-source/gradient_table.cpp>

#include <cilk-adept-source/tfk_shadowmem.cpp>

#include <cilk-adept-source/sp_node.cpp>

#include <adept.h>

#include <vector>
#include <map>
#include "../../../common/utils.h"
#include "../../../common/blockRadixSort.h"
#include "../../../common/gettime.h"
#include "../../../common/semisort.h"
#define SPTREE_spawn cilk_spawn 
#define SPTREE_parfor cilk_for
#define SPTREE_sync cilk_sync


extern wl_stacks* worker_local_stacks;
extern tfkdiff tfk_reducer;

//__attribute__((always_inline))
//void helper_set_bit(uint8_t* byte_array, int idx) {
//  byte_array[idx/8] |= (1 << (idx%8));
//  //byte_array[idx] = true;
//}
//
//__attribute__((always_inline))
//bool helper_get_bit(uint8_t* byte_array, int idx) {
//  //return byte_array[idx];
//  return (byte_array[idx/8] & (1 << (idx % 8)));
//}


timer r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r_copy,r_call, r_small;

void SP_Tree::walk_tree_process_semisort(SP_Node* n, float** worker_local_grad_table, bool* appears_in_statement, float* gradient_) {

  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
       //delete n;
       return;
    }
    int wid = __cilkrts_get_worker_number();
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        //assert(statement.index != -1 && "Why is statement index -1?\n");
        if (statement.index == -1) continue;

        // Perform the extraction.
        //adept::Real a = gradient_[statement.index];
        //for (int j = 0; j < __cilkrts_get_nworkers(); j++) {
        //  a += worker_local_grad_table[j][statement.index];
        //  worker_local_grad_table[j][statement.index] = 0;
        //}
        float* extract_arr = worker_local_stacks[stack.worker_id].statement_stack_deposit_location[ist];
        int extract_arr_len = worker_local_stacks[stack.worker_id].statement_stack_deposit_location_len[ist];
        adept::Real a = gradient_[statement.index];
        gradient_[statement.index] = 0;
        //if (extract_arr_len > 5000) printf("extract arr len is %d\n", extract_arr_len);
        int nonzero_count = 0;

        if (extract_arr_len > 5000) {
          cilk::reducer_opadd<float> red_a(a);
          cilk_for (int i = 0; i < extract_arr_len; i++) {
            *red_a += extract_arr[i];
            //a += extract_arr[i];
            extract_arr[i] = 0;
          }
          a += red_a.get_value();
        } else {
           for (int i = 0; i < extract_arr_len; i++) {
            a += extract_arr[i];
            extract_arr[i] = 0;
          }
        }

        if (a != 0.0) {
           if (statement.end_plus_one - worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one <= 5000 || true) {


             // partition the statement stack.
             //int size = statement.end_plus_one - worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
             //adept::uIndex* indices1 = (adept::uIndex*) malloc(sizeof(float*) * size);
             //adept::uIndex* indices2 = (adept::uIndex*) malloc(sizeof(float*) * size);
             //int count1 = 0;
             //int count2 = 0;
             //for (adept::uIndex j =
             //       worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
             //       j < statement.end_plus_one; j++) {
             //  if(worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid[j]) {
             //    indices1[count1++] = j;
             //  } else {
             //    indices2[count2++] = j;
             //  }
             //}

             //for (int _j = 0; _j < count1; _j++) {
             //  int j = indices1[_j];
             //  adept::Real multiplier_test =
             //      worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             //  adept::uIndex operation_stack_index =
             //      worker_local_stacks[stack.worker_id].operation_stack_arr[j];
             //  *(worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j]) += multiplier_test*a;
             //}

             //for (int _j = 0; _j < count2; _j++) {
             //  int j = indices2[_j];
             //  adept::Real multiplier_test =
             //      worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             //  adept::uIndex operation_stack_index =
             //      worker_local_stacks[stack.worker_id].operation_stack_arr[j];
             //  worker_local_grad_table[wid][operation_stack_index] += multiplier_test*a;
             //}
             //free(indices1);
             //free(indices2);
             for (adept::uIndex j =
                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               adept::Real multiplier_test =
                   worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               if (/*appears_in_statement[operation_stack_index] &&*/ worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid[j]) {
                 *(worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j]) += multiplier_test*a;
                 //gradient_[operation_stack_index] += multiplier_test*a;
               } else {
               //gradient_[operation_stack_index] += multiplier_test*a;
               worker_local_grad_table[wid][operation_stack_index] += multiplier_test*a;
               }
             }
           } else {
             cilk_for (adept::uIndex j =
                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               adept::Real multiplier_test =
                   worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               if (appears_in_statement[operation_stack_index] && worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j] != NULL) {
                 *(worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j]) += multiplier_test*a;
                 //gradient_[operation_stack_index] += multiplier_test*a;
               } else {
                 //gradient_[operation_stack_index] += multiplier_test*a;
                 worker_local_grad_table[wid][operation_stack_index] += multiplier_test*a;
               }
             }

           }
       }
     }
    return;
  }

  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    }

  } else if (n->type == 2) {
    //if (n->children->size() < 8) {
    for (int j = 0; j < n->children->size(); j++) {
      cilk_spawn walk_tree_process_semisort((*(n->children))[n->children->size()-j-1], worker_local_grad_table, appears_in_statement, gradient_);
    }
    cilk_sync;
    //}
    //#pragma cilk grainsize 1
    //cilk_for (int j = 0; j < n->children->size(); j++) {
    //  int i = n->children->size()-j-1;
    //  walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    //}
  }

}


void SP_Tree::collect_ops_for_semisort(SP_Node* n, args_for_collect_ops* args, worker_local_vector<OperationReference>& wl_ops) {
  if (n->type == 3) {
    triple_vector_wl stack = n->data;


    adept::uIndex*__restrict operation_stack_arr = worker_local_stacks[stack.worker_id].operation_stack_arr;
    //adept::Real*__restrict multiplier_stack_arr = worker_local_stacks[stack.worker_id].multiplier_stack_arr;
    const adept::Statement*__restrict statement_stack_arr = worker_local_stacks[stack.worker_id].statement_stack_arr;


    float** __restrict operation_stack_deposit_location =
       worker_local_stacks[stack.worker_id].operation_stack_deposit_location;

    bool* __restrict operation_stack_deposit_location_valid =
       worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid;

    bool*__restrict idx_in_statement = args->idx_in_statement;

    if (stack.statement_stack_end != stack.statement_stack_start) {
      int wid = __cilkrts_get_worker_number();
      for (adept::uIndex ist = stack.statement_stack_start; ist < stack.statement_stack_end; ist++) {
        //const adept::Statement& statement =
        //    worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        const adept::Statement& statement = statement_stack_arr[ist];

        if (statement.index == -1) continue;

        //for (adept::uIndex j =
        //     worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
        //     j < statement.end_plus_one; j++) {
        for (adept::uIndex j =
             statement_stack_arr[ist-1].end_plus_one;
             j < statement.end_plus_one; j++) {
          //adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
          adept::uIndex op_index = operation_stack_arr[j];
          //adept::Real op_mul = multiplier_stack_arr[j];
          //if (op_mul == 0.0) continue;
          if (idx_in_statement[op_index]) {
            if (stack.worker_id == args->last_statement_worker[op_index] &&
                stack.statement_stack_start <= args->last_statement_index[op_index] &&
                stack.statement_stack_end > args->last_statement_index[op_index]) {
              // In this case just deposit directly into the global gradient table.
              //worker_local_stacks[stack.worker_id].operation_stack_deposit_location[j] = &args->gradient_[op_index];
              //worker_local_stacks[stack.worker_id].operation_stack_deposit_location_valid[j] = true;
              operation_stack_deposit_location[j] = &args->gradient_[op_index];
              operation_stack_deposit_location_valid[j] = true;
              //helper_set_bit(operation_stack_deposit_location_valid, j);
            } else {
              OperationReference ref;
              ref.statement_wid = args->last_statement_worker[op_index];
              ref.statement_ist = args->last_statement_index[op_index];
              ref.operation_wid = stack.worker_id;
              ref.operation_j = j;
              ref.gradient_index = op_index;
              wl_ops.push_back(wid, ref);
            }
          }
        }

        args->last_statement_worker[statement.index] = stack.worker_id;
        args->last_statement_index[statement.index] = ist;

      }


      //for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      //  const adept::Statement& statement =
      //      worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      //  if (statement.index == -1) continue;
      //    last_statement_worker[statement.index] = stack.worker_id;
      //    last_statement_index[statement.index] = ist;

      //}

    }
    return;
  }

  if (n->type == 2) {
    for (int i = 0; i < n->children->size(); i++) {
      cilk_spawn collect_ops_for_semisort((*(n->children))[i], args, wl_ops);
    }
    cilk_sync;
    //#pragma cilk grainsize 1
    //cilk_for (int i = 0; i < n->children->size(); i++) {
    //  collect_ops_for_semisort((*(n->children))[i], idx_in_statement, last_statement_worker, last_statement_index, wl_ops);
    //}
  } else {
    for (int i = 0; i < n->children->size(); i++) {
      collect_ops_for_semisort((*(n->children))[i], args, wl_ops);
    }
  }

}


void SP_Tree::test(int64_t n_gradients, float* _gradient) {


  r8.start();
  //r0.start();
  r_copy.start();


  r_small.start();
  // First identify all gradient indices that appear in statements.
  bool* appears_in_statement = new bool[n_gradients];
  cilk_for (int i = 0; i < n_gradients; i++) {
    appears_in_statement[i] = false;
  }

  cilk::reducer_opadd<int> red_nstatements(0);
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    wl_stacks worker_stack = worker_local_stacks[i];
    *red_nstatements += worker_stack.statement_stack_arr_len;
    cilk_for (int j = 0; j < worker_stack.statement_stack_arr_len; j++) {
      if (worker_stack.statement_stack_arr[j].index >= 0 && !appears_in_statement[worker_stack.statement_stack_arr[j].index]) {
        appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
      }
    }
  }
  int64_t nstatements = red_nstatements.get_value()+1;
  //r0.stop();

  //r1.start();
  // now do a left first walk of the tree.
  int8_t* last_statement_worker = new int8_t[n_gradients];
  int32_t* last_statement_index = new int32_t[n_gradients];
  SP_Node** last_statement_node = new SP_Node*[n_gradients];

  cilk_for (uint64_t i = 0; i < n_gradients; i++) {
    last_statement_worker[i] = -1;
    last_statement_index[i] = -1;
    last_statement_node[i] = NULL;
  }

  //std::vector<OperationReference> ops;
  OperationReference* ops;


  //std::vector<OperationReference>* wl_ops = new std::vector<OperationReference>[__cilkrts_get_nworkers()]();

  worker_local_vector<OperationReference> wl_ops;

  int64_t op_stack_len = 0;
  for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    op_stack_len += worker_local_stacks[i].operation_stack_arr_len;
  }


  int* statement_offsets = new int[__cilkrts_get_nworkers()];
  statement_offsets[0] = 0;
  //int nstatements = worker_local_stacks[0].statement_stack_arr_len;
  for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
    statement_offsets[i] = statement_offsets[i-1] + worker_local_stacks[i-1].statement_stack_arr_len;
    //nstatements += worker_local_stacks[i].statement_stack_arr_len;
  }

  wl_ops.reserve((op_stack_len*2)/__cilkrts_get_nworkers());
  //r1.stop();
  r_small.stop();


  r_copy.stop();
  r_call.start();


  cilk_for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
     cilk_for (int64_t i = 0; i < worker_local_stacks[wid].statement_stack_arr_len; i++) {
       worker_local_stacks[wid].statement_stack_deposit_location_len[i] = 0;
     }
     //memset(worker_local_stacks[wid].operation_stack_deposit_location_valid, 0, worker_local_stacks[wid].operation_stack_arr_len);
     cilk_for (int64_t i = 0; i < worker_local_stacks[wid].operation_stack_arr_len; i++) {
       worker_local_stacks[wid].operation_stack_deposit_location_valid[i] = 0;
       //arr[i] = 0;
     }
  //  if (worker_local_stacks[wid].statement_stack_deposit_location == NULL) {
  //    worker_local_stacks[wid].statement_stack_deposit_location = (float**)
  //        malloc(sizeof(float*) * worker_local_stacks[wid].statement_stack_arr_len);
  //    worker_local_stacks[wid].statement_stack_deposit_location_len = (int*)
  //        calloc(worker_local_stacks[wid].statement_stack_arr_len, sizeof(int));
  //    worker_local_stacks[wid].operation_stack_deposit_location = (float**)
  //        malloc(sizeof(float*) * worker_local_stacks[wid].operation_stack_arr_len);
  //  } else {
  //    worker_local_stacks[wid].statement_stack_deposit_location = (float**)
  //        realloc(worker_local_stacks[wid].statement_stack_deposit_location, sizeof(float*) * worker_local_stacks[wid].statement_stack_arr_len);
  //    worker_local_stacks[wid].statement_stack_deposit_location_len = (int*)
  //        realloc(worker_local_stacks[wid].statement_stack_deposit_location_len, worker_local_stacks[wid].statement_stack_arr_len*sizeof(int));
  //    worker_local_stacks[wid].operation_stack_deposit_location = (float**)
  //        realloc(worker_local_stacks[wid].operation_stack_deposit_location, sizeof(float*) * worker_local_stacks[wid].operation_stack_arr_len);
  //    memset(worker_local_stacks[wid].statement_stack_deposit_location_len, 0, sizeof(int)*worker_local_stacks[wid].statement_stack_arr_len);
  //  }
  }



  r2.start();
  args_for_collect_ops args;
  args.idx_in_statement = appears_in_statement;
  args.last_statement_worker = last_statement_worker;
  args.last_statement_index = last_statement_index;
  //args.last_statement_node = last_statement_node;
  args.gradient_ = _gradient;
  collect_ops_for_semisort(get_root(), &args, wl_ops);
  r2.stop();
  r_call.stop();

  r_copy.start();
  r3.start();
  int64_t ops_size = wl_ops.collect(ops);
  r3.stop();

  //for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  for (int j = 0; j < wl_ops[i].size(); j++) {
  //    ops.push_back(wl_ops[i][j]);
  //  }
  //}

  r4.start();
  //std::vector<std::pair<int64_t, int> > mapped_ops(ops_size);

  r9.start();
  std::pair<int, int>* mapped_ops = (std::pair<int, int>*) malloc(sizeof(std::pair<int, int>) * ops_size);
  int64_t mapped_ops_size = ops_size;

  //std::sort(ops.begin(), ops.end());

  // Map for sort.
  cilk_for (uint64_t i = 0; i < ops_size; i++) {
    //mapped_ops[i] = std::make_pair((int64_t) &(worker_local_stacks[ops[i].statement_wid].statement_stack_arr[ops[i].statement_ist]) /*ops[i].gradient_index*/, (int)i);
    //mapped_ops[i] = std::make_pair(((((int64_t) ops[i].statement_wid)*nstatements) + (int64_t)ops[i].statement_ist) /*ops[i].gradient_index*/, (int64_t)i);
    mapped_ops[i] = std::make_pair(statement_offsets[ops[i].statement_wid] + ops[i].statement_ist /*ops[i].gradient_index*/, (int)i);
  }
  r9.stop();
  r_copy.stop();
  r_call.start();
  r10.start();
  //int maxV = sequence::mapReduce<int>(&mapped_ops[0], mapped_ops_size, utils::maxF<int>(), utils::firstF<int, int>());
  intSort::iSort(&mapped_ops[0], mapped_ops_size, nstatements+1, utils::firstF<int, int>());
  //semisort(&mapped_ops[0], ops_size);
  r10.stop();

  r_call.stop();
  r_copy.start();

  //std::sort(mapped_ops.begin(), mapped_ops.end());

  r11.start();
  // Now identify blocks.
  int* boundaries;

  worker_local_vector<int> wl_boundaries;
  //int* wl_boundaries;

  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    if (i == 0 || mapped_ops[i].first != mapped_ops[i-1].first) {
      wl_boundaries.push_back(__cilkrts_get_worker_number(), i);
    }
  }
  int64_t boundaries_size = wl_boundaries.collect(boundaries);
  r11.stop();

  r_copy.stop();
  r_call.start();

  r12.start();
  intSort::iSort(&boundaries[0], boundaries_size, mapped_ops_size, utils::identityF<int>());
  r12.stop();

  r_call.stop();
  r_copy.start();
  //std::sort(boundaries.begin(), boundaries.end());

  r13.start();
  std::pair<int,int>* blocks = (std::pair<int,int>*) malloc(sizeof(std::pair<int,int>)*boundaries_size);
  int64_t blocks_size = boundaries_size;
  //std::vector<std::pair<int, int> > blocks(boundaries_size);

  cilk_for (uint64_t i = 1; i < boundaries_size; i++) {
    blocks[i-1] = (std::make_pair(boundaries[i-1], boundaries[i]));
  }
  blocks[boundaries_size-1] = std::make_pair(boundaries[boundaries_size-1], mapped_ops_size);


  // now augment each worker's statement stack with a pointer to extra data.

  float* deposit_locations = new float[mapped_ops_size];
  cilk_for (uint64_t i = 0; i < mapped_ops_size; i++) {
    deposit_locations[i] = 0;
  }

  r13.stop();
  r4.stop();

  r5.start();
  //cilk_for (uint64_t i = 0; i < blocks_size; i++) {
  //  //printf("block %llu size is %llu\n", i, blocks[i].second-blocks[i].first);


  //  // perform the write for the statement.
  //  if (blocks[i].second > blocks[i].first) {
  //      OperationReference& opref = ops[mapped_ops[blocks[i].first].second];
  //      if (opref.statement_wid != -1) {
  //        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
  //        worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
  //      }
  //  }
  //  //cilk_for (uint64_t j = blocks[i].first; j < blocks[i].second; j++) {
  //  //}

  //}

  

  if (false) {
    //mapped_ops_size;
    std::pair<int, int>* j_to_pair = (std::pair<int,int>*) malloc(sizeof(std::pair<int,int>)*mapped_ops_size);
    cilk_for (uint64_t i = 0; i < blocks_size; i++) {
      cilk_for (uint64_t j = blocks[i].first; j < blocks[i].second; j++) {
        // map mapped_ops[j].second to 
        j_to_pair[j] = std::make_pair(mapped_ops[j].second, (int)j);
      }
    }

    intSort::iSort(j_to_pair, mapped_ops_size, mapped_ops_size, utils::firstF<int, int>());

    cilk_for (int64_t i = 0; i < mapped_ops_size; i++) {
      //ops[j_to_pair[i].first]
      OperationReference& opref = ops[j_to_pair[i].first];

        // Map the statement.
        if (opref.statement_wid != -1) {
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = deposit_locations + j_to_pair[i].second;
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = true;//deposit_locations + j_to_pair[i].second;
          //helper_set_bit((uint8_t*)worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid, opref.operation_j);
        } else {
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = false;
        }
    }
    free(j_to_pair);
  } else {
    cilk_for (uint64_t i = 0; i < blocks_size; i++) {
      //printf("block %llu size is %llu\n", i, blocks[i].second-blocks[i].first);


      // perform the write for the statement.
      if (blocks[i].second > blocks[i].first) {
          OperationReference& opref = ops[mapped_ops[blocks[i].first].second];
          if (opref.statement_wid != -1) {
            worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
            worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
          }
      }
      cilk_for (uint64_t j = blocks[i].first; j < blocks[i].second; j++) {
        //if (mapped_ops[j].first != mapped_ops[blocks[i].first].first) {
        //  printf("ERROR!!!!\n "); assert(false);
        //}
        OperationReference& opref = ops[mapped_ops[j].second];

        // first we need to group ones with the same statement index together.

        // Map the statement.
        if (opref.statement_wid != -1) {
          //if (worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] == 0) {
          //  worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = deposit_locations + blocks[i].first;
          //  worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = blocks[i].second - blocks[i].first;
          //}
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location[opref.operation_j] = deposit_locations + j;
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = true;
          //helper_set_bit((uint8_t*)worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid, opref.operation_j);
        } else {
          worker_local_stacks[opref.operation_wid].operation_stack_deposit_location_valid[opref.operation_j] = false;
        } /*else {
          worker_local_stacks[opref.statement_wid].statement_stack_deposit_location[opref.statement_ist] = NULL; //deposit_locations + blocks[i].first;
          worker_local_stacks[opref.statement_wid].statement_stack_deposit_location_len[opref.statement_ist] = 0; //blocks[i].second - blocks[i].first;
        }*/

        //if (blocks[i].second - blocks[i].first <= 0) {
        //  printf("error block size zero!\n");
        //  assert(false);
        //}
      }

    }
  }
  r14.start();
  float** worker_local_grad_table = (float**) malloc(sizeof(float*) * __cilkrts_get_nworkers());
  cilk_for (int i = 0; i < __cilkrts_get_nworkers(); i++) {
    worker_local_grad_table[i] = (float*) calloc(n_gradients, sizeof(float));
  }
  r14.stop();
  r5.stop();


  r_copy.stop();
  r_call.start();
  r6.start();
  walk_tree_process_semisort(get_root(), worker_local_grad_table, appears_in_statement, _gradient);
  r6.stop();

  r_call.stop();
  r_copy.start();

  r7.start();
  int n_workers = __cilkrts_get_nworkers();

  //cilk_for (int64_t i = 0; i < mapped_ops.size(); i++) {
  //  if(deposit_locations[i] != 0.0) {
  //    printf("deposit location isn't zero %e, %llu\n", deposit_locations[i], i);
  //    //assert(false);
  //  }
  //}

  int64_t max_gradient = tfk_reducer.max_gradient;
  printf("max gradient is %llu\n", max_gradient);
  cilk_for (int64_t i = 0; i < max_gradient; i++) {
    _gradient[i] = 0;
    for (int wid = 0; wid < n_workers; wid++) {
      _gradient[i] += worker_local_grad_table[wid][i];
    }
  }

  r_copy.stop();

  cilk_for (int i = 0; i < n_workers; i++) {
    free(worker_local_grad_table[i]);
  }
  free(worker_local_grad_table);


  //cilk_for (int wid = 0; wid < __cilkrts_get_nworkers(); wid++) {
  //  free(worker_local_stacks[wid].statement_stack_deposit_location);
  //  free(worker_local_stacks[wid].statement_stack_deposit_location_len);
  //  free(worker_local_stacks[wid].operation_stack_deposit_location);
  //}
  delete[] deposit_locations;
  delete[] last_statement_worker;
  delete[] last_statement_index;
  delete[] last_statement_node;
  delete[] appears_in_statement;
  free(blocks);
  free(boundaries);
  free(ops);
  free(mapped_ops);
  r7.stop();
  r8.stop();

  //r0.reportTotal("r0");
  //r1.reportTotal("r1");
  r_small.reportTotal("r_small");
  r2.reportTotal("r2");
  r3.reportTotal("r3");
  r4.reportTotal("r4");
  r5.reportTotal("r5");
  r6.reportTotal("r6");
  r7.reportTotal("r7");
  r8.reportTotal("r8");
  r9.reportTotal("r9");
  r10.reportTotal("r10");
  r11.reportTotal("r11");
  r12.reportTotal("r12");
  r13.reportTotal("r13");
  r14.reportTotal("r14");
  r_copy.reportTotal("r_copy");
  r_call.reportTotal("r_call");
  return;
  //int64_t* offsets = new int64_t[__cilkrts_get_nworkers()]();
  //int64_t total_operations = worker_local_stacks[0].operation_stack_arr_len;

  //float* deposit_locations = new float[total_operations];



  //offsets[0] = 0;
  //for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
  //  offsets[i] = offsets[i-1] + worker_local_stacks[i-1].operation_stack_arr_len;
  //  total_operations += worker_local_stacks[i].operation_stack_arr_len;
  //  //worker_local_stacks[i].deposit_index = new float[worker_local_stacks[i].operation_stack_arr_len]();
  //}


  //// associate each operation with a statement.

  ////std::vector<SP_Node*> nodes;
  ////walk_tree_flatten_datanodes(this->get_root(), nodes);


  //std::pair<int, int>* op_pairs = (std::pair<int, int>*) malloc(sizeof(std::pair<int, int >) * total_operations);

  //bool* appears_in_statement = new bool[n_gradients];
  //SPTREE_parfor (int i = 0; i < n_gradients; i++) {
  //  appears_in_statement[i] = false;
  //}

  //SPTREE_parfor (int i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  wl_stacks worker_stack = worker_local_stacks[i];

  //  SPTREE_parfor (int j = 0; j < worker_stack.statement_stack_arr_len; j++) {
  //    if (worker_stack.statement_stack_arr[j].index >= 0) {
  //      appears_in_statement[worker_stack.statement_stack_arr[j].index] = true;
  //    }
  //  }
  //}

  //printf("before the copy total operations %llu\n", total_operations);

  //int64_t operations_after_filter = 0;

  //std::vector<std::pair<int64_t, int64_t> >* wl_op_vectors = new std::vector<std::pair<int64_t, int64_t> >[__cilkrts_get_nworkers()]();

  //SPTREE_parfor (int i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  wl_stacks worker_stack = worker_local_stacks[i];
  //  auto wl_op_pairs = op_pairs + offsets[i];

  //  for (int j = 0; j < worker_stack.operation_stack_arr_len; j++) {
  //    if (appears_in_statement[worker_stack.operation_stack_arr[j]]) {
  //      //operations_after_filter++;
  //      wl_op_vectors[__cilkrts_get_worker_number()].push_back(std::make_pair(worker_stack.operation_stack_arr[j], i*total_operations + j));
  //      //wl_op_pairs[j] = std::make_pair(worker_stack.operation_stack_arr[j], i*total_operations + j);
  //    }
  //  }
  //}

  //int64_t* offsets2 = new int64_t[__cilkrts_get_nworkers()];
  //offsets2[0] = 0;
  //int64_t total_size = wl_op_vectors[0].size();
  //for (int i = 1; i < __cilkrts_get_nworkers(); i++) {
  //  offsets2[i] = offsets2[i-1] + wl_op_vectors[i-1].size();
  //  total_size += wl_op_vectors[i].size();
  //}

  //std::vector<std::pair<int64_t, int64_t> > op_vector(total_size);
  //SPTREE_parfor (int64_t i = 0; i < __cilkrts_get_nworkers(); i++) {
  //  SPTREE_parfor (int64_t j = 0; j < wl_op_vectors[i].size(); j++) {
  //    op_vector[offsets2[i] + j] = wl_op_vectors[i][j];
  //  }
  //}
  //delete[] offsets2;
  //delete[] wl_op_vectors;
  //printf("operations after filter %llu\n", op_vector.size());


  ////int64_t count = 0;
  ////for (int i = 0; i < n_gradients; i++) {
  ////  if (appears_in_statement[i]) count++;
  ////}
  ////printf("number of gradients appearing in statement is %d\n", count);

  ////intSort::iSort(op_pairs, total_operations, n_gradients, utils::firstF<int, int>());

  //printf("before the sort total operations %llu, %llu\n", total_operations, op_pairs[0].first);

  //intSort::iSort(&(op_vector[0]), op_vector.size(), n_gradients, utils::firstF<int64_t, int64_t>());
  //printf("after the sort %llu,%llu\n", op_vector[0].first, op_vector[0].second);
  ////// replace with semisort.
  ////std::sort(op_pairs, op_pairs + total_operations);

  ////

  ////printf("done with the sort total operations %llu\n", total_operations);

  ////for (int i = 0; i < total_operations; i++) {
  ////  if (i==0 || op_pairs[i].first != op_pairs[i-1].first) {
  ////    // record this location for the statement.
  ////  }
  ////  worker_local_stacks[op_pairs[i].second.first].deposit_index[op_pairs[i].second.second] = offsets[i] + op_pairs[i].second.second;
  ////}
  //free(op_pairs);
  //delete[] offsets;
  //delete[] deposit_locations;
}


// init can happen at the root of the program, and upon a steal.
// Upon a steal: a continuation was stolen. Upon a sync the parent node ought to be a P node.
void SP_Tree::init() {
  SP_Node*& current_node = imp_.view();
  //current_node = get_root();
  current_node->type = 1;
  current_node->parent = NULL;
  recording = false;
  if (current_node->children != NULL) {
    //#pragma cilk grainsize 1
    SPTREE_parfor (int i = 0; i < current_node->children->size(); i++) {
      delete (*(current_node->children))[i];
    }
    delete current_node->children;
  }
  current_node->children = new std::vector<SP_Node*>();
  recording = true;
}

SP_Node* SP_Tree::get_root() {
  SP_Node* current_node = imp_.view();
  while (current_node->parent != NULL) {
    current_node = current_node->parent;
  }
  return current_node;
}

// currently has a memory leak.
void SP_Tree::clear() {
  //bool saved_recording = recording; 
  //recording = false;
  this->init();
  //recording = saved_recording;
}


int SP_Tree::walk_tree_rootset_transform(SP_Node* n, int dep_count) {
  // Data node.
  if (n->type == 3) {
    n->rootset_id = dep_count;
    assert(n->children->size() == 0 && "A data node should not have any children.\n");
    //printf("D node dep_count is %d\n", dep_count);
    return dep_count;
  }
  n->rootset_id = 0;
  // Series node.
  if (n->type == 1 || n->type == 0) {
    int number_of_data_nodes = 0;
    int added_dep_count = 0;
    for (int i = n->children->size()-1; i >= 0; i--) {
      //printf("the node type is %d\n", (*n->children)[i]->type);
      dep_count = walk_tree_rootset_transform((*n->children)[i], dep_count);
      if ((*n->children)[i]->type == 3) {
        n->rootset_id = 1;
        number_of_data_nodes++;
      }

      if ((*n->children)[i]->type == 3 || (*n->children)[i]->rootset_id == 1) {
        dep_count += 1;
        added_dep_count++;
      }
    }
    if (n->children->size() > 0 && added_dep_count > 0) dep_count--;

    //for (int i = n->children->size()-1; i >= 0; i--) {
    //  printf("the node type is %d\n", (*n->children)[i]->type);
    //}
    //printf("children %d num data nodes %d\n", n->children->size(), number_of_data_nodes);
    //if (number_of_data_nodes == 1) {
    //  dep_count--; 
    //} else if (number_of_data_nodes != 0) printf();
    //printf("S node dep_count is %d\n", dep_count);
    return dep_count;
  }

  if (n->type == 2) {
    int max_child_dep_count = dep_count;
    for (int i = 0; i < n->children->size(); i++) {
      int child_dep_count = walk_tree_rootset_transform((*n->children)[i], dep_count);
      if (child_dep_count > max_child_dep_count) max_child_dep_count = child_dep_count;

      if ((*n->children)[i]->type == 3 || (*n->children)[i]->rootset_id == 1) {
        n->rootset_id = 1;
      }
    }
     
    //printf("P node dep_count is %d\n", max_child_dep_count);
    return max_child_dep_count; // returned dep_count is the maximum of all child dependence counts.
  }
  assert(false && "Illegal fall through.\n");
  return 0;
}

void SP_Tree::walk_tree_flatten_allnodes(SP_Node* n, std::vector<SP_Node*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {
      ret.push_back(n);
    }
    return;
  }
  ret.push_back(n);
  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten_allnodes((*(n->children))[i], ret);
  }
}

void SP_Tree::walk_tree_flatten_datanodes(SP_Node* n, std::vector<SP_Node*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end != stack.statement_stack_start) {
      ret.push_back(n);
    }
    return;
  }

  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten_datanodes((*(n->children))[i], ret);
  }

}


void SP_Tree::make_ids_deterministic(int64_t n_gradients) {
  std::vector<SP_Node*> data_nodes;
  walk_tree_flatten_datanodes(get_root(), data_nodes);

  int64_t* remap = new int64_t[n_gradients];
  int64_t next_id = 0;
  for (int i = 0; i < n_gradients; i++) {
    remap[i] = -1;
  }

  for (int i = 0; i < data_nodes.size(); i++) {
    triple_vector_wl stack = data_nodes[i]->data;
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      const adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      if (statement.index == -1) continue;
        if (remap[statement.index] == -1) {
          remap[statement.index] = next_id++;
        }
        if (ist == stack.statement_stack_start) {
          for (adept::uIndex j = stack.operation_stack_start;
               j < statement.end_plus_one; j++) {
            adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
            if (remap[op_index] == -1) remap[op_index] = next_id++;
          }
        } else {
          for (adept::uIndex j =
                 worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                 j < statement.end_plus_one; j++) {
            adept::uIndex op_index = worker_local_stacks[stack.worker_id].operation_stack_arr[j];
            if (remap[op_index] == -1) remap[op_index] = next_id++;
          }
        }
    }

  }

  for (int i = 0; i < data_nodes.size(); i++) {
    triple_vector_wl stack = data_nodes[i]->data;
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      if (statement.index == -1) continue;
        statement.index = remap[statement.index];
        if (ist == stack.statement_stack_start) {
          for (adept::uIndex j = stack.operation_stack_start;
               j < statement.end_plus_one; j++) {
            worker_local_stacks[stack.worker_id].operation_stack_arr[j] = remap[worker_local_stacks[stack.worker_id].operation_stack_arr[j]];
          }
        } else {
          for (adept::uIndex j =
                 worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                 j < statement.end_plus_one; j++) {
            worker_local_stacks[stack.worker_id].operation_stack_arr[j] = remap[worker_local_stacks[stack.worker_id].operation_stack_arr[j]];
          }
        }
    }
  }

  delete[] remap;

}


//void SP_Tree::walk_tree_count_gradients(SP_Node* n, int* counts) {
//  // If its a data node it must be a terminal node.
//  if (n->type == 3) {
//    triple_vector_wl stack = n->data;
//    if (stack.statement_stack_end != stack.statement_stack_start) {
//
//      for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
//          const adept::Statement& statement =
//              worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
//          if (statement.index == -1) continue;
//          if (ist == stack.statement_stack_start) {
//            for (adept::uIndex j = stack.operation_stack_start;
//                    j < statement.end_plus_one; j++) {
//               adept::uIndex operation_stack_index =
//                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
//               counts[operation_stack_index]++;
//             }
//           } else {
//             for (adept::uIndex j =
//                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
//                    j < statement.end_plus_one; j++) {
//               adept::uIndex operation_stack_index =
//                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
//               counts[operation_stack_index]++;
//             }
//           }
//       }
//    }
//    return;
//  }
//  for (int i = 0; i < n->children->size(); i++) {
//    walk_tree_count_gradients((*(n->children))[i], ret);
//  }
//}


SP_Tree* SP_Tree::transform_to_rootset_form() {
  int n_rootsets = walk_tree_rootset_transform(this->get_root(), 0);
  printf("Number of rootsets is %d\n", n_rootsets);
  std::vector<SP_Node*> data_nodes;
  std::vector<SP_Node*> all_nodes;
  walk_tree_flatten_datanodes(this->get_root(), data_nodes);
  walk_tree_flatten_allnodes(this->get_root(), all_nodes);
  printf("Number of data nodes is %llu, all nodes %llu\n", data_nodes.size(), all_nodes.size());

  std::map<int, std::vector<SP_Node*> > rootset_to_nodes;

  int max_rootset_id = 0;
  for (int i = 0; i < data_nodes.size(); i++) {
    rootset_to_nodes[data_nodes[i]->rootset_id].push_back(data_nodes[i]);
    if (data_nodes[i]->rootset_id > max_rootset_id) max_rootset_id = data_nodes[i]->rootset_id;
  }

  SP_Tree* new_tree = new SP_Tree();
  new_tree->init();

  SP_Node* new_root = new_tree->get_root();

  //new_root->children->push_back(new SP_Node(1, new_root, 0));

  //new_tree->open_S_node();
  for (int i = max_rootset_id; i >= 0; i--) {
    //new_tree->open_P_node((void*)(i+1));
    SP_Node* P_node = new SP_Node(2, new_root, 0);
    for (int j = 0; j < rootset_to_nodes[i].size(); j++) {
      SP_Node* S_node = new SP_Node(1, P_node);
      P_node->children->push_back(S_node);
      S_node->children->push_back(rootset_to_nodes[i][j]);
      //new_tree->add_D_node(rootset_to_nodes[i][j]->data);
    }
    new_root->children->push_back(P_node);
    //new_tree->close_P_node();
  }
  //new_tree->close_S_node();

  return new_tree;
}


void SP_Tree::add_D_node(triple_vector_wl data) {
  if (!recording) return;
  SP_Node* data_node = new SP_Node(data);
  SP_Node* current_node = imp_.view();
  current_node->children->push_back(data_node);
}

void SP_Tree::open_P_node(void* sync_id) {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();

  SP_Node* new_node = new SP_Node(2, current_node, sync_id);
  current_node->children->push_back(new_node);
  current_node = new_node;
}

void SP_Tree::open_P_node() {
  if (!recording) return;
  printf("DEBUG: Open P node should not be called right now without sync id\n");
  SP_Node*& current_node = imp_.view();

  if (current_node == NULL) printf("Error current node is null in open P node\n");

  SP_Node* new_node = new SP_Node(2, current_node);
  current_node->children->push_back(new_node);

  current_node = new_node;
}

void SP_Tree::close_P_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close P node\n");

  // pop up.
  SP_Node* parent = current_node->parent;
  current_node = parent;
}

void SP_Tree::sync_P_nodes(void* sync_id) {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close P node\n");

  // we need to walk up the tree to get the outer-most-nested P node to close.
  std::vector<SP_Node*> ancestors;

  SP_Node* parent = current_node;

  int num_closes_needed = 0;
  int num_closes = 0;
  while (parent != NULL) {
    num_closes++;
    if (parent->type == 2 && parent->sync_id == sync_id) {
      parent->sync_id = NULL;
      num_closes_needed = num_closes;
      //break;
    }
    if (parent->type == 0) parent->sync_id = sync_id;
    parent = parent->parent;
  }

  for (int i = 0; i < num_closes_needed; i++) {
    //printf("Close %p num_closes_needed %d\n", sync_id, num_closes_needed);
    close_P_node();
  }
}


void SP_Tree::open_S_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in open S node\n");
  SP_Node* new_node = new SP_Node(1, current_node);
  current_node->children->push_back(new_node);
  current_node = new_node;
}

void SP_Tree::close_S_node() {
  if (!recording) return;
  SP_Node*& current_node = imp_.view();
  if (current_node == NULL) printf("Error current node is null in close S node\n");

  // pop up.
  SP_Node* parent = current_node->parent;
  current_node = parent;
}


std::vector<triple_vector_wl*> SP_Tree::flatten_to_array() {
  std::vector<triple_vector_wl*> ret(0);
  this->walk_tree_flatten(this->get_root(), ret);
  return ret;
}

void SP_Tree::walk_tree_flatten(SP_Node* n, std::vector<triple_vector_wl*>& ret) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    ret.push_back(&(n->data));
    return;
  }

  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_flatten((*(n->children))[i], ret);
  }
}

tfk_gradient_table* SP_Tree::merge_gradient_table_list(
    std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {
  if (gradient_table_list.size() == 1) return gradient_table_list[0];
  if (end-start >  4) {
    int mid = start + (end-start)/2;
    tfk_gradient_table* left = SPTREE_spawn merge_gradient_table_list(gradient_table_list, start,
                                                                    mid);
    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
    SPTREE_sync;

    left->merge_into_me(right);
    return left;
    //adept::uIndex* active_entries_right = right->get_active_entries();
    //int n_active_entries_right = right->get_n_active_entries();
    //for (int i = 0; i < n_active_entries_right; i++) {
    //  //left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
    //  left->accumulate(active_entries_right[i], right->gradient_table_local[active_entries_right[i]]);
    //}
    //return left;
  }


  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];

  for (int j = start+1; j < end; j++) {
    my_gradient_table->merge_into_me(gradient_table_list[j]);
  }

  return my_gradient_table;
}


//tfk_gradient_table* SP_Tree::merge_gradient_table_list(
//    std::vector<tfk_gradient_table*>& gradient_table_list, int start, int end) {
//
//  if (end-start > 4) {
//    int mid = start + (end-start)/2;
//    tfk_gradient_table* left = cilk_spawn merge_gradient_table_list(gradient_table_list, start,
//                                                                    mid);
//    tfk_gradient_table* right = merge_gradient_table_list(gradient_table_list, mid, end);
//    cilk_sync;
//
//    adept::uIndex* active_entries_right = right->get_active_entries();
//    int n_active_entries_right = right->get_n_active_entries();
//    for (int i = 0; i < n_active_entries_right; i++) {
//      //left->accumulate(active_entries_right[i], right->extract_value(active_entries_right[i]));
//      left->accumulate(active_entries_right[i], right->gradient_table_local[active_entries_right[i]]);
//    }
//    return left;
//  }
//
//
//  tfk_gradient_table* my_gradient_table =  gradient_table_list[start];
//
//  for (int j = start+1; j < end; j++) {
//    adept::uIndex* active_entries = gradient_table_list[j]->get_active_entries();
//    int64_t n_active_entries = gradient_table_list[j]->get_n_active_entries();
//
//    for (int i = 0; i < n_active_entries; i++) {
//      //my_gradient_table->accumulate(active_entries[i],
//      //                              gradient_table_list[j]->extract_value(active_entries[i]));
//      my_gradient_table->accumulate(active_entries[i],
//                                    gradient_table_list[j]->gradient_table_local[active_entries[i]]);
//    }
//  }
//
//  return my_gradient_table;
//}

// need to disable recording when walking over the tree.
void SP_Tree::set_recording(bool recording_) {
  this->recording = recording_;
}



void SP_Tree::walk_tree_process_one_worker(float* gradient_table) {

  std::vector<SP_Node*> ret;
  walk_tree_flatten_datanodes(get_root(), ret);

  //FILE* f = fopen("process.debug", "a");

  adept::uIndex count_ist = -1;
  for (int i = ret.size(); i-- > 0;) {
    triple_vector_wl stack = ret[i]->data;
    //for (adept::uIndex ist = worker_local_stacks[stack.worker_id].statement_stack_arr_len; ist-- > 0;) {
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
      count_ist++;
      //fprintf(f, "ist %llu\n", count_ist);
      const adept::Statement& statement =
          worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
      adept::uIndex idx = statement.index;
      if (idx == (adept::uIndex) -1) continue;
      float a = gradient_table[idx];
      //fprintf(f, "ist %llu float a=%e index %llu\n", count_ist, a, idx);
      gradient_table[idx] = 0.0;

      if (a != 0.0f) {
         adept::uIndex count = 0;
         for (adept::uIndex j =
                    worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                    j < statement.end_plus_one; j++) {
               adept::Real multiplier_test =
                   worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
               adept::uIndex operation_stack_index =
                   worker_local_stacks[stack.worker_id].operation_stack_arr[j];
               gradient_table[operation_stack_index] += multiplier_test*a;
               //fprintf(f, "ist %llu j=%llu mul=%e a=%f into idx %llu\n", count_ist, count++, multiplier_test, a, operation_stack_index);
          }
      }
    }
  }
  //fclose(f);
  //for (adept::uIndex ist = worker_local_stacks[0].statement_stack_arr_len; ist-- > 0;) {
  //  const adept::Statement& statement = worker_local_stacks[0].statement_stack_arr[ist];
  //  adept::uIndex idx = statement.index;
  //  if (idx == -1) continue;
  //  float a = gradient_table[idx];
  //  gradient_table[idx] = 0;

  //  if (a != 0.0) {
  //     for (adept::uIndex j =
  //                worker_local_stacks[0].statement_stack_arr[ist-1].end_plus_one;
  //                j < statement.end_plus_one; j++) {
  //           adept::Real multiplier_test =
  //               worker_local_stacks[0].multiplier_stack_arr[j];
  //           adept::uIndex operation_stack_index =
  //               worker_local_stacks[0].operation_stack_arr[j];
  //           gradient_table[operation_stack_index] += multiplier_test*a;
  //      }
  //  }
  //}
}

tfk_gradient_table* SP_Tree::walk_tree_process(SP_Node* n, tfk_gradient_table* my_gradient_table,
                                uint64_t n_gradients) {
  //printf("walk tree process\n");
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
       //delete n;
       return my_gradient_table;
    }


    //int wid = __cilkrts_get_worker_number();



    //if (my_gradient_table->dense_rep != NULL) {
    //  printf("Dense representation is being used for %d statements\n", stack.statement_stack_end-stack.statement_stack_start);
    //}
    //printf("statement stack end is %d, start is %d\n", stack.statement_stack_end, stack.statement_stack_start);
    //printf("new node\n");
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        //assert(statement.index != -1 && "Why is statement index -1?\n");
        if (statement.index == -1) continue;
        int op_count = 0;
        //printf("new statement for index %d\n", statement.index);
        adept::Real a = my_gradient_table->extract_value(statement.index);
        //if (stack.statement_stack_end - stack.statement_stack_start == 450758-449756) {
        //printf("ist is %d, areal is %f, statement index is %d\n", ist, a, statement.index);
        //}
        if (a != 0.0) {
         #ifdef TFK_DEBUG_PRINTS
         printf("statement %d edges:", statement.index);
         #endif
         if (ist == 0/*stack.statement_stack_start*/ && false) {
           for (adept::uIndex j = stack.operation_stack_start;
                  j < statement.end_plus_one; j++) {
             op_count++;
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             if (multiplier_test == 0.112143141) printf("test\n");
             adept::uIndex operation_stack_index =
                 worker_local_stacks[stack.worker_id].operation_stack_arr[j];
             my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);

             #ifdef TFK_DEBUG_PRINTS
             printf("%d,", operation_stack_index);
             #endif
           }
           #ifdef TFK_DEBUG_PRINTS
           printf("; ist %d start %d end %d\n", ist, stack.statement_stack_end,
                  stack.statement_stack_start);
           #endif
         } else {
           for (adept::uIndex j =
                  worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                  j < statement.end_plus_one; j++) {
             op_count++;
             adept::Real multiplier_test =
                 worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
             if (multiplier_test == 0.112143141) printf("test\n");
             adept::uIndex operation_stack_index =
                 worker_local_stacks[stack.worker_id].operation_stack_arr[j];

             my_gradient_table->accumulate(operation_stack_index, multiplier_test*a);

             #ifdef TFK_DEBUG_PRINTS
             printf("%d,", operation_stack_index);
             #endif
           }
           #ifdef TFK_DEBUG_PRINTS
           printf(": ist %d start %d end %d\n", ist, stack.statement_stack_end,
                  stack.statement_stack_start);
           #endif
         }
       }
     }
    //delete n;
    return my_gradient_table;
  }

  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      if (n->type == 0) {
      my_gradient_table = walk_tree_process((*(n->children))[i], my_gradient_table, n_gradients);
      } else {
      my_gradient_table = walk_tree_process((*(n->children))[i], my_gradient_table, n_gradients);
      }
    }

  } else if (n->type == 2) {
    std::vector<tfk_gradient_table*> gradient_table_list;
    //for (int i = n->children.size()-1; i >= 0; i--) {
    //  gradient_table_list.push_back(NULL);//new tfk_gradient_table(n_gradients, my_gradient_table));
    //}


    //#pragma cilk grainsize 1



    int* wids = (int*) malloc(sizeof(int)*n->children->size()+8);
   
    tfk_gradient_table** tables = (tfk_gradient_table**) malloc(sizeof(tfk_gradient_table*)*n->children->size() + 8);
 
    for (int i = 0; i < n->children->size(); i++) {
      wids[i] = -1;
      tables[i] = NULL;
    }

    //#pragma cilk grainsize 1
    SPTREE_parfor (int j = 0; j < n->children->size(); j++) {
      int i = n->children->size()-j-1;
      tfk_gradient_table* table;
      if (i == 0) {
        tables[i] = my_gradient_table;
      }
      else if (i > 0 && /*&& wids[i-1] == __cilkrts_get_worker_number() wids[i-1] != -1*/ false) {
        tables[i] = tables[i-1];
      } else {
        table = new tfk_gradient_table(n_gradients, my_gradient_table);
        tables[i] = table;
      }
      tables[i] = walk_tree_process((*(n->children))[i], /*gradient_table_list[i]*/tables[i], n_gradients);
      wids[i] = __cilkrts_get_worker_number();
    }


    for (int i = 0; i < n->children->size(); i++) {
      if (i == 0 || tables[i] != tables[i-1]) {
        if (tables[i] != NULL) {
          gradient_table_list.push_back(tables[i]);
        }
      }
    }


    //if (gradient_table_list.size() == 0) return my_gradient_table;
    //printf("gradient table list len %d\n", gradient_table_list.size());

    
    tfk_gradient_table* merged_table = my_gradient_table;
    if (gradient_table_list.size() > 0) merged_table = merge_gradient_table_list(gradient_table_list, 0,
                                                                 gradient_table_list.size());
    
    //adept::uIndex* active_entries = merged_table->get_active_entries();
    //int64_t n_active_entries = merged_table->get_n_active_entries();


    //adept::uIndex* my_active_entries = my_gradient_table->get_active_entries();
    //int64_t my_n_active_entries = my_gradient_table->get_n_active_entries();

    //if (my_n_active_entries == 0 && my_gradient_table->raw_gradient_table != NULL && false) {
    //  //printf("my entries zero, others is %d\n", n_active_entries);
    //  //merged_table->gradient_table = my_gradient_table->gradient_table;
    //  //my_gradient_table->active_entries = NULL;
    //  //my_gradient_table->gradient_table_local = merged_table->gradient_table_local;

    //  for (int i = n->children.size()-1; i >= 0; i--) {
    //      if (gradient_table_list[i] != merged_table) {
    //        delete gradient_table_list[i];
    //      }
    //  }

    //  //merged_table->gradient_table = my_gradient_table;
    //  merged_table->n_active_entries = 0;
    //  //merged_table->raw_gradient_table = my_gradient_table->raw_gradient_table;
    //  //free(active_entries);
    //  return merged_table;
    //  //delete my_gradient_table;
    //  //my_gradient_table = merged_table;
    //  //my_gradient_table->active_entries = NULL;
    //  //return my_gradient_table;
    //} /*else {
    //  my_gradient_table->n_active_entries = 0;
    //}*/

    free(wids);// = (int*) malloc(sizeof(int)*n->children.size()+8);
    free(tables);// = (tfk_gradient_table**) malloc(sizeof(tfk_gradient_table*)*n->children.size() + 8);

    if (my_gradient_table->gradient_table_local.size() == 0 && my_gradient_table->raw_gradient_table == NULL && false) {
      //my_gradient_table->gradient_table_local = merged_table->gradient_table_local;
      merged_table->n_active_entries = 0;
      free(merged_table->active_entries);
      merged_table->gradient_table = my_gradient_table;
        for (int i = gradient_table_list.size()-1; i >= 0; i--) {
          if (gradient_table_list[i] != merged_table) {
            delete gradient_table_list[i];
          }
        }
      //delete n;
      return merged_table;
    } else {
      assert(merged_table == my_gradient_table && "merged table must be the merged gradient table.\n");
      //if (merged_table != my_gradient_table) {
      //  my_gradient_table->merge_into_me(merged_table);
      //}
      //for (int i = 0; i < n_active_entries; i++) {
      //  my_gradient_table->accumulate(active_entries[i],
      //                                merged_table->gradient_table_local[active_entries[i]]);
      //}
    }

    for (int i = gradient_table_list.size()-1; i >= 0; i--) {
      if (my_gradient_table != gradient_table_list[i] && gradient_table_list[i] != NULL) {
        delete gradient_table_list[i];
      }
    }
    //delete n;
    return my_gradient_table; 
  } else {
    printf("Odd error with node types in SP_Tree during reverse-pass processing.\n");
  }
  //delete n;
  return my_gradient_table;
}




void SP_Tree::walk_tree_debug(SP_Node* n, int nest_depth,FILE* f) {

  for (int i = 0; i < nest_depth; i++) {
    fprintf(f, "  ");
  }

  nest_depth += 1;

  if (n->type == 1) {
    fprintf(f, "(S:\n");
  } else if (n->type == 2) {
    fprintf(f, "(P:\n");
  }

  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    fprintf(f, "D\n");
    return;
  }

  for (int i = 0; i < n->children->size(); i++) {
    walk_tree_debug((*(n->children))[i], nest_depth, f);
  }

  for (int i = 0; i < nest_depth-1; i++) {
    fprintf(f, "  ");
  }
  fprintf(f, ")\n");
}


void SP_Tree::walk_tree_debug(SP_Node* n) {
  FILE* f = fopen("sptree.debug", "a");
  walk_tree_debug(n, 0, f);
  fclose(f);
  return;
  //return walk_tree_debug(n, 0,f);

  //if (n->type == 1) {
  //  printf("(S:");
  //} else if (n->type == 2) {
  //  printf("(P:");
  //}

  //// If its a data node it must be a terminal node.
  //if (n->type == 3) {
  //  printf("D");
  //  return;
  //}


  //for (int i = 0; i < n->children.size(); i++) {
  //  walk_tree_debug(n->children[i]);
  //}
  //printf(")");
}

void SP_Tree::walk_tree_process_locks(SP_Node* n, float* gradient_, int64_t* locks) {
  // If its a data node it must be a terminal node.
  if (n->type == 3) {
    // We are going to process one of the stacks.
    triple_vector_wl stack = n->data;
    if (stack.statement_stack_end == stack.statement_stack_start) {
       //delete n;
       return;
    }
    int wid = __cilkrts_get_worker_number();
    for (adept::uIndex ist = stack.statement_stack_end; ist-- > stack.statement_stack_start;) {
        const adept::Statement& statement =
            worker_local_stacks[stack.worker_id].statement_stack_arr[ist];
        //assert(statement.index != -1 && "Why is statement index -1?\n");
        if (statement.index == -1) continue;

        // acquire lock.
        {
        bool succ = false;
        do {
          succ = __sync_bool_compare_and_swap(&locks[statement.index], 0, 1);
        } while (!succ || locks[statement.index] == 0);
        }

        adept::Real a = gradient_[statement.index];
        gradient_[statement.index] = 0;
        locks[statement.index] = 0; // release lock.

        if (a != 0.0) {
          for (adept::uIndex j =
                 worker_local_stacks[stack.worker_id].statement_stack_arr[ist-1].end_plus_one;
                 j < statement.end_plus_one; j++) {
            adept::Real multiplier_test =
                worker_local_stacks[stack.worker_id].multiplier_stack_arr[j];
            adept::uIndex operation_stack_index =
                worker_local_stacks[stack.worker_id].operation_stack_arr[j];

            // acquire lock.
            {
            bool succ = false;
            do {
              succ = __sync_bool_compare_and_swap(&locks[operation_stack_index], 0, 1);
            } while (!succ || locks[operation_stack_index] == 0);
            }
            gradient_[operation_stack_index] += multiplier_test*a;
            locks[operation_stack_index] = 0; // Release lock.
          }
       }
     }
    return;
  }

  if (n->type == 1 || n->type == 0) {
    for (int i = n->children->size()-1; i >= 0; i--) {
      walk_tree_process_locks((*(n->children))[i], gradient_, locks);
    }

  } else if (n->type == 2) {
    //if (n->children->size() < 8) {
    for (int j = 0; j < n->children->size(); j++) {
      cilk_spawn walk_tree_process_locks((*(n->children))[n->children->size()-j-1], gradient_, locks);
    }
    cilk_sync;
    //}
    //#pragma cilk grainsize 1
    //cilk_for (int j = 0; j < n->children->size(); j++) {
    //  int i = n->children->size()-j-1;
    //  walk_tree_process_semisort((*(n->children))[i], worker_local_grad_table, appears_in_statement, gradient_);
    //}
  }

}



