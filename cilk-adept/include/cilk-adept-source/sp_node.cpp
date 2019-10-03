
// SP_Node Class
  SP_Node::SP_Node(triple_vector_wl data_) {
    data = data_;
    children = new std::vector<SP_Node*>();
    type = 3;
  }


  // nx1 = nxm ** mx1
  // assumes nxm is static weight matrix.
  //SP_Node::SP_Node(aMatrix& left, aMatrix& right, aMatrix& ans) {

  //   /*
  //     for each of n statements.

  //      	for statement i
  //      		


  //   */

  //}

  SP_Node::~SP_Node() {
    if (children != NULL ) {
      //#pragma cilk
      //#pragma cilk grainsize 1
      for (int i = 0; i < children->size(); i++) {
        delete (*children)[i];
      }
      delete children;
    }
    //children = new std::vector<SP_Node*>();
  }


  SP_Node::SP_Node(int type_, SP_Node* parent_) {
    if (type_ == 2) printf("DEBUG: This should not be called for P nodes.\n");
    type = type_;
    parent = parent_;
    children = new std::vector<SP_Node*>();
    sync_id = NULL;
  }


  SP_Node::SP_Node(int type_, SP_Node* parent_, void* sync_id_) {
    if (type_ != 2) printf("DEBUG: Error this should only be called for P nodes.\n");
    type = type_;
    parent = parent_;
    children = new std::vector<SP_Node*>();
    sync_id = sync_id_;
  }
