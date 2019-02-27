
// SP_Node Class
  SP_Node::SP_Node(triple_vector_wl data_) {
    data = data_;
    type = 3;
  }

  SP_Node::SP_Node(int type_, SP_Node* parent_) {
    if (type_ == 2) printf("DEBUG: This should not be called for P nodes.\n");
    type = type_;
    parent = parent_;
    children = std::vector<SP_Node*>();
    sync_id = NULL;
  }


  SP_Node::SP_Node(int type_, SP_Node* parent_, void* sync_id_) {
    if (type_ != 2) printf("DEBUG: Error this should only be called for P nodes.\n");
    type = type_;
    parent = parent_;
    children = std::vector<SP_Node*>();
    sync_id = sync_id_;
  }
