
// SP_Node Class
  SP_Node::SP_Node(triple_vector_wl data_) {
    data = data_;
    type = 3;
  }

  SP_Node::SP_Node(int type_, SP_Node* parent_) {
    type = type_;
    parent = parent_;
    children = std::vector<SP_Node*>();
  }

