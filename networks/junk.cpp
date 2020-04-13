/*
aReal _max(aReal a, aReal b, aReal c, aReal d) {
  float max_val = a.value();
  int cas = 0;
  if (b.value() > max_val) {
    max_val = b.value();
    cas = 1;
  }
  if (c.value() > max_val) {
    max_val = c.value();
    cas = 2;
  }
  if (d.value() > max_val) {
    max_val = d.value();
    cas = 3;
  }
  if (cas == 0) return a;
  if (cas == 1) return b;
  if (cas == 2) return c;
  if (cas == 3) return d;
  return a;
}
*/

void test_opt() {
  using adept::Stack;
  Stack stack;

  aMatrix A(2,10);
  aMatrix v(10,1);
  for (int i = 0; i < 10; i++) {
    A(0,i) = i;
    A(1,i) = 2*i;
    v(i,0) = 2;
  }

  stack.new_recording();
  stack.pause_recording();
  aMatrix B = A**v;
  /***
  B[0] = sum_i=0..9 A[i]*V[i]

  statement stack		operation stack
  B[0]			idxA[0]		v>0
  B[1]			idxA[1]		v>0

  A**V <- B
  ***/
  stack.continue_recording();
  for (int i = 0; i < 10; i++) {
    if (v(i,0)>0) {
      stack.push_rhs(1.0,A(0,i).gradient_index());
    }
  }
  stack.push_lhs(B[0].gradient_index());
  for (int i = 0; i < 10; i++) {
    if (v(i,1)>0) {
      stack.push_rhs(1.0,A(1,i).gradient_index());
    }
  }
  stack.push_lhs(B[1].gradient_index());

  std::cout << B << std::endl;

  aReal val = sum(B);
  val.set_gradient(1.0);
  stack.reverse();
  std::cout << A.get_gradient() << std::endl;
}

void test_bug() {
  using adept::Stack;
  Stack stack;

  aReal a = 4.0;
  aReal b = 2.0;

  stack.new_recording();

  aReal d = a*a + b;
  d.set_gradient(1.0);

  stack.reverse();

  std::cout << a.get_gradient() << "," << b.get_gradient()<< std::endl;
}

/*
void standard_2dconvolution(aMatrix& input, aMatrix& conv_weights, 
                            int input_stride_x, input_stride_y, 
                            int output_stride_x, int output_stride_y, 
                            aMatrix& output) {
  for (int x = 0; x < ; x++) {
    for (int y = 0; y < dim2; y++) {
      output[x*dim2+y] = conv_weights[
          conv_weights.dimensions()[0]*conv_weights.dimensions()[1] - 1]; // bias
      for (int dx = -2; dx < 3; dx++) {
        for (int dy = -2; dy < 3; dy++) {
            output[x*output_stride_y+y*output_stride_x] +=
                input[(x+dx+2)*32 + (y+dy+2)][0]*conv1_weights[k][(dx+2)*5+dy+2];
        }
      }
    }
  }
}
*/

aReal compute_connect(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                      std::vector<Real>& labels, double* accuracy,
                      double* test_set_loss, bool recording) {
  aReal loss = 0.0;

  aReal* losses = new aReal[data.size()];

  bool* correct = new bool[data.size()];

  cilk_for (int j = 0; j < data.size(); j += 10) {
    int start_i = j;
    int end_i = j+10;
    if (end_i > data.size()) end_i = data.size();

    for (int i = start_i; i < end_i; i++) {
      losses[i] = 0.0;

      std::vector<aMatrix> results = std::vector<aMatrix>(weights.size()-1);
      results[0] = activations::relu(weights[0]**data[i]);
      for (int k = 1; k < weights.size()-1; k++) {
        // bias term.
        results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
        results[k] = activations::relu(weights[k]**results[k-1]);
      }
      aMatrix mat_prediction = activations::softmax(results[results.size()-1], 1.0);

      int argmax = 0;
      double argmaxvalue = mat_prediction[0][0].value();
      for (int k = 0; k < 3; k++) {
        if (argmaxvalue <= mat_prediction[k][0].value()) {
          argmaxvalue = mat_prediction[k][0].value();
          argmax = k;
        }
      }

      Matrix groundtruth(3, 1);
      for (int k = 0; k < 3; k++) {
        groundtruth[k][0] = 0.0;
      }
      if (labels[i] > 0.5) groundtruth[0][0] = 1.0;
      if (labels[i] < -0.5) groundtruth[1][0] = 1.0;
      if (fabs(labels[i]) < 0.5) groundtruth[2][0] = 1.0;

      losses[i] += activations::crossEntropy(mat_prediction, groundtruth);

      correct[i] = false;
      if ((argmax == 0 && labels[i] > 0.5) ||
          (argmax == 1 && labels[i] < -0.5) ||
          (argmax == 2 && fabs(labels[i]) < 0.5)) {
        correct[i] = true;
      }
    }
  }
  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || recording) {
      loss += losses[i];
    } else {
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    }
  }

  delete[] losses;
  delete[] correct;
  *accuracy = (100.0*ncorrect)/total;

  return loss;
}

void learn_connect4() {
  using adept::Stack;
  Stack stack;

  std::vector<Matrix> data;
  std::vector<Real> labels;
  read_connect4("datasets/connect-4.data", data, labels);
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  // Initialize the weights.
  std::vector<aMatrix> weight_list;
  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(3, 43));  // 3 x 1
  for (int i = 0; i < weight_list.size(); i++) {
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i](j,k) = distribution(generator) /
              (weight_list[i].dimensions()[0] * weight_list[i].dimensions()[1]);
      }
    }
  }

  double* weights_raw = allocate_weights(weight_list);
  double* weights_raw_old = allocate_weights(weight_list);
  double* gradients = allocate_weights(weight_list);
  double* momentums = allocate_weights_zero(weight_list);
  double* velocities = allocate_weights_zero(weight_list);

  read_values(weight_list, weights_raw);
  read_values(weight_list, weights_raw_old);

  double learning_rate = 0.01;

  int NUM_ITERS = 20000;

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    set_values(weight_list, weights_raw);
    stack.new_recording();                 // Clear any existing differential statements

    std::uniform_int_distribution<int> dis(0, (data.size()-1)/2);
    std::vector<Matrix> batch_data;
    std::vector<Real> batch_labels;
    for (int i = 0; i < 100; i++) {
      int _random = dis(generator);
      int random = 2*_random;
      batch_data.push_back(data[random]);
      batch_labels.push_back(labels[random]);
    }


    double accuracy = 0.0;
    double test_set_loss = 0.0;
    aReal loss;

    if (iter%200 == 0) {
      stack.pause_recording();
      loss = compute_connect(weight_list, data, labels, &accuracy, &test_set_loss, false);
      stack.continue_recording();
    } else {
      loss = compute_connect(weight_list, batch_data, batch_labels, &accuracy, &test_set_loss,
                             true);
      loss.set_gradient(1.0);
      stack.reverse();
      read_gradients(weight_list, gradients);

      if (std::isnan(loss.value())) {
        std::cout << std::endl << std::endl << "Got nan, doing reset " << std::endl << std::endl;
        store_values_into_old(weight_list, weights_raw_old, weights_raw);  // move old into raw.
        continue;
      }
    }

    std::cout.precision(9);
    std::cout.setf(ios::fixed, ios::floatfield);
    if (iter % 200 == 0) {
      std::cout << std::endl;
      std::cout << "loss:" << loss.value() << ",\t lr: " << learning_rate <<
          "\t accuracy: " << accuracy << "% \t Test set loss: " << test_set_loss <<
          "\r\r"<< std::endl << std::endl;
      continue;
    } else {
      std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
          "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_set_loss <<
          "\r" << std::flush;
    }

    double norm = compute_gradient_norm(weight_list, gradients);
    if (norm < 1.0) norm = 1.0;

    store_values_into_old(weight_list, weights_raw, weights_raw_old);


    #ifdef LINE_SEARCH
    aReal newLoss = loss+0.01;
    double local_lr = learning_rate * 1.1;
    if (local_lr > 1.0) local_lr = 1.0;

    if (local_lr < 0.000001) local_lr = 0.000001;

    while (newLoss.value() > loss.value()) {
      stack.new_recording();
      stack.pause_recording();

      apply_gradient_update(weight_list, weights_raw, weights_raw_old, gradients,
                            local_lr*(1.0/norm));

      set_values(weight_list, weights_raw);

      double test_loss = 0.0;
      newLoss = compute_connect(weight_list, data, labels, &accuracy, &test_loss, false);
      if (newLoss.value() > loss.value()) {
        local_lr = local_lr*0.9;
      }
      stack.continue_recording();
    }
    learning_rate = local_lr;
    #endif

    apply_gradient_update_ADAM(weight_list, weights_raw, weights_raw_old, gradients, momentums,
                               velocities, 1.0, learning_rate, iter+1);
  }
}
