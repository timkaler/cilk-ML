// Copyright 2019 Tim Kaler MIT License

#include <adept_source.h>
#include <adept_arrays.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
#include <random>

#include <iostream>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
#include <map>
#include <vector>

#include "../activations.hpp"
#include "../cxxopts.hpp"
#include "../io_helpers.hpp"
#include "../Graph.hpp"
#include "../mnist_parser.h"
#include "../optimization.hpp"

#include "../common/gettime.h"

// Defined by Makefile_serial
#ifdef TFK_ADEPT_SERIAL
#include <cilk/cilk_stub.h>
#endif

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::ios;
using std::vector;


aMatrix lenet_pool2_maxpool(aMatrix& output3) {
  aMatrix _output4(16*5*5+1, 1);
  for (int x = 0; x < 10; x += 2) {
    for (int y = 0; y < 10; y += 2) {
      _output4(0*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(0,(x+0)*10 + (y+0)),
              output3(0,(x+0)*10 + (y+1)),
              output3(0,(x+1)*10 + (y+0)),
              output3(0,(x+1)*10 + (y+1)));
      _output4(1*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(1,(x+0)*10 + (y+0)),
              output3(1,(x+0)*10 + (y+1)),
              output3(1,(x+1)*10 + (y+0)),
              output3(1,(x+1)*10 + (y+1)));
      _output4(2*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(2,(x+0)*10 + (y+0)),
              output3(2,(x+0)*10 + (y+1)),
              output3(2,(x+1)*10 + (y+0)),
              output3(2,(x+1)*10 + (y+1)));
      _output4(3*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(3,(x+0)*10 + (y+0)),
              output3(3,(x+0)*10 + (y+1)),
              output3(3,(x+1)*10 + (y+0)),
              output3(3,(x+1)*10 + (y+1)));
      _output4(4*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(4,(x+0)*10 + (y+0)),
              output3(4,(x+0)*10 + (y+1)),
              output3(4,(x+1)*10 + (y+0)),
              output3(4,(x+1)*10 + (y+1)));
      _output4(5*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(5,(x+0)*10 + (y+0)),
              output3(5,(x+0)*10 + (y+1)),
              output3(5,(x+1)*10 + (y+0)),
              output3(5,(x+1)*10 + (y+1)));
      _output4(6*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(5,(x+0)*10 + (y+0)),
              output3(5,(x+0)*10 + (y+1)),
              output3(5,(x+1)*10 + (y+0)),
              output3(5,(x+1)*10 + (y+1)));
      _output4(6*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(6,(x+0)*10 + (y+0)),
              output3(6,(x+0)*10 + (y+1)),
              output3(6,(x+1)*10 + (y+0)),
              output3(6,(x+1)*10 + (y+1)));
      _output4(7*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(7,(x+0)*10 + (y+0)),
              output3(7,(x+0)*10 + (y+1)),
              output3(7,(x+1)*10 + (y+0)),
              output3(7,(x+1)*10 + (y+1)));
      _output4(8*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(8,(x+0)*10 + (y+0)),
              output3(8,(x+0)*10 + (y+1)),
              output3(8,(x+1)*10 + (y+0)),
              output3(8,(x+1)*10 + (y+1)));
      _output4(9*5*5 + (x/2)*5+(y/2),0) = _max(
              output3(9,(x+0)*10 + (y+0)),
              output3(9,(x+0)*10 + (y+1)),
              output3(9,(x+1)*10 + (y+0)),
              output3(9,(x+1)*10 + (y+1)));
      _output4(10*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(10,(x+0)*10 + (y+0)),
               output3(10,(x+0)*10 + (y+1)),
               output3(10,(x+1)*10 + (y+0)),
               output3(10,(x+1)*10 + (y+1)));
      _output4(11*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(11,(x+0)*10 + (y+0)),
               output3(11,(x+0)*10 + (y+1)),
               output3(11,(x+1)*10 + (y+0)),
               output3(11,(x+1)*10 + (y+1)));
      _output4(12*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(12,(x+0)*10 + (y+0)),
               output3(12,(x+0)*10 + (y+1)),
               output3(12,(x+1)*10 + (y+0)),
               output3(12,(x+1)*10 + (y+1)));
      _output4(13*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(13,(x+0)*10 + (y+0)),
               output3(13,(x+0)*10 + (y+1)),
               output3(13,(x+1)*10 + (y+0)),
               output3(13,(x+1)*10 + (y+1)));
      _output4(14*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(14,(x+0)*10 + (y+0)),
               output3(14,(x+0)*10 + (y+1)),
               output3(14,(x+1)*10 + (y+0)),
               output3(14,(x+1)*10 + (y+1)));
      _output4(15*5*5 + (x/2)*5+(y/2),0) = _max(
               output3(15,(x+0)*10 + (y+0)),
               output3(15,(x+0)*10 + (y+1)),
               output3(15,(x+1)*10 + (y+0)),
               output3(15,(x+1)*10 + (y+1)));
    }
  }
  return _output4;
}

aMatrix lenet_pool2(aMatrix& output3, aMatrix& pool2_weights) {
  aMatrix _output4(16*5*5+1, 1);
  for (int x = 0; x < 10; x += 2) {
    for (int y = 0; y < 10; y += 2) {
      _output4(0*5*5 + (x/2)*5+(y/2),0) = pool2_weights(0,4) +
              output3(0,(x+0)*10 + (y+0))*pool2_weights(0,0*2+0) +
              output3(0,(x+0)*10 + (y+1))*pool2_weights(0,0*2+1) +
              output3(0,(x+1)*10 + (y+0))*pool2_weights(0,1*2+0) +
              output3(0,(x+1)*10 + (y+1))*pool2_weights(0,1*2+1) ;
      _output4(1*5*5 + (x/2)*5+(y/2),0) = pool2_weights(1,4) +
              output3(1,(x+0)*10 + (y+0))*pool2_weights(1,0*2+0) +
              output3(1,(x+0)*10 + (y+1))*pool2_weights(1,0*2+1) +
              output3(1,(x+1)*10 + (y+0))*pool2_weights(1,1*2+0) +
              output3(1,(x+1)*10 + (y+1))*pool2_weights(1,1*2+1) ;
      _output4(2*5*5 + (x/2)*5+(y/2),0) = pool2_weights(2,4) +
              output3(2,(x+0)*10 + (y+0))*pool2_weights(2,0*2+0) +
              output3(2,(x+0)*10 + (y+1))*pool2_weights(2,0*2+1) +
              output3(2,(x+1)*10 + (y+0))*pool2_weights(2,1*2+0) +
              output3(2,(x+1)*10 + (y+1))*pool2_weights(2,1*2+1) ;
      _output4(3*5*5 + (x/2)*5+(y/2),0) = pool2_weights(3,4) +
              output3(3,(x+0)*10 + (y+0))*pool2_weights(3,0*2+0) +
              output3(3,(x+0)*10 + (y+1))*pool2_weights(3,0*2+1) +
              output3(3,(x+1)*10 + (y+0))*pool2_weights(3,1*2+0) +
              output3(3,(x+1)*10 + (y+1))*pool2_weights(3,1*2+1) ;
      _output4(4*5*5 + (x/2)*5+(y/2),0) = pool2_weights(4,4) +
              output3(4,(x+0)*10 + (y+0))*pool2_weights(4,0*2+0) +
              output3(4,(x+0)*10 + (y+1))*pool2_weights(4,0*2+1) +
              output3(4,(x+1)*10 + (y+0))*pool2_weights(4,1*2+0) +
              output3(4,(x+1)*10 + (y+1))*pool2_weights(4,1*2+1) ;
      _output4(5*5*5 + (x/2)*5+(y/2),0) = pool2_weights(5,4) +
              output3(5,(x+0)*10 + (y+0))*pool2_weights(5,0*2+0) +
              output3(5,(x+0)*10 + (y+1))*pool2_weights(5,0*2+1) +
              output3(5,(x+1)*10 + (y+0))*pool2_weights(5,1*2+0) +
              output3(5,(x+1)*10 + (y+1))*pool2_weights(5,1*2+1) ;
      _output4(6*5*5 + (x/2)*5+(y/2),0) = pool2_weights(6,4) +
              output3(6,(x+0)*10 + (y+0))*pool2_weights(6,0*2+0) +
              output3(6,(x+0)*10 + (y+1))*pool2_weights(6,0*2+1) +
              output3(6,(x+1)*10 + (y+0))*pool2_weights(6,1*2+0) +
              output3(6,(x+1)*10 + (y+1))*pool2_weights(6,1*2+1) ;
      _output4(7*5*5 + (x/2)*5+(y/2),0) = pool2_weights(7,4) +
              output3(7,(x+0)*10 + (y+0))*pool2_weights(7,0*2+0) +
              output3(7,(x+0)*10 + (y+1))*pool2_weights(7,0*2+1) +
              output3(7,(x+1)*10 + (y+0))*pool2_weights(7,1*2+0) +
              output3(7,(x+1)*10 + (y+1))*pool2_weights(7,1*2+1) ;
      _output4(8*5*5 + (x/2)*5+(y/2),0) = pool2_weights(8,4) +
              output3(8,(x+0)*10 + (y+0))*pool2_weights(8,0*2+0) +
              output3(8,(x+0)*10 + (y+1))*pool2_weights(8,0*2+1) +
              output3(8,(x+1)*10 + (y+0))*pool2_weights(8,1*2+0) +
              output3(8,(x+1)*10 + (y+1))*pool2_weights(8,1*2+1) ;
      _output4(9*5*5 + (x/2)*5+(y/2),0) = pool2_weights(9,4) +
              output3(9,(x+0)*10 + (y+0))*pool2_weights(9,0*2+0) +
              output3(9,(x+0)*10 + (y+1))*pool2_weights(9,0*2+1) +
              output3(9,(x+1)*10 + (y+0))*pool2_weights(9,1*2+0) +
              output3(9,(x+1)*10 + (y+1))*pool2_weights(9,1*2+1) ;
      _output4(10*5*5 + (x/2)*5+(y/2),0) = pool2_weights(10,4) +
              output3(10,(x+0)*10 + (y+0))*pool2_weights(10,0*2+0) +
              output3(10,(x+0)*10 + (y+1))*pool2_weights(10,0*2+1) +
              output3(10,(x+1)*10 + (y+0))*pool2_weights(10,1*2+0) +
              output3(10,(x+1)*10 + (y+1))*pool2_weights(10,1*2+1) ;
      _output4(11*5*5 + (x/2)*5+(y/2),0) = pool2_weights(11,4) +
              output3(11,(x+0)*10 + (y+0))*pool2_weights(11,0*2+0) +
              output3(11,(x+0)*10 + (y+1))*pool2_weights(11,0*2+1) +
              output3(11,(x+1)*10 + (y+0))*pool2_weights(11,1*2+0) +
              output3(11,(x+1)*10 + (y+1))*pool2_weights(11,1*2+1) ;
      _output4(12*5*5 + (x/2)*5+(y/2),0) = pool2_weights(12,4) +
              output3(12,(x+0)*10 + (y+0))*pool2_weights(12,0*2+0) +
              output3(12,(x+0)*10 + (y+1))*pool2_weights(12,0*2+1) +
              output3(12,(x+1)*10 + (y+0))*pool2_weights(12,1*2+0) +
              output3(12,(x+1)*10 + (y+1))*pool2_weights(12,1*2+1) ;
      _output4(13*5*5 + (x/2)*5+(y/2),0) = pool2_weights(13,4) +
              output3(13,(x+0)*10 + (y+0))*pool2_weights(13,0*2+0) +
              output3(13,(x+0)*10 + (y+1))*pool2_weights(13,0*2+1) +
              output3(13,(x+1)*10 + (y+0))*pool2_weights(13,1*2+0) +
              output3(13,(x+1)*10 + (y+1))*pool2_weights(13,1*2+1) ;
      _output4(14*5*5 + (x/2)*5+(y/2),0) = pool2_weights(14,4) +
              output3(14,(x+0)*10 + (y+0))*pool2_weights(14,0*2+0) +
              output3(14,(x+0)*10 + (y+1))*pool2_weights(14,0*2+1) +
              output3(14,(x+1)*10 + (y+0))*pool2_weights(14,1*2+0) +
              output3(14,(x+1)*10 + (y+1))*pool2_weights(14,1*2+1) ;
      _output4(15*5*5 + (x/2)*5+(y/2),0) = pool2_weights(15,4) +
              output3(15,(x+0)*10 + (y+0))*pool2_weights(15,0*2+0) +
              output3(15,(x+0)*10 + (y+1))*pool2_weights(15,0*2+1) +
              output3(15,(x+1)*10 + (y+0))*pool2_weights(15,1*2+0) +
              output3(15,(x+1)*10 + (y+1))*pool2_weights(15,1*2+1) ;

      //for (int k = 0; k < 16; k++) {
      //  _output4(k*5*5 + (x/2)*5+(y/2),0) = pool2_weights(k,4); // bias.
      //}
      //for (int dx = 0; dx < 2; dx++) {
      //  for (int dy = 0; dy < 2; dy++) {
      //    for (int k = 0; k < 16; k++) {
      //      _output4(k*5*5 + (x/2)*5+(y/2),0) += output3(k,(x+dx)*10 + (y+dy))*pool2_weights(k,dx*2+dy);
      //    }
      //  }
      //}
    }
  }
  return _output4;
}

aMatrix lenet_conv2(aMatrix& output2, aMatrix& conv2_weights) {
  aMatrix _output3(16, 10*10);
  for (int x = 0; x < 10; x++) {
    for (int y = 0; y < 10; y++) {
      for (int k = 0; k < 16; k++) {
        _output3(k,x*10+y) = conv2_weights(k,25); // bias.
      }
      for (int dx = -2; dx < 3; dx++) {
        for (int dy = -2; dy < 3; dy++) {
          _output3(0, x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2);
          _output3(1, x*10+y) += 
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2);
          _output3(2,x*10+y) += 
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2);
          _output3(3,x*10+y) += 
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2);
          _output3(4,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2);
          _output3(5,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2);
          _output3(6,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2);
          _output3(7,x*10+y) += 
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2);
          _output3(8,x*10+y) += 
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2);
          _output3(9,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2);
          _output3(10,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2);
          _output3(11,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2);
          _output3(12,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2);
          _output3(13,x*10+y) += 
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2);
          _output3(14,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2);
          _output3(15,x*10+y) += 
              output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2) +
              output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2) +
              output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2) +
              output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2) +
              output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2) +
              output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2);
        }
      }
    }
  }
  return _output3;
}

aReal compute_mnist_lenet5_fast_maxpool(std::vector<aMatrix>& weights, 
                                        std::vector<Matrix>& data,
                                        std::vector<uint8_t>& labels,
                                        int max_label, double* accuracy, 
                                        double* test_set_loss) {
  aMatrix& conv1_weights = weights[0];
  aMatrix& pool1_weights = weights[1];
  aMatrix& conv2_weights = weights[2];
  aMatrix& pool2_weights = weights[3];
  aMatrix& fully_connected_weights = weights[4];
  aMatrix& fully_connected_weights2 = weights[5];
  aMatrix& output_layer_weights = weights[6];

  aReal loss = 0.0;

  bool* correct = new bool[data.size()]();
  aReal* losses = new aReal[data.size()]();

  cilk_for (int j = 0; j < data.size(); j++) {
    losses[j] = 0.0;

    // Convolution 1.
    // aMatrix conv1_weights(26,1);
    aMatrix output(6,28*28);
    for (int a = 0; a < output.dimensions()[0]; a++) {
      for (int b = 0; b < output.dimensions()[1]; b++) {
        output(a,b) = 0.0;
      }
    }

    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        for (int k = 0; k < 6; k++) {
          output(k,x*28+y) += conv1_weights(k,25); // bias.
        }
        //for (int dx = -2; dx < 3; dx++) {
        //  for (int dy = -2; dy < 3; dy++) {
            for (int k = 0; k < 6; k++) {
              //_output(k,x*28+y) += data[j]((x+dx+2)*32 + (y+dy+2),0)*conv1_weights(k,(dx+2)*5+dy+2);
              //_output(k,x*28+y) += data[j]((x+dx+2)*32 + (y+dy+2),0)*conv1_weights(k,(dx+2)*5+dy+2);
              output(k,x*28+y) += data[j]((x+-2+2)*32 + (y+-2+2),0)*conv1_weights(k,(-2+2)*5+-2+2)
                                + data[j]((x+-2+2)*32 + (y+-1+2),0)*conv1_weights(k,(-2+2)*5+-1+2)
                                + data[j]((x+-2+2)*32 + (y+0+2),0)*conv1_weights(k,(-2+2)*5+0+2)
                                + data[j]((x+-2+2)*32 + (y+1+2),0)*conv1_weights(k,(-2+2)*5+1+2)
                                + data[j]((x+-2+2)*32 + (y+2+2),0)*conv1_weights(k,(-2+2)*5+2+2)
                                + data[j]((x+-1+2)*32 + (y+-2+2),0)*conv1_weights(k,(-1+2)*5+-2+2)
                                + data[j]((x+-1+2)*32 + (y+-1+2),0)*conv1_weights(k,(-1+2)*5+-1+2)
                                + data[j]((x+-1+2)*32 + (y+0+2),0)*conv1_weights(k,(-1+2)*5+0+2)
                                + data[j]((x+-1+2)*32 + (y+1+2),0)*conv1_weights(k,(-1+2)*5+1+2)
                                + data[j]((x+-1+2)*32 + (y+2+2),0)*conv1_weights(k,(-1+2)*5+2+2)
                                + data[j]((x+0+2)*32 + (y+-2+2),0)*conv1_weights(k,(0+2)*5+-2+2)
                                + data[j]((x+0+2)*32 + (y+-1+2),0)*conv1_weights(k,(0+2)*5+-1+2)
                                + data[j]((x+0+2)*32 + (y+0+2),0)*conv1_weights(k,(0+2)*5+0+2)
                                + data[j]((x+0+2)*32 + (y+1+2),0)*conv1_weights(k,(0+2)*5+1+2)
                                + data[j]((x+0+2)*32 + (y+2+2),0)*conv1_weights(k,(0+2)*5+2+2)
                                + data[j]((x+1+2)*32 + (y+-2+2),0)*conv1_weights(k,(1+2)*5+-2+2)
                                + data[j]((x+1+2)*32 + (y+-1+2),0)*conv1_weights(k,(1+2)*5+-1+2)
                                + data[j]((x+1+2)*32 + (y+0+2),0)*conv1_weights(k,(1+2)*5+0+2)
                                + data[j]((x+1+2)*32 + (y+1+2),0)*conv1_weights(k,(1+2)*5+1+2)
                                + data[j]((x+1+2)*32 + (y+2+2),0)*conv1_weights(k,(1+2)*5+2+2)
                                + data[j]((x+2+2)*32 + (y+-2+2),0)*conv1_weights(k,(2+2)*5+-2+2)
                                + data[j]((x+2+2)*32 + (y+-1+2),0)*conv1_weights(k,(2+2)*5+-1+2)
                                + data[j]((x+2+2)*32 + (y+0+2),0)*conv1_weights(k,(2+2)*5+0+2)
                                + data[j]((x+2+2)*32 + (y+1+2),0)*conv1_weights(k,(2+2)*5+1+2)
                                + data[j]((x+2+2)*32 + (y+2+2),0)*conv1_weights(k,(2+2)*5+2+2);
            //}
          //}
        }
      }
    }


    // pooling layer 1
    //   aMatrix pool1_weights(5)
    aMatrix _output2(6,14*14);

    for (int x = 0; x < 28; x += 2) {
      for (int y = 0; y < 28; y += 2) {
        for (int k = 0; k < 6; k++) {
          _output2(k,(x/2)*14+(y/2)) = pool1_weights(k,4); // bias.
        }


        for (int k = 0; k < 6; k++) {
          float max_val = output(k,(x+0)*28 + y+0).value();
          int max_dx = 0;
          int max_dy = 0;
          for (int dx = 0; dx < 2; dx++) {
            for (int dy = 0; dy < 2; dy++) {
                if (output(k,(x+dx)*28 + y + dy).value() > max_val) {
                  max_val = output(k,(x+dx)*28 + y + dy).value();
                  max_dx = dx;
                  max_dy = dy;
                }
                //_output2(k,(x/2)*14+(y/2)) += output(k,(x+dx)*28 + (y+dy))*pool1_weights(k,dx*2+dy);
            }
          }
          _output2(k,(x/2)*14+(y/2)) += output(k,(x+max_dx)*28 + y + max_dy);
        }
      }
    }
    aMatrix output2 = activations::relu(_output2);


    aMatrix _output3 = lenet_conv2(output2, conv2_weights);

    //aMatrix output3 = tanh(_output3);
    //aMatrix _output4 = lenet_pool2(output3, pool2_weights);
    //aMatrix output5 = tanh(_output4);
    aMatrix output5 = activations::relu(lenet_pool2_maxpool(_output3));

    output5(400,0) = 1.0; // bias.

    // weight 120*120
    // fully_connected_weights(120,120)
    aMatrix output6 = activations::relu(fully_connected_weights**output5);//(120,1);

    output6(120,0) = 1.0; // bias.


    aMatrix output7 = activations::relu(fully_connected_weights2**output6);
    output7(84,0) = 1.0; //bias

    aMatrix final_output = activations::softmax(output_layer_weights**output7,1.0);


    //final_output[final_output.dimensions()[0]-1,0] = 1.0; // bias.

    int argmax = 0;
    double argmaxvalue = final_output(0,0).value();
    for (int k = 0; k < 10; k++) {
      if (argmaxvalue <= final_output(k,0).value()) {
        argmaxvalue = final_output(k,0).value();
        argmax = k;
      }
    }

    Matrix groundtruth(10,1);
    for (int k = 0; k < 10; k++) {
      groundtruth(k,0) = 0.0;
    }
    groundtruth(labels[j],0) = 1.0;

    //std::cout << std::endl;
    //std::cout << labels[j] << std::endl;
    //std::cout << groundtruth << std::endl;
    //std::cout << mat_prediction << std::endl;
    //std::cout << std::endl;
    losses[j] += activations::crossEntropy(final_output, groundtruth);

    correct[j] = false;
    if (argmax == labels[j]) {
      correct[j] = true;
    }

  }


  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || true) {
      loss += losses[i];
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
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

aReal compute_mnist_lenet5_fast(std::vector<aMatrix>& weights, 
                                std::vector<Matrix>& data,
                                std::vector<uint8_t>& labels,
                                int max_label, double* accuracy, 
                                double* test_set_loss) {
  aMatrix& conv1_weights = weights[0];
  aMatrix& pool1_weights = weights[1];
  aMatrix& conv2_weights = weights[2];
  aMatrix& pool2_weights = weights[3];
  aMatrix& fully_connected_weights = weights[4];
  aMatrix& fully_connected_weights2 = weights[5];
  aMatrix& output_layer_weights = weights[6];

  aReal loss = 0.0;

  bool* correct = new bool[data.size()];
  aReal* losses = new aReal[data.size()];

  cilk_for (int j = 0; j < data.size(); j += 1) {
    //int _end = _j+4;
    //if (_end > data.size()) _end = data.size();
    //for (int j = _j; j < _end; j++) {
    losses[j] = 0.0;


    // Convolution 1.
    // aMatrix conv1_weights(26,1);
    aMatrix _output(6,28*28);


    for (int a = 0; a < _output.dimensions()[0]; a++) {
      for (int b = 0; b < _output.dimensions()[1]; b++) {
        _output(a,b) = 0.0;
      }
    }


    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        for (int k = 0; k < 6; k++) {
          _output(k,x*28+y) += conv1_weights(k,25); // bias.
        }

        //for (int dx = -2; dx < 3; dx++) {
        //  for (int dy = -2; dy < 3; dy++) {
            for (int k = 0; k < 6; k++) {
              //_output(k,x*28+y) += data[j]((x+dx+2)*32 + (y+dy+2),0)*conv1_weights(k,(dx+2)*5+dy+2);
              //_output(k,x*28+y) += data[j]((x+dx+2)*32 + (y+dy+2),0)*conv1_weights(k,(dx+2)*5+dy+2);
              _output(k,x*28+y) += data[j]((x+-2+2)*32 + (y+-2+2),0)*conv1_weights(k,(-2+2)*5+-2+2)
                                + data[j]((x+-2+2)*32 + (y+-1+2),0)*conv1_weights(k,(-2+2)*5+-1+2)
                                + data[j]((x+-2+2)*32 + (y+0+2),0)*conv1_weights(k,(-2+2)*5+0+2)
                                + data[j]((x+-2+2)*32 + (y+1+2),0)*conv1_weights(k,(-2+2)*5+1+2)
                                + data[j]((x+-2+2)*32 + (y+2+2),0)*conv1_weights(k,(-2+2)*5+2+2)
                                + data[j]((x+-1+2)*32 + (y+-2+2),0)*conv1_weights(k,(-1+2)*5+-2+2)
                                + data[j]((x+-1+2)*32 + (y+-1+2),0)*conv1_weights(k,(-1+2)*5+-1+2)
                                + data[j]((x+-1+2)*32 + (y+0+2),0)*conv1_weights(k,(-1+2)*5+0+2)
                                + data[j]((x+-1+2)*32 + (y+1+2),0)*conv1_weights(k,(-1+2)*5+1+2)
                                + data[j]((x+-1+2)*32 + (y+2+2),0)*conv1_weights(k,(-1+2)*5+2+2)
                                + data[j]((x+0+2)*32 + (y+-2+2),0)*conv1_weights(k,(0+2)*5+-2+2)
                                + data[j]((x+0+2)*32 + (y+-1+2),0)*conv1_weights(k,(0+2)*5+-1+2)
                                + data[j]((x+0+2)*32 + (y+0+2),0)*conv1_weights(k,(0+2)*5+0+2)
                                + data[j]((x+0+2)*32 + (y+1+2),0)*conv1_weights(k,(0+2)*5+1+2)
                                + data[j]((x+0+2)*32 + (y+2+2),0)*conv1_weights(k,(0+2)*5+2+2)
                                + data[j]((x+1+2)*32 + (y+-2+2),0)*conv1_weights(k,(1+2)*5+-2+2)
                                + data[j]((x+1+2)*32 + (y+-1+2),0)*conv1_weights(k,(1+2)*5+-1+2)
                                + data[j]((x+1+2)*32 + (y+0+2),0)*conv1_weights(k,(1+2)*5+0+2)
                                + data[j]((x+1+2)*32 + (y+1+2),0)*conv1_weights(k,(1+2)*5+1+2)
                                + data[j]((x+1+2)*32 + (y+2+2),0)*conv1_weights(k,(1+2)*5+2+2)
                                + data[j]((x+2+2)*32 + (y+-2+2),0)*conv1_weights(k,(2+2)*5+-2+2)
                                + data[j]((x+2+2)*32 + (y+-1+2),0)*conv1_weights(k,(2+2)*5+-1+2)
                                + data[j]((x+2+2)*32 + (y+0+2),0)*conv1_weights(k,(2+2)*5+0+2)
                                + data[j]((x+2+2)*32 + (y+1+2),0)*conv1_weights(k,(2+2)*5+1+2)
                                + data[j]((x+2+2)*32 + (y+2+2),0)*conv1_weights(k,(2+2)*5+2+2);
            //}
          //}
        }
      }
    }

    aMatrix output = tanh(_output);

    // pooling layer 1
    //   aMatrix pool1_weights(5)
    aMatrix _output2(6,14*14);

    for (int x = 0; x < 28; x += 2) {
      for (int y = 0; y < 28; y += 2) {
        for (int k = 0; k < 6; k++) {
          _output2(k,(x/2)*14+(y/2)) = pool1_weights(k,4); // bias.
        }

        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int k = 0; k < 6; k++) {
              _output2(k,(x/2)*14+(y/2)) += output(k,(x+dx)*28 + (y+dy))*pool1_weights(k,dx*2+dy);
            }
          }
        }
      }
    }
    aMatrix output2 = tanh(_output2);


    aMatrix _output3 = lenet_conv2(output2, conv2_weights);

    aMatrix output3 = tanh(_output3);

    aMatrix _output4 = lenet_pool2(output3, pool2_weights);
    aMatrix output5 = tanh(_output4);

    output5(400,0) = 1.0; // bias.

    // weight 120*120
    // fully_connected_weights(120,120)
    aMatrix output6 = tanh(fully_connected_weights**output5);//(120,1);

    output6(120,0) = 1.0; // bias.


    aMatrix output7 = tanh(fully_connected_weights2**output6);
    output7(84,0) = 1.0; //bias

    aMatrix final_output = activations::softmax(output_layer_weights**output7,1.0);


    //final_output[final_output.dimensions()[0]-1,0] = 1.0; // bias.

    int argmax = 0;
    double argmaxvalue = final_output(0,0).value();
    for (int k = 0; k < 10; k++) {
      if (argmaxvalue <= final_output(k,0).value()) {
        argmaxvalue = final_output(k,0).value();
        argmax = k;
      }
    }

    Matrix groundtruth(10,1);
    for (int k = 0; k < 10; k++) {
      groundtruth(k,0) = 0.0;
    }
    groundtruth(labels[j],0) = 1.0;

    //std::cout << std::endl;
    //std::cout << labels[j] << std::endl;
    //std::cout << groundtruth << std::endl;
    //std::cout << mat_prediction << std::endl;
    //std::cout << std::endl;
    losses[j] += activations::crossEntropy(final_output, groundtruth);

    correct[j] = false;
    if (argmax == labels[j]) {
      correct[j] = true;
    }

  }


  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || true) {
      loss += losses[i];
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
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

aReal compute_mnist_lenet5(std::vector<aMatrix>& weights,
                           std::vector<Matrix>& data,
                           std::vector<uint8_t>& labels,
                           int max_label, double* accuracy, 
                           double* test_set_loss) {
  aMatrix& conv1_weights = weights[0];
  aMatrix& pool1_weights = weights[1];
  aMatrix& conv2_weights = weights[2];
  aMatrix& pool2_weights = weights[3];
  aMatrix& fully_connected_weights = weights[4];
  aMatrix& output_layer_weights = weights[5];

  aReal loss = 0.0;

  bool* correct = new bool[data.size()];
  aReal* losses = new aReal[data.size()];

  cilk_for (int j = 0; j < data.size(); j++) {
    losses[j] = 0.0;


    // Convolution 1.
    // aMatrix conv1_weights(26,1);
    aMatrix _output(6,28*28);

    for (int a = 0; a < _output.dimensions()[0]; a++) {
      for (int b = 0; b < _output.dimensions()[1]; b++) {
        _output[a][b] = 0.0;
      }
    }

    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        for (int k = 0; k < 6; k++) {
          _output[k][x*28+y] += conv1_weights[k][25]; // bias.
        }
        for (int dx = -2; dx < 3; dx++) {
          for (int dy = -2; dy < 3; dy++) {
            for (int k = 0; k < 6; k++) {
              _output[k][x*28+y] += data[j][(x+dx+2)*32 + (y+dy+2)][0]*conv1_weights[k][(dx+2)*5+dy+2];
            }
          }
        }
      }
    }

    aMatrix output = tanh(_output);


    // pooling layer 1
    //   aMatrix pool1_weights(5)
    aMatrix _output2(6,14*14);
    for (int x = 0; x < 28; x += 2) {
      for (int y = 0; y < 28; y += 2) {
        for (int k = 0; k < 6; k++) {
          _output2[k][(x/2)*14+(y/2)] = pool1_weights[k][4]; // bias.
        }
        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int k = 0; k < 6; k++) {
              _output2[k][(x/2)*14+(y/2)] += output[k][(x+dx)*28 + (y+dy)]*pool1_weights[k][dx*2+dy];
            }
          }
        }
      }
    }
    aMatrix output2 = tanh(_output2);


    // convolution layer 2
    //   aMatrix conv2_weights(26,1)
    aMatrix _output3(16, 10*10);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        for (int k = 0; k < 16; k++) {
          _output3[k][x*10+y] = conv2_weights[k][25]; // bias.
        }
        for (int dx = -2; dx < 3; dx++) {
          for (int dy = -2; dy < 3; dy++) {

            _output3[0][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[0][(dx+2)*5+dy+2];
            _output3[0][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[0][(dx+2)*5+dy+2];
            _output3[0][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[0][(dx+2)*5+dy+2];

            _output3[1][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[1][(dx+2)*5+dy+2];
            _output3[1][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[1][(dx+2)*5+dy+2];
            _output3[1][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[1][(dx+2)*5+dy+2];

            _output3[2][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[2][(dx+2)*5+dy+2];
            _output3[2][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[2][(dx+2)*5+dy+2];
            _output3[2][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[2][(dx+2)*5+dy+2];
  
            _output3[3][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[3][(dx+2)*5+dy+2];
            _output3[3][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[3][(dx+2)*5+dy+2];
            _output3[3][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[3][(dx+2)*5+dy+2];
  
            _output3[4][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[4][(dx+2)*5+dy+2];
            _output3[4][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[4][(dx+2)*5+dy+2];
            _output3[4][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[4][(dx+2)*5+dy+2];
  
            _output3[5][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[5][(dx+2)*5+dy+2];
            _output3[5][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[5][(dx+2)*5+dy+2];
            _output3[5][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[5][(dx+2)*5+dy+2];
  
            _output3[6][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[6][(dx+2)*5+dy+2];
            _output3[6][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[6][(dx+2)*5+dy+2];
            _output3[6][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[6][(dx+2)*5+dy+2];
            _output3[6][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[6][(dx+2)*5+dy+2];
  
            _output3[7][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[7][(dx+2)*5+dy+2];
            _output3[7][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[7][(dx+2)*5+dy+2];
            _output3[7][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[7][(dx+2)*5+dy+2];
            _output3[7][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[7][(dx+2)*5+dy+2];
  
            _output3[8][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[8][(dx+2)*5+dy+2];
            _output3[8][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[8][(dx+2)*5+dy+2];
            _output3[8][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[8][(dx+2)*5+dy+2];
            _output3[8][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[8][(dx+2)*5+dy+2];
  
            _output3[9][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[9][(dx+2)*5+dy+2];
            _output3[9][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[9][(dx+2)*5+dy+2];
            _output3[9][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[9][(dx+2)*5+dy+2];
            _output3[9][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[9][(dx+2)*5+dy+2];
  
            _output3[10][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[10][(dx+2)*5+dy+2];
            _output3[10][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[10][(dx+2)*5+dy+2];
            _output3[10][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[10][(dx+2)*5+dy+2];
            _output3[10][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[10][(dx+2)*5+dy+2];
  
            _output3[11][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[11][(dx+2)*5+dy+2];
            _output3[11][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[11][(dx+2)*5+dy+2];
            _output3[11][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[11][(dx+2)*5+dy+2];
            _output3[11][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[11][(dx+2)*5+dy+2];
  
            _output3[12][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[12][(dx+2)*5+dy+2];
            _output3[12][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[12][(dx+2)*5+dy+2];
            _output3[12][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[12][(dx+2)*5+dy+2];
            _output3[12][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[12][(dx+2)*5+dy+2];
  
            _output3[13][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[13][(dx+2)*5+dy+2];
            _output3[13][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[13][(dx+2)*5+dy+2];
            _output3[13][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[13][(dx+2)*5+dy+2];
            _output3[13][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[13][(dx+2)*5+dy+2];
  
            _output3[14][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[14][(dx+2)*5+dy+2];
            _output3[14][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[14][(dx+2)*5+dy+2];
            _output3[14][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[14][(dx+2)*5+dy+2];
            _output3[14][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[14][(dx+2)*5+dy+2];
  
            _output3[15][x*10+y] += output2[0][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
            _output3[15][x*10+y] += output2[1][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
            _output3[15][x*10+y] += output2[2][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
            _output3[15][x*10+y] += output2[3][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
            _output3[15][x*10+y] += output2[4][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
            _output3[15][x*10+y] += output2[5][(x+dx+2)*14 + y+dy+2] * conv2_weights[15][(dx+2)*5+dy+2];
          }
        }
      }
    }
  
    aMatrix output3 = tanh(_output3);
  
  
    // pooling layer 2
    //   aMatrix pool2_weights(5)
    aMatrix _output4(16,5*5);
    for (int x = 0; x < 10; x += 2) {
      for (int y = 0; y < 10; y += 2) {
        for (int k = 0; k < 16; k++) {
          _output4[k][(x/2)*5+(y/2)] = pool2_weights[k][4]; // bias.
        }
        for (int dx = 0; dx < 2; dx++) {
          for (int dy = 0; dy < 2; dy++) {
            for (int k = 0; k < 16; k++) {
              _output4[k][(x/2)*5+(y/2)] += output3[k][(x+dx)*10 + (y+dy)]*pool2_weights[k][dx*2+dy];
            }
          }
        }
      }
    }
    aMatrix output4 = tanh(_output4);
  
  
    aMatrix output5(5*5*16+1,1);
    for (int k = 0; k < 16; k++) {
      for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
          output5[k*5*5 + x*5+y][0] = output4[k][x*5+y];
        }
      }
    }


    output5[400][0] = 1.0; // bias.

    // weight 120*120
    // fully_connected_weights(120,120)
    aMatrix output6 = tanh(fully_connected_weights**output5);//(120,1);

    output6[120][0] = 1.0; // bias.

    aMatrix final_output = activations::softmax(output_layer_weights**output6,1.0);

    //final_output[final_output.dimensions()[0]-1][0] = 1.0; // bias.

    int argmax = 0;
    double argmaxvalue = final_output[0][0].value();
    for (int k = 0; k < 10; k++) {
      if (argmaxvalue <= final_output[k][0].value()) {
        argmaxvalue = final_output[k][0].value();
        argmax = k;
      }
    }

    Matrix groundtruth(10,1);
    for (int k = 0; k < 10; k++) {
      groundtruth[k][0] = 0.0;
    }
    groundtruth[labels[j]][0] = 1.0;

    //std::cout << std::endl;
    //std::cout << labels[j] << std::endl;
    //std::cout << groundtruth << std::endl;
    //std::cout << mat_prediction << std::endl;
    //std::cout << std::endl;
    losses[j] += activations::crossEntropy(final_output, groundtruth);

    correct[j] = false;
    if (argmax == labels[j]) {
      correct[j] = true;
    }


  }

  int ncorrect = 0;
  int total = 0;
  *test_set_loss = 0.0;
  for (int i = 0; i < data.size(); i++) {
    if (i%2 == 0 || true) {
      loss += losses[i];
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    } else {
      *test_set_loss += losses[i].value();
      if (correct[i]) ncorrect++;
      total++;
    }
  }

  delete[] losses;
  delete[] correct;
  *accuracy = (100.0*ncorrect)/total;

  return loss/data.size();
}

// =============================================================================

void learn_mnist_lenet5_tanh() {
  timer s0,s1,s2,s3,s4;
  using adept::Stack;
  Stack stack;

  std::string data_dir_path = "datasets";

  // load MNIST dataset
  std::vector<uint8_t> train_labels, test_labels;
  std::vector<Matrix> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, -1.0, 1.0, 2, 2);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, -1.0, 1.0, 2, 2);

  int dim1 = train_images[0].dimensions()[0];
  int dim2 = train_images[0].dimensions()[1];
  int max_label = 0;
  int min_label = 100;

  for (int i = 0; i < train_labels.size(); i++) {
    if (train_labels[i] > max_label) max_label = train_labels[i];
    if (train_labels[i] < min_label) min_label = train_labels[i];
  }
  printf("dim1 is %d, dim2 is %d, max label %d, min label %d\n", dim1, dim2, max_label, min_label);

  std::vector<aMatrix> weight_list;
  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_list.push_back(aMatrix(6, 26)); // conv1_weights
  weight_list.push_back(aMatrix(6, 5)); // pool1_weights
  weight_list.push_back(aMatrix(16, 26)); // conv2_weights
  weight_list.push_back(aMatrix(16, 5)); // pool2_weights
  weight_list.push_back(aMatrix(121, 401)); // fully_connected_weights
  weight_list.push_back(aMatrix(85, 121)); // fully_connected_weights2
  weight_list.push_back(aMatrix(10, 85)); // output_layer_weights
  weight_hyper_list.push_back(&weight_list);

  // Initialize the weights.
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < weight_list.size(); i++) {
    double mul = sqrt(1.0/(weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]));
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i][j][k] = distribution(generator)*mul;
                               //(weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);*/
      }
    }
  }

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);
  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.001;

  int TIME_THRESH = GLOBAL_ITER_THRESH;
  for (int iter = 1; iter < 30*1; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    std::vector<Matrix> batch_data;
    std::vector<uint8_t> batch_labels;
    std::uniform_int_distribution<int> dis(0, train_images.size()-1);
    for (int i = 0; i < 1000; i++) {
      int _random = dis(generator);
      int random = _random;
      batch_data.push_back(train_images[random]);
      batch_labels.push_back(train_labels[random]);
    }
    double accuracy = 0.0;
    double test_loss = 0.0;
    aReal loss;
    if (iter%600 == 0 && false) {
      stack.pause_recording();
      loss = compute_mnist_lenet5_fast(*weight_hyper_list[0], test_images, test_labels, max_label, &accuracy, &test_loss);

        std::cout.precision(14);
        std::cout.setf(ios::fixed, ios::floatfield);
        std::cout << std::endl << std::endl << "loss:" << loss.value() << ",\t\t lr: " <<
            learning_rate <<
            "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
            "\r" << std::endl << std::endl;
      stack.continue_recording();
      continue;
    } else {
      //stack.pause_recording();
      if (iter > TIME_THRESH) {
      s2.start();
      s0.start();
      }
      loss = compute_mnist_lenet5_fast(*weight_hyper_list[0], batch_data, batch_labels, max_label, &accuracy, &test_loss);
      if (iter > TIME_THRESH) {
      s0.stop();
      }
      //stack.continue_recording();
      //continue;
    }
    loss.set_gradient(1.0);
    if (iter > TIME_THRESH) {
    s1.start();
    }
    stack.reverse();
    if (iter > TIME_THRESH) {
    s1.stop();
    s2.stop();
    }
    read_gradients(weight_hyper_list, gradients);


    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);

    std::cout.precision(4);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter << ", loss: " << loss.value()
              << ", accuracy: " << accuracy
              << ", time (s): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0
              << std::endl;
  }
  s0.reportTotal("Forward pass");
  s1.reportTotal("Reverse pass");
  s2.reportTotal("Forward+Reverse pass");
}

void learn_mnist_lenet5() {
  timer s0,s1,s2,s3,s4;
  using adept::Stack;
  Stack stack;

  std::string data_dir_path = "datasets";

  // load MNIST dataset
  std::vector<uint8_t> train_labels, test_labels;
  std::vector<Matrix> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, -1.0, 1.0, 2, 2);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, -1.0, 1.0, 2, 2);


  int dim1 = train_images[0].dimensions()[0];
  int dim2 = train_images[0].dimensions()[1];


  int max_label = 0;
  int min_label = 100;

  for (int i = 0; i < train_labels.size(); i++) {
    if (train_labels[i] > max_label) max_label = train_labels[i];
    if (train_labels[i] < min_label) min_label = train_labels[i];
  }



  printf("dim1 is %d, dim2 is %d, max label %d, min label %d\n", dim1, dim2, max_label, min_label);


  std::vector<aMatrix> weight_list;
  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_list.push_back(aMatrix(6, 26)); // conv1_weights
  weight_list.push_back(aMatrix(6, 5)); // pool1_weights
  weight_list.push_back(aMatrix(16, 26)); // conv2_weights
  weight_list.push_back(aMatrix(16, 5)); // pool2_weights
  weight_list.push_back(aMatrix(121, 401)); // fully_connected_weights
  weight_list.push_back(aMatrix(85, 121)); // fully_connected_weights2
  weight_list.push_back(aMatrix(10, 85)); // output_layer_weights
  weight_hyper_list.push_back(&weight_list);



  // Initialize the weights.
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < weight_list.size(); i++) {
    double mul = sqrt(1.0/(weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]));
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i][j][k] = distribution(generator)*mul;
                               //(weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);*/
      }
    }
  }




  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);

  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.001;

  //printf("train image size %d\n", train_images.size());





  int TIME_THRESH = GLOBAL_ITER_THRESH;
  for (int iter = 1; iter < 30*1; iter++) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    std::vector<Matrix> batch_data;
    std::vector<uint8_t> batch_labels;
    std::uniform_int_distribution<int> dis(0, train_images.size()-1);
    for (int i = 0; i < 1000; i++) {
      int _random = dis(generator);
      int random = _random;
      batch_data.push_back(train_images[random]);
      batch_labels.push_back(train_labels[random]);
    }
    double accuracy = 0.0;
    double test_loss = 0.0;
    aReal loss;
    if (iter%600 == 0 && false) {
      stack.pause_recording();
      loss = compute_mnist_lenet5_fast_maxpool(*weight_hyper_list[0], test_images, test_labels, max_label, &accuracy, &test_loss);

        std::cout.precision(14);
        std::cout.setf(ios::fixed, ios::floatfield);
        std::cout << std::endl << std::endl << "loss:" << loss.value() << ",\t\t lr: " <<
            learning_rate <<
            "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
            "\r" << std::endl << std::endl;
      stack.continue_recording();
      continue;
    } else {
      //stack.pause_recording();
      if (iter > TIME_THRESH) {
      s2.start();
      s0.start();
      }
      loss = compute_mnist_lenet5_fast_maxpool(*weight_hyper_list[0], batch_data, batch_labels, max_label, &accuracy, &test_loss);
      if (iter > TIME_THRESH) {
      s0.stop();
      }
      //stack.continue_recording();
      //continue;
    }
    loss.set_gradient(1.0);
    if (iter > TIME_THRESH) {
    s1.start();
    }
    stack.reverse();
    if (iter > TIME_THRESH) {
    s1.stop();
    s2.stop();
    }
    read_gradients(weight_hyper_list, gradients);
    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);

    std::cout.precision(4);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "iter: " << iter
              << ", loss: " << loss.value() 
              << ", accuracy: " << accuracy
              << ", time (s): " << (std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count()) / 1000000.0
              << std::endl;
    /*
    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\n";// << std::flush;
    */
  }

  s0.reportTotal("Forward pass");
  s1.reportTotal("Reverse pass");
  s2.reportTotal("Forward+Reverse pass");
}
