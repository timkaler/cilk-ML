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
#include <map>
#include <vector>

#include "./activations.hpp"
#include "./Graph.hpp"
#include "./optimization.hpp"
#include "./io_helpers.hpp"

#include "./mnist_parser.h"







//int sched_yield(void) {
//for (int i=0; i< 4000; i++) _mm_pause(); //usleep(1);
//
//return 0;
//
//}








// #define TFK_ADEPT_SERIAL

using adept::Real;
using adept::aReal;
using adept::Matrix;
using adept::aMatrix;
using adept::Vector;
using adept::aVector;

using std::vector;

std::default_random_engine generator(44);

void tfk_init() {
  thread_local_worker_id = __cilkrts_get_worker_number();
  tfk_reducer.get_tls_references();
}




#define _max(a,b,c,d) max(max(a,b), max(c,d))

//aReal _max(aReal a, aReal b, aReal c, aReal d) {
//
//  float max_val = a.value();
//  int cas = 0;
//  if (b.value() > max_val) {
//    max_val = b.value();
//    cas = 1;
//  }
//
//  if (c.value() > max_val) {
//    max_val = c.value();
//    cas = 2;
//  }
//  if (d.value() > max_val) {
//    max_val = d.value();
//    cas = 3;
//  }
//
//  if (cas == 0) return a;
//  if (cas == 1) return b;
//  if (cas == 2) return c;
//  if (cas == 3) return d;
//  return a;
//}

aMatrix lenet_pool2_maxpool(aMatrix& output3) {
    // pooling layer 2
    //   aMatrix pool2_weights(5)
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
    // pooling layer 2
    //   aMatrix pool2_weights(5)
    aMatrix _output4(16*5*5+1, 1);
    for (int x = 0; x < 10; x += 2) {
      for (int y = 0; y < 10; y += 2) {
        //#include "kernel4.txt"
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
    // convolution layer 2
    //   aMatrix conv2_weights(26,1)
    aMatrix _output3(16, 10*10);
    for (int x = 0; x < 10; x++) {
      for (int y = 0; y < 10; y++) {
        for (int k = 0; k < 16; k++) {
          _output3(k,x*10+y) = conv2_weights(k,25); // bias.
        }
        for (int dx = -2; dx < 3; dx++) {
          for (int dy = -2; dy < 3; dy++) {


            _output3(0,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2)
                               + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2)
                               + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(0,(dx+2)*5+dy+2);

            _output3(1,x*10+y) += output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2)
                               + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(1,(dx+2)*5+dy+2);

            _output3(2,x*10+y) += output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(2,(dx+2)*5+dy+2);
  
            _output3(3,x*10+y) += output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2)
                               + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(3,(dx+2)*5+dy+2);
  
            _output3(4,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2)
                               + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(4,(dx+2)*5+dy+2);
  
            _output3(5,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2)
                               + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2)
                               + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(5,(dx+2)*5+dy+2);
  
            _output3(6,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2)
                               + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2)
                               + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(6,(dx+2)*5+dy+2);
  
            _output3(7,x*10+y) += output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2)
                               + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(7,(dx+2)*5+dy+2);

            _output3(8,x*10+y) += output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2)
                               + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(8,(dx+2)*5+dy+2);
  
            _output3(9,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2)
                               + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2)
                               + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2)
                               + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(9,(dx+2)*5+dy+2);
  
            _output3(10,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2)
                                + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2)
                                + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2)
                                + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(10,(dx+2)*5+dy+2);
  
            _output3(11,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2)
                                + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2)
                                + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2)
                                + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(11,(dx+2)*5+dy+2);
  
            _output3(12,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2)
                                + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2)
                                + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2)
                                + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(12,(dx+2)*5+dy+2);
  
            _output3(13,x*10+y) += output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2)
                                + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2)
                                + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2)
                                + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(13,(dx+2)*5+dy+2);
  
            _output3(14,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2)
                                + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2)
                                + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2)
                                + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(14,(dx+2)*5+dy+2);
  
            _output3(15,x*10+y) += output2(0,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2)
                                + output2(1,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2)
                                + output2(2,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2)
                                + output2(3,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2)
                                + output2(4,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2)
                                + output2(5,(x+dx+2)*14 + y+dy+2) * conv2_weights(15,(dx+2)*5+dy+2);
          }
        }
      }
    }

  return _output3;
}


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

cilk_for(int i = 0; i < 10; i++) {
  if (i == __cilkrts_get_worker_number()) {
    a += b;
    b = a;
  }
}


aReal d = a*a + b;


d.set_gradient(1.0);
stack.reverse();

std::cout << a.get_gradient() << "," << b.get_gradient()<< std::endl;





}



//void standard_2dconvolution(aMatrix& input, aMatrix& conv_weights, int input_stride_x, input_stride_y, int output_stride_x, int output_stride_y,
//    aMatrix& output) {
//
//
//  for (int x = 0; x < ; x++) {
//    for (int y = 0; y < dim2; y++) {
//      output[x*dim2+y] =
//          conv_weights[conv_weights.dimensions()[0]*conv_weights.dimensions()[1]-1];  // bias
//
//      for (int dx = -2; dx < 3; dx++) {
//        for (int dy = -2; dy < 3; dy++) {
//            output[x*output_stride_y+y*output_stride_x] +=
//                input[(x+dx+2)*32 + (y+dy+2)][0]*conv1_weights[k][(dx+2)*5+dy+2];
//        }
//      }
//
//    }
//  }
//}



aReal compute_mnist_lenet5_fast_maxpool(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                    std::vector<uint8_t>& labels,
                    int max_label, double* accuracy, double* test_set_loss) {
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

  cilk_for (int j = 0; j < data.size(); j += 1) {
    //int _end = _j+4;
    //if (_end > data.size()) _end = data.size();
    //for (int j = _j; j < _end; j++) {
    losses[j] = 0.0;


    // Convolution 1.
    // aMatrix conv1_weights(26,1);
    aMatrix output(6,28*28);


    for (int a = 0; a < output.dimensions()[0]; a++) {
      for (int b = 0; b < output.dimensions()[1]; b++) {
        output(a,b) = 0.0;
      }
    }


    cilk_for (int x = 0; x < 28; x++) {
      cilk_for (int y = 0; y < 28; y++) {
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

    cilk_for (int x = 0; x < 28; x += 2) {
      cilk_for (int y = 0; y < 28; y += 2) {
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
    aMatrix output2 = tfksig(_output2);


    aMatrix _output3 = lenet_conv2(output2, conv2_weights);

    //aMatrix output3 = tanh(_output3);
    //aMatrix _output4 = lenet_pool2(output3, pool2_weights);
    //aMatrix output5 = tanh(_output4);
    aMatrix output5 = tfksig(lenet_pool2_maxpool(_output3));

    output5(400,0) = 1.0; // bias.

    // weight 120*120
    // fully_connected_weights(120,120)
    aMatrix output6 = tfksig(fully_connected_weights**output5);//(120,1);

    output6(120,0) = 1.0; // bias.


    aMatrix output7 = tfksig(fully_connected_weights2**output6);
    output7(84,0) = 1.0; //bias

    aMatrix final_output = tfksoftmax(output_layer_weights**output7,1.0);


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
    losses[j] += crossEntropy(final_output, groundtruth);

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





















aReal compute_mnist_lenet5_fast(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                    std::vector<uint8_t>& labels,
                    int max_label, double* accuracy, double* test_set_loss) {
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

    aMatrix final_output = tfksoftmax(output_layer_weights**output7,1.0);


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
    losses[j] += crossEntropy(final_output, groundtruth);

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





















aReal compute_mnist_lenet5(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                    std::vector<uint8_t>& labels,
                    int max_label, double* accuracy, double* test_set_loss) {
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

    aMatrix final_output = tfksoftmax(output_layer_weights**output6,1.0);

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
    losses[j] += crossEntropy(final_output, groundtruth);

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


aReal compute_mnist(std::vector<aMatrix>& weights, std::vector<Matrix>& data,
                    std::vector<uint8_t>& labels,
                    int max_label, double* accuracy, double* test_set_loss) {


  aReal loss = 0.0;

  bool* correct = new bool[data.size()];
  aReal* losses = new aReal[data.size()];

  aMatrix& biases = weights[weights.size()-1];

  #pragma cilk grainsize 1
  cilk_for (int j = 0; j < data.size(); j++) {
  //cilk_for (int _j = 0; _j < data.size(); _j += 10) {
    //int j_start = _j; 
    //int j_end = j_start + 10;
    //if (j_end > data.size()) j_end = data.size();
    //for (int j = j_start; j < j_end; j++) { 

    losses[j] = 0.0;

    std::vector<aMatrix> results = std::vector<aMatrix>(weights.size()-1);

    //data[j][data[j].dimensions()[0]-1][0] = 1.0;
    //aMatrix& biases = weights[weights.size()-1];
    results[0] = tfksig(weights[0]**data[j]);

    //results[0] += biases(0,0);

    //std::cout << data[j] << std::endl;
    //exit(0);


    for (int k = 1; k < weights.size()-1; k++) {
      //results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
      if (k != weights.size()-2) {
      results[k] = tfksig(weights[k]**results[k-1]);
      } else {
      results[k] = weights[k]**results[k-1];
      }
      results[k] += biases(k,0);
    }
    aMatrix mat_prediction = tfksoftmax(results[results.size()-1], 1.0);
    //printf("dimensions %d, %d\n", mat_prediction.dimensions()[0], mat_prediction.dimensions()[1]);

    //std::cout << mat_prediction <<std::endl;

    int argmax = 0;
    double argmaxvalue = mat_prediction(0,0).value();
    for (int k = 0; k < 10; k++) {
      if (argmaxvalue <= mat_prediction(k,0).value()) {
        argmaxvalue = mat_prediction(k,0).value();
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
    losses[j] += logitCrossEntropy(mat_prediction, groundtruth);

    correct[j] = false;
    if (argmax == labels[j]) {
      correct[j] = true;
    }
    //}
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

  return loss/(1.0*data.size());
}

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
      results[0] = tfksig(weights[0]**data[i]);
      for (int k = 1; k < weights.size()-1; k++) {
        // bias term.
        results[k-1][results[k-1].dimensions()[0]-1][0] = 1.0;
        results[k] = tfksig(weights[k]**results[k-1]);
      }
      aMatrix mat_prediction = tfksoftmax(results[results.size()-1], 1.0);

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

      losses[i] += crossEntropy(mat_prediction, groundtruth);

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

  Stack stack;                           // Object to store differential statements

  std::vector<Matrix > data;
  std::vector<Real> labels;
  read_connect4("datasets/connect-4.data", data, labels);


  std::vector<aMatrix> weight_list;

  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(43, 43));  // 43 x 1
  weight_list.push_back(aMatrix(3, 43));  // 3 x 1


  // Initialize the weights.
  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  for (int i = 0; i < weight_list.size(); i++) {
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i](j,k) =
            distribution(generator) /
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



aReal compute_gcn_pubmed(Graph& G, std::vector<Matrix>& groundtruth_labels, bool* is_train,
                         bool* is_val, int max_labels, double* accuracy, double* test_set_loss) {

  std::vector<std::vector<aMatrix> > embeddings;
  embeddings.resize(G.embedding_dim_list.size()-1);
  aReal loss = 0;
  aReal loss_norm = 0.0;
  for (int i = 0; i < G.embedding_dim_list.size()-1; i++) {
    embeddings[i].resize(G.num_vertices);
  }

  for (int l = 0; l < G.embedding_dim_list.size()-1; l++) {

    bool last = (l == (G.embedding_dim_list.size()-2));

    cilk_for (int i = 0; i < G.num_vertices; i += 1) {
    //cilk_for (int i = 0; i < G.num_vertices; i++) {
      int end = i+1;
      if (end > G.num_vertices) end = G.num_vertices;
      if (i == end) continue;
      for (int j = i; j < end; j++) {
        if (last) {
          embeddings[l][j] = tfksoftmax(G.get_embedding(j,l, embeddings), 1.0);
        } else {
          embeddings[l][j] = G.get_embedding(j,l, embeddings);
        }
      }
      //embeddings[l][i] = G.get_embedding(i,l, embeddings);
    }
  }
  int total_predictions = 0;
  int total_correct = 0;

  int num_train_items = 0;
  for (int i = 0; i < G.num_vertices; i++) {
    aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
    //aMatrix yhat = tfksoftmax(yhat_, 0.5);

    Matrix y(max_labels,1);
    double max_label_val = 0.0;
    int max_label = 0;

    int gt_label = 0;

    for (int j = 0; j < y.dimensions()[0]; j++) {
      y[j][0] = 0.0;
      if (yhat[j][0].value() > max_label_val) {
        max_label_val = yhat[j][0].value();
        max_label = j;
      }
      if (groundtruth_labels[i](j,0) > 0.5) gt_label = j;
    }


 
    y[gt_label][0] = 1.0;
    //if (!G.vertex_training[i]) {
    if (!is_train[i] && !is_val[i]) {
      if (max_label == gt_label) total_correct++;
      total_predictions++;
      //aReal tmp = sum((yhat-y)*(yhat-y));//crossEntropy(yhat, y);
      aReal tmp = crossEntropy(yhat, y);
      *test_set_loss += tmp.value();
    } else if (is_train[i]) {
      loss += crossEntropy(yhat,y);
      loss_norm += 1.0;
      //num_train_items++;
    }
  }

  loss = loss / loss_norm;
  //loss /= (1.0*num_train_items);

  printf("num_train_items %d\n", num_train_items);
  //printf("total predictions %d\n", total_predictions);

  //tfk_reducer.sp_tree.walk_tree_debug(tfk_reducer.sp_tree.get_root());
  //Real norm = 1.0*num_train_items;
  //aReal loss_normalized = loss / norm;

  *accuracy = ((100.0*total_correct)/(1.0*total_predictions));

  return loss;
}




aReal compute_gcn(Graph& G, std::map<int, int >& department_labels, int max_label,
                  double* accuracy, double* test_set_loss) {

  std::vector<std::vector<aMatrix> > embeddings;
  embeddings.resize(G.embedding_dim_list.size()-1);
  aReal loss = 0;
  for (int i = 0; i < G.embedding_dim_list.size()-1; i++) {
    embeddings[i].resize(G.num_vertices);
  }

  for (int l = 0; l < G.embedding_dim_list.size()-1; l++) {

    bool last = (l == (G.embedding_dim_list.size()-2));

    cilk_for (int i = 0; i < G.num_vertices; i += 10) {
    //cilk_for (int i = 0; i < G.num_vertices; i++) {
      int end = i+10;
      if (end > G.num_vertices) end = G.num_vertices;
      if (i == end) continue;
      for (int j = i; j < end; j++) {
        if (last) {
          embeddings[l][j] = tfksoftmax(G.get_embedding(j,l, embeddings), 0.5);
        } else {
          embeddings[l][j] = G.get_embedding(j,l, embeddings);
        }
      }
      //embeddings[l][i] = G.get_embedding(i,l, embeddings);
    }
  }
  int total_predictions = 0;
  int total_correct = 0;

  for (int i = 0; i < G.num_vertices; i++) {
    aMatrix yhat = (embeddings[G.embedding_dim_list.size()-2][i]);
    //aMatrix yhat = tfksoftmax(yhat_, 0.5);

    Matrix y(max_label,1);
    double max_label_val = 0.0;
    int max_label = 0;
    for (int j = 0; j < y.dimensions()[0]; j++) {
      y[j][0] = 0.0;
      if (yhat[j][0].value() > max_label_val) {
        max_label_val = yhat[j][0].value();
        max_label = j;
      }
    }
    y[department_labels[i]][0] = 1.0;
    if (/*i%2==1*/!G.vertex_training[i]) {
      if (max_label == department_labels[i]) total_correct++;
      total_predictions++;
      //aReal tmp = sum((yhat-y)*(yhat-y));//crossEntropy(yhat, y);
      aReal tmp = crossEntropy(yhat, y);
      *test_set_loss += tmp.value();
    } else if (G.vertex_training_active[i]) {
      loss += crossEntropy(yhat,y);
    }
  }


  //tfk_reducer.sp_tree.walk_tree_debug(tfk_reducer.sp_tree.get_root());

  *accuracy = ((100.0*total_correct)/(1.0*total_predictions));
  return loss;
}

void learn_gcn() {
  using adept::Stack;

  Stack stack;

  Graph G(0);
  edge_list_to_graph("datasets/email-Eu-core.txt", G);

  std::map<int, int> department_labels;
  std::vector<std::pair<int,int> > pairs;
  read_pair_list("datasets/email-Eu-core-department-labels.txt", pairs);
  int max_label = 0;
  for (int i = 0; i < pairs.size(); i++) {
    department_labels[pairs[i].first] = pairs[i].second;
    if (pairs[i].second > max_label) max_label = pairs[i].second;
  }
  max_label = max_label + 1;
  G.max_label = max_label;

  std::vector<int> counts(max_label+1);
  for (int i = 0; i < pairs.size(); i++) {
    counts[pairs[i].second]++;
  }


  std::vector<int> _embedding_dim_list;
  _embedding_dim_list.push_back(64);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(128);
  _embedding_dim_list.push_back(max_label);


  G.setup_embeddings(_embedding_dim_list);
  G.generate_random_initial_embeddings();

  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_hyper_list.push_back(&G.weights);
  weight_hyper_list.push_back(&G.skip_weights);

  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  G.vertex_values.resize(G.num_vertices);
  G.vertex_training.resize(G.num_vertices);
  G.vertex_training_active.resize(G.num_vertices);

  // Randomly divide into training and test set.
  for (int i = 0; i < G.num_vertices; i++) {
    G.vertex_values[i] = department_labels[i];
    if (distribution(generator) < 0.5) {
      G.vertex_training[i] = true;
      G.vertex_training_active[i] = true;
    } else {
      G.vertex_training[i] = false;
      G.vertex_training_active[i] = false;
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

  for (int iter = 0; iter < 10000; iter++) {
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    double accuracy = 0.0;
    double test_loss = 0.0;

    aReal loss = compute_gcn(G, department_labels, max_label, &accuracy, &test_loss);

    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\r" << std::flush;

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    //double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
    //apply_gradient_update(weight_hyper_list, weights_raw, weights_raw_old, gradients,
    //                      learning_rate/norm);

  }
}


void learn_gcn_pubmed() {
  using adept::Stack;

  Stack stack;

  Graph G(0);
  edge_list_to_graph("datasets/pubmed.edges", G);

  int n_vertices = G.num_vertices;
  int feature_dim = 500;

  bool* is_train = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_val = (bool*) calloc(n_vertices, sizeof(bool));
  bool* is_test = (bool*) calloc(n_vertices, sizeof(bool));

  std::vector<Matrix> groundtruth_labels;
  std::vector<Matrix> feature_vectors;
  for (int i = 0; i < n_vertices; i++) {
    Matrix tmp(3,1);
    for (int j = 0; j < 3; j++) {
      tmp(j,0) = 0.0;
    }
    groundtruth_labels.push_back(tmp);

    Matrix tmp2(feature_dim,1);
    for (int j = 0; j < feature_dim; j++) {
      tmp2(j,0) = 0.0;
    }
    feature_vectors.push_back(tmp2);
  }
  int max_label = 3;
  parse_pubmed_data("datasets/pubmed.trainlabels", "datasets/pubmed.vallabels",
                    "datasets/pubmed.testlabels", "datasets/pubmed_features", is_train, is_val, is_test, groundtruth_labels,
                    feature_vectors);

  std::vector<int> _embedding_dim_list;
  _embedding_dim_list.push_back(feature_dim);
  _embedding_dim_list.push_back(32);
  //_embedding_dim_list.push_back(16);
  _embedding_dim_list.push_back(max_label);


  G.setup_embeddings(_embedding_dim_list);
  G.set_initial_embeddings(feature_vectors);
  //G.generate_random_initial_embeddings();

  std::vector<std::vector<aMatrix>*> weight_hyper_list;
  weight_hyper_list.push_back(&G.weights);
  weight_hyper_list.push_back(&G.skip_weights);

  std::default_random_engine generator(1000);
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  //G.vertex_values.resize(G.num_vertices);
  //G.vertex_training.resize(G.num_vertices);
  //G.vertex_training_active.resize(G.num_vertices);

  //// Randomly divide into training and test set.
  //for (int i = 0; i < G.num_vertices; i++) {
  //  G.vertex_values[i] = department_labels[i];
  //  if (distribution(generator) < 0.5) {
  //    G.vertex_training[i] = true;
  //    G.vertex_training_active[i] = true;
  //  } else {
  //    G.vertex_training[i] = false;
  //    G.vertex_training_active[i] = false;
  //  }
  //}

  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);

  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);

  double learning_rate = 0.1;//0.01;

  for (int iter = 0; iter < 100; iter++) {
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    double accuracy = 0.0;
    double test_loss = 0.0;

    aReal loss = compute_gcn_pubmed(G, groundtruth_labels, is_train, is_val, max_label, &accuracy,
                                    &test_loss);

    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "" << std::endl;

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    //double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
    //apply_gradient_update(weight_hyper_list, weights_raw, weights_raw_old, gradients,
    //                      learning_rate/norm);

  }
}




void learn_mnist_lenet5() {
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





  for (int iter = 1; iter < 60*1; iter++) {
    set_values(weight_hyper_list, weights_raw);
    stack.new_recording();

    std::vector<Matrix> batch_data;
    std::vector<uint8_t> batch_labels;
    std::uniform_int_distribution<int> dis(0, train_images.size()-1);
    for (int i = 0; i < 100; i++) {
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
      loss = compute_mnist_lenet5_fast_maxpool(*weight_hyper_list[0], batch_data, batch_labels, max_label, &accuracy, &test_loss);
      //stack.continue_recording();
      //continue;
    }
    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\r" << std::flush;

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
  }


}



void learn_mnist() {
  using adept::Stack;
  Stack stack;
  std::string data_dir_path = "datasets";

  // load MNIST dataset
  std::vector<uint8_t> train_labels, test_labels;
  std::vector<Matrix> train_images, test_images;

  tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               &train_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               &train_images, 0.0, 1.0, 0, 0);
  tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               &test_labels);
  tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               &test_images, 0.0, 1.0, 0, 0);


  //for (int i = 0; i < train_images.size(); i++) {
  //  train_images[i] /= 28.0*28.0;
  //}
  //for (int i = 0; i < test_images.size(); i++) {
  //  test_images[i] /= 28.0*28.0;
  //}

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
  weight_list.push_back(aMatrix(800, 28*28));
  weight_list.push_back(aMatrix(10, 800));
  weight_list.push_back(aMatrix(2,1));
  //weight_list.push_back(aMatrix(256+1, 1024));
  //weight_list.push_back(aMatrix(64+1, 256+1));
  //weight_list.push_back(aMatrix(16+1, 64+1));
  //weight_list.push_back(aMatrix(10, 16+1));
  weight_hyper_list.push_back(&weight_list);



  // Initialize the weights.
  std::default_random_engine generator(1000);
  for (int i = 0; i < weight_list.size(); i++) {
    float range = sqrt(6.0 / (weight_list[i].dimensions()[0] + weight_list[i].dimensions()[1]));
    std::uniform_real_distribution<double> distribution(-range, range);
    if (i == weight_list.size()-1) {
      for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
        for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
          weight_list[i][j][k] = 0.0;//distribution(generator);// / (weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);
        }
      }
      continue;
    }
    for (int j = 0; j < weight_list[i].dimensions()[0]; j++) {
      for (int k = 0; k < weight_list[i].dimensions()[1]; k++) {
        weight_list[i](j,k) = distribution(generator) / (weight_list[i].dimensions()[0]*weight_list[i].dimensions()[1]);
      }
    }
  }




  double* weights_raw = allocate_weights(weight_hyper_list);
  double* weights_raw_old = allocate_weights(weight_hyper_list);
  double* gradients = allocate_weights_zero(weight_hyper_list);
  double* momentums = allocate_weights_zero(weight_hyper_list);
  double* velocities = allocate_weights_zero(weight_hyper_list);

  read_values(weight_hyper_list, weights_raw);
  read_values(weight_hyper_list, weights_raw_old);


  double learning_rate = 0.01;

  for (int iter = 0; iter < 60*1; iter++) {
    set_values(weight_hyper_list, weights_raw);

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
    stack.new_recording();
    aReal loss;
    if (iter%100 == 0 && false) {
      stack.pause_recording();
      loss = compute_mnist(*weight_hyper_list[0], test_images, test_labels, max_label, &accuracy, &test_loss);

        std::cout.precision(14);
        std::cout.setf(ios::fixed, ios::floatfield);
        std::cout << std::endl << std::endl << "loss:" << loss.value() << ",\t\t lr: " <<
            learning_rate <<
            "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
            "\n" << std::endl << std::endl;
      stack.continue_recording();
      continue;
    } else {
      //stack.pause_recording();
      loss += compute_mnist(*weight_hyper_list[0], batch_data, batch_labels, max_label, &accuracy, &test_loss);
      //stack.continue_recording();
    }
    //stack.initialize_gradients();
    loss.set_gradient(1.0);
    stack.reverse();
    read_gradients(weight_hyper_list, gradients);

    std::cout.precision(14);
    std::cout.setf(ios::fixed, ios::floatfield);
    std::cout << "loss:" << loss.value() << ",\t\t lr: " << learning_rate <<
        "\t\t accuracy z: " << accuracy << "% \t\t Test set loss: " << test_loss <<
        "\n" << std::endl;

    store_values_into_old(weight_hyper_list, weights_raw, weights_raw_old);


    double norm = compute_gradient_norm(weight_hyper_list, gradients);
    //printf("gradient norm is %f\n", norm);
    //if (norm < 1.0) norm = 1.0;
    apply_gradient_update_ADAM(weight_hyper_list, weights_raw, weights_raw_old, gradients,
                               momentums, velocities, 1.0, learning_rate, iter+1);
  }


}


int main(int argc, const char** argv) {
  //learn_connect4();
  //learn_gcn_pubmed();
  //test_matvec();
  //test_matvec_slow();
  //test_matvec();

  learn_mnist();
  //learn_mnist_lenet5();
  //test_bug();
  //test_opt();
  //test_tb3(atoi(argv[1]));
  //learn_mnist_lenet5_fast();
  return 0;
}























