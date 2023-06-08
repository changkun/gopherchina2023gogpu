// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

#include <metal_stdlib>
using namespace metal;

struct params {
    uint colA;
    uint colB;
};

kernel void mul(device const float*  inA     [[ buffer(0) ]],
                device const float*  inB     [[ buffer(1) ]],
                device       float*  out     [[ buffer(2) ]],
                device const params& params  [[ buffer(3) ]],
                uint                 index   [[thread_position_in_grid]]) {

    uint i = index / uint(params.colB);
    uint j = index % uint(params.colB);

    float sum = 0.0;
    for (uint k = 0; k < params.colA; k++) {
        float a = inA[i * int(params.colA) + k];
        float b = inB[k * int(params.colB) + j];
        sum += a * b;
    }
    out[index] = sum;
}