// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

package main_test

import (
	"fmt"
	"testing"

	"changkun.de/x/gogpu/gpu"
	"changkun.de/x/gogpu/math"
)

func TestMul(t *testing.T) {
	if !gpu.Driver().Available() {
		t.Skip("no GPU device available")
	}

	tests := []struct {
		m1, m2 math.Mat[float32]
	}{
		{
			m1: math.Mat[float32]{
				Row: 2, Col: 2,
				Data: []float32{
					1, 2,
					3, 4,
				},
			},
			m2: math.Mat[float32]{
				Row: 2, Col: 2,
				Data: []float32{
					5, 6,
					7, 8,
				},
			},
		},
		{
			m1: math.Mat[float32]{
				Row: 2, Col: 1,
				Data: []float32{
					1,
					2,
				},
			},
			m2: math.Mat[float32]{
				Row: 1, Col: 2,
				Data: []float32{
					1, 2,
				},
			},
		},
		{
			m1: math.Mat[float32]{
				Row: 1, Col: 2,
				Data: []float32{
					1, 2,
				},
			},
			m2: math.Mat[float32]{
				Row: 2, Col: 1,
				Data: []float32{
					1,
					2,
				},
			},
		},
		{
			m1: math.Mat[float32]{
				Row: 2, Col: 1,
				Data: []float32{
					1,
					2,
				},
			},
			m2: math.Mat[float32]{
				Row: 1, Col: 10,
				Data: []float32{
					1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				},
			},
		},
		{
			m1: math.Mat[float32]{
				Row: 1, Col: 2,
				Data: []float32{
					1, 2,
				},
			},
			m2: math.Mat[float32]{
				Row: 2, Col: 3,
				Data: []float32{
					1, 2, 3,
					4, 5, 6,
				},
			},
		},
		{
			m1: math.NewRandMat[float32](7, 6),
			m2: math.NewRandMat[float32](6, 3),
		},
	}

	for _, tt := range tests {
		out1 := gpu.Mul(tt.m1, tt.m2)
		out2 := tt.m1.Mul(tt.m2)
		out3 := tt.m1.MulNaive(tt.m2)

		if !out1.Eq(out2) || !out1.Eq(out3) {
			t.Fatalf("GPU Mul receives different results compare to CPU: GPU(%v) vs. CPU(%v, %v): m1(%v), m2(%v)", out1, out2, out3, tt.m1, tt.m2)
		}
	}
}

func BenchmarkMul(b *testing.B) {
	if !gpu.Driver().Available() {
		b.Skip("no Metal device available")
	}

	for size := 1 << 1; size < 2<<13; size *= 2 {
		m1 := math.NewRandMat[float32](size, size)
		m2 := math.NewRandMat[float32](size, size)

		var outGPU math.Mat[float32]
		var outCPU math.Mat[float32]

		b.Run(fmt.Sprintf("GPU(%vx%v)", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				outGPU = gpu.Mul(m1, m2)
			}
		})
		b.Run(fmt.Sprintf("CPU-block(%vx%v)", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				outCPU = m1.Mul(m2)
			}
		})
		b.Run(fmt.Sprintf("CPU-naive(%vx%v)", size, size), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				outCPU = m1.MulNaive(m2)
			}
		})

		if !outCPU.Eq(outGPU) {
			b.Fatal("inconsistent results")
		}
	}
}
