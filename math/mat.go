// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

package math

import (
	"math"
	"math/rand"
)

// Type defines all supported data types.
type Type interface {
	~uint8 | ~uint32 | ~int32 | ~float32
}

// TypeSize returns the corresponding type size.
func TypeSize[T Type]() int {
	var v T
	switch any(v).(type) {
	case uint8:
		return 1
	case int32:
		return 4
	case uint32:
		return 4
	case float32:
		return 4
	}
	panic("unknown data size for type")
}

// Mat represents a WxH matrix.
type Mat[T Type] struct {
	Row  int
	Col  int
	Data []T
}

// NewRandMat returns a random matrix.
func NewRandMat[T Type](row, col int) Mat[T] {
	m := Mat[T]{
		Row:  row,
		Col:  col,
		Data: make([]T, row*col),
	}

	for i := range m.Data {
		m.Data[i] = T(rand.Float64())
	}
	return m
}

// Eq returns true if two matrices are equal.
func (m Mat[T]) Eq(n Mat[T]) bool {
	if m.Row != n.Row || m.Col != n.Col {
		return false
	}

	approxEq := func(v1, v2 float64) bool { return math.Abs(v1-v2) <= 1e-7 }
	for i := range m.Data {
		if !approxEq(float64(m.Data[i]), float64(n.Data[i])) {
			return false
		}
	}
	return true
}

// Index returns the element index of Data at (i, j)
func (m Mat[T]) Index(i, j int) int {
	return i*m.Col + j
}

// Get gets the corresponding element at (i, j)
func (m Mat[T]) Get(i, j int) T {
	return m.Data[m.Index(i, j)]
}

// Set sets the given value to matrix at (i, j)
func (m Mat[T]) Set(i, j int, v T) {
	m.Data[m.Index(i, j)] = v
}

// MulNaive applies matrix multiplication of two given matrix, and returns
// the resulting matrix: r = m*n
func (m Mat[T]) MulNaive(n Mat[T]) Mat[T] {
	if m.Col != n.Row {
		panic("math: mismatched matrix dimension")
	}

	r := Mat[T]{
		Row:  m.Row,
		Col:  n.Col,
		Data: make([]T, m.Row*n.Col),
	}

	for i := 0; i < m.Row; i++ {
		for j := 0; j < n.Col; j++ {
			sum := T(0)
			for k := 0; k < m.Col; k++ {
				sum += m.Get(i, k) * n.Get(k, j)
			}
			r.Set(i, j, sum)
		}
	}
	return r
}

// Mul applies matrix multiplication of two given matrix, and returns
// the resulting matrix: r = m*n
//
// This is a blocking version of matrix multiplication in jki order.
func (m Mat[T]) Mul(n Mat[T]) Mat[T] {
	if m.Col != n.Row {
		panic("math: mismatched matrix dimension")
	}

	blockSize := 4

	r := Mat[T]{
		Row:  m.Row,
		Col:  n.Col,
		Data: make([]T, m.Row*n.Col),
	}

	min := m.Row
	if m.Col < min {
		min = m.Col
	}
	if n.Col < min {
		min = n.Col
	}
	var (
		kk, jj, i, j, k int
		rr              T
		en              = blockSize * (min / blockSize)
	)

	for kk = 0; kk < en; kk += blockSize {
		for jj = 0; jj < en; jj += blockSize {
			for k = kk; k < kk+blockSize; k++ {
				for j = jj; j < jj+blockSize; j++ {
					rr = n.Get(k, j)
					for i = 0; i < m.Row; i++ {
						r.Set(i, j, r.Get(i, j)+rr*m.Get(i, k))
					}
				}
			}
		}
		for k = kk; k < kk+blockSize; k++ {
			for j = en; j < n.Col; j++ {
				rr = n.Get(k, j)
				for i = 0; i < m.Row; i++ {
					r.Set(i, j, r.Get(i, j)+rr*m.Get(i, k))
				}
			}
		}
	}

	for jj = 0; jj < en; jj += blockSize {
		for k = en; k < m.Col; k++ {
			for j = jj; j < jj+blockSize; j++ {
				rr = n.Get(k, j)
				for i = 0; i < m.Row; i++ {
					r.Set(i, j, r.Get(i, j)+rr*m.Get(i, k))
				}
			}
		}
	}

	for k = en; k < m.Col; k++ {
		for j = en; j < n.Col; j++ {
			rr = n.Get(k, j)
			for i = 0; i < m.Row; i++ {
				r.Set(i, j, r.Get(i, j)+rr*m.Get(i, k))
			}
		}
	}

	return r
}
