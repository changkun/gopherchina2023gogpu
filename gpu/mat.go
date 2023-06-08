// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

package gpu

import (
	_ "embed"
	"errors"
	"sync"
	"unsafe"

	"changkun.de/x/gogpu/gpu/mtl"
	"changkun.de/x/gogpu/math"
)

// Device is a GPU device.
type Device interface {
	Available() bool
}

// Driver returns an avaliable GPU device.
func Driver() Device { return device }

// Mul is a GPU version of math.Mat[T].Mul method and it multiplies
// two matrices m1 and m2 and returns the result.
func Mul[T math.Type](m1, m2 math.Mat[T]) math.Mat[T] {
	if m1.Col != m2.Row {
		panic("math: mismatched matrix dimension")
	}

	// Allocate GPU buffers
	a := device.MakeBuffer(unsafe.Pointer(&m1.Data[0]), uintptr(math.TypeSize[T]()*len(m1.Data)), mtl.ResourceStorageModeShared)
	defer a.Release()
	b := device.MakeBuffer(unsafe.Pointer(&m2.Data[0]), uintptr(math.TypeSize[T]()*len(m2.Data)), mtl.ResourceStorageModeShared)
	defer b.Release()
	out := device.MakeBuffer(nil, uintptr(math.TypeSize[T]()*m1.Row*m2.Col), mtl.ResourceStorageModeShared)
	defer out.Release()
	dp := device.MakeBuffer(unsafe.Pointer(&params[T]{
		ColA: int32(m1.Col),
		ColB: int32(m2.Col),
	}), unsafe.Sizeof(params[T]{}), mtl.ResourceStorageModeShared)
	defer dp.Release()

	// Create command buffer
	cb := cq.MakeCommandBuffer()
	defer cb.Release()

	// Encode, dispatch threads, then commit and wait for completion
	ce := cb.MakeComputeCommandEncoder()
	ce.SetComputePipelineState(fn.funcMul.cps)
	ce.SetBuffer(a, 0, 0)
	ce.SetBuffer(b, 0, 1)
	ce.SetBuffer(out, 0, 2)
	ce.SetBuffer(dp, 0, 3)
	ce.DispatchThreads(
		mtl.Size{Width: m1.Row * m2.Col, Height: 1, Depth: 1},
		mtl.Size{Width: 1, Height: 1, Depth: 1})
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	// Copy data from GPU buffer to CPU buffer
	data := make([]T, m1.Row*m2.Col)
	copy(data, unsafe.Slice((*T)(out.Content()), m1.Row*m2.Col))
	return math.Mat[T]{
		Row:  m1.Row,
		Col:  m2.Col,
		Data: data,
	}
}

var (
	//go:embed mul.metal
	mathMetal string

	once   sync.Once
	fn     gpuFunc
	device mtl.Device
	cq     mtl.CommandQueue
)

type gpuFunc struct {
	lib     mtl.Library
	funcMul struct {
		fn  mtl.Function
		cps mtl.ComputePipelineState
	}
}

func init() {
	defer handle(func(err error) {
		if err != nil {
			panic(err)
		}
	})

	once.Do(func() {
		device = try(mtl.CreateSystemDefaultDevice())
		cq = device.MakeCommandQueue()

		lib := try(device.MakeLibrary(mathMetal, mtl.CompileOptions{
			LanguageVersion: mtl.LanguageVersion2_4,
		}))

		fn = gpuFunc{lib: lib}
		fn.funcMul.fn = try(lib.MakeFunction("mul"))
		fn.funcMul.cps = try(device.MakeComputePipelineState(fn.funcMul.fn))
	})
}

type params[T math.Type] struct {
	ColA int32
	ColB int32
}

func try[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func handle(f func(err error)) {
	if r := recover(); r != nil {
		var err error
		switch x := r.(type) {
		case string:
			err = errors.New(x)
		case error:
			err = x
		default:
			err = errors.New("unknown panic")
		}
		f(err)
	}
}
