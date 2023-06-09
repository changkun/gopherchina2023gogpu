package enhance

import (
	_ "embed"
	"errors"
	"image"
	"sync"
	"unsafe"

	"changkun.de/x/gogpu/gpu/mtl"
	"changkun.de/x/gogpu/math"
)

// ImageGPU is a GPU version of Image.
func ImageGPU(m *image.RGBA, params Params) *image.RGBA {
	pixes := make([]float32, len(m.Pix))
	for i := range m.Pix {
		pixes[i] = float32(m.Pix[i]) / 255
	}

	buf := device.MakeBuffer(unsafe.Pointer(&pixes[0]), uintptr(math.TypeSize[float32]()*len(pixes)), mtl.ResourceStorageModeShared)
	defer buf.Release()
	out := device.MakeBuffer(nil, uintptr(math.TypeSize[float32]()*len(pixes)), mtl.ResourceStorageModeShared)
	defer out.Release()
	pp := device.MakeBuffer(unsafe.Pointer(&params), unsafe.Sizeof(params), mtl.ResourceStorageModeShared)
	defer pp.Release()

	// Create command buffer
	cb := cq.MakeCommandBuffer()
	defer cb.Release()

	// Encode, dispatch threads, then commit and wait for completion
	ce := cb.MakeComputeCommandEncoder()
	ce.SetComputePipelineState(fn.funcProc.cps)
	ce.SetBuffer(buf, 0, 0)
	ce.SetBuffer(out, 0, 1)
	ce.SetBuffer(pp, 0, 2)
	ce.DispatchThreads(
		mtl.Size{Width: len(pixes) / 4, Height: 1, Depth: 1},
		mtl.Size{Width: 1, Height: 1, Depth: 1})
	ce.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()

	proc := make([]float32, len(pixes))
	copy(proc, unsafe.Slice((*float32)(out.Content()), len(pixes)))

	for i := range m.Pix {
		m.Pix[i] = uint8(proc[i] * 255)
	}
	return m
}

var (
	//go:embed image_gpu.metal
	imageMetal string

	onceGPU sync.Once
	fn      gpuFunc
	device  mtl.Device
	cq      mtl.CommandQueue
)

type gpuFunc struct {
	lib      mtl.Library
	funcProc struct {
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

	onceGPU.Do(func() {
		device = try(mtl.CreateSystemDefaultDevice())
		cq = device.MakeCommandQueue()

		lib := try(device.MakeLibrary(imageMetal, mtl.CompileOptions{
			LanguageVersion: mtl.LanguageVersion2_4,
		}))

		fn = gpuFunc{lib: lib}
		fn.funcProc.fn = try(lib.MakeFunction("proc"))
		fn.funcProc.cps = try(device.MakeComputePipelineState(fn.funcProc.fn))
	})
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
