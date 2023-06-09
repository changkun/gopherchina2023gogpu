// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

package main_test

import (
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"log"
	"os"
	"testing"

	"changkun.de/x/gogpu/enhance"
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

func TestImageEnhance(t *testing.T) {
	f, err := os.Open("testdata/1.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	pp := enhance.Params{
		Brightness:  .6,
		Contrast:    .6,
		Saturation:  .6,
		Temperature: .6,
		Tint:        .1,
	}
	enhancedCPU := enhance.Image(imageToRGBA(img), pp)
	f, err = os.Create("testdata/enhanced_cpu.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := jpeg.Encode(f, enhancedCPU, nil); err != nil {
		log.Fatal(err)
	}

	enhancedGPU := enhance.ImageGPU(imageToRGBA(img), pp)
	f, err = os.Create("testdata/enhanced_gpu.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := jpeg.Encode(f, enhancedGPU, nil); err != nil {
		log.Fatal(err)
	}

	diffSum := 0
	for i := 0; i < len(enhancedCPU.Pix); i++ {
		if enhancedCPU.Pix[i] != enhancedGPU.Pix[i] {
			diffSum++
		}
	}
	if diffSum > 50 {
		t.Fatalf("inconsistent results CPU vs. GPU: %v", diffSum)
	}
	t.Log("diffSum:", diffSum)
}

func BenchmarkImageEnhance(b *testing.B) {
	f, err := os.Open("testdata/1.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	img, err := jpeg.Decode(f)
	if err != nil {
		log.Fatal(err)
	}

	pp := enhance.Params{
		Brightness:  .6,
		Contrast:    .6,
		Saturation:  .6,
		Temperature: .6,
		Tint:        .7,
	}

	b.Run("CPU", func(b *testing.B) {
		m := imageToRGBA(img)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			enhance.Image(m, pp)
		}
	})
	b.Run("GPU", func(b *testing.B) {
		m := imageToRGBA(img)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			enhance.ImageGPU(m, pp)
		}
	})
}

func imageToRGBA(src image.Image) *image.RGBA {
	// No conversion needed if image is an *image.RGBA.
	if dst, ok := src.(*image.RGBA); ok {
		return dst
	}

	// Use the image/draw package to convert to *image.RGBA.
	b := src.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), src, b.Min, draw.Src)
	return dst
}
