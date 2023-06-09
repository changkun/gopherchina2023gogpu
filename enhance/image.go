// Copyright 2022 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

// Package enhance provides image enhancement algorithms for adjusting brightness, contrast, saturation, temperature, and tint.
package enhance

import (
	"image"
)

// Params defines the parameters for image enhancement.
// The values should in range [0, 1].
type Params struct {
	Brightness  float32
	Contrast    float32
	Saturation  float32
	Temperature float32
	Tint        float32
}

// Image enhances a given image.Image and returns a new image.RGBA.
//
// This method reproduces https://github.com/yuki-koyama/enhancer.
func Image(m *image.RGBA, params Params) *image.RGBA {
	b := m.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := NewColor(m.RGBAAt(x, y))
			p := Pixel(c, params)
			m.Set(x, y, p.ToRGBA())
		}
	}
	return m
}
