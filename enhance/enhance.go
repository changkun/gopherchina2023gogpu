// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

package enhance

import (
	"image/color"
	"math"
)

// Color RGB from 0 to 1.
type Color struct {
	R, G, B float32
}

func NewColor(c color.RGBA) Color {
	return Color{
		R: float32(c.R) / 255.0,
		G: float32(c.G) / 255.0,
		B: float32(c.B) / 255.0,
	}
}

func (c Color) ToRGBA() color.RGBA {
	return color.RGBA{
		R: uint8(c.R * 255),
		G: uint8(c.G * 255),
		B: uint8(c.B * 255),
		A: 255,
	}
}

// Pixel changes the color of a pixel based on the given parameters.
func Pixel(p Color, params Params) Color {
	brightness := clamp(params.Brightness) - 0.5
	contrast := clamp(params.Contrast) - 0.5
	saturation := clamp(params.Saturation) - 0.5
	temperature := clamp(params.Temperature) - 0.5
	tint := clamp(params.Tint) - 0.5

	c := Color{
		R: fromsRGB2Linear(p.R),
		G: fromsRGB2Linear(p.G),
		B: fromsRGB2Linear(p.B),
	}
	c = applyTemperatureTintEffect(c, temperature, tint)
	c = applyBrightnessEffect(c, brightness)
	c = applyContrastEffect(c, contrast)
	c = applySaturationEffect(c, saturation)
	return Color{
		R: clamp(fromLinear2sRGB(c.R)),
		G: clamp(fromLinear2sRGB(c.G)),
		B: clamp(fromLinear2sRGB(c.B)),
	}
}

func clamp(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Y'UV (BT.709) to linear RGB
// Values are from https://en.wikipedia.org/wiki/YUV
func yuv2rgb(c Color) Color {
	return Color{
		R: +1.00000*c.R + 0.00000*c.G + 1.28033*c.B,
		G: +1.00000*c.R - 0.21482*c.G - 0.38059*c.B,
		B: +1.00000*c.R + 2.12798*c.G + 0.00000*c.B,
	}
}

// Linear RGB to Y'UV (BT.709)
// Values are from https://en.wikipedia.org/wiki/YUV
func rgb2yuv(c Color) Color {
	return Color{
		R: +0.21260*c.R + 0.71520*c.G + 0.07220*c.B,
		G: -0.09991*c.R - 0.33609*c.G + 0.43600*c.B,
		B: +0.61500*c.R - 0.55861*c.G - 0.05639*c.B,
	}
}

func applyTemperatureTintEffect(c Color, temperature, tint float32) Color {
	const scale = 0.10
	cc := rgb2yuv(c)
	cc = Color{
		R: (cc.R),
		G: (cc.G - temperature*scale + tint*scale),
		B: (cc.B + temperature*scale + tint*scale),
	}
	cc = yuv2rgb(cc)
	return Color{
		R: clamp(cc.R),
		G: clamp(cc.G),
		B: clamp(cc.B),
	}
}

func applyBrightnessEffect(c Color, brightness float32) Color {
	const scale = 1.5
	return Color{
		R: float32(math.Pow(float64(c.R), float64(1.0/(1.0+scale*brightness)))),
		G: float32(math.Pow(float64(c.G), float64(1.0/(1.0+scale*brightness)))),
		B: float32(math.Pow(float64(c.B), float64(1.0/(1.0+scale*brightness)))),
	}
}

func applyContrastEffect(c Color, contrast float32) Color {
	const pi4 = 3.14159265358979 * 0.25
	contrastCoef := float32(math.Tan(float64(contrast)+1) * pi4)
	return Color{
		R: fromsRGB2Linear(float32(
			math.Max(0, float64((fromLinear2sRGB(c.R)-0.5)*contrastCoef+0.5)))),
		G: fromsRGB2Linear(float32(
			math.Max(0, float64((fromLinear2sRGB(c.G)-0.5)*contrastCoef+0.5)))),
		B: fromsRGB2Linear(float32(
			math.Max(0, float64((fromLinear2sRGB(c.B)-0.5)*contrastCoef+0.5)))),
	}
}

func rgb2h(c Color) float32 {
	r := float64(c.R)
	g := float64(c.G)
	b := float64(c.B)
	M := float32(math.Max(math.Max(r, g), b))
	m := float32(math.Min(math.Min(r, g), b))

	h := float32(0.0)
	if M == m {
		h = 0.0
	} else if m == c.B {
		h = 60.0*(c.G-c.R)/(M-m) + 60.0
	} else if m == c.R {
		h = 60.0*(c.B-c.G)/(M-m) + 180.0
	} else if m == c.G {
		h = 60.0*(c.R-c.B)/(M-m) + 300.0
	}

	h /= 360.0
	if h < 0.0 {
		h += 1
	} else if h > 1.0 {
		h -= 1
	}
	return h
}

func rgb2s4hsv(c Color) float32 {
	r := float64(c.R)
	g := float64(c.G)
	b := float64(c.B)
	M := float32(math.Max(math.Max(r, g), b))
	m := float32(math.Min(math.Min(r, g), b))

	if M < 1e-14 {
		return 0.0
	}
	return (M - m) / M
}
func rgb2hsv(c Color) Color {
	r := float64(c.R)
	g := float64(c.G)
	b := float64(c.B)

	v := float32(math.Max(math.Max(r, g), b))
	h := rgb2h(c)
	s := rgb2s4hsv(c)
	return Color{R: h, G: s, B: v}
}
func hsv2rgb(c Color) Color {
	h := c.R
	s := c.G
	v := c.B

	if s < 1e-14 {
		return Color{R: v, G: v, B: v}
	}

	h6 := h * 6.0
	i := int(math.Floor(float64(h6))) % 6
	f := h6 - float32(i)
	p := v * (1 - s)
	q := v * (1 - (s * f))
	t := v * (1 - (s * (1 - f)))

	r, g, b := float32(0.), float32(0.), float32(0.)
	switch i {
	case 0:
		r = v
		g = t
		b = p
	case 1:
		r = q
		g = v
		b = p
	case 2:
		r = p
		g = v
		b = t
	case 3:
		r = p
		g = q
		b = v
	case 4:
		r = t
		g = p
		b = v
	case 5:
		r = v
		g = p
		b = q
	}
	return Color{R: r, G: g, B: b}
}

func applySaturationEffect(c Color, saturation float32) Color {
	hsv := rgb2hsv(c)
	hsv.G = hsv.G * (saturation + 1)
	return hsv2rgb(hsv)
}
