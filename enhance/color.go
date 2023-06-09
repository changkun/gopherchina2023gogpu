package enhance

import (
	"math"
	"sync"
)

// fromLinear2sRGB converts a given value from linear space to
// sRGB space.
func fromLinear2sRGB(v float32) float32 {
	if !useLut {
		return linear2sRGB(v)
	}
	if v <= 0 {
		return 0
	}
	if v == 1 {
		return 1
	}
	i := v * lutSize
	ifloor := int(i) & (lutSize - 1)
	v0 := float32(lin2sRGBLUfloat32[ifloor])
	v1 := float32(lin2sRGBLUfloat32[ifloor+1])
	i -= float32(ifloor)
	return v0*(1.0-i) + v1*i
}

// fromsRGB2Linear converts a given value from linear space to
// sRGB space.
func fromsRGB2Linear(v float32) float32 {
	if !useLut {
		return sRGB2linear(v)
	}
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 1
	}

	i := v * lutSize
	ifloor := int(i) & (lutSize - 1)
	v0 := float32(sRGB2linLUfloat32[ifloor])
	v1 := float32(sRGB2linLUfloat32[ifloor+1])
	i -= float32(ifloor)
	return v0*(1.0-i) + v1*i
}

const (
	lutSize = 1024 // keep a power of 2
)

var (
	once              sync.Once
	useLut            = false
	lin2sRGBLUfloat32 [lutSize + 1]float32
	sRGB2linLUfloat32 [lutSize + 1]float32
)

func init() {
	// Initialize look up table.
	once.Do(func() {
		for i := 0; i < lutSize; i++ {
			lin2sRGBLUfloat32[i] = linear2sRGB(float32(i) / lutSize)
			sRGB2linLUfloat32[i] = sRGB2linear(float32(i) / lutSize)
		}
		lin2sRGBLUfloat32[lutSize] = lin2sRGBLUfloat32[lutSize-1]
		sRGB2linLUfloat32[lutSize] = sRGB2linLUfloat32[lutSize-1]
	})
}

func sRGB2linear(v float32) float32 {
	if v <= 0.04045 {
		v /= 12.92
	} else {
		v = float32(math.Pow((float64(v)+0.055)/1.055, 2.4))
	}
	return v
}

func linear2sRGB(v float32) float32 {
	if v <= 0.0031308 {
		v *= 12.92
	} else {
		v = float32(1.055*math.Pow(float64(v), 1.0/2.4) - 0.055)
	}
	return v
}
