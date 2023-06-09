// Copyright 2023 Changkun Ou <changkun.de>. All rights reserved.
// Use of this source code is governed by a MIT license that
// can be found in the LICENSE file.

#include <metal_common>
#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

struct color {
    float r;
    float g;
    float b;
};

float clamp(float v) {
    if (v < 0.0) {
        return 0.0;
    } else if (v > 1.0) {
        return 1.0;
    }
    return v;
}

float srgb2linear(float v) {
    if (v <= 0.04045) {
        return v / 12.92;
    } else {
        return pow((v + 0.055) / 1.055, 2.4);
    }
}

float linear2srgb(float v) {
    if (v <= 0.0031308) {
        return v * 12.92;
    } else {
        return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
    }
}

color rgb2yuv(color rgb) {
    color yuv;
    yuv.r = +0.21260*rgb.r + 0.71520*rgb.g + 0.07220*rgb.b;
    yuv.g = -0.09991*rgb.r - 0.33609*rgb.g + 0.43600*rgb.b;
    yuv.b = +0.61500*rgb.r - 0.55861*rgb.g - 0.05639*rgb.b;
    return yuv;
}

color yuv2rgb(color yuv) {
    color rgb;
    rgb.r = +1.00000*yuv.r + 0.00000*yuv.g + 1.28033*yuv.b;
    rgb.g = +1.00000*yuv.r - 0.21482*yuv.g - 0.38059*yuv.b;
    rgb.b = +1.00000*yuv.r + 2.12798*yuv.g + 0.00000*yuv.b;
    return rgb;
}

float rgb2h(color c) {
    float r = c.r;
    float g = c.g;
    float b = c.b;
    float M = max(max(r, g), b);
    float m = min(min(r, g), b);

    float h = 0;
    if (M == m) {
        h = 0;
    } else if (m == b) {
        h = 60*(g-r)/(M-m) + 60;
    } else if (m == r) {
        h = 60*(b-g)/(M-m) + 180;
    } else if (m == g) {
        h = 60*(r-b)/(M-m) + 300;
    }

    h /= 360;
    if (h < 0) {
        h += 1;
    } else if (h > 1) {
        h -= 1;
    }
    return h;
}

float rgb2s4hsv(color c) {
    float r = c.r;
    float g = c.g;
    float b = c.b;
    float M = max(max(r, g), b);
    float m = min(min(r, g), b);

    if (M < 1e-14) {
        return 0.0;
    }
    return (M - m) / M;
}

color rgb2hsv(color c) {
    float v = max(max(c.r, c.g), c.b);
    float h = rgb2h(c);
    float s = rgb2s4hsv(c);
    return color{h, s, v};
}

color hsv2rgb(color c) {
    float h = c.r;
    float s = c.g;
    float v = c.b;

    if (s < 1e-14) {
        return color{v, v, v};
    }

    float h6 = h * 6.0;
    int i = int(floor(h6)) % 6;
    float f = h6 - float(i);
    float p = v * (1 - s);
    float q = v * (1 - (s * f));
    float t = v * (1 - (s * (1 - f)));
    float r = 0, g = 0, b = 0;
    switch (i) {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    }
    return color{r, g, b};
}

color apply_temperature_tint(color c, float temperature, float tint) {
    const float scale = 0.10;
    color cc = rgb2yuv(c);
    cc.r = cc.r;
    cc.g = cc.g - temperature*scale + tint*scale;
    cc.b = cc.b + temperature*scale + tint*scale;
    cc = yuv2rgb(cc);
    cc.r = clamp(cc.r);
    cc.g = clamp(cc.g);
    cc.b = clamp(cc.b);
    return cc;
}

color apply_brightness(color c, float brightness) {
    const float scale = 1.5;
    c.r = pow(c.r, 1.0/(1.0+scale*brightness));
    c.g = pow(c.g, 1.0/(1.0+scale*brightness));
    c.b = pow(c.b, 1.0/(1.0+scale*brightness));
    return c;
}

color apply_contrast(color c, float contrast) {
    const float pi4 = 3.14159265358979 * 0.25;
    float t = tan(contrast + 1) * pi4;
    c.r = srgb2linear(max(0.0, (linear2srgb(c.r)-0.5)*t + 0.5));
    c.g = srgb2linear(max(0.0, (linear2srgb(c.g)-0.5)*t + 0.5));
    c.b = srgb2linear(max(0.0, (linear2srgb(c.b)-0.5)*t + 0.5));
    return c;
}

color apply_saturation(color c, float saturation) {
    color hsv = rgb2hsv(c);
    hsv.g = hsv.g * (saturation+1);
    return hsv2rgb(hsv);
}

struct params {
    float brightness;
    float contrast;
    float saturation;
    float temperature;
    float tint;
};

kernel void proc(device const float* img       [[ buffer(0) ]],
                 device       float* out       [[ buffer(1) ]],
                 device const params& params  [[ buffer(2) ]],
                 uint                 index   [[thread_position_in_grid]]) {
    float brightness = clamp(params.brightness) - 0.5;
    float contrast = clamp(params.contrast) - 0.5;
    float saturation = clamp(params.saturation) - 0.5;
    float temperature = clamp(params.temperature) - 0.5;
    float tint = clamp(params.tint) - 0.5;

    float r = srgb2linear(img[index * 4 + 0]);
    float g = srgb2linear(img[index * 4 + 1]);
    float b = srgb2linear(img[index * 4 + 2]);
    color c = color{r, g, b};

    c = apply_temperature_tint(c, temperature, tint);
    c = apply_brightness(c, brightness);
    c = apply_contrast(c, contrast);
    c = apply_saturation(c, saturation);

    out[index * 4 + 0] = clamp(linear2srgb(c.r));
    out[index * 4 + 1] = clamp(linear2srgb(c.g));
    out[index * 4 + 2] = clamp(linear2srgb(c.b));
    out[index * 4 + 3] = img[index * 4 + 3];
}