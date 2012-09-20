//  NEON path
//

#include "m_pd.h"
#include <arm_neon.h>

t_int *plus_perf8_neon(t_int *w)
{
    t_sample *in1 = (t_sample *)(w[1]);
    t_sample *in2 = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in1 += 8, in2 += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in1[0]);
        float32x4_t g0 = vld1q_f32((float32_t*)&in2[0]);
        float32x4_t r0 = vaddq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in1[4]);
        float32x4_t g4 = vld1q_f32((float32_t*)&in2[4]);
        float32x4_t r4 = vaddq_f32(f4, g4);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r4);
    }
    return (w+5);
}

t_int *scalarplus_perf8_neon(t_int *w)
{
    t_sample *in = (t_sample *)(w[1]);
    t_float g = *(t_float *)(w[2]);
    float32x4_t g0 = { g, g, g, g };
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in[0]);
        float32x4_t r0 = vaddq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in[4]);
        float32x4_t r1 = vaddq_f32(f4, g0);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r1);
    }
    return (w+5);
}

t_int *minus_perf8_neon(t_int *w)
{
    t_sample *in1 = (t_sample *)(w[1]);
    t_sample *in2 = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in1 += 8, in2 += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in1[0]);
        float32x4_t g0 = vld1q_f32((float32_t*)&in2[0]);
        float32x4_t r0 = vsubq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in1[4]);
        float32x4_t g4 = vld1q_f32((float32_t*)&in2[4]);
        float32x4_t r1 = vsubq_f32(f4, g4);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r1);
    }
    return (w+5);
}

t_int *scalarminus_perf8_neon(t_int *w)
{
    t_sample *in = (t_sample *)(w[1]);
    t_float g = *(t_float *)(w[2]);
    float32x4_t g0 = { g, g, g, g };
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in[0]);
        float32x4_t r0 = vsubq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in[4]);
        float32x4_t r1 = vsubq_f32(f4, g0);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r1);
    }
    return (w+5);
}

t_int *times_perf8_neon(t_int *w)
{
    t_sample *in1 = (t_sample *)(w[1]);
    t_sample *in2 = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in1 += 8, in2 += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in1[0]);
        float32x4_t g0 = vld1q_f32((float32_t*)&in2[0]);
        float32x4_t r0 = vmulq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in1[4]);
        float32x4_t g4 = vld1q_f32((float32_t*)&in2[4]);
        float32x4_t r1 = vmulq_f32(f4, g4);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r1);
    }
    return (w+5);
}

t_int *scalartimes_perf8_neon(t_int *w)
{
    t_sample *in = (t_sample *)(w[1]);
    t_float g = *(t_float *)(w[2]);
    float32x4_t g0 = { g, g, g, g };
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in[0]);
        float32x4_t r0 = vmulq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in[4]);
        float32x4_t r1 = vmulq_f32(f4, g0);
        vst1q_f32((float32_t*)&out[0], r0);
        vst1q_f32((float32_t*)&out[4], r1);
    }
    return (w+5);
}

t_int *max_perf8_neon(t_int *w)
{
    t_sample *in1 = (t_sample *)(w[1]);
    t_sample *in2 = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in1 += 8, in2 += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in1[0]);
        float32x4_t g0 = vld1q_f32((float32_t*)&in2[0]);
        float32x4_t r0 = vmaxq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in1[4]);
        float32x4_t g4 = vld1q_f32((float32_t*)&in2[4]);
        float32x4_t r4 = vmaxq_f32(f4, g4);
        vst1q_f32((float32_t*)&out[0], r0);        
        vst1q_f32((float32_t*)&out[4], r4);        
    }
    return (w+5);
}

t_int *scalarmax_perf8_neon(t_int *w)
{
    t_sample *in = (t_sample *)(w[1]);
    t_float g = *(t_float *)(w[2]);
    float32x4_t g0 = { g, g, g, g };
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in[0]);
        float32x4_t r0 = vmaxq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in[4]);
        float32x4_t r4 = vmaxq_f32(f4, g0);
        vst1q_f32((float32_t*)&out[0], r0);        
        vst1q_f32((float32_t*)&out[4], r4);        
    }
    return (w+5);
}

t_int *min_perf8_neon(t_int *w)
{
    t_sample *in1 = (t_sample *)(w[1]);
    t_sample *in2 = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in1 += 8, in2 += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in1[0]);
        float32x4_t g0 = vld1q_f32((float32_t*)&in2[0]);
        float32x4_t r0 = vminq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in1[4]);
        float32x4_t g4 = vld1q_f32((float32_t*)&in2[4]);
        float32x4_t r4 = vminq_f32(f4, g4);
        vst1q_f32((float32_t*)&out[0], r0);        
        vst1q_f32((float32_t*)&out[4], r4);
    }
    return (w+5);
}

t_int *scalarmin_perf8_neon(t_int *w)
{
    t_sample *in = (t_sample *)(w[1]);
    t_float g = *(t_float *)(w[2]);
    float32x4_t g0 = { g, g, g, g };
    t_float *out = (t_float *)(w[3]);
    int n = (int)(w[4]);
    for (; n; n -= 8, in += 8, out += 8)
    {
        float32x4_t f0 = vld1q_f32((float32_t*)&in[0]);
        float32x4_t r0 = vmaxq_f32(f0, g0);
        float32x4_t f4 = vld1q_f32((float32_t*)&in[4]);
        float32x4_t r4 = vmaxq_f32(f4, g0);
        vst1q_f32((float32_t*)&out[0], r0);        
        vst1q_f32((float32_t*)&out[4], r4);
    }
    return (w+5);
}

