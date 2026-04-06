//! AVX-512 SIMD kernels for QOR3B CPU inference.

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const Q4_GROUP_SIZE: usize = 32;

pub fn has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    { is_x86_feature_detected!("avx512f") }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_ps(v: __m512) -> f32 {
    let a = _mm512_extractf32x4_ps(v, 0);
    let b = _mm512_extractf32x4_ps(v, 1);
    let c = _mm512_extractf32x4_ps(v, 2);
    let d = _mm512_extractf32x4_ps(v, 3);
    let sum = _mm_add_ps(_mm_add_ps(a, b), _mm_add_ps(c, d));
    let hi = _mm_movehl_ps(sum, sum);
    let sum2 = _mm_add_ps(sum, hi);
    let hi2 = _mm_shuffle_ps(sum2, sum2, 1);
    let sum3 = _mm_add_ss(sum2, hi2);
    _mm_cvtss_f32(sum3)
}

static Q4_FACTORS: [f32; 16] = [
    -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
];
static INTERLEAVE_FIRST: [i32; 16] = [0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23];
static INTERLEAVE_SECOND: [i32; 16] = [8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_q4_avx512(
    input: &[f32], packed: &[u8], scales: &[f16],
    n: usize, k_start: usize, k_end: usize,
) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;
    let mut output = vec![0.0f32; n];
    let factors = _mm512_loadu_ps(Q4_FACTORS.as_ptr());
    let interleave_lo = _mm512_loadu_si512(INTERLEAVE_FIRST.as_ptr() as *const _);
    let interleave_hi = _mm512_loadu_si512(INTERLEAVE_SECOND.as_ptr() as *const _);
    let nibble_mask = _mm512_set1_epi32(0x0F);
    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 { continue; }
        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;
        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * val;
            if s == 0.0 { continue; }
            let s_vec = _mm512_set1_ps(s);
            let lut = _mm512_mul_ps(factors, s_vec);
            let po = pack_base + g * packed_per_group;
            let bytes = _mm_loadu_si128(packed.as_ptr().add(po) as *const __m128i);
            let bytes_i32 = _mm512_cvtepu8_epi32(bytes);
            let lo_nib = _mm512_and_epi32(bytes_i32, nibble_mask);
            let hi_nib = _mm512_srli_epi32(bytes_i32, 4);
            let lo_vals = _mm512_permutexvar_ps(lo_nib, lut);
            let hi_vals = _mm512_permutexvar_ps(hi_nib, lut);
            let first_16 = _mm512_permutex2var_ps(lo_vals, interleave_lo, hi_vals);
            let second_16 = _mm512_permutex2var_ps(lo_vals, interleave_hi, hi_vals);
            let oo = g * Q4_GROUP_SIZE;
            let acc1 = _mm512_loadu_ps(output.as_ptr().add(oo));
            let acc2 = _mm512_loadu_ps(output.as_ptr().add(oo + 16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo), _mm512_add_ps(acc1, first_16));
            _mm512_storeu_ps(output.as_mut_ptr().add(oo + 16), _mm512_add_ps(acc2, second_16));
        }
    }
    output
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn fused_gate_up_q4_avx512(
    input: &[f32], gate_packed: &[u8], gate_scales: &[f16],
    up_packed: &[u8], up_scales: &[f16],
    n: usize, k_start: usize, k_end: usize,
) -> (Vec<f32>, Vec<f32>) {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;
    let mut gate_out = vec![0.0f32; n];
    let mut up_out = vec![0.0f32; n];
    let factors = _mm512_loadu_ps(Q4_FACTORS.as_ptr());
    let interleave_lo = _mm512_loadu_si512(INTERLEAVE_FIRST.as_ptr() as *const _);
    let interleave_hi = _mm512_loadu_si512(INTERLEAVE_SECOND.as_ptr() as *const _);
    let nibble_mask = _mm512_set1_epi32(0x0F);
    for ki in k_start..k_end {
        let val = input[ki];
        if val == 0.0 { continue; }
        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;
        for g in 0..groups_per_row {
            let gs = gate_scales[scale_base + g].to_f32() * val;
            let us = up_scales[scale_base + g].to_f32() * val;
            if gs == 0.0 && us == 0.0 { continue; }
            let po = pack_base + g * packed_per_group;
            let oo = g * Q4_GROUP_SIZE;
            if gs != 0.0 {
                let gs_vec = _mm512_set1_ps(gs);
                let g_lut = _mm512_mul_ps(factors, gs_vec);
                let g_bytes = _mm_loadu_si128(gate_packed.as_ptr().add(po) as *const __m128i);
                let g_i32 = _mm512_cvtepu8_epi32(g_bytes);
                let g_lo = _mm512_and_epi32(g_i32, nibble_mask);
                let g_hi = _mm512_srli_epi32(g_i32, 4);
                let g_lo_vals = _mm512_permutexvar_ps(g_lo, g_lut);
                let g_hi_vals = _mm512_permutexvar_ps(g_hi, g_lut);
                let g_first = _mm512_permutex2var_ps(g_lo_vals, interleave_lo, g_hi_vals);
                let g_second = _mm512_permutex2var_ps(g_lo_vals, interleave_hi, g_hi_vals);
                let ga1 = _mm512_loadu_ps(gate_out.as_ptr().add(oo));
                let ga2 = _mm512_loadu_ps(gate_out.as_ptr().add(oo + 16));
                _mm512_storeu_ps(gate_out.as_mut_ptr().add(oo), _mm512_add_ps(ga1, g_first));
                _mm512_storeu_ps(gate_out.as_mut_ptr().add(oo + 16), _mm512_add_ps(ga2, g_second));
            }
            if us != 0.0 {
                let us_vec = _mm512_set1_ps(us);
                let u_lut = _mm512_mul_ps(factors, us_vec);
                let u_bytes = _mm_loadu_si128(up_packed.as_ptr().add(po) as *const __m128i);
                let u_i32 = _mm512_cvtepu8_epi32(u_bytes);
                let u_lo = _mm512_and_epi32(u_i32, nibble_mask);
                let u_hi = _mm512_srli_epi32(u_i32, 4);
                let u_lo_vals = _mm512_permutexvar_ps(u_lo, u_lut);
                let u_hi_vals = _mm512_permutexvar_ps(u_hi, u_lut);
                let u_first = _mm512_permutex2var_ps(u_lo_vals, interleave_lo, u_hi_vals);
                let u_second = _mm512_permutex2var_ps(u_lo_vals, interleave_hi, u_hi_vals);
                let ua1 = _mm512_loadu_ps(up_out.as_ptr().add(oo));
                let ua2 = _mm512_loadu_ps(up_out.as_ptr().add(oo + 16));
                _mm512_storeu_ps(up_out.as_mut_ptr().add(oo), _mm512_add_ps(ua1, u_first));
                _mm512_storeu_ps(up_out.as_mut_ptr().add(oo + 16), _mm512_add_ps(ua2, u_second));
            }
        }
    }
    (gate_out, up_out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn gemv_f16_avx512(
    input: &[f32], weight_data: &[f16], k: usize, n: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    let n16 = n / 16 * 16;
    for ki in 0..k {
        let val = input[ki];
        if val == 0.0 { continue; }
        let val_vec = _mm512_set1_ps(val);
        let row = ki * n;
        let mut j = 0usize;
        while j < n16 {
            let w_f16 = _mm256_loadu_si256(weight_data.as_ptr().add(row + j) as *const __m256i);
            let w_f32 = _mm512_cvtph_ps(w_f16);
            let acc = _mm512_loadu_ps(output.as_ptr().add(j));
            let result = _mm512_fmadd_ps(val_vec, w_f32, acc);
            _mm512_storeu_ps(output.as_mut_ptr().add(j), result);
            j += 16;
        }
        while j < n {
            output[j] += val * weight_data[row + j].to_f32();
            j += 1;
        }
    }
    output
}
