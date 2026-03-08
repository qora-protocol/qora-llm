//! Fast GEMV (General Matrix-Vector multiply) for single-token decode.
//!
//! Supports two weight formats:
//! - **F16**: Half precision (~6GB for 3B model). Better quality.
//! - **Q4**: 4-bit symmetric quantization (~1.7GB). Faster, lower memory.
//!
//! Q4 uses per-group (32 values) symmetric quantization:
//!   scale = absmax / 7, q = round(val/scale) + 8, packed 2 per byte.
//!   Dequant: val = (q - 8) * scale
//!
//! KV cache and activations remain f32 for accuracy.

use half::f16;
use rayon::prelude::*;

// ============================================================
// Weight format types
// ============================================================

const Q4_GROUP_SIZE: usize = 32;

/// A weight matrix stored as f16 with shape metadata.
struct F16Weight {
    data: Vec<f16>,
    k: usize, // rows (input dim)
    n: usize, // cols (output dim)
}

/// A weight matrix stored in 4-bit symmetric quantization (Q4_0 style).
/// Group size: 32 values share one f16 scale factor.
/// Packing: consecutive pairs — byte = q[2i] | (q[2i+1] << 4), where q in [0,15].
/// Dequant: value = (q - 8) * scale
struct Q4Weight {
    packed: Vec<u8>,   // k * groups_per_row * 16 bytes
    scales: Vec<f16>,  // k * groups_per_row scales
    k: usize,          // rows (input dim)
    n: usize,          // cols (output dim)
}

/// Polymorphic weight — either f16 or Q4.
enum Weight {
    F16(F16Weight),
    Q4(Q4Weight),
}

impl Weight {
    fn n(&self) -> usize {
        match self {
            Weight::F16(w) => w.n,
            Weight::Q4(w) => w.n,
        }
    }
    fn memory_bytes(&self) -> usize {
        match self {
            Weight::F16(w) => w.data.len() * 2,
            Weight::Q4(w) => w.packed.len() + w.scales.len() * 2,
        }
    }
}

// ============================================================
// Public data types for serialization (used by save.rs)
// ============================================================

/// Reference to raw weight data for GPU loading.
#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
pub enum WeightRef<'a> {
    F16 { data: &'a [f16], k: usize, n: usize },
    Q4 { packed: &'a [u8], scales: &'a [f16], k: usize, n: usize },
}

/// Per-layer weight references for GPU loading.
#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
pub struct GpuLayerRef<'a> {
    pub q_proj: WeightRef<'a>,
    pub k_proj: WeightRef<'a>,
    pub v_proj: WeightRef<'a>,
    pub o_proj: WeightRef<'a>,
    pub gate_proj: WeightRef<'a>,
    pub up_proj: WeightRef<'a>,
    pub down_proj: WeightRef<'a>,
    pub input_norm_gamma: &'a [f16],
    pub post_attn_norm_gamma: &'a [f16],
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
}

#[cfg(any(feature = "gpu", feature = "gpu-metal"))]
fn weight_ref(w: &Weight) -> WeightRef<'_> {
    match w {
        Weight::F16(w) => WeightRef::F16 { data: &w.data, k: w.k, n: w.n },
        Weight::Q4(w) => WeightRef::Q4 { packed: &w.packed, scales: &w.scales, k: w.k, n: w.n },
    }
}

/// Serializable weight data — public mirror of the internal Weight enum.
pub enum WeightData {
    F16 { data: Vec<f16>, k: usize, n: usize },
    Q4 { packed: Vec<u8>, scales: Vec<f16>, k: usize, n: usize },
}

/// Serializable per-layer weight data.
pub struct LayerWeightData {
    pub q_proj: WeightData,
    pub k_proj: WeightData,
    pub v_proj: WeightData,
    pub o_proj: WeightData,
    pub gate_proj: WeightData,
    pub up_proj: WeightData,
    pub down_proj: WeightData,
    pub input_norm_gamma: Vec<f16>,
    pub post_attn_norm_gamma: Vec<f16>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
}

impl From<WeightData> for Weight {
    fn from(wd: WeightData) -> Self {
        match wd {
            WeightData::F16 { data, k, n } => Weight::F16(F16Weight { data, k, n }),
            WeightData::Q4 { packed, scales, k, n } => Weight::Q4(Q4Weight { packed, scales, k, n }),
        }
    }
}

impl From<LayerWeightData> for LayerWeights {
    fn from(ld: LayerWeightData) -> Self {
        LayerWeights {
            q_proj: ld.q_proj.into(),
            k_proj: ld.k_proj.into(),
            v_proj: ld.v_proj.into(),
            o_proj: ld.o_proj.into(),
            gate_proj: ld.gate_proj.into(),
            up_proj: ld.up_proj.into(),
            down_proj: ld.down_proj.into(),
            input_norm_gamma: ld.input_norm_gamma,
            post_attn_norm_gamma: ld.post_attn_norm_gamma,
            num_heads: ld.num_heads,
            num_kv_heads: ld.num_kv_heads,
            head_dim: ld.head_dim,
            num_kv_groups: ld.num_kv_groups,
        }
    }
}

// ============================================================
// KV cache types
// ============================================================

/// Raw KV cache: per-layer (k_data, v_data, seq_len).
/// Layout: [seq_len, kv_heads, head_dim] token-major, f32.
pub type RawKvCache = Vec<(Vec<f32>, Vec<f32>, usize)>;

/// Create an empty raw KV cache for all layers.
pub fn empty_kv_cache(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> RawKvCache {
    (0..num_layers)
        .map(|_| {
            let cap = num_kv_heads * 512 * head_dim;
            (Vec::with_capacity(cap), Vec::with_capacity(cap), 0usize)
        })
        .collect()
}

// ============================================================
// Quantization
// ============================================================

// ============================================================
// Per-layer and model weight structures
// ============================================================

/// Per-layer weights (f16 or Q4 for projections, f16 for norms).
struct LayerWeights {
    q_proj: Weight,
    k_proj: Weight,
    v_proj: Weight,
    o_proj: Weight,
    gate_proj: Weight,
    up_proj: Weight,
    down_proj: Weight,
    input_norm_gamma: Vec<f16>,
    post_attn_norm_gamma: Vec<f16>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
}

/// All model weights for inference.
/// Supports f16 or Q4 weight formats. Norms are always f16.
pub struct DecodeWeights {
    layers: Vec<LayerWeights>,
    embed: Weight,           // [vocab, hidden]
    embed_vocab: usize,
    embed_hidden: usize,
    final_norm_gamma: Vec<f16>,
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    rope_half_dim: usize,
    no_rope_layers: Vec<bool>,
    format_name: &'static str,
}

impl DecodeWeights {
    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for l in &self.layers {
            total += l.q_proj.memory_bytes();
            total += l.k_proj.memory_bytes();
            total += l.v_proj.memory_bytes();
            total += l.o_proj.memory_bytes();
            total += l.gate_proj.memory_bytes();
            total += l.up_proj.memory_bytes();
            total += l.down_proj.memory_bytes();
            total += (l.input_norm_gamma.len() + l.post_attn_norm_gamma.len()) * 2;
        }
        total += self.embed.memory_bytes();
        total += self.final_norm_gamma.len() * 2;
        total += (self.rope_cos.len() + self.rope_sin.len()) * 4;
        total
    }

    pub fn format_name(&self) -> &str {
        self.format_name
    }

    // --- Accessors for save.rs ---

    pub fn num_layers(&self) -> usize { self.layers.len() }
    pub fn vocab(&self) -> usize { self.embed_vocab }
    pub fn hidden(&self) -> usize { self.embed_hidden }
    pub fn rope_half_dim(&self) -> usize { self.rope_half_dim }
    pub fn no_rope_layers_ref(&self) -> &[bool] { &self.no_rope_layers }
    pub fn final_norm_ref(&self) -> &[f16] { &self.final_norm_gamma }
    pub fn rope_cos_ref(&self) -> &[f32] { &self.rope_cos }
    pub fn rope_sin_ref(&self) -> &[f32] { &self.rope_sin }

    // --- Accessors for gpu_loader.rs ---

    /// Get per-layer weight references for GPU loading.
    /// Returns (q, k, v, o, gate, up, down, input_norm, post_attn_norm, num_heads, num_kv_heads, head_dim, num_kv_groups)
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    pub fn layer_ref(&self, i: usize) -> GpuLayerRef<'_> {
        let lw = &self.layers[i];
        GpuLayerRef {
            q_proj: weight_ref(&lw.q_proj),
            k_proj: weight_ref(&lw.k_proj),
            v_proj: weight_ref(&lw.v_proj),
            o_proj: weight_ref(&lw.o_proj),
            gate_proj: weight_ref(&lw.gate_proj),
            up_proj: weight_ref(&lw.up_proj),
            down_proj: weight_ref(&lw.down_proj),
            input_norm_gamma: &lw.input_norm_gamma,
            post_attn_norm_gamma: &lw.post_attn_norm_gamma,
            num_heads: lw.num_heads,
            num_kv_heads: lw.num_kv_heads,
            head_dim: lw.head_dim,
            num_kv_groups: lw.num_kv_groups,
        }
    }

    /// Get embedding weight reference for GPU loading.
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    pub fn embed_ref(&self) -> WeightRef<'_> {
        weight_ref(&self.embed)
    }

    /// Format id: 0 = F16, 1 = Q4.
    pub fn format_id(&self) -> u8 {
        match &self.embed {
            Weight::F16(_) => 0,
            Weight::Q4(_) => 1,
        }
    }

    /// Write all weights for layer i.
    pub fn write_layer(&self, w: &mut impl std::io::Write, i: usize) -> std::io::Result<()> {
        let lw = &self.layers[i];
        write_weight_io(w, &lw.q_proj)?;
        write_weight_io(w, &lw.k_proj)?;
        write_weight_io(w, &lw.v_proj)?;
        write_weight_io(w, &lw.o_proj)?;
        write_weight_io(w, &lw.gate_proj)?;
        write_weight_io(w, &lw.up_proj)?;
        write_weight_io(w, &lw.down_proj)?;
        write_f16_vec_io(w, &lw.input_norm_gamma)?;
        write_f16_vec_io(w, &lw.post_attn_norm_gamma)?;
        write_u32_io(w, lw.num_heads as u32)?;
        write_u32_io(w, lw.num_kv_heads as u32)?;
        write_u32_io(w, lw.head_dim as u32)?;
        write_u32_io(w, lw.num_kv_groups as u32)?;
        Ok(())
    }

    /// Write the embedding weight.
    pub fn write_embed(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        write_weight_io(w, &self.embed)
    }

    /// Construct DecodeWeights from deserialized parts.
    pub fn from_parts(
        layer_data: Vec<LayerWeightData>,
        embed_data: WeightData,
        vocab: usize,
        hidden: usize,
        final_norm_gamma: Vec<f16>,
        rope_cos: Vec<f32>,
        rope_sin: Vec<f32>,
        rope_half_dim: usize,
        no_rope_layers: Vec<bool>,
        format_id: u8,
    ) -> Self {
        let layers: Vec<LayerWeights> = layer_data.into_iter().map(|ld| ld.into()).collect();
        let embed: Weight = embed_data.into();
        let format_name = match format_id {
            0 => "F16",
            1 => "Q4",
            _ => "unknown",
        };
        Self {
            layers,
            embed,
            embed_vocab: vocab,
            embed_hidden: hidden,
            final_norm_gamma,
            rope_cos,
            rope_sin,
            rope_half_dim,
            no_rope_layers,
            format_name,
        }
    }
}

// I/O helpers used by DecodeWeights methods
fn write_weight_io(w: &mut impl std::io::Write, weight: &Weight) -> std::io::Result<()> {
    match weight {
        Weight::F16(fw) => {
            write_u64_io(w, fw.k as u64)?;
            write_u64_io(w, fw.n as u64)?;
            write_f16_vec_io(w, &fw.data)?;
        }
        Weight::Q4(qw) => {
            write_u64_io(w, qw.k as u64)?;
            write_u64_io(w, qw.n as u64)?;
            write_bytes_io(w, &qw.packed)?;
            write_f16_vec_io(w, &qw.scales)?;
        }
    }
    Ok(())
}

fn write_u32_io(w: &mut impl std::io::Write, val: u32) -> std::io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_u64_io(w: &mut impl std::io::Write, val: u64) -> std::io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_bytes_io(w: &mut impl std::io::Write, data: &[u8]) -> std::io::Result<()> {
    write_u64_io(w, data.len() as u64)?;
    w.write_all(data)
}
fn write_f16_vec_io(w: &mut impl std::io::Write, data: &[f16]) -> std::io::Result<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2)
    };
    write_u64_io(w, data.len() as u64)?;
    w.write_all(bytes)
}

// ============================================================
// GEMV dispatch (single-token decode)
// ============================================================

/// GEMV: output = input @ weight, dispatches to f16 or Q4 kernel.
#[inline]
fn gemv(input: &[f32], weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemv_f16(input, w),
        Weight::Q4(w) => gemv_q4(input, w),
    }
}

/// Fused gate+up GEMV: computes silu(gate(x)) * up(x) in one pass.
/// Reads input once instead of twice, halving memory traffic for MLP.
#[inline]
fn fused_gate_up_gemv(input: &[f32], gate_w: &Weight, up_w: &Weight) -> Vec<f32> {
    match (gate_w, up_w) {
        (Weight::Q4(gw), Weight::Q4(uw)) => fused_gate_up_q4(input, gw, uw),
        _ => {
            // Fallback: separate gate+up for f16
            let gate = gemv(input, gate_w);
            let up = gemv(input, up_w);
            let n = gate.len();
            let mut out = vec![0.0f32; n];
            for j in 0..n {
                let g = gate[j];
                let silu_g = g / (1.0 + (-g).exp());
                out[j] = silu_g * up[j];
            }
            out
        }
    }
}

/// Fused gate+up Q4 GEMV: single input read, dual accumulation.
fn fused_gate_up_q4(input: &[f32], gate_w: &Q4Weight, up_w: &Q4Weight) -> Vec<f32> {
    let k = gate_w.k;
    let n = gate_w.n;
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;

    // Parallel across output columns (chunks of n)
    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;

    let partials: Vec<(Vec<f32>, Vec<f32>)> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k { return None; }

            let mut gate_out = vec![0.0f32; n];
            let mut up_out = vec![0.0f32; n];

            for ki in k_start..k_end {
                let input_val = input[ki];
                if input_val == 0.0 { continue; }

                let scale_base = ki * groups_per_row;
                let pack_base = ki * groups_per_row * packed_per_group;

                for g in 0..groups_per_row {
                    let gs = gate_w.scales[scale_base + g].to_f32() * input_val;
                    let us = up_w.scales[scale_base + g].to_f32() * input_val;
                    if gs == 0.0 && us == 0.0 { continue; }

                    let g_lut = [
                        gs * -8.0, gs * -7.0, gs * -6.0, gs * -5.0,
                        gs * -4.0, gs * -3.0, gs * -2.0, gs * -1.0,
                        0.0,       gs,        gs * 2.0,  gs * 3.0,
                        gs * 4.0,  gs * 5.0,  gs * 6.0,  gs * 7.0,
                    ];
                    let u_lut = [
                        us * -8.0, us * -7.0, us * -6.0, us * -5.0,
                        us * -4.0, us * -3.0, us * -2.0, us * -1.0,
                        0.0,       us,        us * 2.0,  us * 3.0,
                        us * 4.0,  us * 5.0,  us * 6.0,  us * 7.0,
                    ];

                    let pack_offset = pack_base + g * packed_per_group;
                    let out_offset = g * Q4_GROUP_SIZE;

                    for j in 0..packed_per_group {
                        let gb = gate_w.packed[pack_offset + j];
                        let ub = up_w.packed[pack_offset + j];
                        gate_out[out_offset + j * 2] += g_lut[(gb & 0x0F) as usize];
                        gate_out[out_offset + j * 2 + 1] += g_lut[(gb >> 4) as usize];
                        up_out[out_offset + j * 2] += u_lut[(ub & 0x0F) as usize];
                        up_out[out_offset + j * 2 + 1] += u_lut[(ub >> 4) as usize];
                    }
                }
            }
            Some((gate_out, up_out))
        })
        .collect();

    // Sum partials and apply SiLU fusion
    let mut gate_final = vec![0.0f32; n];
    let mut up_final = vec![0.0f32; n];
    for (gp, up) in &partials {
        for j in 0..n {
            gate_final[j] += gp[j];
            up_final[j] += up[j];
        }
    }

    // silu(gate) * up
    let mut out = vec![0.0f32; n];
    for j in 0..n {
        let g = gate_final[j];
        let silu_g = g / (1.0 + (-g).exp());
        out[j] = silu_g * up_final[j];
    }
    out
}

/// GEMM: [seq_len, k] @ [k, n] -> [seq_len, n], dispatches to f16 or Q4.
#[inline]
fn gemm(x: &[f32], seq_len: usize, weight: &Weight) -> Vec<f32> {
    match weight {
        Weight::F16(w) => gemm_f16(x, seq_len, w),
        Weight::Q4(w) => gemm_q4(x, seq_len, w),
    }
}

/// Embedding lookup: dequantize one row from f16 or Q4 weight.
#[inline]
fn embed_lookup(weight: &Weight, token_id: usize, hidden: usize) -> Vec<f32> {
    match weight {
        Weight::F16(w) => {
            let row_start = token_id * hidden;
            w.data[row_start..row_start + hidden]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        }
        Weight::Q4(w) => embed_lookup_q4(w, token_id),
    }
}

// ============================================================
// F16 compute kernels
// ============================================================

/// GEMV with f16 weights: outer-product accumulation.
#[inline]
fn gemv_f16(input: &[f32], weight: &F16Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;
    let w = &weight.data;
    let mut output = vec![0.0f32; n];
    for ki in 0..k {
        let input_val = input[ki];
        let row_start = ki * n;
        for j in 0..n {
            output[j] += input_val * w[row_start + j].to_f32();
        }
    }
    output
}

/// GEMM with f16 weights: [seq_len, k] @ [k, n] -> [seq_len, n].
#[inline]
fn gemm_f16(x: &[f32], seq_len: usize, weight: &F16Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;
    let w = &weight.data;
    let mut output = vec![0.0f32; seq_len * n];
    for t in 0..seq_len {
        let x_row = &x[t * k..(t + 1) * k];
        let out_row = &mut output[t * n..(t + 1) * n];
        for ki in 0..k {
            let input_val = x_row[ki];
            let w_start = ki * n;
            for j in 0..n {
                out_row[j] += input_val * w[w_start + j].to_f32();
            }
        }
    }
    output
}

// ============================================================
// Q4 compute kernels
// ============================================================

/// Single-threaded Q4 GEMV with LUT optimization.
#[inline]
fn gemv_q4_inner(input: &[f32], packed: &[u8], scales: &[f16],
                  _k: usize, n: usize, k_start: usize, k_end: usize) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    for ki in k_start..k_end {
        let input_val = input[ki];
        if input_val == 0.0 { continue; }

        let scale_base = ki * groups_per_row;
        let pack_base = ki * packed_per_row;

        for g in 0..groups_per_row {
            let s = scales[scale_base + g].to_f32() * input_val;
            if s == 0.0 { continue; }

            let lut = [
                s * -8.0, s * -7.0, s * -6.0, s * -5.0,
                s * -4.0, s * -3.0, s * -2.0, s * -1.0,
                0.0,      s,        s * 2.0,  s * 3.0,
                s * 4.0,  s * 5.0,  s * 6.0,  s * 7.0,
            ];

            let pack_offset = pack_base + g * packed_per_group;
            let out_offset = g * Q4_GROUP_SIZE;

            for j in 0..packed_per_group {
                let byte = packed[pack_offset + j];
                output[out_offset + j * 2] += lut[(byte & 0x0F) as usize];
                output[out_offset + j * 2 + 1] += lut[(byte >> 4) as usize];
            }
        }
    }

    output
}

/// Q4 GEMV — multi-threaded via rayon for large matrices, single-threaded for small.
/// Splits by k (input rows), each thread produces a partial output, then sums.
#[inline]
fn gemv_q4(input: &[f32], weight: &Q4Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    // Only parallelize large matrices (gate/up/down projections at n=11008)
    if k * n < 4_000_000 {
        return gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, 0, k);
    }

    let num_threads = rayon::current_num_threads();
    let chunk_k = (k + num_threads - 1) / num_threads;

    let partials: Vec<Vec<f32>> = (0..num_threads)
        .into_par_iter()
        .filter_map(|t| {
            let k_start = t * chunk_k;
            let k_end = ((t + 1) * chunk_k).min(k);
            if k_start >= k { return None; }
            Some(gemv_q4_inner(input, &weight.packed, &weight.scales, k, n, k_start, k_end))
        })
        .collect();

    let mut output = vec![0.0f32; n];
    for partial in &partials {
        for j in 0..n {
            output[j] += partial[j];
        }
    }
    output
}

/// GEMM with Q4 weights: [seq_len, k] @ [k, n] -> [seq_len, n].
/// Multi-threaded via rayon: parallelizes across tokens for large matrices.
#[inline]
fn gemm_q4(x: &[f32], seq_len: usize, weight: &Q4Weight) -> Vec<f32> {
    let k = weight.k;
    let n = weight.n;

    if seq_len <= 1 || k * n < 4_000_000 {
        let mut output = vec![0.0f32; seq_len * n];
        for t in 0..seq_len {
            let row = gemv_q4_inner(
                &x[t * k..(t + 1) * k],
                &weight.packed, &weight.scales, k, n, 0, k,
            );
            output[t * n..(t + 1) * n].copy_from_slice(&row);
        }
        return output;
    }

    // Rayon parallel across tokens
    let mut output = vec![0.0f32; seq_len * n];
    output.par_chunks_mut(n)
        .enumerate()
        .for_each(|(t, out_row)| {
            let x_row = &x[t * k..(t + 1) * k];
            let row = gemv_q4_inner(x_row, &weight.packed, &weight.scales, k, n, 0, k);
            out_row.copy_from_slice(&row);
        });
    output
}

/// Dequantize one row from Q4 embedding for token lookup.
#[inline]
fn embed_lookup_q4(weight: &Q4Weight, token_id: usize) -> Vec<f32> {
    let n = weight.n;
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;

    let scale_base = token_id * groups_per_row;
    let pack_base = token_id * groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; n];

    for g in 0..groups_per_row {
        let scale = weight.scales[scale_base + g].to_f32();
        let pack_offset = pack_base + g * packed_per_group;
        let out_offset = g * Q4_GROUP_SIZE;

        for j in 0..packed_per_group {
            let byte = weight.packed[pack_offset + j];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;

            output[out_offset + j * 2] = scale * q0 as f32;
            output[out_offset + j * 2 + 1] = scale * q1 as f32;
        }
    }

    output
}

// ============================================================
// Shared compute kernels (format-independent)
// ============================================================

/// RmsNorm with f16 gamma.
#[inline]
fn rms_norm_f16(x: &[f32], gamma: &[f16]) -> Vec<f32> {
    let size = x.len();
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_rms = 1.0 / (sum_sq / size as f32 + 1e-6).sqrt();
    let mut out = vec![0.0f32; size];
    for i in 0..size {
        out[i] = x[i] * inv_rms * gamma[i].to_f32();
    }
    out
}

/// Apply RoPE in-place on a flat buffer [num_heads * head_dim].
#[inline]
fn apply_rope_raw(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    half_dim: usize,
    position: usize,
) {
    let cos_offset = position * half_dim;
    for h in 0..num_heads {
        let base = h * head_dim;
        for i in 0..half_dim {
            let x1 = data[base + i];
            let x2 = data[base + half_dim + i];
            let c = cos_table[cos_offset + i];
            let s = sin_table[cos_offset + i];
            data[base + i] = x1 * c - x2 * s;
            data[base + half_dim + i] = x2 * c + x1 * s;
        }
    }
}

/// In-place softmax.
#[inline]
fn softmax_raw(scores: &mut [f32]) {
    let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max_val).exp();
        sum += *s;
    }
    let inv_sum = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

/// Parallel lm_head using Weight enum (f16 or Q4).
#[inline(never)]
fn lm_head_parallel(input: &[f32], embed: &Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    match embed {
        Weight::F16(w) => lm_head_parallel_f16(input, &w.data, vocab, hidden),
        Weight::Q4(w) => lm_head_parallel_q4(input, w, vocab, hidden),
    }
}

/// Parallel lm_head with f16 embedding weights (rayon).
fn lm_head_parallel_f16(input: &[f32], embed_data: &[f16], vocab: usize, hidden: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; vocab];
    output.par_chunks_mut(256)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * 256;
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let r = start + i;
                let row = &embed_data[r * hidden..(r + 1) * hidden];
                let mut sum = 0.0f32;
                for j in 0..hidden {
                    sum += input[j] * row[j].to_f32();
                }
                *out = sum;
            }
        });
    output
}

/// Parallel lm_head with Q4 embedding weights (rayon).
fn lm_head_parallel_q4(input: &[f32], embed: &Q4Weight, vocab: usize, hidden: usize) -> Vec<f32> {
    let groups_per_row = hidden / Q4_GROUP_SIZE;
    let packed_per_group = Q4_GROUP_SIZE / 2;
    let packed_per_row = groups_per_row * packed_per_group;

    let mut output = vec![0.0f32; vocab];
    output.par_chunks_mut(256)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let start = chunk_idx * 256;
            for (i, out) in out_chunk.iter_mut().enumerate() {
                let v = start + i;
                let mut dot = 0.0f32;
                let scale_base = v * groups_per_row;
                let pack_base = v * packed_per_row;

                for g in 0..groups_per_row {
                    let scale = embed.scales[scale_base + g].to_f32();
                    if scale == 0.0 { continue; }
                    let pack_offset = pack_base + g * packed_per_group;
                    let inp_offset = g * Q4_GROUP_SIZE;

                    let lut = [
                        scale * -8.0, scale * -7.0, scale * -6.0, scale * -5.0,
                        scale * -4.0, scale * -3.0, scale * -2.0, scale * -1.0,
                        0.0,          scale,         scale * 2.0,  scale * 3.0,
                        scale * 4.0,  scale * 5.0,  scale * 6.0,  scale * 7.0,
                    ];

                    for j in 0..packed_per_group {
                        let byte = embed.packed[pack_offset + j];
                        dot += input[inp_offset + j * 2] * lut[(byte & 0x0F) as usize];
                        dot += input[inp_offset + j * 2 + 1] * lut[(byte >> 4) as usize];
                    }
                }
                *out = dot;
            }
        });
    output
}

// ============================================================
// Forward decode (single token)
// ============================================================

/// Fast single-token decode. Works with both f16 and Q4 weights.
/// All computation uses raw f32 buffers. KV cache is raw Vec<f32>.
pub fn forward_decode_raw(
    weights: &DecodeWeights,
    token_id: usize,
    kv_cache: &mut RawKvCache,
) -> Vec<f32> {
    let hidden = weights.embed_hidden;
    let num_layers = weights.layers.len();

    // Token embedding
    let mut x = embed_lookup(&weights.embed, token_id, hidden);

    for i in 0..num_layers {
        let lw = &weights.layers[i];
        let use_rope = weights.no_rope_layers[i];
        let (ref mut cached_k, ref mut cached_v, ref mut cached_len) = kv_cache[i];
        let offset = *cached_len;

        let num_heads = lw.num_heads;
        let num_kv_heads = lw.num_kv_heads;
        let head_dim = lw.head_dim;
        let num_kv_groups = lw.num_kv_groups;

        // Pre-attention RmsNorm
        let x_norm = rms_norm_f16(&x, &lw.input_norm_gamma);

        // QKV projections
        let mut q = gemv(&x_norm, &lw.q_proj);
        let mut k_new = gemv(&x_norm, &lw.k_proj);
        let v_new = gemv(&x_norm, &lw.v_proj);

        // Apply RoPE
        if use_rope {
            apply_rope_raw(&mut q, num_heads, head_dim, &weights.rope_cos, &weights.rope_sin, weights.rope_half_dim, offset);
            apply_rope_raw(&mut k_new, num_kv_heads, head_dim, &weights.rope_cos, &weights.rope_sin, weights.rope_half_dim, offset);
        }

        // Append to KV cache (token-major: [seq_len, kv_heads, head_dim])
        cached_k.extend_from_slice(&k_new);
        cached_v.extend_from_slice(&v_new);
        *cached_len = offset + 1;
        let kv_seq_len = *cached_len;

        // Attention — parallel across heads for long contexts
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kv_stride = num_kv_heads * head_dim;

        let attn_output: Vec<f32> = if kv_seq_len >= 64 {
            // Parallel: each head computed independently via rayon
            let head_results: Vec<Vec<f32>> = (0..num_heads)
                .into_par_iter()
                .map(|h| {
                    let kv_h = h / num_kv_groups;
                    let q_offset = h * head_dim;
                    let q_vec = &q[q_offset..q_offset + head_dim];

                    let mut scores = vec![0.0f32; kv_seq_len];
                    for s in 0..kv_seq_len {
                        let k_offset = s * kv_stride + kv_h * head_dim;
                        let k_vec = &cached_k[k_offset..k_offset + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[s] = dot * scale;
                    }

                    softmax_raw(&mut scores);

                    let mut head_out = vec![0.0f32; head_dim];
                    for s in 0..kv_seq_len {
                        let v_offset = s * kv_stride + kv_h * head_dim;
                        let score = scores[s];
                        for d in 0..head_dim {
                            head_out[d] += score * cached_v[v_offset + d];
                        }
                    }
                    head_out
                })
                .collect();

            // Flatten: [num_heads][head_dim] -> [num_heads * head_dim]
            let mut out = Vec::with_capacity(num_heads * head_dim);
            for hr in head_results {
                out.extend_from_slice(&hr);
            }
            out
        } else {
            // Short context: serial is faster (no rayon overhead)
            let mut attn_output = vec![0.0f32; num_heads * head_dim];
            for h in 0..num_heads {
                let kv_h = h / num_kv_groups;
                let q_offset = h * head_dim;
                let q_vec = &q[q_offset..q_offset + head_dim];

                let mut scores = vec![0.0f32; kv_seq_len];
                for s in 0..kv_seq_len {
                    let k_offset = s * kv_stride + kv_h * head_dim;
                    let k_vec = &cached_k[k_offset..k_offset + head_dim];
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    scores[s] = dot * scale;
                }

                softmax_raw(&mut scores);

                let out_offset = h * head_dim;
                for s in 0..kv_seq_len {
                    let v_offset = s * kv_stride + kv_h * head_dim;
                    let score = scores[s];
                    for d in 0..head_dim {
                        attn_output[out_offset + d] += score * cached_v[v_offset + d];
                    }
                }
            }
            attn_output
        };

        // O projection
        let attn_out = gemv(&attn_output, &lw.o_proj);

        // Residual
        for j in 0..hidden {
            x[j] += attn_out[j];
        }

        // Pre-MLP RmsNorm
        let x_norm = rms_norm_f16(&x, &lw.post_attn_norm_gamma);

        // MLP: silu(gate(x)) * up(x) -> down(...)
        // Fused gate+up: compute both in one pass to halve input reads
        let intermediate = fused_gate_up_gemv(&x_norm, &lw.gate_proj, &lw.up_proj);

        let mlp_out = gemv(&intermediate, &lw.down_proj);

        // Residual
        for j in 0..hidden {
            x[j] += mlp_out[j];
        }
    }

    // Final norm
    x = rms_norm_f16(&x, &weights.final_norm_gamma);

    // lm_head (parallel)
    lm_head_parallel(&x, &weights.embed, weights.embed_vocab, weights.embed_hidden)
}

// ============================================================
// Raw prefill (full prompt)
// ============================================================

/// Raw prefill: process entire prompt through all layers.
/// Returns (last_token_logits, raw_kv_cache).
pub fn raw_prefill(
    weights: &DecodeWeights,
    token_ids: &[u32],
) -> (Vec<f32>, RawKvCache) {
    let hidden = weights.embed_hidden;
    let seq_len = token_ids.len();
    let num_layers = weights.layers.len();

    // Embedding: [seq_len, hidden]
    let mut x = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let row = embed_lookup(&weights.embed, tid as usize, hidden);
        x[t * hidden..(t + 1) * hidden].copy_from_slice(&row);
    }

    let mut kv_cache = Vec::with_capacity(num_layers);

    for i in 0..num_layers {
        let lw = &weights.layers[i];
        let use_rope = weights.no_rope_layers[i];
        let num_heads = lw.num_heads;
        let num_kv_heads = lw.num_kv_heads;
        let head_dim = lw.head_dim;
        let num_kv_groups = lw.num_kv_groups;

        // Pre-attention RmsNorm: per-token
        let mut x_norm = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = &x[t * hidden..(t + 1) * hidden];
            let normed = rms_norm_f16(row, &lw.input_norm_gamma);
            x_norm[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // QKV: [seq_len, hidden] @ weight -> [seq_len, out_dim]
        let mut q_all = gemm(&x_norm, seq_len, &lw.q_proj);
        let mut k_all = gemm(&x_norm, seq_len, &lw.k_proj);
        let v_all = gemm(&x_norm, seq_len, &lw.v_proj);

        // Apply RoPE per-token
        if use_rope {
            for t in 0..seq_len {
                let q_start = t * num_heads * head_dim;
                apply_rope_raw(
                    &mut q_all[q_start..q_start + num_heads * head_dim],
                    num_heads, head_dim,
                    &weights.rope_cos, &weights.rope_sin, weights.rope_half_dim, t,
                );
                let k_start = t * num_kv_heads * head_dim;
                apply_rope_raw(
                    &mut k_all[k_start..k_start + num_kv_heads * head_dim],
                    num_kv_heads, head_dim,
                    &weights.rope_cos, &weights.rope_sin, weights.rope_half_dim, t,
                );
            }
        }

        // KV cache: already [seq_len, kv_heads, head_dim] from GEMM
        let cached_k = k_all.clone();
        let cached_v = v_all.clone();

        // Causal attention — parallel across heads
        let scale = 1.0 / (head_dim as f32).sqrt();
        let q_stride = num_heads * head_dim;
        let kv_stride = num_kv_heads * head_dim;

        let head_results: Vec<Vec<f32>> = (0..num_heads)
            .into_par_iter()
            .map(|h| {
                let kv_h = h / num_kv_groups;
                let mut head_out = vec![0.0f32; seq_len * head_dim];

                for t1 in 0..seq_len {
                    let attend_len = t1 + 1;
                    let q_off = t1 * q_stride + h * head_dim;
                    let q_vec = &q_all[q_off..q_off + head_dim];

                    let mut scores = vec![0.0f32; attend_len];
                    for t2 in 0..attend_len {
                        let k_off = t2 * kv_stride + kv_h * head_dim;
                        let k_vec = &cached_k[k_off..k_off + head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        scores[t2] = dot * scale;
                    }

                    softmax_raw(&mut scores);

                    let out_base = t1 * head_dim;
                    for t2 in 0..attend_len {
                        let v_off = t2 * kv_stride + kv_h * head_dim;
                        let score = scores[t2];
                        for d in 0..head_dim {
                            head_out[out_base + d] += score * cached_v[v_off + d];
                        }
                    }
                }
                head_out
            })
            .collect();

        // Interleave head results: [seq_len, num_heads, head_dim]
        let mut attn_output = vec![0.0f32; seq_len * num_heads * head_dim];
        for (h, hr) in head_results.iter().enumerate() {
            for t in 0..seq_len {
                let src = &hr[t * head_dim..(t + 1) * head_dim];
                let dst_off = t * q_stride + h * head_dim;
                attn_output[dst_off..dst_off + head_dim].copy_from_slice(src);
            }
        }

        // O projection
        let o_out = gemm(&attn_output, seq_len, &lw.o_proj);

        // Residual
        for j in 0..seq_len * hidden {
            x[j] += o_out[j];
        }

        // Pre-MLP RmsNorm
        let mut x_norm2 = vec![0.0f32; seq_len * hidden];
        for t in 0..seq_len {
            let row = &x[t * hidden..(t + 1) * hidden];
            let normed = rms_norm_f16(row, &lw.post_attn_norm_gamma);
            x_norm2[t * hidden..(t + 1) * hidden].copy_from_slice(&normed);
        }

        // MLP: gate+up are separate GEMMs for prefill (fused per-token not beneficial for GEMM)
        let gate = gemm(&x_norm2, seq_len, &lw.gate_proj);
        let up = gemm(&x_norm2, seq_len, &lw.up_proj);

        let inter_size = lw.gate_proj.n();
        let mut intermediate = vec![0.0f32; seq_len * inter_size];
        // Fuse SiLU activation: silu(gate) * up
        for j in 0..seq_len * inter_size {
            let g = gate[j];
            let silu_g = g / (1.0 + (-g).exp());
            intermediate[j] = silu_g * up[j];
        }

        let mlp_out = gemm(&intermediate, seq_len, &lw.down_proj);

        // Residual
        for j in 0..seq_len * hidden {
            x[j] += mlp_out[j];
        }

        kv_cache.push((cached_k, cached_v, seq_len));

        if i % 6 == 0 || i == num_layers - 1 {
            eprintln!("  Prefill layer {i}/{num_layers}");
        }
    }

    // Final norm (last token only)
    let last_row = &x[(seq_len - 1) * hidden..seq_len * hidden];
    let normed = rms_norm_f16(last_row, &weights.final_norm_gamma);

    // lm_head
    let logits = lm_head_parallel(&normed, &weights.embed, weights.embed_vocab, hidden);

    (logits, kv_cache)
}

