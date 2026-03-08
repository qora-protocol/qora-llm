//! GPU weight loading for QOR3B.
//!
//! Converts CPU DecodeWeights (Q4 or F16) into Cortex GPU tensors.
//! Q4 weights are uploaded as packed quantized tensors with on-the-fly
//! GPU dequantization during matmul.

use cortex::prelude::*;
use cortex::tensor::{DType, TensorData};
use cortex::tensor::quantization::{
    BlockSize, QuantLevel, QuantMode, QuantParam, QuantScheme, QuantStore, QuantValue,
};
use half::f16;

use crate::gemv::{DecodeWeights, WeightRef};

// ============================================================
// GPU model structures
// ============================================================

/// All model weights on GPU.
pub struct GpuModel<B: Backend> {
    pub layers: Vec<GpuLayer<B>>,
    pub embed_f32: Vec<f32>,          // [vocab, hidden] kept on CPU for lookup
    pub embed_vocab: usize,
    pub embed_hidden: usize,
    pub final_norm_gamma: Tensor<B, 1>,
    pub rope_cos: Tensor<B, 2>,       // [max_seq, half_dim]
    pub rope_sin: Tensor<B, 2>,
    pub no_rope_layers: Vec<bool>,
}

/// Per-layer weights on GPU.
pub struct GpuLayer<B: Backend> {
    pub q_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub gate_proj: Tensor<B, 2>,
    pub up_proj: Tensor<B, 2>,
    pub down_proj: Tensor<B, 2>,
    pub input_norm_gamma: Tensor<B, 1>,
    pub post_attn_norm_gamma: Tensor<B, 1>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
}

// ============================================================
// Q4 format conversion
// ============================================================

/// Quantization scheme for Q4 block-32 symmetric.
fn q4_scheme() -> QuantScheme {
    QuantScheme {
        value: QuantValue::Q4S,
        param: QuantParam::F32,
        store: QuantStore::PackedU32(0), // packed on innermost dim
        level: QuantLevel::Block(BlockSize::new([32])), // group_size=32
        mode: QuantMode::Symmetric,
    }
}

/// Convert QOR3B Q4 packed bytes to Burn Q4S packed bytes.
///
/// QOR3B stores unsigned nibbles [0-15] with offset -8: val = (q - 8) * scale
/// Burn Q4S uses signed two's complement nibbles: val = sign_extend(q) * scale
///
/// Conversion: XOR each nibble's sign bit → byte ^ 0x88
fn convert_q4_packed(qor3b_packed: &[u8]) -> Vec<u8> {
    qor3b_packed.iter().map(|&b| b ^ 0x88).collect()
}

/// Convert a Q4 weight to a Cortex GPU tensor using raw packed bytes.
fn q4_weight_to_gpu<B: Backend>(
    packed: &[u8],
    scales: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    // Convert nibble encoding: QOR3B unsigned offset → Burn signed two's complement
    let burn_packed = convert_q4_packed(packed);

    // Convert scales: f16 → f32, then to raw bytes (little-endian)
    let scales_f32: Vec<f32> = scales.iter().map(|s| s.to_f32()).collect();
    let scale_bytes: Vec<u8> = scales_f32.iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();

    // Concatenate packed data + scale bytes
    let total_len = burn_packed.len() + scale_bytes.len();
    let mut combined = Vec::with_capacity(total_len);
    combined.extend_from_slice(&burn_packed);
    combined.extend_from_slice(&scale_bytes);

    let scheme = q4_scheme();
    let data = TensorData::from_bytes_vec(combined, vec![k, n], DType::QFloat(scheme));
    Tensor::from_data(data, device)
}

/// Convert an F16 weight to a Cortex GPU f32 tensor.
fn f16_weight_to_gpu<B: Backend>(
    data: &[f16],
    k: usize,
    n: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
    let td = TensorData::new(f32_data, [k, n]);
    Tensor::from_data(td, device)
}

/// Convert any WeightRef to a GPU tensor.
fn weight_to_gpu<B: Backend>(w: &WeightRef<'_>, device: &B::Device) -> Tensor<B, 2> {
    match w {
        WeightRef::F16 { data, k, n } => f16_weight_to_gpu::<B>(data, *k, *n, device),
        WeightRef::Q4 { packed, scales, k, n } => q4_weight_to_gpu::<B>(packed, scales, *k, *n, device),
    }
}

/// Convert f16 norm gamma to GPU f32 tensor.
fn norm_to_gpu<B: Backend>(gamma: &[f16], device: &B::Device) -> Tensor<B, 1> {
    let f32_data: Vec<f32> = gamma.iter().map(|v| v.to_f32()).collect();
    let td = TensorData::new(f32_data, [gamma.len()]);
    Tensor::from_data(td, device)
}

// ============================================================
// Model loading
// ============================================================

/// Load DecodeWeights to GPU, creating a GpuModel.
pub fn load_model_gpu<B: Backend>(
    weights: &DecodeWeights,
    device: &B::Device,
) -> GpuModel<B> {
    let num_layers = weights.num_layers();
    let hidden = weights.hidden();

    eprintln!("Loading {} layers to GPU ({})...",
        num_layers, weights.format_name());

    let mut layers = Vec::with_capacity(num_layers);
    for i in 0..num_layers {
        if i % 6 == 0 {
            eprintln!("  Layer {i}/{num_layers}...");
        }
        let lr = weights.layer_ref(i);
        layers.push(GpuLayer {
            q_proj: weight_to_gpu::<B>(&lr.q_proj, device),
            k_proj: weight_to_gpu::<B>(&lr.k_proj, device),
            v_proj: weight_to_gpu::<B>(&lr.v_proj, device),
            o_proj: weight_to_gpu::<B>(&lr.o_proj, device),
            gate_proj: weight_to_gpu::<B>(&lr.gate_proj, device),
            up_proj: weight_to_gpu::<B>(&lr.up_proj, device),
            down_proj: weight_to_gpu::<B>(&lr.down_proj, device),
            input_norm_gamma: norm_to_gpu::<B>(lr.input_norm_gamma, device),
            post_attn_norm_gamma: norm_to_gpu::<B>(lr.post_attn_norm_gamma, device),
            num_heads: lr.num_heads,
            num_kv_heads: lr.num_kv_heads,
            head_dim: lr.head_dim,
            num_kv_groups: lr.num_kv_groups,
        });
    }
    eprintln!("  Layer {num_layers}/{num_layers}... done");

    // Embedding: keep on CPU as Vec<f32> for fast single-token lookup
    let embed_f32 = match weights.embed_ref() {
        WeightRef::F16 { data, .. } => data.iter().map(|v| v.to_f32()).collect(),
        WeightRef::Q4 { packed, scales, k, n } => dequant_q4_to_f32(packed, scales, k, n),
    };

    // Final norm gamma
    let final_norm_gamma = norm_to_gpu::<B>(weights.final_norm_ref(), device);

    // RoPE tables
    let rope_half_dim = weights.rope_half_dim();
    let rope_cos_data = weights.rope_cos_ref();
    let rope_sin_data = weights.rope_sin_ref();
    let max_seq = rope_cos_data.len() / rope_half_dim;

    let rope_cos = Tensor::from_data(
        TensorData::new(rope_cos_data.to_vec(), [max_seq, rope_half_dim]),
        device,
    );
    let rope_sin = Tensor::from_data(
        TensorData::new(rope_sin_data.to_vec(), [max_seq, rope_half_dim]),
        device,
    );

    GpuModel {
        layers,
        embed_f32,
        embed_vocab: weights.vocab(),
        embed_hidden: hidden,
        final_norm_gamma,
        rope_cos,
        rope_sin,
        no_rope_layers: weights.no_rope_layers_ref().to_vec(),
    }
}

/// Dequantize Q4 packed data to f32 (for embedding table kept on CPU).
fn dequant_q4_to_f32(packed: &[u8], scales: &[f16], k: usize, n: usize) -> Vec<f32> {
    let groups_per_row = n / 32;
    let mut out = vec![0.0f32; k * n];

    for ki in 0..k {
        for g in 0..groups_per_row {
            let group_idx = ki * groups_per_row + g;
            let scale = scales[group_idx].to_f32();
            let col_start = g * 32;
            let pack_start = group_idx * 16;

            for j in (0..32).step_by(2) {
                let byte = packed[pack_start + j / 2];
                let q0 = (byte & 0x0F) as f32 - 8.0;
                let q1 = (byte >> 4) as f32 - 8.0;
                out[ki * n + col_start + j] = q0 * scale;
                out[ki * n + col_start + j + 1] = q1 * scale;
            }
        }
    }
    out
}
