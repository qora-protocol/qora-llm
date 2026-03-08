//! GPU inference for QORA.
//!
//! Implements single-token decode and prompt prefill using Cortex GPU tensors.
//! Uses quantized matmul (Q4) or f32 matmul depending on weight format.

use cortex::prelude::*;
use cortex::tensor::activation;
use std::io::Write;
use std::time::Instant;

use crate::gemv::DecodeWeights;
use crate::generate::GenerateParams;
use crate::gpu_loader::{GpuModel, load_model_gpu};
use crate::tokenizer::QoraTokenizer;

// ============================================================
// KV Cache
// ============================================================

/// GPU KV cache: per-layer key/value tensors.
/// Layout: k[layer] = [kv_heads, seq_len, head_dim], v same.
pub struct GpuKvCache<B: Backend> {
    pub k: Vec<Option<Tensor<B, 3>>>,
    pub v: Vec<Option<Tensor<B, 3>>>,
    pub seq_len: usize,
}

impl<B: Backend> GpuKvCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            k: (0..num_layers).map(|_| None).collect(),
            v: (0..num_layers).map(|_| None).collect(),
            seq_len: 0,
        }
    }
}

// ============================================================
// Helper ops
// ============================================================

/// RMS norm: x * gamma / sqrt(mean(x^2) + eps)
fn rms_norm<B: Backend>(x: Tensor<B, 2>, gamma: &Tensor<B, 1>) -> Tensor<B, 2> {
    let eps = 1e-5;
    // x: [batch, hidden]
    let x_sq = x.clone().powf_scalar(2.0);
    let mean_sq = x_sq.mean_dim(1); // [batch, 1]
    let rms = (mean_sq + eps).sqrt();
    let normed = x / rms;
    // Broadcast gamma [hidden] over batch dim
    normed * gamma.clone().unsqueeze::<2>()
}

/// Apply Llama-style split-half RoPE rotation.
/// q_or_k: [1, num_heads * head_dim] or [seq_len, num_heads * head_dim]
/// cos, sin: sliced to [seq_len, half_dim]
fn apply_rope<B: Backend>(
    x: Tensor<B, 2>,
    cos: Tensor<B, 2>,
    sin: Tensor<B, 2>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    let seq_len = x.dims()[0];
    let half_dim = head_dim / 2;

    // Reshape to [seq_len, num_heads, head_dim]
    let x = x.reshape([seq_len, num_heads, head_dim]);

    // Split into first half and second half
    let x_first = x.clone().slice([0..seq_len, 0..num_heads, 0..half_dim]);
    let x_second = x.slice([0..seq_len, 0..num_heads, half_dim..head_dim]);

    // cos, sin: [seq_len, half_dim] → [seq_len, 1, half_dim] for broadcast
    let cos: Tensor<B, 3> = cos.unsqueeze_dim::<3>(1);
    let sin: Tensor<B, 3> = sin.unsqueeze_dim::<3>(1);

    // Rotate: [first, second] → [first*cos - second*sin, second*cos + first*sin]
    let new_first = x_first.clone() * cos.clone() - x_second.clone() * sin.clone();
    let new_second = x_second * cos + x_first * sin;

    // Concatenate back
    Tensor::cat(vec![new_first, new_second], 2)
        .reshape([seq_len, num_heads * head_dim])
}

/// Grouped Query Attention for single-token decode.
/// q: [1, num_heads * head_dim]
/// k_cache: [kv_heads, seq_len, head_dim]
/// v_cache: [kv_heads, seq_len, head_dim]
fn gqa_decode<B: Backend>(
    q: Tensor<B, 2>,
    k_cache: &Tensor<B, 3>,
    v_cache: &Tensor<B, 3>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    let kv_seq_len = k_cache.dims()[1];
    let num_kv_groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // q: [1, num_heads, head_dim] → [num_heads, 1, head_dim]
    let q = q.reshape([1, num_heads, head_dim])
        .swap_dims(0, 1); // [num_heads, 1, head_dim]

    // Expand KV heads: [kv_heads, seq, head_dim] → [num_heads, seq, head_dim]
    let k = if num_kv_groups > 1 {
        // [kv_heads, seq, dim] → [kv_heads, 1, seq, dim] → [kv_heads, groups, seq, dim] → [num_heads, seq, dim]
        let k4: Tensor<B, 4> = k_cache.clone()
            .unsqueeze_dim::<4>(1)
            .repeat_dim(1, num_kv_groups);
        k4.reshape([num_heads, kv_seq_len, head_dim])
    } else {
        k_cache.clone()
    };

    let v = if num_kv_groups > 1 {
        let v4: Tensor<B, 4> = v_cache.clone()
            .unsqueeze_dim::<4>(1)
            .repeat_dim(1, num_kv_groups);
        v4.reshape([num_heads, kv_seq_len, head_dim])
    } else {
        v_cache.clone()
    };

    // Attention scores: [num_heads, 1, head_dim] @ [num_heads, head_dim, seq] → [num_heads, 1, seq]
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);
    let attn_weights = activation::softmax(scores, 2);

    // Weighted sum: [num_heads, 1, seq] @ [num_heads, seq, head_dim] → [num_heads, 1, head_dim]
    let out = attn_weights.matmul(v);

    // Reshape back: [num_heads, 1, head_dim] → [1, num_heads * head_dim]
    out.swap_dims(0, 1).reshape([1, num_heads * head_dim])
}

/// Grouped Query Attention for prefill (with causal mask).
/// q: [seq_len, num_heads * head_dim]
/// k, v: [seq_len, kv_heads * head_dim]
fn gqa_prefill<B: Backend>(
    q: Tensor<B, 2>,
    k: Tensor<B, 2>,
    v: Tensor<B, 2>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor<B, 2> {
    let seq_len = q.dims()[0];
    let num_kv_groups = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Reshape: [seq, num_heads, head_dim] → [num_heads, seq, head_dim]
    let q = q.reshape([seq_len, num_heads, head_dim]).swap_dims(0, 1);
    let k = k.reshape([seq_len, num_kv_heads, head_dim]).swap_dims(0, 1);
    let v = v.reshape([seq_len, num_kv_heads, head_dim]).swap_dims(0, 1);

    // Expand KV heads
    let k = if num_kv_groups > 1 {
        let k4: Tensor<B, 4> = k.unsqueeze_dim::<4>(1)
            .repeat_dim(1, num_kv_groups);
        k4.reshape([num_heads, seq_len, head_dim])
    } else {
        k
    };
    let v = if num_kv_groups > 1 {
        let v4: Tensor<B, 4> = v.unsqueeze_dim::<4>(1)
            .repeat_dim(1, num_kv_groups);
        v4.reshape([num_heads, seq_len, head_dim])
    } else {
        v
    };

    // Scores: [num_heads, seq, seq]
    let scores = q.matmul(k.swap_dims(1, 2)).mul_scalar(scale);

    // Causal mask: upper triangle = -inf
    let mask = Tensor::<B, 2>::ones([seq_len, seq_len], &scores.device())
        .triu(1) // upper triangle (excluding diagonal)
        .mul_scalar(-1e9f32); // large negative
    let scores = scores + mask.unsqueeze_dim::<3>(0); // broadcast over heads

    let attn_weights = activation::softmax(scores, 2);
    let out = attn_weights.matmul(v);

    // [num_heads, seq, head_dim] → [seq, num_heads * head_dim]
    out.swap_dims(0, 1).reshape([seq_len, num_heads * head_dim])
}

/// SiLU activation: x * sigmoid(x)
fn silu<B: Backend>(x: Tensor<B, 2>) -> Tensor<B, 2> {
    activation::silu(x)
}

// ============================================================
// Forward pass
// ============================================================

/// Single-token decode on GPU.
fn forward_decode_gpu<B: Backend>(
    model: &GpuModel<B>,
    token_id: usize,
    kv_cache: &mut GpuKvCache<B>,
) -> Vec<f32> {
    let device = model.final_norm_gamma.device();
    let hidden = model.embed_hidden;
    let offset = kv_cache.seq_len;

    // Token embedding: lookup from CPU f32 table, upload to GPU
    let embed_start = token_id * hidden;
    let embed_slice = &model.embed_f32[embed_start..embed_start + hidden];
    let mut x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(embed_slice.to_vec(), [1, hidden]),
        &device,
    );

    for (i, layer) in model.layers.iter().enumerate() {
        let use_rope = model.no_rope_layers[i];
        let num_heads = layer.num_heads;
        let num_kv_heads = layer.num_kv_heads;
        let head_dim = layer.head_dim;

        // Pre-attention RmsNorm
        let x_norm = rms_norm(x.clone(), &layer.input_norm_gamma);

        // QKV projections: [1, hidden] @ [hidden, out] → [1, out]
        let q = x_norm.clone().matmul(layer.q_proj.clone());
        let k_new = x_norm.clone().matmul(layer.k_proj.clone());
        let v_new = x_norm.matmul(layer.v_proj.clone());

        // Apply RoPE
        let (q, k_new) = if use_rope {
            let cos = model.rope_cos.clone().slice([offset..offset + 1, 0..head_dim / 2]);
            let sin = model.rope_sin.clone().slice([offset..offset + 1, 0..head_dim / 2]);
            let q = apply_rope(q, cos.clone(), sin.clone(), num_heads, head_dim);
            let k_new = apply_rope(k_new, cos, sin, num_kv_heads, head_dim);
            (q, k_new)
        } else {
            (q, k_new)
        };

        // Reshape K, V for cache: [1, kv_heads * head_dim] → [kv_heads, 1, head_dim]
        let k_new_3d = k_new.reshape([1, num_kv_heads, head_dim]).swap_dims(0, 1);
        let v_new_3d = v_new.reshape([1, num_kv_heads, head_dim]).swap_dims(0, 1);

        // Update KV cache
        let (k_cache, v_cache) = if let (Some(prev_k), Some(prev_v)) =
            (kv_cache.k[i].take(), kv_cache.v[i].take())
        {
            // Concatenate along seq dimension (dim 1)
            let k = Tensor::cat(vec![prev_k, k_new_3d], 1);
            let v = Tensor::cat(vec![prev_v, v_new_3d], 1);
            (k, v)
        } else {
            (k_new_3d, v_new_3d)
        };

        // Attention
        let attn_out = gqa_decode(q, &k_cache, &v_cache, num_heads, num_kv_heads, head_dim);

        // Store updated cache
        kv_cache.k[i] = Some(k_cache);
        kv_cache.v[i] = Some(v_cache);

        // O projection
        let attn_out = attn_out.matmul(layer.o_proj.clone());

        // Residual
        x = x + attn_out;

        // Pre-MLP RmsNorm
        let x_norm = rms_norm(x.clone(), &layer.post_attn_norm_gamma);

        // MLP: silu(gate) * up → down
        let gate = x_norm.clone().matmul(layer.gate_proj.clone());
        let up = x_norm.matmul(layer.up_proj.clone());
        let intermediate = silu(gate) * up;
        let mlp_out = intermediate.matmul(layer.down_proj.clone());

        // Residual
        x = x + mlp_out;
    }

    kv_cache.seq_len = offset + 1;

    // Final norm
    x = rms_norm(x, &model.final_norm_gamma);

    // lm_head: x @ embed.T — done on CPU for efficiency (vocab=128K is large)
    let x_data = x.to_data().to_vec::<f32>().unwrap();
    lm_head_cpu(&x_data, &model.embed_f32, model.embed_vocab, hidden)
}

/// Prompt prefill on GPU.
fn forward_prefill_gpu<B: Backend>(
    model: &GpuModel<B>,
    token_ids: &[u32],
    kv_cache: &mut GpuKvCache<B>,
) -> Vec<f32> {
    let device = model.final_norm_gamma.device();
    let hidden = model.embed_hidden;
    let seq_len = token_ids.len();

    // Embedding: lookup all tokens
    let mut embed_data = vec![0.0f32; seq_len * hidden];
    for (t, &tid) in token_ids.iter().enumerate() {
        let start = tid as usize * hidden;
        embed_data[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&model.embed_f32[start..start + hidden]);
    }
    let mut x: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(embed_data, [seq_len, hidden]),
        &device,
    );

    for (i, layer) in model.layers.iter().enumerate() {
        let use_rope = model.no_rope_layers[i];
        let num_heads = layer.num_heads;
        let num_kv_heads = layer.num_kv_heads;
        let head_dim = layer.head_dim;

        // Pre-attention RmsNorm
        let x_norm = rms_norm(x.clone(), &layer.input_norm_gamma);

        // QKV projections: [seq, hidden] @ [hidden, out] → [seq, out]
        let q = x_norm.clone().matmul(layer.q_proj.clone());
        let k = x_norm.clone().matmul(layer.k_proj.clone());
        let v = x_norm.matmul(layer.v_proj.clone());

        // Apply RoPE
        let (q, k) = if use_rope {
            let cos = model.rope_cos.clone().slice([0..seq_len, 0..head_dim / 2]);
            let sin = model.rope_sin.clone().slice([0..seq_len, 0..head_dim / 2]);
            let q = apply_rope(q, cos.clone(), sin.clone(), num_heads, head_dim);
            let k = apply_rope(k, cos, sin, num_kv_heads, head_dim);
            (q, k)
        } else {
            (q, k)
        };

        // Store KV cache: [seq, kv_heads*head_dim] → [kv_heads, seq, head_dim]
        let k_cache = k.clone().reshape([seq_len, num_kv_heads, head_dim]).swap_dims(0, 1);
        let v_cache = v.clone().reshape([seq_len, num_kv_heads, head_dim]).swap_dims(0, 1);
        kv_cache.k[i] = Some(k_cache);
        kv_cache.v[i] = Some(v_cache);

        // Causal attention
        let attn_out = gqa_prefill(q, k, v, num_heads, num_kv_heads, head_dim);

        // O projection
        let attn_out = attn_out.matmul(layer.o_proj.clone());

        // Residual
        x = x + attn_out;

        // Pre-MLP RmsNorm
        let x_norm = rms_norm(x.clone(), &layer.post_attn_norm_gamma);

        // MLP
        let gate = x_norm.clone().matmul(layer.gate_proj.clone());
        let up = x_norm.matmul(layer.up_proj.clone());
        let intermediate = silu(gate) * up;
        let mlp_out = intermediate.matmul(layer.down_proj.clone());

        // Residual
        x = x + mlp_out;

        if i % 6 == 0 || i == model.layers.len() - 1 {
            eprintln!("  Prefill layer {i}/{}", model.layers.len());
        }
    }

    kv_cache.seq_len = seq_len;

    // Final norm (last token only)
    let last = x.slice([seq_len - 1..seq_len, 0..hidden]);
    let normed = rms_norm(last, &model.final_norm_gamma);
    let x_data = normed.to_data().to_vec::<f32>().unwrap();
    lm_head_cpu(&x_data, &model.embed_f32, model.embed_vocab, hidden)
}

/// CPU lm_head: hidden → vocab logits via embed matrix.
fn lm_head_cpu(x: &[f32], embed: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    use rayon::prelude::*;

    // Parallel over vocab chunks
    let chunk_size = 4096;
    let mut logits = vec![0.0f32; vocab];

    logits
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            for (j, out) in chunk.iter_mut().enumerate() {
                let v = start + j;
                if v >= vocab { break; }
                let row = &embed[v * hidden..(v + 1) * hidden];
                let mut dot = 0.0f32;
                for d in 0..hidden {
                    dot += x[d] * row[d];
                }
                *out = dot;
            }
        });

    logits
}

// ============================================================
// Sampling (reused from generate.rs logic)
// ============================================================

struct Rng { state: u64 }

impl Rng {
    fn new() -> Self {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self { state: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0
    }
}

fn sample_token_top_k(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val { max_val = v; max_idx = i; }
        }
        return max_idx as u32;
    }

    let inv_temp = 1.0 / temperature;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k > 0 { top_k.min(indexed.len()) } else { indexed.len() };
    let top_k_candidates = &indexed[..k];

    let mut probs: Vec<(u32, f32)> = top_k_candidates.iter()
        .map(|&(id, l)| (id, ((l - max_logit) * inv_temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 { return probs[0].0; }
    let inv = 1.0 / sum;
    for (_, p) in probs.iter_mut() { *p *= inv; }

    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p { cutoff = i + 1; break; }
    }
    let candidates = &probs[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 { return candidates[0].0; }

    let rand_val = rng.next_f32() * total;
    let mut accum = 0.0f32;
    for &(id, prob) in candidates {
        accum += prob;
        if accum >= rand_val { return id; }
    }
    candidates[0].0
}

fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 { return; }
    let window = generated.len().min(64);
    let recent = &generated[generated.len() - window..];
    for &tok in recent {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 { logits[idx] /= penalty; }
            else { logits[idx] *= penalty; }
        }
    }
}

fn apply_presence_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 0.0 || generated.is_empty() { return; }
    let mut seen = vec![false; logits.len()];
    for &tok in generated {
        let idx = tok as usize;
        if idx < seen.len() { seen[idx] = true; }
    }
    for (idx, &was_seen) in seen.iter().enumerate() {
        if was_seen { logits[idx] -= penalty; }
    }
}

// ============================================================
// Main entry point
// ============================================================

/// Run GPU inference. Returns Ok(()) on success, Err if GPU not available.
pub fn generate_gpu(
    decode_weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> Result<(), String> {
    // Deep lazy computation graphs need large stack.
    // Use thread::scope so we can borrow data without Send requirement.
    // Wrap in catch_unwind so GPU OOM panics fall back to CPU gracefully.
    let result = std::cell::RefCell::new(None);

    std::thread::scope(|s| {
        let builder = std::thread::Builder::new()
            .name("gpu-llm".into())
            .stack_size(128 * 1024 * 1024);
        let handle = builder.spawn_scoped(s, || {
            // catch_unwind prevents GPU OOM panics from propagating
            // through the scope (which would crash the whole program)
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_gpu_inner(decode_weights, tokenizer, prompt, params)
            }))
        });
        match handle {
            Ok(h) => {
                let join_result = h.join();
                *result.borrow_mut() = Some(match join_result {
                    // Thread finished, catch_unwind succeeded, inner returned Result
                    Ok(Ok(inner)) => inner,
                    // Thread finished, catch_unwind caught a panic (OOM, etc.)
                    Ok(Err(panic_payload)) => {
                        let msg = if let Some(s) = panic_payload.downcast_ref::<String>() {
                            format!("GPU panic: {s}")
                        } else if let Some(s) = panic_payload.downcast_ref::<&str>() {
                            format!("GPU panic: {s}")
                        } else {
                            "GPU panic: unknown error (likely out of VRAM)".to_string()
                        };
                        Err(msg)
                    }
                    // Thread itself panicked (shouldn't happen with catch_unwind, but just in case)
                    Err(_) => Err("GPU thread panicked unexpectedly".to_string()),
                });
            }
            Err(e) => {
                *result.borrow_mut() = Some(Err(format!("Failed to spawn GPU thread: {e}")));
            }
        }
    });

    result.into_inner().unwrap_or(Err("No result from GPU thread".into()))
}

fn generate_gpu_inner(
    weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> Result<(), String> {
    type B = cortex::backend::Wgpu;

    // Try GPU initialization
    let device: <B as Backend>::Device = Default::default();
    let _test: Tensor<B, 1> = Tensor::zeros([1], &device);
    eprintln!("GPU initialized successfully");

    // VRAM probe: try allocating a ~256MB tensor to verify GPU has enough memory.
    // Full model needs ~2.3GB (Q4 weights + KV cache + activations).
    // If we can't even allocate 256MB, there's no point loading the model.
    {
        let probe_size = 256 * 1024 * 1024 / 4; // 256MB of f32
        let probe_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _probe: Tensor<B, 1> = Tensor::zeros([probe_size], &device);
            // Force the allocation to actually happen by reading a value
            let _ = _probe.slice([0..1]).to_data();
        }));
        if probe_result.is_err() {
            return Err("GPU has insufficient VRAM (failed 256MB probe)".to_string());
        }
        eprintln!("VRAM probe: 256MB OK");
    }

    // Estimate VRAM needed and warn
    let model_mb = weights.memory_bytes() / (1024 * 1024);
    let estimated_vram_mb = model_mb + 800; // weights + KV cache + activations overhead
    eprintln!("Estimated VRAM needed: ~{}MB (weights {}MB + KV cache + buffers)",
        estimated_vram_mb, model_mb);

    // Load weights to GPU
    let t0 = Instant::now();
    let model = load_model_gpu::<B>(weights, &device);
    eprintln!("Weights loaded to GPU in {:.1?}", t0.elapsed());

    // Tokenize
    let tokens = tokenizer.format_chat(prompt, params.think, params.max_new_tokens);
    eprintln!("Prompt tokens: {}", tokens.len());

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut in_think_block = false;
    let mut think_content = String::new();
    let mut rng = Rng::new();

    // Prefill
    let num_layers = model.layers.len();
    let mut kv_cache = GpuKvCache::<B>::new(num_layers);

    let t0 = Instant::now();
    let mut logits = forward_prefill_gpu(&model, &tokens, &mut kv_cache);
    let prefill_time = t0.elapsed();
    let seq_len = tokens.len();
    eprintln!("Prefill: {seq_len} tokens in {prefill_time:.1?} ({:.1} tok/s)",
        seq_len as f64 / prefill_time.as_secs_f64());

    if params.temperature > 0.0 {
        apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
    }
    let mut next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
    eprintln!("First token: {} (id={})", tokenizer.decode(&[next_token_id]), next_token_id);

    let mut decode_tokens = 0u32;
    let mut think_tokens = 0u32;
    let decode_start = Instant::now();

    // Decode loop
    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id {
            eprintln!("\n[EOS after {step} tokens]");
            break;
        }

        generated_tokens.push(next_token_id);

        // Handle think blocks
        if next_token_id == crate::tokenizer::THINK_START {
            in_think_block = true;
            if params.show_think {
                eprint!("<think>");
                std::io::stderr().flush().ok();
            }
        } else if next_token_id == crate::tokenizer::THINK_END {
            in_think_block = false;
            eprintln!("\n[thinking done: {} tokens, {} chars]", think_tokens, think_content.len());
            if params.show_think { eprintln!("</think>"); }
            think_content.clear();
            think_tokens = 0;
        } else {
            let token_text = tokenizer.decode(&[next_token_id]);
            if in_think_block {
                think_content.push_str(&token_text);
                think_tokens += 1;
                if params.show_think {
                    eprint!("{token_text}");
                    std::io::stderr().flush().ok();
                } else if think_tokens % 50 == 0 {
                    eprint!("[thinking: {think_tokens} tokens...] ");
                    std::io::stderr().flush().ok();
                }
            } else {
                print!("{token_text}");
                std::io::stdout().flush().ok();
            }
        }

        // Sentence-boundary stop near token budget (85%) to avoid mid-sentence cutoff
        if !in_think_block && step >= params.max_new_tokens * 85 / 100 {
            let piece = tokenizer.decode(&[next_token_id]);
            let trimmed = piece.trim_end();
            if trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?') {
                eprintln!("\n[clean stop near token limit at step {step}]");
                break;
            }
        }

        // Detect repetition loops and force EOS
        if crate::generate::is_stuck_in_loop(&generated_tokens) {
            eprintln!("\n[loop detected at step {}, forcing EOS]", decode_tokens);
            break;
        }

        // GPU decode
        let mut logits = forward_decode_gpu(&model, next_token_id as usize, &mut kv_cache);

        // Presence penalty prevents thinking spirals
        if in_think_block {
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
            logits[params.eos_token_id as usize] = f32::NEG_INFINITY;
        } else if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
        }

        // Think budget: force </think> if thinking too long
        if in_think_block && think_tokens >= params.think_budget as u32 {
            next_token_id = crate::tokenizer::THINK_END;
            eprintln!("\n[think budget reached ({} tokens), forcing </think>]", params.think_budget);
        } else {
            // Lower temperature during thinking for more focused reasoning
            let temp = if in_think_block { (params.temperature * 0.5).max(0.1) } else { params.temperature };
            next_token_id = sample_token_top_k(&logits, temp, params.top_p, params.top_k, &mut rng);
        }
        decode_tokens += 1;

        if decode_tokens % 50 == 0 {
            let elapsed = decode_start.elapsed();
            let tps = decode_tokens as f64 / elapsed.as_secs_f64();
            eprint!("[{decode_tokens} tokens, {tps:.1} tok/s] ");
            std::io::stderr().flush().ok();
        }
    }

    if in_think_block {
        eprintln!("\n[WARNING: thinking did not finish within {} tokens]", params.max_new_tokens);
        eprintln!("[Think content preview: {}...]", &think_content[..think_content.len().min(200)]);
    }

    println!();

    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!("Decode: {} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens,
            decode_tokens as f64 / decode_elapsed.as_secs_f64());
    }

    Ok(())
}
