use std::io::Write;
use std::time::Instant;

use crate::gemv::{DecodeWeights, forward_decode_raw, raw_prefill};
use crate::tokenizer::QoraTokenizer;

/// Generation parameters.
pub struct GenerateParams {
    pub max_new_tokens: usize,
    pub eos_token_id: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub think: bool,
    pub show_think: bool,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub think_budget: usize,  // max thinking tokens before forcing </think>
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024,
            eos_token_id: 128012, // <|im_end|>
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            think: true,
            show_think: false,
            repetition_penalty: 1.1,
            presence_penalty: 1.5,
            think_budget: 1024,
        }
    }
}

/// Stateful PRNG (xorshift64).
struct Rng {
    state: u64,
}

impl Rng {
    fn new() -> Self {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        // Ensure non-zero seed
        Self { state: seed | 1 }
    }

    /// Returns a float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 40) as f32 / 16777216.0 // 24-bit mantissa
    }
}

/// Apply repetition penalty to logits: reduce probability of recently generated tokens.
fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty <= 1.0 { return; }
    // Penalize tokens seen in the last 64 generated tokens
    let window = generated.len().min(64);
    let recent = &generated[generated.len() - window..];
    for &tok in recent {
        let idx = tok as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Apply presence penalty — subtract flat value from logits of any token seen at least once.
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

/// Detect if the model is stuck in a repeating loop.
/// Returns true if a pattern of length 4+ repeats 3+ times at the end.
pub fn is_stuck_in_loop(tokens: &[u32]) -> bool {
    if tokens.len() < 12 { return false; }
    // Check pattern lengths 4..20
    for pat_len in 4..=20.min(tokens.len() / 3) {
        let end = tokens.len();
        let pat = &tokens[end - pat_len..end];
        let mut repeats = 1;
        for r in 1..=5 {
            let start = end.saturating_sub(pat_len * (r + 1));
            if start + pat_len > end - pat_len * r { break; }
            let candidate = &tokens[end - pat_len * (r + 1)..end - pat_len * r];
            if candidate == pat {
                repeats += 1;
            } else {
                break;
            }
        }
        if repeats >= 3 { return true; }
    }
    false
}

/// Text generation with raw prefill + raw GEMV decode.
pub fn generate(
    weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> String {
    let tokens = tokenizer.format_chat(prompt, params.think, params.max_new_tokens);
    eprintln!("Prompt tokens: {}", tokens.len());

    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut in_think_block = false;
    let mut think_content = String::new();
    let mut rng = Rng::new();

    // === Prefill ===
    let t0 = Instant::now();
    let (mut logits, mut kv_cache) = raw_prefill(weights, &tokens);
    let prefill_time = t0.elapsed();
    let seq_len = tokens.len();
    eprintln!("Prefill: {seq_len} tokens in {prefill_time:.1?} ({:.1} tok/s)",
        seq_len as f64 / prefill_time.as_secs_f64());

    // Apply repetition penalty to initial logits (skip in greedy mode)
    if params.temperature > 0.0 {
        apply_repetition_penalty(&mut logits, &generated_tokens, params.repetition_penalty);
    }
    let mut next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);

    // Debug: show first token
    eprintln!("First token: {} (id={})", tokenizer.decode(&[next_token_id]), next_token_id);

    let mut decode_tokens = 0u32;
    let mut think_tokens = 0u32;
    let decode_start = Instant::now();

    // === Decode loop ===
    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id {
            eprintln!("\n[EOS after {step} tokens]");
            break;
        }

        generated_tokens.push(next_token_id);

        // Handle think block display
        if next_token_id == crate::tokenizer::THINK_START {
            in_think_block = true;
            if params.show_think {
                eprint!("<think>");
                std::io::stderr().flush().ok();
            }
        } else if next_token_id == crate::tokenizer::THINK_END {
            in_think_block = false;
            eprintln!("\n[thinking done: {} tokens, {} chars]", think_tokens, think_content.len());
            if params.show_think {
                eprintln!("</think>");
            }
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
                } else {
                    // Show progress every 50 tokens
                    if think_tokens % 50 == 0 {
                        eprint!("[thinking: {think_tokens} tokens...] ");
                        std::io::stderr().flush().ok();
                    }
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
        if is_stuck_in_loop(&generated_tokens) {
            eprintln!("\n[loop detected at step {step}, forcing EOS]");
            break;
        }

        // Fast raw GEMV decode
        let mut logits = forward_decode_raw(weights, next_token_id as usize, &mut kv_cache);

        // Presence penalty prevents thinking spirals
        if in_think_block {
            apply_presence_penalty(&mut logits, &generated_tokens, params.presence_penalty);
            logits[params.eos_token_id as usize] = f32::NEG_INFINITY;
        } else if params.temperature > 0.0 {
            // Repetition + presence penalty for final answer (not during thinking)
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

        // Show timing every 50 tokens (not every token — reduces stderr spam)
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

    let output: Vec<u32> = generated_tokens
        .iter()
        .copied()
        .filter(|&t| t != crate::tokenizer::THINK_START && t != crate::tokenizer::THINK_END)
        .collect();

    tokenizer.decode(&output)
}

/// Raw text completion with raw prefill + raw GEMV decode.
pub fn generate_raw(
    weights: &DecodeWeights,
    tokenizer: &QoraTokenizer,
    prompt: &str,
    params: &GenerateParams,
) -> String {
    let tokens = tokenizer.encode(prompt);
    eprintln!("Prompt tokens: {}", tokens.len());

    let mut all_tokens = tokens.clone();
    let mut rng = Rng::new();

    // === Prefill ===
    let t0 = Instant::now();
    let (mut logits, mut kv_cache) = raw_prefill(weights, &tokens);
    let prefill_time = t0.elapsed();
    let seq_len = tokens.len();
    eprintln!("Prefill: {seq_len} tokens in {prefill_time:.1?}");

    if params.temperature > 0.0 {
        apply_repetition_penalty(&mut logits, &all_tokens, params.repetition_penalty);
    }
    let mut next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
    let decode_start = Instant::now();
    let mut decode_tokens = 0u32;

    // === Decode loop ===
    for step in 0..params.max_new_tokens {
        if next_token_id == params.eos_token_id {
            eprintln!("\n[EOS after {step} tokens]");
            break;
        }

        all_tokens.push(next_token_id);

        let token_text = tokenizer.decode(&[next_token_id]);
        print!("{token_text}");
        std::io::stdout().flush().ok();

        // Detect repetition loops and force EOS
        if is_stuck_in_loop(&all_tokens) {
            eprintln!("\n[loop detected at step {step}, forcing EOS]");
            break;
        }

        // Fast raw GEMV decode
        let mut logits = forward_decode_raw(weights, next_token_id as usize, &mut kv_cache);
        if params.temperature > 0.0 {
            apply_repetition_penalty(&mut logits, &all_tokens, params.repetition_penalty);
        }
        next_token_id = sample_token_top_k(&logits, params.temperature, params.top_p, params.top_k, &mut rng);
        decode_tokens += 1;

        if decode_tokens % 50 == 0 {
            let elapsed = decode_start.elapsed();
            let tps = decode_tokens as f64 / elapsed.as_secs_f64();
            eprint!("[{decode_tokens} tokens, {tps:.1} tok/s] ");
            std::io::stderr().flush().ok();
        }
    }
    println!();

    let decode_elapsed = decode_start.elapsed();
    if decode_tokens > 0 {
        eprintln!("Decode: {} tokens in {decode_elapsed:.1?} ({:.2} tok/s)",
            decode_tokens,
            decode_tokens as f64 / decode_elapsed.as_secs_f64());
    }

    tokenizer.decode(&all_tokens)
}

/// Sample with top-k + top-p (nucleus) filtering.
fn sample_token_top_k(logits: &[f32], temperature: f32, top_p: f32, top_k: usize, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 {
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > max_val { max_val = v; max_idx = i; }
        }
        return max_idx as u32;
    }

    // Step 1: top-k filtering — keep only the k highest logits
    let inv_temp = 1.0 / temperature;
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut indexed: Vec<(u32, f32)> = logits.iter().enumerate()
        .map(|(i, &l)| (i as u32, l))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = if top_k > 0 { top_k.min(indexed.len()) } else { indexed.len() };
    let top_k_candidates = &indexed[..k];

    // Step 2: softmax over top-k candidates
    let mut probs: Vec<(u32, f32)> = top_k_candidates.iter()
        .map(|&(id, l)| (id, ((l - max_logit) * inv_temp).exp()))
        .collect();
    let sum: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum <= 0.0 { return probs[0].0; }
    let inv = 1.0 / sum;
    for (_, p) in probs.iter_mut() { *p *= inv; }

    // Step 3: top-p (nucleus) filtering within top-k
    let mut cumsum = 0.0f32;
    let mut cutoff = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p { cutoff = i + 1; break; }
    }
    let candidates = &probs[..cutoff];
    let total: f32 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 { return candidates[0].0; }

    // Step 4: sample
    let rand_val = rng.next_f32() * total;
    let mut accum = 0.0f32;
    for &(id, prob) in candidates {
        accum += prob;
        if accum >= rand_val { return id; }
    }
    candidates[0].0
}
