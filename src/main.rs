use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse flags
    let no_think = args.iter().any(|a| a == "--no-think");
    let raw_mode = args.iter().any(|a| a == "--raw");
    let greedy = args.iter().any(|a| a == "--greedy");
    let show_think = args.iter().any(|a| a == "--show-think");
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    let force_cpu = args.iter().any(|a| a == "--cpu");

    // Parse key-value arguments
    let mut prompt = String::from("Hello, how are you?");
    let mut max_tokens: usize = 1024;
    let mut think_budget: usize = 1024;
    let mut load_path = PathBuf::from("model.qora");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" => {
                if i + 1 < args.len() {
                    prompt = args[i + 1].clone();
                    i += 1;
                }
            }
            "--max-tokens" => {
                if i + 1 < args.len() {
                    max_tokens = args[i + 1].parse().unwrap_or(256);
                    i += 1;
                }
            }
            "--think-budget" => {
                if i + 1 < args.len() {
                    think_budget = args[i + 1].parse().unwrap_or(1024);
                    i += 1;
                }
            }
            "--load" => {
                if i + 1 < args.len() {
                    load_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let mode_str = if raw_mode {
        "raw"
    } else if no_think {
        "no-think"
    } else {
        "think"
    };

    eprintln!("QORA - Native Rust LLM Inference Engine");

    // Detect system resources and apply smart limits
    let sys_info = qora::system::SystemInfo::detect();
    eprintln!("System: {} MB total RAM, {} MB available, {} threads",
        sys_info.total_ram_mb, sys_info.available_ram_mb, sys_info.cpu_threads);
    let limits = sys_info.smart_limits();
    if let Some(warning) = limits.warning {
        eprintln!("WARNING: {warning}");
    }

    // Apply smart defaults if user didn't specify
    let user_specified_tokens = args.iter().any(|a| a == "--max-tokens");
    let user_specified_budget = args.iter().any(|a| a == "--think-budget");
    if !user_specified_tokens {
        max_tokens = limits.default_max_tokens;
    }
    if !user_specified_budget {
        think_budget = limits.default_think_budget;
    }
    // Clamp to hard limits
    max_tokens = max_tokens.min(limits.max_tokens);
    think_budget = think_budget.min(limits.max_think_budget);
    // Ensure max_tokens > think_budget so model has room to answer
    if !no_think && max_tokens <= think_budget {
        max_tokens = think_budget + 1024;
    }

    eprintln!("Prompt: {prompt}");
    eprintln!("Mode: {mode_str}, max_tokens: {max_tokens}, think_budget: {think_budget}");

    // === Load model from .qor3b binary ===
    eprintln!("Loading model from {}...", load_path.display());
    let t0 = Instant::now();
    let decode_weights = qora::save::load_model(&load_path)
        .expect("Failed to load .qor3b model");
    let mem_mb = decode_weights.memory_bytes() / (1024 * 1024);
    eprintln!("Model loaded in {:.1?} ({} format, {mem_mb} MB)", t0.elapsed(), decode_weights.format_name());

    // Architecture constants
    let num_kv_heads = 4;
    let head_dim = 128;
    let num_layers = decode_weights.num_layers();

    // Decode-only benchmark
    {
        eprintln!("Decode-only benchmark ({} weights)...", decode_weights.format_name());
        let mut kv_cache = qora::gemv::empty_kv_cache(num_layers, num_kv_heads, head_dim);
        let t = Instant::now();
        let _logits = qora::gemv::forward_decode_raw(&decode_weights, 1, &mut kv_cache);
        eprintln!("  forward_decode_raw: {:.1?}", t.elapsed());

        let mut kv_cache2 = qora::gemv::empty_kv_cache(num_layers, num_kv_heads, head_dim);
        let t = Instant::now();
        let _logits = qora::gemv::forward_decode_raw(&decode_weights, 1, &mut kv_cache2);
        eprintln!("  forward_decode_raw (warm): {:.1?}", t.elapsed());
    }

    // Load tokenizer — look next to the .qor3b file first, then current dir
    let tokenizer_path = load_path.parent()
        .unwrap_or(std::path::Path::new("."))
        .join("tokenizer.json");
    let tokenizer = qora::tokenizer::QoraTokenizer::from_file(&tokenizer_path)
        .expect("Failed to load tokenizer");

    let eos_token_id = 128012u32; // IM_END

    let temperature = if greedy { 0.0 } else if no_think { 0.7 } else { 0.6 };

    let params = qora::generate::GenerateParams {
        max_new_tokens: max_tokens,
        eos_token_id,
        temperature,
        top_p: 0.95,
        top_k: 20,
        think: if raw_mode { true } else { !no_think },
        show_think,
        repetition_penalty: 1.1,
        presence_penalty: 1.5,
        think_budget,
    };

    // === Try GPU inference (auto-detect) ===
    #[cfg(any(feature = "gpu", feature = "gpu-metal"))]
    if !force_cpu {
        eprintln!("Attempting GPU inference...");
        match qora::gpu_inference::generate_gpu(&decode_weights, &tokenizer, &prompt, &params) {
            Ok(()) => return,
            Err(e) => eprintln!("GPU not available ({}), falling back to CPU", e),
        }
    }

    // === CPU inference ===
    if raw_mode {
        qora::generate::generate_raw(&decode_weights, &tokenizer, &prompt, &params);
    } else {
        qora::generate::generate(&decode_weights, &tokenizer, &prompt, &params);
    }
}
