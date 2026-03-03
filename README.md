# QORA - Native Rust LLM Inference Engine

<img width="1395" height="926" alt="Screenshot 2026-02-27 174517" src="https://github.com/user-attachments/assets/b02d0a77-bf76-40fd-b61e-bb19dc868d88" />

## Downlod 🤗: https://huggingface.co/qoranet/QORA-LLM

Pure Rust inference engine for the SmolLM3-3B language model. No Python runtime, no CUDA, no external dependencies. Single executable + quantized weights = portable AI on any machine.

## Overview

| Property | Value |
|----------|-------|
| **Engine** | QORA (Pure Rust) |

| **Base Model** | SmolLM3-3B (HuggingFaceTB/SmolLM3-3B) |
| **Parameters** | 3.07 Billion |
| **Quantization** | Q4 (4-bit symmetric, group_size=32) |
| **Model Size** | 1.68 GB (Q4) / ~6 GB (F16) |
| **Executable** | 6.7 MB |
| **Context Length** | 65,536 tokens (up to 128K with YARN) |
| **Platform** | Windows x86_64 (CPU-only) |

## Architecture

SmolLM3-3B is a decoder-only transformer with several advanced features:

| Component | Details |
|-----------|---------|
| **Layers** | 36 decoder layers |
| **Hidden Size** | 2,048 |
| **Attention Heads** | 16 (Query) / 4 (KV) — Grouped Query Attention |
| **Head Dimension** | 128 |
| **MLP (Intermediate)** | 11,008 (SwiGLU: gate + up + down) |
| **Vocabulary** | 128,256 tokens |
| **Normalization** | RMSNorm (eps=1e-6) |
| **Position Encoding** | NoPE scheme — RoPE on every 4th layer only (9/36 layers) |
| **RoPE Theta** | 5,000,000 |
| **Activation** | SiLU (Sigmoid Linear Unit) |
| **Embeddings** | Tied (input = output projection) |

### Key Architectural Innovation: NoPE (No Position Encoding)

SmolLM3 uses a 3:1 NoPE ratio — 75% of layers have **no positional encoding** at all. Only layers 3, 7, 11, 15, 19, 23, 27, 31, 35 apply RoPE. This reduces computational overhead and enables better long-context generalization.

## Files

```
model/
  qora.exe          — 6.7 MB    Inference engine (single binary)
  model.qora        — 1.68 GB   Q4 quantized weights (4-bit)
  tokenizer.json    — 16.4 MB   Tokenizer vocabulary
  README.md         — This file
```

## Usage

```bash
# Basic chat (with thinking mode)
qora.exe --load model.qora --prompt "What is photosynthesis?"

# Direct answer (no thinking)
qora.exe --load model.qora --prompt "What is the capital of France?" --no-think

# Greedy decoding (deterministic)
qora.exe --load model.qora --prompt "Write hello world in Python" --greedy --no-think

# Control output length
qora.exe --load model.qora --prompt "Tell me a story" --max-tokens 512

# Raw text completion (no chat template)
qora.exe --load model.qora --prompt "Once upon a time" --raw --max-tokens 128
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--load <path>` | — | Load from .qora binary (fast, ~2s) |
| `--model-path <path>` | `.` | Path to safetensors model directory |
| `--prompt <text>` | "Hello, how are you?" | Input prompt |
| `--max-tokens <n>` | 256 | Maximum tokens to generate |
| `--no-think` | off | Disable reasoning/thinking mode |
| `--greedy` | off | Greedy decoding (temperature=0) |
| `--raw` | off | Raw text completion (no chat template) |
| `--f16` | off | Use F16 weights instead of Q4 |
| `--save <path>` | — | Save model as .qora binary |

## Performance Benchmarks

**Test Hardware:** Windows 11, CPU-only (no GPU acceleration)

### Inference Speed

| Metric | Value |
|--------|-------|
| **Model Load (binary)** | ~2-5s (single instance) |
| **Prefill Speed** | ~0.5 tok/s (123 tokens in ~270s) |
| **Decode Speed (warm)** | ~3.7s per token (single decode) |
| **Decode Throughput** | 0.20-0.29 tok/s (sustained generation) |
| **Memory (Q4)** | 1,681 MB |
| **Memory (F16)** | ~6,000 MB |

### Quality Test Results

All tests run with `--greedy --no-think` for deterministic, reproducible results.

#### Test 1: Factual Knowledge

**Prompt:** "What is the capital of France? Answer in one sentence."

**Output:**
> The capital of France is Paris.

| Metric | Value |
|--------|-------|
| Correctness | PASS |
| Tokens Generated | 7 |
| Prefill | 123 tokens in 269.1s |

---

#### Test 2: Mathematical Reasoning

**Prompt:** "Solve step by step: If a train travels at 60 km/h for 2.5 hours, how far does it travel?"

**Output:**
> To solve this problem, we can use the formula for distance:
>
> Distance = Speed x Time
>
> Given:
> Speed = 60 km/h
> Time = 2.5 hours
>
> Step 1: Plug the values into the formula
> Distance = 60 km/h x 2.5 hours
>
> Step 2: Perform the multiplication
> Distance = 150 km
>
> So, the train travels 150 kilometers in 2.5 hours.

| Metric | Value |
|--------|-------|
| Correctness | PASS (150 km is correct) |
| Tokens Generated | 110 |
| Decode Speed | 0.20 tok/s |

---

#### Test 3: Code Generation

**Prompt:** "Write a Python function to check if a number is prime. Include docstring."

**Output:**
```python
def is_prime(n):
    """
    Checks if a number is prime.

    Parameters:
    n (int): The number to check.

    Returns:
    bool: True if the number is prime, False otherwise.

    Raises:
    ValueError: If n is less than 1, since prime numbers are
    defined for natural numbers greater than 1.
    """
    if n < 1:
        raise ValueError("Prime numbers are defined for natural numbers greater than 1.")
    if n == 1 or n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
```


| Metric | Value |
|--------|-------|
| Correctness | PARTIAL (n==1 should return False; otherwise correct algorithm) |
| Tokens Generated | 200 |
| Decode Speed | 0.29 tok/s |
| Code Quality | Good structure, docstring, error handling, efficient trial division |

---

#### Test 4: Explanation / Simplification

**Prompt:** "Explain quantum computing in 3 sentences for a 10-year-old."

**Output:**
> Quantum computing is like a super-fast version of regular computers that use tiny particles called qubits to perform calculations much faster than today's computers. In quantum computing, these qubits can be in multiple states at once, allowing them to solve complex problems much more quickly than classical computers. However, quantum computing is still a new and complex field that is still being developed, so it's like a futuristic computer that is still in the process of learning and growing!

| Metric | Value |
|--------|-------|
| Correctness | PASS |
| Age-Appropriate | Yes (simple language, relatable analogy) |
| Tokens Generated | 92 |
| Decode Speed | 0.24 tok/s |

---

#### Test 5: Science Knowledge

**Prompt:** "What are the three laws of thermodynamics? Be concise."

**Output:**
> 1. **First Law of Thermodynamics**: The law of energy conservation states that energy cannot be created or destroyed, only converted from one form to another.
>
> 2. **Second Law of Thermodynamics**: This law describes the direction of spontaneous processes, stating that the entropy (a measure of disorder or randomness) of an isolated system always increases over time, or that the entropy of a closed system will always increase or remain constant.
>
> 3. **Third Law of Thermodynamics**: This law states that as the temperature of a system approaches absolute zero, the entropy of the system approaches a constant value...

| Metric | Value |
|--------|-------|
| Correctness | PASS (all three laws accurately described) |
| Tokens Generated | 128 |
| Decode Speed | 0.25 tok/s |

---

#### Test 6: Logical Reasoning

**Prompt:** "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."

**Output:**
> Yes, we can conclude that some roses fade quickly based on the given information.
>
> Reasoning:
> 1. All roses are flowers (given).
> 2. Some flowers fade quickly (given).
> 3. Since roses are a subset of flowers, any characteristic or property that applies to flowers can also be applied to roses.
>
> Therefore, since some flowers fade quickly, and roses are flowers, it logically follows that some roses also fade quickly.

| Metric | Value |
|--------|-------|
| Correctness | FAIL (this is a syllogistic fallacy — "some flowers fade" does not imply roses specifically fade) |
| Tokens Generated | 128 |
| Decode Speed | 0.25 tok/s |
| Note | This is a known weakness of small LMs — the correct answer is "No, we cannot conclude this" |

### Quality Summary

| Test Category | Result | Score |
|---------------|--------|-------|
| Factual Knowledge | PASS | 1/1 |
| Math Reasoning | PASS | 1/1 |
| Code Generation | PARTIAL | 0.5/1 |
| Explanation | PASS | 1/1 |
| Science Knowledge | PASS | 1/1 |
| Logical Reasoning | FAIL | 0/1 |
| **Total** | | **4.5/6 (75%)** |

## Published Benchmark Scores (SmolLM3-3B Base Model)

Official scores from the HuggingFace model card. QORA runs the same weights with Q4 quantization (minimal accuracy loss).

### Reasoning and Commonsense

| Benchmark | SmolLM3-3B | Qwen2.5-3B | Llama3.2-3B | Qwen3-4B |
|-----------|-----------|-----------|-----------|---------|
| **HellaSwag** | **76.15** | 74.19 | 75.52 | 74.37 |
| **ARC-CF** | **65.61** | 59.81 | 58.58 | 62.11 |
| **BoolQ** | **78.99** | 73.61 | 75.33 | 74.28 |
| **PIQA** | **78.89** | 78.35 | 78.51 | 77.58 |
| **Winogrande** | 58.88 | **61.41** | 58.72 | 59.59 |
| **CommonsenseQA** | 55.28 | 49.14 | **60.60** | 52.99 |

### Knowledge and Understanding

| Benchmark | SmolLM3-3B | Qwen2.5-3B | Llama3.2-3B | Qwen3-4B |
|-----------|-----------|-----------|-----------|---------|
| **MMLU-CF** | 44.13 | 42.93 | 41.32 | **47.65** |
| **MMLU Pro CF** | 19.61 | 16.66 | 16.42 | **24.92** |
| **MMLU Pro MCF** | 32.70 | 31.32 | 25.07 | **41.07** |
| **OpenBookQA** | 40.60 | 40.20 | 42.00 | **42.40** |

### Math and Code

| Benchmark | SmolLM3-3B | Qwen2.5-3B | Llama3.2-3B | Qwen3-4B |
|-----------|-----------|-----------|-----------|---------|
| **HumanEval+** | 30.48 | 34.14 | 25.00 | **54.87** |
| **MBPP+** | 52.91 | 52.11 | 38.88 | **63.75** |
| **MATH (4-shot)** | 46.10 | 40.10 | 7.44 | **51.20** |
| **GSM8K (5-shot)** | 67.63 | 70.13 | 25.92 | **74.14** |

### Instruction Following (Chat Model)

| Benchmark | SmolLM3-3B | Qwen2.5-3B | Llama3.1-3B | Qwen3-4B |
|-----------|-----------|-----------|-----------|---------|
| **IFEval** | **76.7** | 65.6 | 71.6 | 68.9 |
| **AIME 2025** | 9.3 | 2.9 | 0.3 | **17.1** |
| **GSM-Plus** | 72.8 | 74.1 | 59.2 | **82.1** |
| **LiveCodeBench** | 15.2 | 10.5 | 3.4 | **24.9** |
| **GPQA Diamond** | 35.7 | 32.2 | 29.4 | **44.4** |
| **Global MMLU** | 53.5 | 50.54 | 46.8 | **65.1** |
| **BFCL (Tools)** | 92.3 | — | 92.3 | **95.0** |

### Extended Thinking Mode

| Benchmark | No Think | With Think | Improvement |
|-----------|----------|------------|-------------|
| **AIME 2025** | 9.3 | **36.7** | +295% |
| **GSM-Plus** | 72.8 | **83.4** | +15% |
| **LiveCodeBench** | 15.2 | **30.0** | +97% |
| **GPQA Diamond** | 35.7 | **41.7** | +17% |
| **Global MMLU** | 53.5 | **64.1** | +20% |

### Long Context

| Benchmark | SmolLM3-3B | Qwen2.5-3B | Llama3.2-3B | Qwen3-4B |
|-----------|-----------|-----------|-----------|---------|
| **RULER 32K** | 76.35 | 75.93 | 77.58 | **83.98** |
| **RULER 64K** | 67.85 | 64.90 | **72.93** | 60.29 |
| **RULER 128K** | 61.03 | 62.23 | **71.30** | 47.23 |

### Multilingual (HellaSwag)

| Language | SmolLM3-3B | Qwen2.5-3B | Llama3.2-3B | Qwen3-4B |
|----------|-----------|-----------|-----------|---------|
| **French** | **63.94** | 57.47 | 57.66 | 61.00 |
| **Spanish** | **65.85** | 58.25 | 59.39 | 61.85 |
| **German** | **59.56** | 49.99 | 53.19 | 56.43 |
| **Italian** | **62.49** | 53.21 | 54.96 | 58.76 |
| **Portuguese** | **63.22** | 57.38 | 56.84 | 59.89 |

## Model Comparison

| Model | Params | Format | Size on Disk | Best At |
|-------|--------|--------|-------------|---------|
| **QORA (SmolLM3-3B)** | 3.07B | Q4 | 1.68 GB | Reasoning, multilingual, instruction following |
| Qwen2.5-3B | 3B | — | ~6 GB | Math (GSM8K), Winogrande |
| Llama3.2-3B | 3.2B | — | ~6 GB | Long context (128K), CommonsenseQA |
| Qwen3-4B | 4B | — | ~8 GB | Overall best (larger model), math, code |

### Why SmolLM3-3B?

- **Best-in-class reasoning** among 3B models (HellaSwag 76.15, ARC 65.61, BoolQ 78.99)
- **Best instruction following** (IFEval 76.7) — beats even Qwen3-4B
- **Best multilingual** performance among 3B models across 5 European languages
- **Thinking mode** boosts AIME by 295% — competitive reasoning from a 3B model
- **128K context** support with strong RULER scores

## Technical Details

### Quantization

QORA uses symmetric 4-bit quantization with group_size=32:
- Each group of 32 float values is quantized to 4-bit integers
- One f32 scale factor per group
- Total: 4 bits/weight + 1 bit/weight overhead = ~5 bits effective
- Memory reduction: 32-bit -> ~5 bits = **6.4x compression**

### Inference Pipeline

```
1. Model Load    — Read .qora binary (Q4 weights + f16 norms)
2. Tokenize      — Encode prompt with chat template
3. Prefill       — Process full prompt through 36 layers (batched)
4. Decode Loop   — Generate tokens one at a time:
   a. Embedding lookup
   b. 36x: RMSNorm -> Attention (GQA, KV cache) -> RMSNorm -> SwiGLU MLP
   c. Final RMSNorm -> LM head (tied weights)
   d. Sample (top-p, temperature)
5. Detokenize    — Decode token IDs back to text
```

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.6 | Controls randomness (0 = greedy) |
| Top-P | 0.95 | Nucleus sampling threshold |
| Max Tokens | 256 | Maximum generation length |

## QORA Model Family

| Engine | Model | Params | Size (Q4) | Purpose |
|--------|-------|--------|-----------|---------|
| **QORA** | SmolLM3-3B | 3.07B | 1.68 GB | Text generation, reasoning, chat |
| **QORA-TTS** | Qwen3-TTS | — | — | Text-to-speech synthesis |
| **QORA-Vision (Image)** | SigLIP 2 Base | 86M | 58 MB | Image embeddings, zero-shot classification |
| **QORA-Vision (Video)** | ViViT Base | 89M | 60 MB | Video action classification (400 classes) |

All engines are pure Rust, CPU-only, single-binary executables with no Python dependencies.

## Building from Source

```bash
cd QOR3B
cargo build --release

# Convert from safetensors to .qora binary:
./target/release/qora.exe --model-path ../SmolLM3-3B/ --save model/model.qora
```

### Dependencies

- `burn` — Rust deep learning framework (for initial weight loading)
- `half` — F16 support
- `serde` / `serde_json` — Config parsing
- `safetensors` — HuggingFace weight format
- `tokenizers` — HuggingFace tokenizer

## License

The QORA inference engine is custom-built. The SmolLM3-3B model weights are released under the [SmolLM3 License](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) by HuggingFace.

---

*Built with QORA — Pure Rust AI Inference*
