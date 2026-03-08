use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

// Special token IDs
pub const IM_START: u32 = 128011;
pub const IM_END: u32 = 128012;
pub const THINK_START: u32 = 128002;
pub const THINK_END: u32 = 128003;

pub struct QoraTokenizer {
    inner: tokenizers::Tokenizer,
}

impl QoraTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let encoding = self.inner.encode(text, false).expect("Failed to encode");
        encoding.get_ids().to_vec()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids, true).expect("Failed to decode")
    }

    /// Build a full chat prompt matching the QORA chat template.
    /// Key: system section has NO <|im_end|> when there are no tools (matches trained format).
    pub fn format_chat(&self, user_message: &str, think: bool, max_tokens: usize) -> Vec<u32> {
        let today = current_date_string();
        let system_content = build_system_prompt(&today, think, max_tokens);

        // Match the exact template format:
        // <|im_start|>system\n...\n\n
        // (NO <|im_end|> for system when no tools!)
        // <|im_start|>user\n{message}<|im_end|>\n
        // <|im_start|>assistant\n[optional: <think>\n\n</think>\n]
        let assistant_suffix = if think {
            ""
        } else {
            "<think>\n\n</think>\n"
        };

        let full_text = format!(
            "<|im_start|>system\n\
             {system_content}\n\
             <|im_start|>user\n\
             {user_message}<|im_end|>\n\
             <|im_start|>assistant\n\
             {assistant_suffix}"
        );

        let encoding = self.inner.encode(full_text, false).expect("Failed to encode chat");
        encoding.get_ids().to_vec()
    }
}

/// Build the system prompt content with metadata, thinking instructions, and length hints.
fn build_system_prompt(today: &str, think: bool, max_tokens: usize) -> String {
    let reasoning_mode = if think { "/think" } else { "/no_think" };

    let thinking_instructions = if think {
        "When solving problems, think step by step in the <think> block.\n\
         Then give ONLY the final answer clearly. Do NOT repeat your reasoning after </think>.\n"
    } else {
        ""
    };

    let length_hint = if max_tokens <= 100 {
        "IMPORTANT: Keep your response very brief — 1-2 sentences only.\n"
    } else if max_tokens <= 300 {
        "IMPORTANT: Keep your response concise — a few sentences. Do not use bullet points or lists.\n"
    } else if max_tokens <= 500 {
        "Keep your response brief — a short paragraph. Avoid lengthy lists or breakdowns.\n"
    } else {
        ""
    };

    format!(
        "## Metadata\n\n\
         Knowledge Cutoff Date: June 2025\n\
         Today Date: {today}\n\
         Reasoning Mode: {reasoning_mode}\n\n\
         ## Instructions\n\n\
         You are QORA, a helpful AI assistant. \
         You provide accurate, clear responses.\n\
         {length_hint}\
         {thinking_instructions}"
    )
}

/// Get current date as "DD Month YYYY" string (e.g. "27 February 2026").
/// Uses Howard Hinnant's algorithm — no external dependencies needed.
fn current_date_string() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let days = (secs / 86400) as i64;

    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let month_name = match m {
        1 => "January", 2 => "February", 3 => "March", 4 => "April",
        5 => "May", 6 => "June", 7 => "July", 8 => "August",
        9 => "September", 10 => "October", 11 => "November", 12 => "December",
        _ => "Unknown",
    };

    format!("{d} {month_name} {y}")
}
