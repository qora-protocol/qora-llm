use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct QoraConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub no_rope_layers: Vec<u8>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl QoraConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Returns true if layer `i` should apply RoPE.
    /// In HF transformers: use_rope = config.no_rope_layers[layer_idx]
    /// So 1 = use RoPE, 0 = no RoPE (despite the misleading field name).
    pub fn layer_uses_rope(&self, i: usize) -> bool {
        self.no_rope_layers[i] == 1
    }
}
