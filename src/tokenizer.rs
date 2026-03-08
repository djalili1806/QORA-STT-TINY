//! Whisper tokenizer wrapper using HuggingFace tokenizers crate.

use std::path::Path;

pub struct WhisperTokenizer {
    inner: tokenizers::Tokenizer,
}

impl WhisperTokenizer {
    pub fn from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
        Ok(Self { inner: tokenizer })
    }

    /// Decode token IDs to text, skipping special tokens (>= 50257).
    pub fn decode(&self, tokens: &[u32]) -> String {
        // Filter out special tokens
        let text_tokens: Vec<u32> = tokens.iter()
            .copied()
            .filter(|&t| t < 50257)
            .collect();

        if text_tokens.is_empty() {
            return String::new();
        }

        self.inner.decode(&text_tokens, true)
            .unwrap_or_else(|_| String::from("[decode error]"))
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.inner.encode(text, false) {
            Ok(encoding) => encoding.get_ids().to_vec(),
            Err(_) => Vec::new(),
        }
    }
}
