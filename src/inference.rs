//! Full Whisper inference pipeline: audio → mel → encoder → decoder → text.

use std::path::Path;
use std::time::Instant;

use crate::config::{self, WhisperConfig};
use crate::weights::WhisperWeights;
use crate::tokenizer::WhisperTokenizer;

/// Transcribe a WAV audio file to text.
pub fn transcribe(
    weights: &WhisperWeights,
    config: &WhisperConfig,
    tokenizer: &WhisperTokenizer,
    audio_path: &Path,
    language: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let d = weights.d_model;

    // 1. Read and resample audio
    eprintln!("Reading audio...");
    let t0 = Instant::now();
    let audio = crate::wav::read_wav_16khz(audio_path)?;
    eprintln!("  Audio loaded in {:.1?}", t0.elapsed());

    // 2. Extract mel spectrogram
    eprintln!("Extracting mel spectrogram...");
    let t0 = Instant::now();
    let mel = crate::mel::extract_mel(&audio);
    eprintln!("  Mel extracted in {:.1?} ({} mels x {} frames)",
        t0.elapsed(), crate::mel::n_mels(), crate::mel::n_frames());

    // 3. Run encoder
    eprintln!("Running encoder...");
    let t0 = Instant::now();
    let encoder_output = crate::encoder::encoder_forward(
        &mel, &weights.encoder, d, weights.encoder_heads, weights.encoder_head_dim,
    );
    let enc_seq_len = encoder_output.len() / d;
    eprintln!("  Encoder done in {:.1?} (output: [{}, {}])", t0.elapsed(), enc_seq_len, d);

    // 4. Compute cross-attention KV cache
    eprintln!("Computing cross-attention cache...");
    let t0 = Instant::now();
    let mut kv_cache = crate::decoder::DecoderKvCache::new(config.decoder_layers);
    crate::decoder::compute_cross_kv(
        &encoder_output, enc_seq_len, &weights.decoder, d, &mut kv_cache,
    );
    eprintln!("  Cross-attention cached in {:.1?}", t0.elapsed());

    // 5. Autoregressive decoding
    eprintln!("Decoding...");
    let t0 = Instant::now();

    // Build initial tokens: SOT + language + transcribe + no_timestamps
    let lang_token = config::language_token(language);
    let forced_tokens: Vec<u32> = vec![
        config::SOT,
        lang_token,
        config::TRANSCRIBE,
        config::NO_TIMESTAMPS,
    ];

    let mut tokens: Vec<u32> = Vec::new();
    let max_tokens = config.max_target_positions;

    // Process forced tokens first
    for (i, &tok) in forced_tokens.iter().enumerate() {
        tokens.push(tok);

        // Run decoder to build KV cache, but don't use logits for forced tokens
        let _logits = crate::decoder::decoder_step(
            tok,
            i,
            &weights.decoder,
            d,
            weights.vocab_size,
            weights.decoder_heads,
            weights.decoder_head_dim,
            &mut kv_cache,
        );
    }
    kv_cache.self_seq_len = forced_tokens.len();

    // Autoregressive loop
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut step_times = Vec::new();

    for step in 0..(max_tokens - forced_tokens.len()) {
        let pos = forced_tokens.len() + step;
        let prev_token = if step == 0 {
            *forced_tokens.last().unwrap()
        } else {
            generated_tokens[step - 1]
        };

        let step_t = Instant::now();
        let mut logits = crate::decoder::decoder_step(
            prev_token,
            pos,
            &weights.decoder,
            d,
            weights.vocab_size,
            weights.decoder_heads,
            weights.decoder_head_dim,
            &mut kv_cache,
        );

        // Apply suppress_tokens
        for &tok in &config.suppress_tokens {
            if tok < logits.len() {
                logits[tok] = f32::NEG_INFINITY;
            }
        }

        // Greedy: argmax
        let next_token = argmax(&logits);
        let step_ms = step_t.elapsed().as_millis();
        step_times.push(step_ms);

        if next_token == config::EOT {
            eprintln!("  EOT at step {} ({:.1?} total)", step, t0.elapsed());
            break;
        }

        generated_tokens.push(next_token);

        // Print progress every 10 tokens
        if step % 10 == 0 && step > 0 {
            let avg_ms: u128 = step_times.iter().sum::<u128>() / step_times.len() as u128;
            eprintln!("  Step {}: avg {avg_ms}ms/token", step);
        }
    }

    let total_time = t0.elapsed();
    let avg_ms = if !step_times.is_empty() {
        step_times.iter().sum::<u128>() / step_times.len() as u128
    } else { 0 };
    eprintln!("  Decoded {} tokens in {:.1?} (avg {}ms/token)",
        generated_tokens.len(), total_time, avg_ms);

    // 6. Decode tokens to text
    let text = tokenizer.decode(&generated_tokens);

    Ok(text)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}
