//! Whisper decoder: autoregressive with self-attention + cross-attention.
//!
//! Self-attention: causal, with KV cache (grows each step).
//! Cross-attention: K/V from encoder output (cached after first computation).
//! Note: k_proj has NO bias in Whisper.

use crate::weights::{DecoderWeights, DecoderLayerWeights};
use crate::encoder::{gemm_bias, gemm_nobias, gemv_bias, gemv_nobias, layer_norm_inplace};

/// KV cache for the decoder.
pub struct DecoderKvCache {
    /// Self-attention KV cache per layer: (k_data, v_data)
    pub self_kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Cross-attention KV cache per layer: (k_data, v_data)
    pub cross_kv: Vec<(Vec<f32>, Vec<f32>)>,
    pub self_seq_len: usize,
    pub cross_seq_len: usize,
}

impl DecoderKvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            self_kv: (0..n_layers).map(|_| (Vec::new(), Vec::new())).collect(),
            cross_kv: (0..n_layers).map(|_| (Vec::new(), Vec::new())).collect(),
            self_seq_len: 0,
            cross_seq_len: 0,
        }
    }
}

/// Compute cross-attention KV cache from encoder output (called once).
pub fn compute_cross_kv(
    encoder_output: &[f32],
    enc_seq_len: usize,
    weights: &DecoderWeights,
    d_model: usize,
    cache: &mut DecoderKvCache,
) {
    for (i, layer) in weights.layers.iter().enumerate() {
        // K has NO bias, V has bias
        let k = gemm_nobias(encoder_output, enc_seq_len, d_model,
            &layer.xk_proj_w, d_model);
        let v = gemm_bias(encoder_output, enc_seq_len, d_model,
            &layer.xv_proj_w, d_model, &layer.xv_proj_b);
        cache.cross_kv[i] = (k, v);
    }
    cache.cross_seq_len = enc_seq_len;
}

/// Run one decoder step (single token).
/// Returns logits [vocab_size].
/// proj_out = embed_tokens (tied weights, [vocab_size, d_model]).
pub fn decoder_step(
    token_id: u32,
    position: usize,
    weights: &DecoderWeights,
    d_model: usize,
    vocab_size: usize,
    n_heads: usize,
    head_dim: usize,
    cache: &mut DecoderKvCache,
) -> Vec<f32> {
    // 1. Token embedding + positional embedding
    let mut hidden = vec![0.0f32; d_model];
    let tok_off = token_id as usize * d_model;
    let pos_off = position * d_model;
    for d in 0..d_model {
        hidden[d] = weights.embed_tokens[tok_off + d] + weights.embed_positions[pos_off + d];
    }

    // 2. Decoder layers
    for (i, layer) in weights.layers.iter().enumerate() {
        hidden = decoder_layer_step(
            &hidden, layer, d_model, n_heads, head_dim, cache, i,
        );
    }

    // 3. Final layer norm
    layer_norm_inplace(&mut hidden, &weights.ln_w, &weights.ln_b);

    // 4. Project to logits using embed_tokens as proj_out (tied)
    // embed_tokens is [vocab_size, d_model], logits = hidden @ embed_tokens^T
    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let mut sum = 0.0f32;
        for d in 0..d_model {
            sum += hidden[d] * weights.embed_tokens[v * d_model + d];
        }
        logits[v] = sum;
    }

    logits
}

/// Single decoder layer step.
fn decoder_layer_step(
    input: &[f32],
    layer: &DecoderLayerWeights,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
    cache: &mut DecoderKvCache,
    layer_idx: usize,
) -> Vec<f32> {
    let mut hidden = input.to_vec();

    // === Self-attention ===
    {
        let mut normed = hidden.clone();
        layer_norm_inplace(&mut normed, &layer.sa_ln_w, &layer.sa_ln_b);

        let q = gemv_bias(&normed, &layer.q_proj_w, d_model, d_model, &layer.q_proj_b);
        let k = gemv_nobias(&normed, &layer.k_proj_w, d_model, d_model); // NO bias
        let v = gemv_bias(&normed, &layer.v_proj_w, d_model, d_model, &layer.v_proj_b);

        cache.self_kv[layer_idx].0.extend_from_slice(&k);
        cache.self_kv[layer_idx].1.extend_from_slice(&v);
        let kv_len = cache.self_kv[layer_idx].0.len() / d_model;

        let attn_out = cached_attention(
            &q, &cache.self_kv[layer_idx].0, &cache.self_kv[layer_idx].1,
            kv_len, d_model, n_heads, head_dim,
        );

        let projected = gemv_bias(&attn_out, &layer.o_proj_w, d_model, d_model, &layer.o_proj_b);
        for d in 0..d_model {
            hidden[d] += projected[d];
        }
    }

    // === Cross-attention ===
    {
        let mut normed = hidden.clone();
        layer_norm_inplace(&mut normed, &layer.xa_ln_w, &layer.xa_ln_b);

        let q = gemv_bias(&normed, &layer.xq_proj_w, d_model, d_model, &layer.xq_proj_b);

        let cross_k = &cache.cross_kv[layer_idx].0;
        let cross_v = &cache.cross_kv[layer_idx].1;
        let enc_len = cache.cross_seq_len;

        let attn_out = cached_attention(
            &q, cross_k, cross_v,
            enc_len, d_model, n_heads, head_dim,
        );

        let projected = gemv_bias(&attn_out, &layer.xo_proj_w, d_model, d_model, &layer.xo_proj_b);
        for d in 0..d_model {
            hidden[d] += projected[d];
        }
    }

    // === FFN ===
    {
        let mut normed = hidden.clone();
        layer_norm_inplace(&mut normed, &layer.ff_ln_w, &layer.ff_ln_b);

        let ffn_dim = layer.fc1_b.len();
        let mut h = gemv_bias(&normed, &layer.fc1_w, d_model, ffn_dim, &layer.fc1_b);
        for v in &mut h {
            *v = gelu(*v);
        }
        let ffn_out = gemv_bias(&h, &layer.fc2_w, ffn_dim, d_model, &layer.fc2_b);

        for d in 0..d_model {
            hidden[d] += ffn_out[d];
        }
    }

    hidden
}

/// Attention: single query Q[d_model] attending to cached K[kv_len, d_model] and V[kv_len, d_model].
fn cached_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_len: usize,
    d_model: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; d_model];

    for h in 0..n_heads {
        let h_off = h * head_dim;

        let mut scores = vec![0.0f32; kv_len];
        for ki in 0..kv_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[h_off + d] * k_cache[ki * d_model + h_off + d];
            }
            scores[ki] = dot * scale;
        }

        softmax_inplace(&mut scores);

        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for ki in 0..kv_len {
                sum += scores[ki] * v_cache[ki * d_model + h_off + d];
            }
            output[h_off + d] = sum;
        }
    }

    output
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax_inplace(x: &mut [f32]) {
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}
