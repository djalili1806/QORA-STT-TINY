//! Load Whisper weights from safetensors format.
//!
//! Weight keys:
//!   model.encoder.conv1.{weight,bias}
//!   model.encoder.conv2.{weight,bias}
//!   model.encoder.embed_positions.weight
//!   model.encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
//!   model.encoder.layers.{i}.self_attn.out_proj.{weight,bias}
//!   model.encoder.layers.{i}.self_attn_layer_norm.{weight,bias}
//!   model.encoder.layers.{i}.fc1.{weight,bias}
//!   model.encoder.layers.{i}.fc2.{weight,bias}
//!   model.encoder.layers.{i}.final_layer_norm.{weight,bias}
//!   model.encoder.layer_norm.{weight,bias}
//!   model.decoder.embed_tokens.weight
//!   model.decoder.embed_positions.weight
//!   model.decoder.layers.{i}.*
//!   model.decoder.layer_norm.{weight,bias}
//!   proj_out.weight

use std::path::Path;
use safetensors::SafeTensors;

use crate::config::WhisperConfig;
use crate::weights::*;

pub fn load_weights(
    model_path: &Path,
    config: &WhisperConfig,
) -> Result<WhisperWeights, Box<dyn std::error::Error>> {
    // Find safetensors file
    let st_path = model_path.join("model.safetensors");
    if !st_path.exists() {
        return Err(format!("model.safetensors not found in {}", model_path.display()).into());
    }

    let data = std::fs::read(&st_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    eprintln!("  Found {} tensors in safetensors", tensors.names().len());

    // Helper to load a tensor as Vec<f32>
    let load = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let view = tensors.tensor(name).map_err(|e| format!("{name}: {e}"))?;
        Ok(tensor_to_f32(view.data(), view.dtype()))
    };

    // Helper to load and transpose a weight matrix [out, in] → [in, out] (row-major for GEMV)
    let load_t = |name: &str| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let view = tensors.tensor(name).map_err(|e| format!("{name}: {e}"))?;
        let shape = view.shape();
        let flat = tensor_to_f32(view.data(), view.dtype());
        if shape.len() == 2 {
            Ok(transpose_2d(&flat, shape[0], shape[1]))
        } else {
            Ok(flat)
        }
    };

    // === Encoder ===
    eprintln!("  Loading encoder...");
    let conv1_w = load("model.encoder.conv1.weight")?;   // [384, 80, 3]
    let conv1_b = load("model.encoder.conv1.bias")?;
    let conv2_w = load("model.encoder.conv2.weight")?;   // [384, 384, 3]
    let conv2_b = load("model.encoder.conv2.bias")?;
    let enc_pos = load("model.encoder.embed_positions.weight")?;

    let mut enc_layers = Vec::with_capacity(config.encoder_layers);
    for i in 0..config.encoder_layers {
        let p = format!("model.encoder.layers.{i}");
        enc_layers.push(EncoderLayerWeights {
            q_proj_w: load_t(&format!("{p}.self_attn.q_proj.weight"))?,
            q_proj_b: load(&format!("{p}.self_attn.q_proj.bias"))?,
            k_proj_w: load_t(&format!("{p}.self_attn.k_proj.weight"))?,
            // k_proj has NO bias in Whisper
            v_proj_w: load_t(&format!("{p}.self_attn.v_proj.weight"))?,
            v_proj_b: load(&format!("{p}.self_attn.v_proj.bias"))?,
            o_proj_w: load_t(&format!("{p}.self_attn.out_proj.weight"))?,
            o_proj_b: load(&format!("{p}.self_attn.out_proj.bias"))?,
            sa_ln_w: load(&format!("{p}.self_attn_layer_norm.weight"))?,
            sa_ln_b: load(&format!("{p}.self_attn_layer_norm.bias"))?,
            fc1_w: load_t(&format!("{p}.fc1.weight"))?,
            fc1_b: load(&format!("{p}.fc1.bias"))?,
            fc2_w: load_t(&format!("{p}.fc2.weight"))?,
            fc2_b: load(&format!("{p}.fc2.bias"))?,
            ff_ln_w: load(&format!("{p}.final_layer_norm.weight"))?,
            ff_ln_b: load(&format!("{p}.final_layer_norm.bias"))?,
        });
    }

    let enc_ln_w = load("model.encoder.layer_norm.weight")?;
    let enc_ln_b = load("model.encoder.layer_norm.bias")?;

    // === Decoder ===
    eprintln!("  Loading decoder...");
    let dec_embed = load("model.decoder.embed_tokens.weight")?;
    let dec_pos = load("model.decoder.embed_positions.weight")?;

    let mut dec_layers = Vec::with_capacity(config.decoder_layers);
    for i in 0..config.decoder_layers {
        let p = format!("model.decoder.layers.{i}");
        dec_layers.push(DecoderLayerWeights {
            q_proj_w: load_t(&format!("{p}.self_attn.q_proj.weight"))?,
            q_proj_b: load(&format!("{p}.self_attn.q_proj.bias"))?,
            k_proj_w: load_t(&format!("{p}.self_attn.k_proj.weight"))?,
            v_proj_w: load_t(&format!("{p}.self_attn.v_proj.weight"))?,
            v_proj_b: load(&format!("{p}.self_attn.v_proj.bias"))?,
            o_proj_w: load_t(&format!("{p}.self_attn.out_proj.weight"))?,
            o_proj_b: load(&format!("{p}.self_attn.out_proj.bias"))?,
            sa_ln_w: load(&format!("{p}.self_attn_layer_norm.weight"))?,
            sa_ln_b: load(&format!("{p}.self_attn_layer_norm.bias"))?,
            xq_proj_w: load_t(&format!("{p}.encoder_attn.q_proj.weight"))?,
            xq_proj_b: load(&format!("{p}.encoder_attn.q_proj.bias"))?,
            xk_proj_w: load_t(&format!("{p}.encoder_attn.k_proj.weight"))?,
            xv_proj_w: load_t(&format!("{p}.encoder_attn.v_proj.weight"))?,
            xv_proj_b: load(&format!("{p}.encoder_attn.v_proj.bias"))?,
            xo_proj_w: load_t(&format!("{p}.encoder_attn.out_proj.weight"))?,
            xo_proj_b: load(&format!("{p}.encoder_attn.out_proj.bias"))?,
            xa_ln_w: load(&format!("{p}.encoder_attn_layer_norm.weight"))?,
            xa_ln_b: load(&format!("{p}.encoder_attn_layer_norm.bias"))?,
            fc1_w: load_t(&format!("{p}.fc1.weight"))?,
            fc1_b: load(&format!("{p}.fc1.bias"))?,
            fc2_w: load_t(&format!("{p}.fc2.weight"))?,
            fc2_b: load(&format!("{p}.fc2.bias"))?,
            ff_ln_w: load(&format!("{p}.final_layer_norm.weight"))?,
            ff_ln_b: load(&format!("{p}.final_layer_norm.bias"))?,
        });
    }

    let dec_ln_w = load("model.decoder.layer_norm.weight")?;
    let dec_ln_b = load("model.decoder.layer_norm.bias")?;

    // Output projection is tied to embed_tokens (no separate proj_out in whisper-tiny)
    let d = config.d_model;

    Ok(WhisperWeights {
        encoder: EncoderWeights {
            conv1_w, conv1_b, conv2_w, conv2_b,
            embed_positions: enc_pos,
            layers: enc_layers,
            ln_w: enc_ln_w, ln_b: enc_ln_b,
        },
        decoder: DecoderWeights {
            embed_tokens: dec_embed,
            embed_positions: dec_pos,
            layers: dec_layers,
            ln_w: dec_ln_w, ln_b: dec_ln_b,
        },
        d_model: d,
        vocab_size: config.vocab_size,
        encoder_heads: config.encoder_attention_heads,
        decoder_heads: config.decoder_attention_heads,
        encoder_head_dim: config.encoder_head_dim(),
        decoder_head_dim: config.decoder_head_dim(),
    })
}

// ============================================================
// Helpers
// ============================================================

fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect()
        }
        safetensors::Dtype::BF16 => {
            data.chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        }
        _ => panic!("Unsupported dtype: {:?}", dtype),
    }
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
