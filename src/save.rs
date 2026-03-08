//! Save/load Whisper weights to binary format (.qora-stt).
//!
//! Format: magic "QSTT" + version(u32) + config + encoder + decoder.
//! Note: k_proj has NO bias, proj_out is tied to embed_tokens.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::weights::*;

const MAGIC: &[u8; 4] = b"QSTT";
const VERSION: u32 = 1;

// ============================================================
// Save
// ============================================================

pub fn save_model(weights: &WhisperWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, File::create(path)?);

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;

    w.write_all(&(weights.d_model as u32).to_le_bytes())?;
    w.write_all(&(weights.vocab_size as u32).to_le_bytes())?;
    w.write_all(&(weights.encoder_heads as u32).to_le_bytes())?;
    w.write_all(&(weights.decoder_heads as u32).to_le_bytes())?;
    w.write_all(&(weights.encoder_head_dim as u32).to_le_bytes())?;
    w.write_all(&(weights.decoder_head_dim as u32).to_le_bytes())?;

    eprintln!("  Saving encoder...");
    save_encoder(&mut w, &weights.encoder)?;
    eprintln!("  Saving decoder...");
    save_decoder(&mut w, &weights.decoder)?;

    w.flush()?;
    Ok(())
}

fn save_encoder(w: &mut impl Write, enc: &EncoderWeights) -> io::Result<()> {
    write_f32_vec(w, &enc.conv1_w)?;
    write_f32_vec(w, &enc.conv1_b)?;
    write_f32_vec(w, &enc.conv2_w)?;
    write_f32_vec(w, &enc.conv2_b)?;
    write_f32_vec(w, &enc.embed_positions)?;

    w.write_all(&(enc.layers.len() as u32).to_le_bytes())?;
    for layer in &enc.layers {
        write_f32_vec(w, &layer.q_proj_w)?;
        write_f32_vec(w, &layer.q_proj_b)?;
        write_f32_vec(w, &layer.k_proj_w)?;
        // k_proj has NO bias
        write_f32_vec(w, &layer.v_proj_w)?;
        write_f32_vec(w, &layer.v_proj_b)?;
        write_f32_vec(w, &layer.o_proj_w)?;
        write_f32_vec(w, &layer.o_proj_b)?;
        write_f32_vec(w, &layer.sa_ln_w)?;
        write_f32_vec(w, &layer.sa_ln_b)?;
        write_f32_vec(w, &layer.fc1_w)?;
        write_f32_vec(w, &layer.fc1_b)?;
        write_f32_vec(w, &layer.fc2_w)?;
        write_f32_vec(w, &layer.fc2_b)?;
        write_f32_vec(w, &layer.ff_ln_w)?;
        write_f32_vec(w, &layer.ff_ln_b)?;
    }

    write_f32_vec(w, &enc.ln_w)?;
    write_f32_vec(w, &enc.ln_b)?;
    Ok(())
}

fn save_decoder(w: &mut impl Write, dec: &DecoderWeights) -> io::Result<()> {
    write_f32_vec(w, &dec.embed_tokens)?;
    write_f32_vec(w, &dec.embed_positions)?;

    w.write_all(&(dec.layers.len() as u32).to_le_bytes())?;
    for layer in &dec.layers {
        // Self-attention
        write_f32_vec(w, &layer.q_proj_w)?;
        write_f32_vec(w, &layer.q_proj_b)?;
        write_f32_vec(w, &layer.k_proj_w)?;
        write_f32_vec(w, &layer.v_proj_w)?;
        write_f32_vec(w, &layer.v_proj_b)?;
        write_f32_vec(w, &layer.o_proj_w)?;
        write_f32_vec(w, &layer.o_proj_b)?;
        write_f32_vec(w, &layer.sa_ln_w)?;
        write_f32_vec(w, &layer.sa_ln_b)?;
        // Cross-attention
        write_f32_vec(w, &layer.xq_proj_w)?;
        write_f32_vec(w, &layer.xq_proj_b)?;
        write_f32_vec(w, &layer.xk_proj_w)?;
        write_f32_vec(w, &layer.xv_proj_w)?;
        write_f32_vec(w, &layer.xv_proj_b)?;
        write_f32_vec(w, &layer.xo_proj_w)?;
        write_f32_vec(w, &layer.xo_proj_b)?;
        write_f32_vec(w, &layer.xa_ln_w)?;
        write_f32_vec(w, &layer.xa_ln_b)?;
        // FFN
        write_f32_vec(w, &layer.fc1_w)?;
        write_f32_vec(w, &layer.fc1_b)?;
        write_f32_vec(w, &layer.fc2_w)?;
        write_f32_vec(w, &layer.fc2_b)?;
        write_f32_vec(w, &layer.ff_ln_w)?;
        write_f32_vec(w, &layer.ff_ln_b)?;
    }

    write_f32_vec(w, &dec.ln_w)?;
    write_f32_vec(w, &dec.ln_b)?;
    Ok(())
}

// ============================================================
// Load
// ============================================================

pub fn load_model(path: &Path) -> io::Result<WhisperWeights> {
    let mut r = BufReader::with_capacity(4 * 1024 * 1024, File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic (expected QSTT)"));
    }
    let version = read_u32(&mut r)?;
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Version {version}, expected {VERSION}")));
    }

    let d_model = read_u32(&mut r)? as usize;
    let vocab_size = read_u32(&mut r)? as usize;
    let encoder_heads = read_u32(&mut r)? as usize;
    let decoder_heads = read_u32(&mut r)? as usize;
    let encoder_head_dim = read_u32(&mut r)? as usize;
    let decoder_head_dim = read_u32(&mut r)? as usize;

    eprintln!("  Loading encoder...");
    let encoder = load_encoder(&mut r)?;
    eprintln!("  Loading decoder...");
    let decoder = load_decoder(&mut r)?;

    Ok(WhisperWeights {
        encoder, decoder,
        d_model, vocab_size, encoder_heads, decoder_heads,
        encoder_head_dim, decoder_head_dim,
    })
}

fn load_encoder(r: &mut impl Read) -> io::Result<EncoderWeights> {
    let conv1_w = read_f32_vec(r)?;
    let conv1_b = read_f32_vec(r)?;
    let conv2_w = read_f32_vec(r)?;
    let conv2_b = read_f32_vec(r)?;
    let embed_positions = read_f32_vec(r)?;

    let n_layers = read_u32(r)? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(EncoderLayerWeights {
            q_proj_w: read_f32_vec(r)?,
            q_proj_b: read_f32_vec(r)?,
            k_proj_w: read_f32_vec(r)?,
            v_proj_w: read_f32_vec(r)?,
            v_proj_b: read_f32_vec(r)?,
            o_proj_w: read_f32_vec(r)?,
            o_proj_b: read_f32_vec(r)?,
            sa_ln_w: read_f32_vec(r)?,
            sa_ln_b: read_f32_vec(r)?,
            fc1_w: read_f32_vec(r)?,
            fc1_b: read_f32_vec(r)?,
            fc2_w: read_f32_vec(r)?,
            fc2_b: read_f32_vec(r)?,
            ff_ln_w: read_f32_vec(r)?,
            ff_ln_b: read_f32_vec(r)?,
        });
    }

    let ln_w = read_f32_vec(r)?;
    let ln_b = read_f32_vec(r)?;

    Ok(EncoderWeights {
        conv1_w, conv1_b, conv2_w, conv2_b, embed_positions,
        layers, ln_w, ln_b,
    })
}

fn load_decoder(r: &mut impl Read) -> io::Result<DecoderWeights> {
    let embed_tokens = read_f32_vec(r)?;
    let embed_positions = read_f32_vec(r)?;

    let n_layers = read_u32(r)? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(DecoderLayerWeights {
            q_proj_w: read_f32_vec(r)?,
            q_proj_b: read_f32_vec(r)?,
            k_proj_w: read_f32_vec(r)?,
            v_proj_w: read_f32_vec(r)?,
            v_proj_b: read_f32_vec(r)?,
            o_proj_w: read_f32_vec(r)?,
            o_proj_b: read_f32_vec(r)?,
            sa_ln_w: read_f32_vec(r)?,
            sa_ln_b: read_f32_vec(r)?,
            xq_proj_w: read_f32_vec(r)?,
            xq_proj_b: read_f32_vec(r)?,
            xk_proj_w: read_f32_vec(r)?,
            xv_proj_w: read_f32_vec(r)?,
            xv_proj_b: read_f32_vec(r)?,
            xo_proj_w: read_f32_vec(r)?,
            xo_proj_b: read_f32_vec(r)?,
            xa_ln_w: read_f32_vec(r)?,
            xa_ln_b: read_f32_vec(r)?,
            fc1_w: read_f32_vec(r)?,
            fc1_b: read_f32_vec(r)?,
            fc2_w: read_f32_vec(r)?,
            fc2_b: read_f32_vec(r)?,
            ff_ln_w: read_f32_vec(r)?,
            ff_ln_b: read_f32_vec(r)?,
        });
    }

    let ln_w = read_f32_vec(r)?;
    let ln_b = read_f32_vec(r)?;

    Ok(DecoderWeights {
        embed_tokens, embed_positions, layers, ln_w, ln_b,
    })
}

// ============================================================
// Helpers
// ============================================================

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len * 4];
    r.read_exact(&mut buf)?;
    Ok(buf.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
