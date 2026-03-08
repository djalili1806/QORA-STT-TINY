//! Weight structures for Whisper Tiny. All f32 (model is only ~151MB).
//! Note: k_proj has NO bias in Whisper (q, v, out all have bias).

pub struct WhisperWeights {
    pub encoder: EncoderWeights,
    pub decoder: DecoderWeights,
    pub d_model: usize,
    pub vocab_size: usize,
    pub encoder_heads: usize,
    pub decoder_heads: usize,
    pub encoder_head_dim: usize,
    pub decoder_head_dim: usize,
}

pub struct EncoderWeights {
    pub conv1_w: Vec<f32>,          // [384, 80, 3]
    pub conv1_b: Vec<f32>,          // [384]
    pub conv2_w: Vec<f32>,          // [384, 384, 3]
    pub conv2_b: Vec<f32>,          // [384]
    pub embed_positions: Vec<f32>,  // [1500, 384]
    pub layers: Vec<EncoderLayerWeights>,
    pub ln_w: Vec<f32>,             // [384]
    pub ln_b: Vec<f32>,             // [384]
}

pub struct EncoderLayerWeights {
    // Self-attention (k_proj has NO bias)
    pub q_proj_w: Vec<f32>,   // [384, 384] (transposed to row-major)
    pub q_proj_b: Vec<f32>,   // [384]
    pub k_proj_w: Vec<f32>,   // [384, 384] — NO bias
    pub v_proj_w: Vec<f32>,
    pub v_proj_b: Vec<f32>,
    pub o_proj_w: Vec<f32>,
    pub o_proj_b: Vec<f32>,
    pub sa_ln_w: Vec<f32>,    // self_attn_layer_norm
    pub sa_ln_b: Vec<f32>,
    // FFN
    pub fc1_w: Vec<f32>,      // [1536, 384]
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,      // [384, 1536]
    pub fc2_b: Vec<f32>,
    pub ff_ln_w: Vec<f32>,    // final_layer_norm
    pub ff_ln_b: Vec<f32>,
}

pub struct DecoderWeights {
    pub embed_tokens: Vec<f32>,     // [51865, 384] — also used as proj_out (tied)
    pub embed_positions: Vec<f32>,  // [448, 384]
    pub layers: Vec<DecoderLayerWeights>,
    pub ln_w: Vec<f32>,
    pub ln_b: Vec<f32>,
}

pub struct DecoderLayerWeights {
    // Self-attention (k_proj has NO bias)
    pub q_proj_w: Vec<f32>,
    pub q_proj_b: Vec<f32>,
    pub k_proj_w: Vec<f32>,   // NO bias
    pub v_proj_w: Vec<f32>,
    pub v_proj_b: Vec<f32>,
    pub o_proj_w: Vec<f32>,
    pub o_proj_b: Vec<f32>,
    pub sa_ln_w: Vec<f32>,
    pub sa_ln_b: Vec<f32>,
    // Cross-attention (k_proj has NO bias)
    pub xq_proj_w: Vec<f32>,
    pub xq_proj_b: Vec<f32>,
    pub xk_proj_w: Vec<f32>,  // NO bias
    pub xv_proj_w: Vec<f32>,
    pub xv_proj_b: Vec<f32>,
    pub xo_proj_w: Vec<f32>,
    pub xo_proj_b: Vec<f32>,
    pub xa_ln_w: Vec<f32>,
    pub xa_ln_b: Vec<f32>,
    // FFN
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
    pub ff_ln_w: Vec<f32>,
    pub ff_ln_b: Vec<f32>,
}

impl WhisperWeights {
    pub fn memory_bytes(&self) -> usize {
        let enc = &self.encoder;
        let mut total = 0usize;

        total += (enc.conv1_w.len() + enc.conv1_b.len()) * 4;
        total += (enc.conv2_w.len() + enc.conv2_b.len()) * 4;
        total += enc.embed_positions.len() * 4;
        total += (enc.ln_w.len() + enc.ln_b.len()) * 4;

        for l in &enc.layers {
            total += (l.q_proj_w.len() + l.q_proj_b.len()) * 4;
            total += l.k_proj_w.len() * 4;
            total += (l.v_proj_w.len() + l.v_proj_b.len()) * 4;
            total += (l.o_proj_w.len() + l.o_proj_b.len()) * 4;
            total += (l.sa_ln_w.len() + l.sa_ln_b.len()) * 4;
            total += (l.fc1_w.len() + l.fc1_b.len()) * 4;
            total += (l.fc2_w.len() + l.fc2_b.len()) * 4;
            total += (l.ff_ln_w.len() + l.ff_ln_b.len()) * 4;
        }

        let dec = &self.decoder;
        total += dec.embed_tokens.len() * 4;
        total += dec.embed_positions.len() * 4;
        total += (dec.ln_w.len() + dec.ln_b.len()) * 4;

        for l in &dec.layers {
            total += (l.q_proj_w.len() + l.q_proj_b.len()) * 4;
            total += l.k_proj_w.len() * 4;
            total += (l.v_proj_w.len() + l.v_proj_b.len()) * 4;
            total += (l.o_proj_w.len() + l.o_proj_b.len()) * 4;
            total += (l.sa_ln_w.len() + l.sa_ln_b.len()) * 4;
            total += (l.xq_proj_w.len() + l.xq_proj_b.len()) * 4;
            total += l.xk_proj_w.len() * 4;
            total += (l.xv_proj_w.len() + l.xv_proj_b.len()) * 4;
            total += (l.xo_proj_w.len() + l.xo_proj_b.len()) * 4;
            total += (l.xa_ln_w.len() + l.xa_ln_b.len()) * 4;
            total += (l.fc1_w.len() + l.fc1_b.len()) * 4;
            total += (l.fc2_w.len() + l.fc2_b.len()) * 4;
            total += (l.ff_ln_w.len() + l.ff_ln_b.len()) * 4;
        }

        total
    }
}
