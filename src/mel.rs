//! Whisper-compatible mel spectrogram extraction.
//!
//! Parameters: n_fft=400, hop=160, n_mels=80, sr=16kHz.
//! Uses power spectrum, log10 compression, and Whisper normalization.

use std::f32::consts::PI;

const SAMPLE_RATE: usize = 16000;
const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const CHUNK_SAMPLES: usize = 480000; // 30 seconds at 16kHz

/// Extract Whisper-compatible mel spectrogram from 16kHz audio.
/// Returns [n_mels, n_frames] = [80, 3000] in channel-first format.
pub fn extract_mel(audio: &[f32]) -> Vec<f32> {
    // 1. Pad or trim to exactly 30 seconds
    let mut padded = vec![0.0f32; CHUNK_SAMPLES];
    let copy_len = audio.len().min(CHUNK_SAMPLES);
    padded[..copy_len].copy_from_slice(&audio[..copy_len]);

    // 2. Compute STFT frames
    let window = hann_window(N_FFT);
    let n_freq = N_FFT / 2 + 1; // 201 frequency bins
    let n_frames = CHUNK_SAMPLES / HOP_LENGTH; // 3000 frames

    // Pad FFT to next power of 2 for fast FFT
    let fft_size = N_FFT.next_power_of_two(); // 512

    let mut power_spectrogram = vec![0.0f32; n_frames * n_freq];

    for t in 0..n_frames {
        let start = t * HOP_LENGTH;

        // Extract and window the frame
        let mut real = vec![0.0f32; fft_size];
        let mut imag = vec![0.0f32; fft_size];
        for i in 0..N_FFT {
            let sample_idx = start + i;
            if sample_idx < padded.len() {
                real[i] = padded[sample_idx] * window[i];
            }
        }
        // Remaining positions 400..512 stay zero (zero-padding)

        // Compute FFT
        fft_in_place(&mut real, &mut imag);

        // Power spectrum: |STFT|^2, take first n_freq bins
        // Use n_fft-based frequency resolution (take bins corresponding to 0..n_fft/2)
        for k in 0..n_freq {
            // Map bin k from n_fft resolution to fft_size resolution
            let fft_bin = k as f32 * fft_size as f32 / N_FFT as f32;
            let bin_lo = fft_bin as usize;
            let bin_hi = (bin_lo + 1).min(fft_size / 2);
            let frac = fft_bin - bin_lo as f32;

            // Interpolate between adjacent FFT bins
            let re = real[bin_lo] * (1.0 - frac) + real[bin_hi] * frac;
            let im = imag[bin_lo] * (1.0 - frac) + imag[bin_hi] * frac;

            power_spectrogram[t * n_freq + k] = re * re + im * im;
        }
    }

    // 3. Apply mel filterbank
    let mel_filters = create_mel_filterbank(N_MELS, N_FFT, SAMPLE_RATE, 0.0, 8000.0);

    let mut mel_spec = vec![0.0f32; N_MELS * n_frames];
    for t in 0..n_frames {
        for m in 0..N_MELS {
            let mut energy = 0.0f32;
            for f in 0..n_freq {
                energy += mel_filters[m * n_freq + f] * power_spectrogram[t * n_freq + f];
            }
            mel_spec[m * n_frames + t] = energy;
        }
    }

    // 4. Whisper normalization: log10, dynamic range compression, scale
    let mut max_val = f32::NEG_INFINITY;
    for v in &mut mel_spec {
        *v = (*v).max(1e-10).log10();
        if *v > max_val { max_val = *v; }
    }

    let threshold = max_val - 8.0;
    for v in &mut mel_spec {
        *v = (*v).max(threshold);
        *v = (*v + 4.0) / 4.0;
    }

    mel_spec
}

/// Return the number of mel frames for the standard 30s chunk.
pub fn n_frames() -> usize {
    CHUNK_SAMPLES / HOP_LENGTH
}

/// Return number of mel bins.
pub fn n_mels() -> usize {
    N_MELS
}

// ============================================================
// Internal helpers
// ============================================================

fn hann_window(length: usize) -> Vec<f32> {
    // Periodic Hann window (PyTorch default)
    (0..length)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / length as f32).cos()))
        .collect()
}

fn create_mel_filterbank(
    n_mels: usize, n_fft: usize, sample_rate: usize,
    fmin: f32, fmax: f32,
) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let fft_freqs: Vec<f32> = (0..n_freq)
        .map(|i| i as f32 * sample_rate as f32 / n_fft as f32)
        .collect();

    let mut filterbank = vec![0.0f32; n_mels * n_freq];

    for m in 0..n_mels {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for k in 0..n_freq {
            let freq = fft_freqs[k];
            let val = if freq >= f_left && freq <= f_center && f_center > f_left {
                (freq - f_left) / (f_center - f_left)
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                (f_right - freq) / (f_right - f_center)
            } else {
                0.0
            };
            filterbank[m * n_freq + k] = val;
        }
    }

    filterbank
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// In-place Cooley-Tukey radix-2 FFT. Requires power-of-2 size.
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    assert!(n.is_power_of_two());

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            real.swap(i, j);
            imag.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle_step = -2.0 * PI / len as f32;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let angle = angle_step * k as f32;
                let wr = angle.cos();
                let wi = angle.sin();
                let a = start + k;
                let b = start + k + half;
                let tr = wr * real[b] - wi * imag[b];
                let ti = wr * imag[b] + wi * real[b];
                real[b] = real[a] - tr;
                imag[b] = imag[a] - ti;
                real[a] += tr;
                imag[a] += ti;
            }
        }
        len <<= 1;
    }
}
