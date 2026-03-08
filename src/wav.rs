//! PCM 16-bit WAV reader (mono) with resampling to 16kHz.

use std::io;
use std::path::Path;

/// Read WAV file and return (samples, sample_rate).
/// Converts to mono f32 samples [-1.0, 1.0].
pub fn read_wav(path: &Path) -> io::Result<(Vec<f32>, u32)> {
    let all_data = std::fs::read(path)?;

    if all_data.len() < 44 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "WAV file too small"));
    }

    if &all_data[0..4] != b"RIFF" || &all_data[8..12] != b"WAVE" {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a valid WAV file"));
    }

    let mut pos = 12;
    let mut num_channels = 0usize;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut fmt_found = false;

    while pos + 8 <= all_data.len() {
        let chunk_id = &all_data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            all_data[pos + 4], all_data[pos + 5],
            all_data[pos + 6], all_data[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            if pos + 8 + chunk_size > all_data.len() { break; }
            num_channels = u16::from_le_bytes([all_data[pos + 10], all_data[pos + 11]]) as usize;
            sample_rate = u32::from_le_bytes([
                all_data[pos + 12], all_data[pos + 13],
                all_data[pos + 14], all_data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([all_data[pos + 22], all_data[pos + 23]]);
            fmt_found = true;
        } else if chunk_id == b"data" && fmt_found {
            if bits_per_sample != 16 {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("Only 16-bit WAV supported, got {bits_per_sample}-bit")));
            }

            let data_size = chunk_size;
            let data_start = pos + 8;
            if data_start + data_size > all_data.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Data chunk exceeds file"));
            }

            let data = &all_data[data_start..data_start + data_size];
            let num_samples = data_size / 2 / num_channels;
            let mut samples = vec![0.0f32; num_samples];

            for i in 0..num_samples {
                let mut sum = 0.0f32;
                for ch in 0..num_channels {
                    let offset = (i * num_channels + ch) * 2;
                    if offset + 1 < data.len() {
                        let val = i16::from_le_bytes([data[offset], data[offset + 1]]);
                        sum += val as f32 / 32768.0;
                    }
                }
                samples[i] = sum / num_channels as f32;
            }

            return Ok((samples, sample_rate));
        }

        pos += 8 + chunk_size;
        if chunk_size % 2 == 1 { pos += 1; }
    }

    Err(io::Error::new(io::ErrorKind::InvalidData, "Missing data chunk"))
}

/// Resample audio to target sample rate using linear interpolation.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = (src_pos - idx as f64) as f32;

        if idx + 1 < samples.len() {
            output.push(samples[idx] * (1.0 - frac) + samples[idx + 1] * frac);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}

/// Read WAV and resample to 16kHz mono.
pub fn read_wav_16khz(path: &Path) -> io::Result<Vec<f32>> {
    let (samples, rate) = read_wav(path)?;
    eprintln!("  Audio: {} samples at {}Hz ({:.1}s)",
        samples.len(), rate, samples.len() as f32 / rate as f32);

    if rate == 16000 {
        Ok(samples)
    } else {
        eprintln!("  Resampling {}Hz → 16000Hz...", rate);
        let resampled = resample(&samples, rate, 16000);
        eprintln!("  Resampled: {} samples ({:.1}s)",
            resampled.len(), resampled.len() as f32 / 16000.0);
        Ok(resampled)
    }
}
