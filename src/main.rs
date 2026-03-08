use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_path = PathBuf::from(".");
    let mut audio_path: Option<PathBuf> = None;
    let mut language = String::from("english");
    let mut save_path: Option<PathBuf> = None;
    let mut load_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-path" => {
                if i + 1 < args.len() {
                    model_path = PathBuf::from(&args[i + 1]);
                    i += 1;
                }
            }
            "--audio" => {
                if i + 1 < args.len() {
                    audio_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--language" => {
                if i + 1 < args.len() {
                    language = args[i + 1].clone();
                    i += 1;
                }
            }
            "--save" => {
                if i + 1 < args.len() {
                    save_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--load" => {
                if i + 1 < args.len() {
                    load_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = Some(PathBuf::from(&args[i + 1]));
                    i += 1;
                }
            }
            "--help" | "-h" => {
                eprintln!("QORA-STT — Pure Rust Speech-to-Text (Whisper Tiny)");
                eprintln!();
                eprintln!("Usage: qora-stt [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --model-path <dir>   Directory with model.safetensors + config.json");
                eprintln!("  --audio <wav>        Input WAV file to transcribe");
                eprintln!("  --language <name>    Language (default: english)");
                eprintln!("  --save <path>        Save binary model to .qora-stt file");
                eprintln!("  --load <path>        Load binary model from .qora-stt file");
                eprintln!("  --output <path>      Write transcription to file");
                eprintln!("  --help               Show this help");
                return;
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("QORA-STT — Pure Rust Speech-to-Text Engine");
    eprintln!("Model: Whisper Tiny (39M params)");
    eprintln!();

    // Load config
    let config = qora_stt::config::WhisperConfig::from_file(model_path.join("config.json"))
        .expect("Failed to load config.json");
    eprintln!("Config: d_model={}, enc_layers={}, dec_layers={}, vocab={}",
        config.d_model, config.encoder_layers, config.decoder_layers, config.vocab_size);

    // Load weights
    let weights = if let Some(ref lp) = load_path {
        eprintln!("Loading binary model from {}...", lp.display());
        let t0 = Instant::now();
        let w = qora_stt::save::load_model(lp).expect("Failed to load binary model");
        let mem_mb = w.memory_bytes() / (1024 * 1024);
        eprintln!("Model loaded in {:.1?} ({mem_mb} MB)", t0.elapsed());
        w
    } else {
        eprintln!("Loading safetensors from {}...", model_path.display());
        let t0 = Instant::now();
        let w = qora_stt::loader::load_weights(&model_path, &config)
            .expect("Failed to load safetensors");
        let mem_mb = w.memory_bytes() / (1024 * 1024);
        eprintln!("Model loaded in {:.1?} ({mem_mb} MB)", t0.elapsed());
        w
    };

    // Save binary if requested
    if let Some(ref sp) = save_path {
        eprintln!("Saving binary model to {}...", sp.display());
        let t0 = Instant::now();
        qora_stt::save::save_model(&weights, sp).expect("Failed to save binary model");
        let size_mb = std::fs::metadata(sp).map(|m| m.len() / (1024 * 1024)).unwrap_or(0);
        eprintln!("Saved in {:.1?} ({size_mb} MB)", t0.elapsed());
    }

    // Transcribe audio
    if let Some(ref ap) = audio_path {
        // Load tokenizer
        let tok_path = model_path.join("tokenizer.json");
        let tokenizer = qora_stt::tokenizer::WhisperTokenizer::from_file(&tok_path)
            .expect("Failed to load tokenizer.json");

        eprintln!();
        let text = qora_stt::inference::transcribe(&weights, &config, &tokenizer, ap, &language)
            .expect("Transcription failed");

        eprintln!();
        eprintln!("=== Transcription ===");
        println!("{}", text);

        if let Some(ref op) = output_path {
            std::fs::write(op, &text).expect("Failed to write output");
            eprintln!("Written to {}", op.display());
        }
    } else if save_path.is_none() {
        eprintln!("No --audio specified. Use --audio <wav> to transcribe.");
    }
}
