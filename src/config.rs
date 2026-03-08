use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct WhisperConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub vocab_size: usize,
    pub max_source_positions: usize,
    pub max_target_positions: usize,
    pub num_mel_bins: usize,
    #[serde(default = "default_decoder_start")]
    pub decoder_start_token_id: usize,
    #[serde(default = "default_eos")]
    pub eos_token_id: usize,
    #[serde(default)]
    pub forced_decoder_ids: Vec<(usize, usize)>,
    #[serde(default)]
    pub suppress_tokens: Vec<usize>,
    #[serde(default)]
    pub begin_suppress_tokens: Vec<usize>,
}

fn default_decoder_start() -> usize { 50258 }
fn default_eos() -> usize { 50257 }

impl WhisperConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&data)?;
        Ok(config)
    }

    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}

// Special token constants
pub const SOT: u32 = 50258;     // <|startoftranscript|>
pub const EOT: u32 = 50257;     // <|endoftext|>
pub const TRANSCRIBE: u32 = 50359;
pub const TRANSLATE: u32 = 50358;
pub const NO_TIMESTAMPS: u32 = 50363;

// Language token IDs (50259 = en, 50260 = zh, ...)
pub fn language_token(lang: &str) -> u32 {
    match lang.to_lowercase().as_str() {
        "english" | "en" => 50259,
        "chinese" | "zh" => 50260,
        "german" | "de" => 50261,
        "spanish" | "es" => 50262,
        "russian" | "ru" => 50263,
        "korean" | "ko" => 50264,
        "french" | "fr" => 50265,
        "japanese" | "ja" => 50266,
        "portuguese" | "pt" => 50267,
        "turkish" | "tr" => 50268,
        "polish" | "pl" => 50269,
        "catalan" | "ca" => 50270,
        "dutch" | "nl" => 50271,
        "arabic" | "ar" => 50272,
        "swedish" | "sv" => 50273,
        "italian" | "it" => 50274,
        "indonesian" | "id" => 50275,
        "hindi" | "hi" => 50276,
        "finnish" | "fi" => 50277,
        "vietnamese" | "vi" => 50278,
        "hebrew" | "he" => 50279,
        "ukrainian" | "uk" => 50280,
        "greek" | "el" => 50281,
        "malay" | "ms" => 50282,
        "czech" | "cs" => 50283,
        "romanian" | "ro" => 50284,
        "danish" | "da" => 50285,
        "hungarian" | "hu" => 50286,
        "tamil" | "ta" => 50287,
        "norwegian" | "no" => 50288,
        "thai" | "th" => 50289,
        "urdu" | "ur" => 50290,
        "croatian" | "hr" => 50291,
        "bulgarian" | "bg" => 50292,
        "lithuanian" | "lt" => 50293,
        "latin" | "la" => 50294,
        "maori" | "mi" => 50295,
        "malayalam" | "ml" => 50296,
        "welsh" | "cy" => 50297,
        "slovak" | "sk" => 50298,
        "telugu" | "te" => 50299,
        "persian" | "fa" => 50300,
        "latvian" | "lv" => 50301,
        "bengali" | "bn" => 50302,
        "serbian" | "sr" => 50303,
        "azerbaijani" | "az" => 50304,
        "slovenian" | "sl" => 50305,
        "kannada" | "kn" => 50306,
        "estonian" | "et" => 50307,
        "macedonian" | "mk" => 50308,
        "breton" | "br" => 50309,
        "basque" | "eu" => 50310,
        "icelandic" | "is" => 50311,
        "armenian" | "hy" => 50312,
        "nepali" | "ne" => 50313,
        "mongolian" | "mn" => 50314,
        "bosnian" | "bs" => 50315,
        "kazakh" | "kk" => 50316,
        "albanian" | "sq" => 50317,
        "swahili" | "sw" => 50318,
        "galician" | "gl" => 50319,
        "marathi" | "mr" => 50320,
        "punjabi" | "pa" => 50321,
        "sinhala" | "si" => 50322,
        "khmer" | "km" => 50323,
        "shona" | "sn" => 50324,
        "yoruba" | "yo" => 50325,
        "somali" | "so" => 50326,
        "afrikaans" | "af" => 50327,
        "occitan" | "oc" => 50328,
        "georgian" | "ka" => 50329,
        "belarusian" | "be" => 50330,
        "tajik" | "tg" => 50331,
        "sindhi" | "sd" => 50332,
        "gujarati" | "gu" => 50333,
        "amharic" | "am" => 50334,
        "yiddish" | "yi" => 50335,
        "lao" | "lo" => 50336,
        "uzbek" | "uz" => 50337,
        "faroese" | "fo" => 50338,
        "haitian" | "ht" => 50339,
        "pashto" | "ps" => 50340,
        "turkmen" | "tk" => 50341,
        "nynorsk" | "nn" => 50342,
        "maltese" | "mt" => 50343,
        "sanskrit" | "sa" => 50344,
        "luxembourgish" | "lb" => 50345,
        "myanmar" | "my" => 50346,
        "tibetan" | "bo" => 50347,
        "tagalog" | "tl" => 50348,
        "malagasy" | "mg" => 50349,
        "assamese" | "as" => 50350,
        "tatar" | "tt" => 50351,
        "hawaiian" | "haw" => 50352,
        "lingala" | "ln" => 50353,
        "hausa" | "ha" => 50354,
        "bashkir" | "ba" => 50355,
        "javanese" | "jw" => 50356,
        "sundanese" | "su" => 50357,
        _ => 50259, // default to English
    }
}
