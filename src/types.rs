use std::fmt;

/// Audio container format for a transcription request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Webm,
    M4a,
    Ogg,
}

impl AudioFormat {
    /// Parse a format from a file extension or format hint string
    /// (e.g. `"wav"`, `"mp4"`, `"m4a"`, `"webm"`, `"ogg"`). Unknown values
    /// default to `Wav`, matching the previous `match` fallback behavior.
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "webm" => AudioFormat::Webm,
            "mp4" | "m4a" => AudioFormat::M4a,
            "ogg" => AudioFormat::Ogg,
            _ => AudioFormat::Wav,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Webm => "webm",
            AudioFormat::M4a => "m4a",
            AudioFormat::Ogg => "ogg",
        }
    }
}

impl fmt::Display for AudioFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// STT provider selection and credentials, sourced from app settings.
#[derive(Debug, Clone, Default)]
pub struct SttConfig {
    /// Which backend to use — `"whisper"` selects local inference (desktop
    /// only), anything else falls back to ElevenLabs. See [`crate::route_provider`].
    pub stt_provider: String,
    /// ElevenLabs API key. Required only when `stt_provider` resolves to
    /// [`crate::Provider::ElevenLabs`].
    pub elevenlabs_api_key: String,
    /// Model id from [`crate::whisper::MODELS`] (e.g. `"whisper-tiny"`).
    /// Required only when `stt_provider` resolves to [`crate::Provider::Whisper`].
    pub whisper_model_size: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_extension_defaults_to_wav() {
        assert_eq!(AudioFormat::from_extension("flac"), AudioFormat::Wav);
    }

    #[test]
    fn mp4_and_m4a_both_map_to_m4a() {
        assert_eq!(AudioFormat::from_extension("mp4"), AudioFormat::M4a);
        assert_eq!(AudioFormat::from_extension("m4a"), AudioFormat::M4a);
    }
}
