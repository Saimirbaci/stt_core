//! Framework-agnostic speech-to-text logic shared by Synapse Voice's
//! desktop and (future) server callers: ElevenLabs cloud transcription and
//! local Whisper/Moonshine inference via `whisper-apr`.
//!
//! This crate has no Tauri dependency and does not spawn its own blocking
//! threads — the local-inference path (`transcribe_whisper`) is synchronous
//! by design, and callers are expected to run it via their own
//! `tokio::task::spawn_blocking` (or equivalent) so async runtimes are never
//! stalled.

pub mod elevenlabs;
pub mod error;
pub mod types;

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub mod whisper;

pub use elevenlabs::transcribe_elevenlabs;
pub use error::SttError;
pub use types::{AudioFormat, SttConfig};

#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use whisper::download::download_and_convert_model;
#[cfg(not(any(target_os = "android", target_os = "ios")))]
pub use whisper::{
    transcribe_local as transcribe_whisper, DownloadProgress, WhisperModelInfo, WhisperState,
};

/// Which backend a transcription request should be routed to, per
/// `SttConfig::stt_provider`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// Cloud transcription via the ElevenLabs API. Available on every platform.
    ElevenLabs,
    /// Local, offline transcription via Whisper/Moonshine. Desktop only.
    Whisper,
}

/// Decide which provider a request should use, mirroring the
/// `stt_provider == "whisper"` switch previously inlined in
/// `transcribe.rs::transcribe_bytes`. Local Whisper is only ever selected on
/// platforms where the `whisper` module is compiled in (desktop).
pub fn route_provider(cfg: &SttConfig) -> Provider {
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    if cfg.stt_provider == "whisper" {
        return Provider::Whisper;
    }
    let _ = cfg;
    Provider::ElevenLabs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_provider_defaults_to_elevenlabs() {
        let cfg = SttConfig {
            stt_provider: "elevenlabs".to_string(),
            elevenlabs_api_key: "key".to_string(),
            whisper_model_size: "whisper-tiny".to_string(),
        };
        assert_eq!(route_provider(&cfg), Provider::ElevenLabs);
    }

    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    #[test]
    fn route_provider_selects_whisper_when_configured() {
        let cfg = SttConfig {
            stt_provider: "whisper".to_string(),
            elevenlabs_api_key: String::new(),
            whisper_model_size: "whisper-tiny".to_string(),
        };
        assert_eq!(route_provider(&cfg), Provider::Whisper);
    }
}
