use std::fmt;

/// Errors produced by stt_core's transcription and model-management paths.
///
/// `Display` is meant for logs, not end users: several variants (`ApiError`,
/// `HttpRequest`, `Io`, `ModelLoad`, `DecodeAudio`, `TranscribeFailed`) wrap
/// raw upstream or internal error text that can contain URLs or filesystem
/// paths. Callers crossing a UI boundary should not forward `Display` output
/// directly — see `synapse_voice`'s `transcribe::map_stt_error` for the
/// pattern of logging the full text and returning a generic message for
/// those variants.
///
/// `#[non_exhaustive]` because this crate is a publish candidate: adding a
/// variant must not be a semver break for downstream exhaustive matches.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SttError {
    MissingApiKey,
    DecodeAudio(String),
    UnsupportedFormat(String),
    HttpRequest(String),
    ApiError { status: String, body: String },
    ModelNotFound(String),
    ModelNotDownloaded(String),
    ModelLoad(String),
    TranscribeFailed(String),
    Io(String),
}

impl fmt::Display for SttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SttError::MissingApiKey => {
                write!(
                    f,
                    "ElevenLabs API key not configured. Go to Settings to add it."
                )
            }
            SttError::DecodeAudio(msg) => write!(f, "{}", msg),
            SttError::UnsupportedFormat(msg) => write!(f, "{}", msg),
            SttError::HttpRequest(msg) => write!(f, "{}", msg),
            SttError::ApiError { status, body } => {
                write!(f, "ElevenLabs API error ({}): {}", status, body)
            }
            SttError::ModelNotFound(msg) => write!(f, "{}", msg),
            SttError::ModelNotDownloaded(msg) => write!(f, "{}", msg),
            SttError::ModelLoad(msg) => write!(f, "{}", msg),
            SttError::TranscribeFailed(msg) => write!(f, "{}", msg),
            SttError::Io(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for SttError {}

impl From<std::io::Error> for SttError {
    fn from(e: std::io::Error) -> Self {
        SttError::Io(e.to_string())
    }
}

impl From<reqwest::Error> for SttError {
    fn from(e: reqwest::Error) -> Self {
        SttError::HttpRequest(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_api_key_message_matches_legacy_string() {
        assert_eq!(
            SttError::MissingApiKey.to_string(),
            "ElevenLabs API key not configured. Go to Settings to add it."
        );
    }

    #[test]
    fn api_error_message_matches_legacy_format() {
        let err = SttError::ApiError {
            status: "500 Internal Server Error".to_string(),
            body: "boom".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "ElevenLabs API error (500 Internal Server Error): boom"
        );
    }
}
