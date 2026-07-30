use reqwest::multipart;

use crate::error::SttError;
use crate::types::AudioFormat;

/// Transcribe audio via ElevenLabs cloud API.
pub async fn transcribe_elevenlabs(
    api_key: &str,
    audio_bytes: Vec<u8>,
    audio_format: AudioFormat,
) -> Result<String, SttError> {
    if api_key.is_empty() {
        return Err(SttError::MissingApiKey);
    }

    let (file_name, mime) = match audio_format {
        AudioFormat::Webm => ("recording.webm", "audio/webm"),
        AudioFormat::M4a => ("recording.m4a", "audio/mp4"),
        AudioFormat::Ogg => ("recording.ogg", "audio/ogg"),
        AudioFormat::Wav => ("recording.wav", "audio/wav"),
    };

    let part = multipart::Part::bytes(audio_bytes)
        .file_name(file_name.to_string())
        .mime_str(mime)
        .map_err(|e| SttError::HttpRequest(e.to_string()))?;

    let form = multipart::Form::new()
        .text("model_id", "scribe_v1")
        .part("file", part);

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.elevenlabs.io/v1/speech-to-text")
        .header("xi-api-key", api_key)
        .multipart(form)
        .send()
        .await
        .map_err(|e| SttError::HttpRequest(format!("Transcription request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status().to_string();
        let body = response.text().await.unwrap_or_default();
        return Err(SttError::ApiError { status, body });
    }

    let result: serde_json::Value = response
        .json()
        .await
        .map_err(|e| SttError::HttpRequest(e.to_string()))?;
    result["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| SttError::TranscribeFailed("No transcription text in response".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_api_key_short_circuits_before_any_request() {
        let result = transcribe_elevenlabs("", vec![1, 2, 3], AudioFormat::Wav).await;
        assert!(matches!(result, Err(SttError::MissingApiKey)));
    }
}
