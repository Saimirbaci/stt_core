pub mod convert;
pub mod download;

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use whisper_apr::{TranscribeOptions, WhisperApr};

use crate::error::SttError;

// =============================================================================
// State
// =============================================================================

/// Holds the lazily-loaded Whisper model.
/// Wrapped in Arc so it can be sent to spawn_blocking.
#[derive(Clone)]
pub struct WhisperState {
    model: Arc<Mutex<Option<WhisperApr>>>,
    loaded_model: Arc<Mutex<Option<String>>>,
}

impl Default for WhisperState {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperState {
    pub fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            loaded_model: Arc::new(Mutex::new(None)),
        }
    }

    /// Unload the in-memory model if `model_id` is the currently loaded one.
    /// No-op if a different (or no) model is loaded.
    pub fn unload_if_loaded(&self, model_id: &str) -> Result<(), SttError> {
        let mut loaded = self
            .loaded_model
            .lock()
            .map_err(|e| SttError::ModelLoad(e.to_string()))?;
        if loaded.as_deref() == Some(model_id) {
            let mut model = self
                .model
                .lock()
                .map_err(|e| SttError::ModelLoad(e.to_string()))?;
            *model = None;
            *loaded = None;
        }
        Ok(())
    }
}

// =============================================================================
// Types
// =============================================================================

/// Caller-facing view of a model's registry metadata plus its on-disk
/// download state, for populating a model picker UI.
#[derive(Serialize, Clone, Debug)]
pub struct WhisperModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size_bytes: u64,
    pub downloaded: bool,
    pub file_size: Option<u64>,
}

/// A snapshot of an in-progress model download/conversion, reported via the
/// `progress` callback passed to [`download::download_and_convert_model`].
#[derive(Serialize, Clone, Debug)]
pub struct DownloadProgress {
    pub model_size: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percent: f64,
    /// `"downloading"` or `"converting"`.
    #[serde(default)]
    pub phase: String,
}

// =============================================================================
// Model Registry
// =============================================================================

/// Static metadata for one supported model in the [`MODELS`] registry.
pub struct ModelEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// HuggingFace repo the SafeTensors weights are downloaded from.
    pub hf_repo: &'static str,
    /// Pinned HuggingFace commit SHA to download from, instead of the
    /// mutable `main` ref — prevents a `main` force-push or repo compromise
    /// from silently substituting different weights on the next download.
    pub revision: &'static str,
    pub is_moonshine: bool,
    /// Approximate size of the *converted* `.apr` file, for display only
    /// (e.g. in the model picker UI). This is roughly half of
    /// `safetensors_size` and must not be used to bound the download.
    pub estimated_size: u64,
    /// Actual size in bytes of `model.safetensors` at the pinned `revision`
    /// above, measured directly against the HuggingFace tree API. Used to
    /// bound the download (see `download::download_and_convert_model`) —
    /// unlike `estimated_size`, this is the real download size, not a
    /// display estimate, so it is safe to use as a security limit.
    pub safetensors_size: u64,
    /// Filename this model is stored under, once converted, inside a
    /// caller-supplied models directory.
    pub apr_filename: &'static str,
}

/// All models this crate knows how to download, convert, and run locally.
pub const MODELS: &[ModelEntry] = &[
    ModelEntry {
        id: "moonshine-tiny",
        name: "Moonshine Tiny",
        description: "54 MB — Ultra-fast, great for live preview",
        hf_repo: "usefulsensors/moonshine-tiny",
        revision: "390624ed33d594443aa4aa221f5b9f283b545b5a",
        is_moonshine: true,
        estimated_size: 54_000_000,
        safetensors_size: 108_389_192,
        apr_filename: "moonshine-tiny.apr",
    },
    ModelEntry {
        id: "moonshine-base",
        name: "Moonshine Base",
        description: "123 MB — Fast with good accuracy",
        hf_repo: "usefulsensors/moonshine-base",
        revision: "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf",
        is_moonshine: true,
        estimated_size: 123_000_000,
        safetensors_size: 246_079_928,
        apr_filename: "moonshine-base.apr",
    },
    ModelEntry {
        id: "whisper-tiny",
        name: "Whisper Tiny",
        description: "78 MB — Fastest Whisper, quick notes",
        hf_repo: "openai/whisper-tiny",
        revision: "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
        is_moonshine: false,
        estimated_size: 78_000_000,
        safetensors_size: 151_061_672,
        apr_filename: "whisper-tiny.apr",
    },
    ModelEntry {
        id: "whisper-base",
        name: "Whisper Base",
        description: "148 MB — Good balance of speed and accuracy",
        hf_repo: "openai/whisper-base",
        revision: "e37978b90ca9030d5170a5c07aadb050351a65bb",
        is_moonshine: false,
        estimated_size: 148_000_000,
        safetensors_size: 290_403_936,
        apr_filename: "whisper-base.apr",
    },
    ModelEntry {
        id: "whisper-small",
        name: "Whisper Small",
        description: "488 MB — Better accuracy, still fast",
        hf_repo: "openai/whisper-small",
        revision: "973afd24965f72e36ca33b3055d56a652f456b4d",
        is_moonshine: false,
        estimated_size: 488_000_000,
        safetensors_size: 966_995_080,
        apr_filename: "whisper-small.apr",
    },
    ModelEntry {
        id: "whisper-medium",
        name: "Whisper Medium",
        description: "1.5 GB — High accuracy",
        hf_repo: "openai/whisper-medium",
        revision: "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
        is_moonshine: false,
        estimated_size: 1_500_000_000,
        safetensors_size: 3_055_544_304,
        apr_filename: "whisper-medium.apr",
    },
    ModelEntry {
        id: "whisper-large",
        name: "Whisper Large v3",
        description: "3.0 GB — Best accuracy",
        hf_repo: "openai/whisper-large-v3",
        revision: "06f233fe06e710322aca913c1bc4249a0d71fce1",
        is_moonshine: false,
        estimated_size: 3_000_000_000,
        safetensors_size: 3_087_130_976,
        apr_filename: "whisper-large.apr",
    },
    ModelEntry {
        id: "whisper-large-v3-turbo",
        name: "Whisper Large v3 Turbo",
        description: "1.6 GB — Fast large model",
        hf_repo: "openai/whisper-large-v3-turbo",
        revision: "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        is_moonshine: false,
        estimated_size: 1_600_000_000,
        safetensors_size: 1_617_824_864,
        apr_filename: "whisper-large-v3-turbo.apr",
    },
];

/// Look up a model's static registry entry by id (e.g. `"whisper-tiny"`).
pub fn find_model_entry(model_id: &str) -> Option<&'static ModelEntry> {
    MODELS.iter().find(|m| m.id == model_id)
}

pub(crate) fn get_model_config(model_id: &str) -> Option<whisper_apr::model::ModelConfig> {
    match model_id {
        "moonshine-tiny" => Some(whisper_apr::model::ModelConfig::moonshine_tiny()),
        "moonshine-base" => Some(whisper_apr::model::ModelConfig::moonshine_base()),
        "whisper-tiny" => Some(whisper_apr::model::ModelConfig::tiny()),
        "whisper-base" => Some(whisper_apr::model::ModelConfig::base()),
        "whisper-small" => Some(whisper_apr::model::ModelConfig::small()),
        "whisper-medium" => Some(whisper_apr::model::ModelConfig::medium()),
        "whisper-large" => Some(whisper_apr::model::ModelConfig::large()),
        "whisper-large-v3-turbo" => Some(whisper_apr::model::ModelConfig::large_v3_turbo()),
        _ => None,
    }
}

// =============================================================================
// Model Loading
// =============================================================================

/// Resolve where a model's converted `.apr` file lives (or would live) inside
/// `models_dir`. Returns `None` for an unknown `model_id`; does not check
/// whether the file actually exists.
pub fn model_path(models_dir: &Path, model_id: &str) -> Option<PathBuf> {
    find_model_entry(model_id).map(|m| models_dir.join(m.apr_filename))
}

/// Load or reuse the WhisperApr model for the given model id.
pub fn ensure_model(
    whisper_state: &WhisperState,
    models_dir: &Path,
    model_id: &str,
) -> Result<(), SttError> {
    let mut loaded = whisper_state
        .loaded_model
        .lock()
        .map_err(|e| SttError::ModelLoad(e.to_string()))?;
    if loaded.as_deref() == Some(model_id) {
        return Ok(());
    }

    let path = model_path(models_dir, model_id)
        .ok_or_else(|| SttError::ModelNotFound(format!("Unknown model: {}", model_id)))?;

    if !path.exists() {
        return Err(SttError::ModelNotDownloaded(format!(
            "Model '{}' not downloaded. Download it in Settings first.",
            model_id
        )));
    }

    log::info!("loading model '{}'", model_id);
    log::debug!("model '{}' path: {:?}", model_id, path);
    let data = std::fs::read(&path)
        .map_err(|e| SttError::ModelLoad(format!("Failed to read model file: {}", e)))?;
    let model = WhisperApr::load_from_apr(&data)
        .map_err(|e| SttError::ModelLoad(format!("Failed to load model: {}", e)))?;

    let mut model_slot = whisper_state
        .model
        .lock()
        .map_err(|e| SttError::ModelLoad(e.to_string()))?;
    *model_slot = Some(model);
    *loaded = Some(model_id.to_string());
    log::info!("model '{}' loaded successfully", model_id);
    Ok(())
}

// =============================================================================
// Audio Decoding
// =============================================================================

/// Decode WAV bytes into 16kHz mono f32 samples.
pub fn decode_wav_to_f32(audio_bytes: &[u8]) -> Result<Vec<f32>, SttError> {
    let cursor = std::io::Cursor::new(audio_bytes);
    let reader = hound::WavReader::new(cursor)
        .map_err(|e| SttError::DecodeAudio(format!("Failed to read WAV: {}", e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
    };

    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    if sample_rate == 16000 {
        return Ok(mono);
    }
    resample_linear(&mono, sample_rate, 16000)
}

/// Simple linear interpolation resampler.
pub fn resample_linear(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>, SttError> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (samples.len() as f64 / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let idx_floor = src_idx.floor() as usize;
        let frac = (src_idx - idx_floor as f64) as f32;

        let sample = if idx_floor + 1 < samples.len() {
            samples[idx_floor] * (1.0 - frac) + samples[idx_floor + 1] * frac
        } else if idx_floor < samples.len() {
            samples[idx_floor]
        } else {
            0.0
        };
        output.push(sample);
    }
    Ok(output)
}

// =============================================================================
// Transcription
// =============================================================================

/// Transcribe audio bytes locally using whisper-apr.
///
/// Synchronous by design — the caller is responsible for running this on a
/// blocking thread (e.g. `tokio::task::spawn_blocking`) so it doesn't stall
/// an async runtime.
pub fn transcribe_local(
    audio_bytes: &[u8],
    audio_format: &str,
    model_id: &str,
    whisper_state: &WhisperState,
    models_dir: &Path,
) -> Result<String, SttError> {
    let samples = match audio_format {
        "wav" => decode_wav_to_f32(audio_bytes)?,
        _ => {
            return Err(SttError::UnsupportedFormat(format!(
                "Local transcription currently supports WAV format only. Got: {}",
                audio_format
            )));
        }
    };

    if samples.is_empty() {
        return Err(SttError::TranscribeFailed(
            "Audio is empty after decoding".to_string(),
        ));
    }

    ensure_model(whisper_state, models_dir, model_id)?;

    let model_guard = whisper_state
        .model
        .lock()
        .map_err(|e| SttError::TranscribeFailed(e.to_string()))?;
    let model = model_guard
        .as_ref()
        .ok_or_else(|| SttError::TranscribeFailed("Model not loaded".to_string()))?;

    let options = TranscribeOptions::default();
    let result = model
        .transcribe(&samples, options)
        .map_err(|e| SttError::TranscribeFailed(format!("Transcription failed: {}", e)))?;

    Ok(result.text.trim().to_string())
}

/// Preload model into memory (called from setup for fast first transcription).
pub fn preload_model(
    model_id: &str,
    models_dir: &Path,
    whisper_state: &WhisperState,
) -> Result<(), SttError> {
    let path = model_path(models_dir, model_id);
    match path {
        Some(p) if p.exists() => ensure_model(whisper_state, models_dir, model_id),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_model_entry_resolves_all_registry_ids() {
        for entry in MODELS {
            let found = find_model_entry(entry.id).expect("model should be found");
            assert_eq!(found.apr_filename, entry.apr_filename);
        }
    }

    #[test]
    fn find_model_entry_unknown_id_returns_none() {
        assert!(find_model_entry("does-not-exist").is_none());
    }

    #[test]
    fn get_model_config_covers_all_registry_ids() {
        for entry in MODELS {
            assert!(
                get_model_config(entry.id).is_some(),
                "missing model config for {}",
                entry.id
            );
        }
    }

    #[test]
    fn download_bound_covers_every_registry_entry() {
        // Regression test for a prior cap that was derived from
        // `estimated_size` (the converted .apr size, roughly half the real
        // download) instead of the actual `safetensors_size`, which silently
        // made 3 of 8 models undownloadable. The bound must be computed
        // from — and stay above — the real measured download size.
        for entry in MODELS {
            let bound = crate::whisper::download::max_download_bytes(entry);
            assert!(
                bound >= entry.safetensors_size,
                "{}: download bound {} is below the real safetensors size {}",
                entry.id,
                bound,
                entry.safetensors_size
            );
        }
    }

    #[test]
    fn resample_linear_upsamples_to_target_length() {
        let samples = vec![0.0, 1.0, 0.0, -1.0];
        let out = resample_linear(&samples, 8000, 16000).unwrap();
        assert_eq!(out.len(), samples.len() * 2);
    }

    #[test]
    fn resample_linear_noop_for_matching_rate() {
        let samples = vec![0.1, 0.2, 0.3];
        let out = resample_linear(&samples, 16000, 16000).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn ensure_model_errors_when_not_downloaded() {
        let dir = tempfile::tempdir().unwrap();
        let state = WhisperState::new();
        let err = ensure_model(&state, dir.path(), "whisper-tiny").unwrap_err();
        assert!(matches!(err, SttError::ModelNotDownloaded(_)));
    }

    #[test]
    fn ensure_model_errors_for_unknown_model_id() {
        let dir = tempfile::tempdir().unwrap();
        let state = WhisperState::new();
        let err = ensure_model(&state, dir.path(), "not-a-model").unwrap_err();
        assert!(matches!(err, SttError::ModelNotFound(_)));
    }
}
