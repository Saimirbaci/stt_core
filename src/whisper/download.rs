use std::path::{Path, PathBuf};

use crate::error::SttError;
use crate::whisper::convert::{convert_moonshine_to_apr, convert_whisper_to_apr};
use crate::whisper::{find_model_entry, ModelEntry};
use crate::DownloadProgress;

/// Upper bound for an auxiliary config/vocab file (`tokenizer.json`,
/// `vocab.json`, `preprocessor_config.json`). The largest of these observed
/// across the registry is a few MB; this leaves generous headroom while
/// still closing off an unbounded-allocation response from the untrusted
/// HuggingFace origin.
const AUX_FILE_MAX_BYTES: u64 = 16_000_000;

/// Bound the `model.safetensors` download for `entry` at 1.25x its actual
/// measured size (`safetensors_size`), which — unlike the display-only
/// `estimated_size` — is the real download size at the pinned revision.
/// This is a security bound, not a display estimate, so it must be derived
/// from real measured sizes rather than an arbitrary multiplier over a
/// different field.
pub(crate) fn max_download_bytes(entry: &ModelEntry) -> u64 {
    entry
        .safetensors_size
        .saturating_add(entry.safetensors_size / 4)
        .max(10_000_000)
}

/// Identifies a single file to fetch from a pinned HuggingFace revision.
struct HfFileRequest<'a> {
    hf_repo: &'a str,
    revision: &'a str,
    filename: &'a str,
    model_id: &'a str,
}

/// Download a file from HuggingFace, reporting progress via `progress`.
/// `max_bytes` bounds both the initial allocation and the total streamed
/// size — the server's `Content-Length` is untrusted input, not a size
/// contract.
async fn download_hf_file(
    client: &reqwest::Client,
    req: HfFileRequest<'_>,
    phase: &str,
    max_bytes: u64,
    mut progress: impl FnMut(DownloadProgress) + Send,
) -> Result<Vec<u8>, SttError> {
    let HfFileRequest {
        hf_repo,
        revision,
        filename,
        model_id,
    } = req;
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        hf_repo, revision, filename
    );

    log::info!("downloading {} from {}", filename, url);

    let response = client.get(&url).send().await.map_err(|e| {
        SttError::HttpRequest(format!("Download request failed for {}: {}", filename, e))
    })?;

    if !response.status().is_success() {
        return Err(SttError::HttpRequest(format!(
            "Download failed for {} with status: {}",
            filename,
            response.status()
        )));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    if total_bytes > max_bytes {
        return Err(SttError::HttpRequest(format!(
            "Refusing to download {}: server-reported size {} bytes exceeds the {} byte limit",
            filename, total_bytes, max_bytes
        )));
    }
    let mut downloaded_bytes: u64 = 0;
    // `total_bytes` is already known <= `max_bytes` from the check above.
    let mut data = Vec::with_capacity(total_bytes as usize);

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| SttError::HttpRequest(format!("Download error: {}", e)))?;
        downloaded_bytes += chunk.len() as u64;
        if downloaded_bytes > max_bytes {
            return Err(SttError::HttpRequest(format!(
                "Aborting download of {}: exceeded the {} byte limit",
                filename, max_bytes
            )));
        }
        data.extend_from_slice(&chunk);

        let percent = if total_bytes > 0 {
            (downloaded_bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };

        progress(DownloadProgress {
            model_size: model_id.to_string(),
            downloaded_bytes,
            total_bytes,
            percent,
            phase: phase.to_string(),
        });
    }

    Ok(data)
}

/// Download a small auxiliary file from HuggingFace (no progress events).
/// Bounded the same way as `download_hf_file` — the server's
/// `Content-Length` and the actual streamed byte count are both untrusted
/// and checked against `AUX_FILE_MAX_BYTES`, so a compromised/misconfigured
/// origin can't force an unbounded allocation via this path either.
async fn download_hf_aux_file(
    client: &reqwest::Client,
    hf_repo: &str,
    revision: &str,
    filename: &str,
) -> Result<String, SttError> {
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        hf_repo, revision, filename
    );

    let response =
        client.get(&url).send().await.map_err(|e| {
            SttError::HttpRequest(format!("Download failed for {}: {}", filename, e))
        })?;

    if !response.status().is_success() {
        return Err(SttError::HttpRequest(format!(
            "Download failed for {} with status: {}",
            filename,
            response.status()
        )));
    }

    let total_bytes = response.content_length().unwrap_or(0);
    if total_bytes > AUX_FILE_MAX_BYTES {
        return Err(SttError::HttpRequest(format!(
            "Refusing to download {}: server-reported size {} bytes exceeds the {} byte limit",
            filename, total_bytes, AUX_FILE_MAX_BYTES
        )));
    }

    use futures_util::StreamExt;
    let mut stream = response.bytes_stream();
    // `total_bytes` is already known <= `AUX_FILE_MAX_BYTES` from the check above.
    let mut data = Vec::with_capacity(total_bytes as usize);
    let mut downloaded_bytes: u64 = 0;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| SttError::HttpRequest(format!("Download error: {}", e)))?;
        downloaded_bytes += chunk.len() as u64;
        if downloaded_bytes > AUX_FILE_MAX_BYTES {
            return Err(SttError::HttpRequest(format!(
                "Aborting download of {}: exceeded the {} byte limit",
                filename, AUX_FILE_MAX_BYTES
            )));
        }
        data.extend_from_slice(&chunk);
    }

    String::from_utf8(data)
        .map_err(|e| SttError::HttpRequest(format!("{} is not valid UTF-8: {}", filename, e)))
}

/// Download a model from HuggingFace, convert SafeTensors to `.apr`, and save it
/// into `models_dir`. Reports progress via `progress` (downloading/converting
/// phases) — callers translate this into whatever transport they use (Tauri
/// events, a WebSocket, etc).
pub async fn download_and_convert_model(
    models_dir: &Path,
    model_size: &str,
    mut progress: impl FnMut(DownloadProgress) + Send,
) -> Result<PathBuf, SttError> {
    let entry = find_model_entry(model_size)
        .ok_or_else(|| SttError::ModelNotFound(format!("Unknown model: {}", model_size)))?;

    let dest = models_dir.join(entry.apr_filename);
    let client = reqwest::Client::new();

    log::info!(
        "starting download+convert for '{}' from {}",
        model_size,
        entry.hf_repo
    );

    // Step 1: Download model.safetensors (with progress). Bound at 1.25x the
    // real measured download size rather than trusting the server's
    // Content-Length, which would otherwise let a malicious/misconfigured
    // response force an unbounded allocation and download.
    let max_bytes = max_download_bytes(entry);
    let safetensors_data = download_hf_file(
        &client,
        HfFileRequest {
            hf_repo: entry.hf_repo,
            revision: entry.revision,
            filename: "model.safetensors",
            model_id: model_size,
        },
        "downloading",
        max_bytes,
        &mut progress,
    )
    .await?;

    // Step 2: Report converting phase
    progress(DownloadProgress {
        model_size: model_size.to_string(),
        downloaded_bytes: 0,
        total_bytes: 0,
        percent: 0.0,
        phase: "converting".to_string(),
    });

    // Step 3: Download auxiliary files and convert
    let apr_data = if entry.is_moonshine {
        let tokenizer_json =
            download_hf_aux_file(&client, entry.hf_repo, entry.revision, "tokenizer.json").await?;
        convert_moonshine_to_apr(&safetensors_data, &tokenizer_json, model_size)?
    } else {
        let vocab_json =
            download_hf_aux_file(&client, entry.hf_repo, entry.revision, "vocab.json").await?;
        let preprocessor_json = download_hf_aux_file(
            &client,
            entry.hf_repo,
            entry.revision,
            "preprocessor_config.json",
        )
        .await?;
        convert_whisper_to_apr(
            &safetensors_data,
            &vocab_json,
            &preprocessor_json,
            model_size,
        )?
    };

    // Step 4: Write .apr file atomically
    let temp_dest = dest.with_extension("apr.part");
    std::fs::write(&temp_dest, &apr_data)
        .map_err(|e| SttError::Io(format!("Failed to write model file: {}", e)))?;
    std::fs::rename(&temp_dest, &dest)
        .map_err(|e| SttError::Io(format!("Failed to finalize model file: {}", e)))?;

    log::info!(
        "'{}' downloaded and converted successfully ({} bytes .apr)",
        model_size,
        apr_data.len()
    );

    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn download_and_convert_model_errors_for_unknown_model() {
        let dir = tempfile::tempdir().unwrap();
        let calls: Arc<Mutex<Vec<DownloadProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let calls_clone = calls.clone();

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(download_and_convert_model(
                dir.path(),
                "not-a-model",
                move |p| {
                    calls_clone.lock().unwrap().push(p);
                },
            ))
            .unwrap_err();

        assert!(matches!(err, SttError::ModelNotFound(_)));
        // No progress should have been reported before model resolution failed.
        assert!(calls.lock().unwrap().is_empty());
    }
}
