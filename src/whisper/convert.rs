use std::collections::HashMap;

use crate::error::SttError;

use super::{find_model_entry, get_model_config};

// =============================================================================
// SafeTensors → .apr Conversion
// =============================================================================

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut e: u32 = 1;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e += 1;
            }
            sign | ((127 - 15 + 1 - e) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | (0xff << 23) | (frac << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

/// Convert bfloat16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Extract f32 data from a SafeTensors tensor view.
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Option<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 => Some(
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        safetensors::Dtype::F16 => Some(
            data.chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        ),
        safetensors::Dtype::BF16 => Some(
            data.chunks_exact(2)
                .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
        ),
        _ => None,
    }
}

/// Build GPT-2 byte decoder (Unicode char → byte value).
fn build_gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut decoder = HashMap::new();
    // Printable ASCII maps to itself
    for b in b'!'..=b'~' {
        decoder.insert(char::from(b), b);
    }
    for b in 0xa1u8..=0xac {
        decoder.insert(char::from(b), b);
    }
    for b in 0xaeu8..=0xff {
        decoder.insert(char::from(b), b);
    }
    // Non-printable bytes get mapped to Unicode starting at U+0100
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !decoder.values().any(|&v| v == b) {
            if let Some(c) = char::from_u32(256 + n) {
                decoder.insert(c, b);
            }
            n += 1;
        }
    }
    decoder
}

/// Load mel filterbank from preprocessor_config.json.
fn load_mel_filters(preprocessor_json: &str) -> Result<(u32, u32, Vec<f32>), SttError> {
    let config: serde_json::Value = serde_json::from_str(preprocessor_json).map_err(|e| {
        SttError::TranscribeFailed(format!("Failed to parse preprocessor_config.json: {}", e))
    })?;

    let mel_array = config.get("mel_filters").ok_or_else(|| {
        SttError::TranscribeFailed("mel_filters not found in preprocessor_config.json".to_string())
    })?;

    let mel_2d: Vec<Vec<f64>> = serde_json::from_value(mel_array.clone())
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to parse mel_filters: {}", e)))?;

    let n_mels = mel_2d.len() as u32;
    let n_freqs = mel_2d.first().map_or(0, |row| row.len()) as u32;

    let data: Vec<f32> = mel_2d
        .into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect();

    Ok((n_mels, n_freqs, data))
}

/// Build Whisper vocabulary from vocab.json with special tokens.
fn build_whisper_vocab(vocab_json: &str) -> Result<whisper_apr::tokenizer::Vocabulary, SttError> {
    let token_map: HashMap<String, u32> = serde_json::from_str(vocab_json)
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to parse vocab.json: {}", e)))?;

    let mut tokens: Vec<(String, u32)> = token_map.into_iter().collect();
    tokens.sort_by_key(|(_, id)| *id);

    let byte_decoder = build_gpt2_byte_decoder();
    let mut vocab = whisper_apr::tokenizer::Vocabulary::new();

    for (token_str, _) in &tokens {
        let bytes: Vec<u8> = token_str
            .chars()
            .filter_map(|c| byte_decoder.get(&c).copied())
            .collect();
        vocab.add_token(bytes);
    }

    // Add special tokens up to SOT (50258)
    while vocab.len() < 50258 {
        vocab.add_token(vec![0]);
    }
    vocab.add_token(b"<|startoftranscript|>".to_vec()); // 50258

    // 99 language tokens (50259-50357)
    let languages = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
        "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
        "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
        "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
    ];
    for lang in languages {
        vocab.add_token(format!("<|{lang}|>").into_bytes());
    }

    // Task tokens (50358-50363)
    vocab.add_token(b"<|translate|>".to_vec());
    vocab.add_token(b"<|transcribe|>".to_vec());
    vocab.add_token(b"<|startoflm|>".to_vec());
    vocab.add_token(b"<|startofprev|>".to_vec());
    vocab.add_token(b"<|nospeech|>".to_vec());
    vocab.add_token(b"<|notimestamps|>".to_vec());

    // Timestamp tokens (50364-51864)
    for i in 0..1501 {
        let seconds = i as f32 * 0.02;
        vocab.add_token(format!("<|{seconds:.2}|>").into_bytes());
    }

    Ok(vocab)
}

/// Build Moonshine vocabulary from tokenizer.json.
fn build_moonshine_vocab(
    tokenizer_json: &str,
) -> Result<whisper_apr::tokenizer::Vocabulary, SttError> {
    let json: serde_json::Value = serde_json::from_str(tokenizer_json).map_err(|e| {
        SttError::TranscribeFailed(format!("Failed to parse tokenizer.json: {}", e))
    })?;

    let vocab_obj = json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            SttError::TranscribeFailed("Missing model.vocab in tokenizer.json".to_string())
        })?;

    let mut id_to_piece: HashMap<u32, String> = HashMap::new();
    for (piece, id_val) in vocab_obj {
        if let Some(id) = id_val.as_u64() {
            id_to_piece.insert(id as u32, piece.clone());
        }
    }

    if let Some(added) = json.get("added_tokens").and_then(|a| a.as_array()) {
        for token in added {
            if let (Some(id), Some(content)) = (
                token.get("id").and_then(|i| i.as_u64()),
                token.get("content").and_then(|c| c.as_str()),
            ) {
                id_to_piece
                    .entry(id as u32)
                    .or_insert_with(|| content.to_string());
            }
        }
    }

    let max_id = id_to_piece.keys().copied().max().unwrap_or(0);
    let mut vocab = whisper_apr::tokenizer::Vocabulary::new();
    for id in 0..=max_id {
        let piece = id_to_piece.get(&id).cloned().unwrap_or_default();
        vocab.add_token(piece.into_bytes());
    }

    Ok(vocab)
}

/// Build APR metadata from a model config.
fn build_metadata(
    config: &whisper_apr::model::ModelConfig,
    source: &str,
) -> whisper_apr::format::AprV2Metadata {
    let mut meta = whisper_apr::format::AprV2Metadata {
        model_type: "speech-recognition".to_string(),
        architecture: Some(
            if config.is_moonshine() {
                "moonshine"
            } else {
                "whisper"
            }
            .to_string(),
        ),
        vocab_size: Some(config.n_vocab as usize),
        hidden_size: Some(config.n_text_state as usize),
        num_layers: Some(config.n_text_layer as usize),
        num_heads: Some(config.n_text_head as usize),
        source: Some(source.to_string()),
        original_format: Some("safetensors".to_string()),
        ..Default::default()
    };

    meta.custom.insert(
        "n_audio_state".into(),
        serde_json::json!(config.n_audio_state),
    );
    meta.custom.insert(
        "n_audio_layer".into(),
        serde_json::json!(config.n_audio_layer),
    );
    meta.custom.insert(
        "n_audio_head".into(),
        serde_json::json!(config.n_audio_head),
    );
    meta.custom
        .insert("n_audio_ctx".into(), serde_json::json!(config.n_audio_ctx));
    meta.custom
        .insert("n_text_ctx".into(), serde_json::json!(config.n_text_ctx));
    meta.custom
        .insert("n_mels".into(), serde_json::json!(config.n_mels));
    meta.custom.insert(
        "audio_frontend".into(),
        serde_json::json!(if config.is_moonshine() {
            "learned_conv"
        } else {
            "mel_filterbank"
        }),
    );
    meta.custom.insert(
        "model_family".into(),
        serde_json::json!(if config.is_moonshine() {
            "moonshine"
        } else {
            "whisper"
        }),
    );

    meta
}

/// Convert Whisper SafeTensors to .apr format.
pub fn convert_whisper_to_apr(
    safetensors_data: &[u8],
    vocab_json: &str,
    preprocessor_json: &str,
    model_id: &str,
) -> Result<Vec<u8>, SttError> {
    use whisper_apr::format::{AprV2Writer, TensorDType};

    let config = get_model_config(model_id)
        .ok_or_else(|| SttError::ModelNotFound(format!("Unknown model config: {}", model_id)))?;

    let tensors = safetensors::SafeTensors::deserialize(safetensors_data)
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to parse SafeTensors: {}", e)))?;

    let source = format!(
        "hf://{}",
        find_model_entry(model_id).map_or(model_id, |m| m.hf_repo)
    );
    let metadata = build_metadata(&config, &source);
    let mut writer = AprV2Writer::new(metadata);

    // Embed mel filterbank
    if let Ok((n_mels, n_freqs, mel_data)) = load_mel_filters(preprocessor_json) {
        writer.add_f32_tensor(
            "__mel_filters__",
            vec![n_mels as usize, n_freqs as usize],
            &mel_data,
        );
    }

    // Embed vocabulary
    if let Ok(vocab) = build_whisper_vocab(vocab_json) {
        let vocab_bytes = vocab.to_bytes();
        writer.add_tensor(
            "__vocab__",
            TensorDType::U8,
            vec![vocab_bytes.len()],
            vocab_bytes,
        );
    }

    // Convert and write tensors. A skipped tensor would silently produce a
    // structurally valid but incomplete/incorrect model, so an unsupported
    // dtype fails the whole conversion rather than being dropped.
    for (name, tensor) in tensors.tensors() {
        let Some(f32_data) = tensor_to_f32(tensor.data(), tensor.dtype()) else {
            log::warn!("tensor {} has unsupported dtype {:?}", name, tensor.dtype());
            return Err(SttError::TranscribeFailed(format!(
                "Tensor '{}' has unsupported dtype {:?}; cannot convert model",
                name,
                tensor.dtype()
            )));
        };
        let our_name = whisper_apr::format::map_tensor_name(&name);
        let shape: Vec<usize> = tensor.shape().to_vec();
        writer.add_f32_tensor(our_name, shape, &f32_data);
    }

    writer
        .write()
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to write APR: {}", e)))
}

/// Convert Moonshine SafeTensors to .apr format.
pub fn convert_moonshine_to_apr(
    safetensors_data: &[u8],
    tokenizer_json: &str,
    model_id: &str,
) -> Result<Vec<u8>, SttError> {
    use whisper_apr::format::{AprV2Writer, TensorDType};

    let config = get_model_config(model_id)
        .ok_or_else(|| SttError::ModelNotFound(format!("Unknown model config: {}", model_id)))?;

    let tensors = safetensors::SafeTensors::deserialize(safetensors_data)
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to parse SafeTensors: {}", e)))?;

    let source = format!(
        "hf://{}",
        find_model_entry(model_id).map_or(model_id, |m| m.hf_repo)
    );
    let metadata = build_metadata(&config, &source);
    let mut writer = AprV2Writer::new(metadata);

    // Embed vocabulary
    if let Ok(vocab) = build_moonshine_vocab(tokenizer_json) {
        let vocab_bytes = vocab.to_bytes();
        writer.add_tensor(
            "__vocab__",
            TensorDType::U8,
            vec![vocab_bytes.len()],
            vocab_bytes,
        );
    }

    // Convert tensors, tracking embed_tokens for tied embeddings
    let mut has_proj_out = false;
    let mut embed_tokens_data: Option<(Vec<usize>, Vec<f32>)> = None;

    for (name, tensor) in tensors.tensors() {
        let Some(f32_data) = tensor_to_f32(tensor.data(), tensor.dtype()) else {
            log::warn!("tensor {} has unsupported dtype {:?}", name, tensor.dtype());
            return Err(SttError::TranscribeFailed(format!(
                "Tensor '{}' has unsupported dtype {:?}; cannot convert model",
                name,
                tensor.dtype()
            )));
        };
        let our_name = whisper_apr::format::map_moonshine_tensor_name(&name);
        let shape: Vec<usize> = tensor.shape().to_vec();

        if our_name == "decoder.proj_out.weight" {
            has_proj_out = true;
        }
        if our_name == "decoder.token_embedding.weight" {
            embed_tokens_data = Some((shape.clone(), f32_data.clone()));
        }

        writer.add_f32_tensor(our_name, shape, &f32_data);
    }

    // Handle tied embeddings: clone embed_tokens → proj_out if missing
    if !has_proj_out {
        if let Some((shape, data)) = embed_tokens_data {
            log::info!("tied embeddings: cloning embed_tokens -> proj_out");
            writer.add_f32_tensor("decoder.proj_out.weight", shape, &data);
        }
    }

    writer
        .write()
        .map_err(|e| SttError::TranscribeFailed(format!("Failed to write APR: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_to_f32_roundtrips_common_values() {
        // 1.0 in f16 is 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0 in f16 is 0x0000
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -2.0 in f16 is 0xC000
        assert!((f16_to_f32(0xC000) - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn bf16_to_f32_matches_truncated_mantissa() {
        // 1.0 in bf16 is 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tensor_to_f32_handles_f32_dtype() {
        let value = 3.5f32;
        let bytes = value.to_le_bytes().to_vec();
        let out = tensor_to_f32(&bytes, safetensors::Dtype::F32).unwrap();
        assert_eq!(out, vec![3.5]);
    }

    #[test]
    fn tensor_to_f32_returns_none_for_unsupported_dtype() {
        assert!(tensor_to_f32(&[0, 0], safetensors::Dtype::I8).is_none());
    }

    #[test]
    fn build_gpt2_byte_decoder_covers_all_256_bytes() {
        let decoder = build_gpt2_byte_decoder();
        let mut seen: Vec<u8> = decoder.values().copied().collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 256);
    }
}
