# stt_core

Framework-agnostic speech-to-text logic for Rust apps: cloud transcription via
[ElevenLabs](https://elevenlabs.io) and local, offline transcription via
Whisper/Moonshine models (through [`whisper-apr`](https://crates.io/crates/whisper-apr),
a pure-Rust inference backend — no C++ toolchain required).

This crate has no dependency on any particular application framework (no Tauri,
no web framework). It does not read configuration from environment variables,
does not manage its own background threads, and does not assume a specific
on-disk layout beyond a caller-supplied model directory. It is designed to be
embedded in a desktop app, a CLI tool, or a server.

## Platform support

- **Cloud (ElevenLabs)** — works on every platform, including mobile.
- **Local (Whisper/Moonshine)** — desktop only. The `whisper` module is compiled
  out on `target_os = "android"` and `target_os = "ios"` via `#[cfg]`, since
  `whisper-apr` targets desktop platforms.

## Usage

```rust
use stt_core::{SttConfig, Provider, route_provider, transcribe_elevenlabs, AudioFormat};

let cfg = SttConfig {
    stt_provider: "elevenlabs".to_string(),
    // Read from your own secret store/config — never commit a real key.
    elevenlabs_api_key: load_elevenlabs_key_from_config(),
    whisper_model_size: "whisper-tiny".to_string(),
};

match route_provider(&cfg) {
    Provider::ElevenLabs => {
        let text = transcribe_elevenlabs(&cfg.elevenlabs_api_key, audio_bytes, AudioFormat::Wav).await?;
    }
    Provider::Whisper => {
        // see below
    }
}
```

### Local Whisper/Moonshine transcription

```rust
use stt_core::{WhisperState, transcribe_whisper};
use std::path::Path;

let whisper_state = WhisperState::new();
let models_dir = Path::new("/path/to/models");

// `transcribe_whisper` is synchronous — run it on a blocking thread
// (e.g. `tokio::task::spawn_blocking`) so it never stalls an async runtime.
let text = transcribe_whisper(&audio_bytes, "wav", "whisper-tiny", &whisper_state, models_dir)?;
```

### Downloading a model

```rust
use stt_core::download_and_convert_model;
use std::path::Path;

download_and_convert_model(Path::new("/path/to/models"), "whisper-tiny", |progress| {
    println!("{}: {:.1}%", progress.phase, progress.percent);
}).await?;
```

`stt_core::whisper::MODELS` lists every supported model id, its HuggingFace
source repo, and estimated download size. Downloads (including the auxiliary
tokenizer/vocab/preprocessor files) are pinned to a specific commit revision
per model rather than the mutable `main` ref, and bounded independent of
whatever `Content-Length` the server reports, so a `main` update or a
misbehaving response can't force an unbounded download. Revision pinning is
not a content-integrity check, though: there is no expected hash for the
downloaded bytes, so a compromised HuggingFace account or repo, or a
TLS-intercepting proxy with a locally trusted CA, can still serve different
(but correctly-sized) bytes for the pinned revision and the app will convert
and load them without detecting it. `find_model_entry` and `model_path`
are cheap, synchronous lookups a caller can use to check whether a model is
present (e.g. `model_path(dir, id).is_some_and(|p| p.exists())`) without
loading anything. `ensure_model` is **not** a cheap state probe — it reads the
entire `.apr` file off disk (hundreds of MB for larger models) and loads it
into memory; `preload_model` is the intended warm-up entry point for that.

## Error handling

All fallible operations return `Result<T, stt_core::SttError>`. `SttError`
implements `std::error::Error` and `Display`, but not every variant's message
is safe to show end users as-is: `MissingApiKey`'s text is user-presentable,
while `ApiError`, `HttpRequest`, `Io`, and `ModelLoad` interpolate raw upstream
response bodies, reqwest errors, or `std::io::Error` messages that can contain
URLs or local filesystem paths. Log those variants and present a generic
message to end users; don't forward `Display` output directly to a UI.

## Adding this crate as a dependency

Until this crate is published to crates.io, reference it as a git dependency
pinned to a tag. The published package is named `synapse-stt-core` (the
import path stays `stt_core` since the library target is named separately
from the package):

```toml
[dependencies]
stt_core = { git = "https://github.com/Saimirbaci/stt_core", tag = "v0.1.0", package = "synapse-stt-core" }
```

Pin to a specific tag or commit — pointing at a branch will silently drift as
the crate evolves.

## License

Dual-licensed under either the [MIT license](LICENSE-MIT) or the
[Apache License, Version 2.0](LICENSE-APACHE), at your option.
