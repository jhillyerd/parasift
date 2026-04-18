//! parasift — classify files in a directory using an OpenAI-compatible
//! chat completions endpoint.
//!
//! See `SPEC.md` for the full contract. This binary is the reference
//! Rust implementation.

mod classify;
mod client;
mod config;
mod example;
mod output;

use anyhow::{bail, Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::classify::Pipeline;
use crate::client::{ChatClient, API_KEY_ENV, CONCURRENCY_ENV, URL_ENV};
use crate::output::emit;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "parasift",
    about = "Classify files in a directory using an OpenAI-compatible chat completions API.",
    version
)]
struct Cli {
    /// Path to the YAML config file (see SPEC §4).
    #[arg(long, short = 'c')]
    config: PathBuf,

    /// Directory containing files to classify (non-recursive).
    #[arg(long, short = 'i')]
    input: PathBuf,

    /// Number of in-flight requests against the inference server.
    /// Should match the server's slot count.
    #[arg(long, short = 'j', env = CONCURRENCY_ENV, default_value_t = 1)]
    concurrency: usize,

    /// Base URL of the OpenAI-compatible chat completions endpoint
    /// (e.g. `http://localhost:8080/v1`). Required.
    #[arg(long, env = URL_ENV, hide_env_values = false)]
    llm_url: String,

    /// Bearer token for the inference endpoint. Optional; only needed
    /// for hosted providers that require auth.
    #[arg(long, env = API_KEY_ENV, hide_env_values = true)]
    llm_api_key: Option<String>,

    /// Hide filenames from the model. By default the file's basename is
    /// included in the prompt; pass this flag for unbiased benchmarking
    /// where filenames may leak the ground-truth label.
    #[arg(long)]
    hide_filename: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize tracing. Respects RUST_LOG; defaults to `warn` so a
    // normal invocation stays quiet. Writes to stderr so stdout stays
    // reserved for the JSONL records (SPEC §5).
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    if cli.concurrency < 1 {
        bail!("--concurrency must be >= 1");
    }

    if cli.llm_url.trim().is_empty() {
        bail!("--llm-url (or ${URL_ENV}) must not be empty");
    }

    // Load & parse config (SPEC §4).
    let cfg = config::load(&cli.config)?;

    // Build the schema validator, bound to draft 2020-12.
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(&cfg.schema)
        .map_err(|e| anyhow::anyhow!("failed to compile JSON schema: {e}"))?;
    let validator = Arc::new(validator);

    // Generate and self-validate the prompt example (SPEC §7).
    let example = example::generate_example(&cfg.schema);
    example::validate_example_against_schema(&cfg.schema, &example)
        .context("generated schema example")?;
    let example_json = serde_json::to_string_pretty(&example).expect("example serialization");

    // Connect to the inference server. The client's idle-connection
    // pool is sized from `--concurrency`.
    let client = ChatClient::new(
        &cli.llm_url,
        cli.llm_api_key.as_deref().filter(|s| !s.is_empty()),
        cli.concurrency,
    );

    // Enumerate input files, non-recursive, files only, skip dot-files.
    let files = list_input_files(&cli.input)?;
    if files.is_empty() {
        // Exit cleanly — the run itself succeeded, there's just nothing to do.
        return Ok(());
    }

    let pipeline = Arc::new(Pipeline {
        client,
        validator,
        instructions: cfg.instructions,
        example_json,
        model: cfg.model,
        max_retries: cfg.max_retries,
        include_filename: !cli.hide_filename,
    });

    let stdout_lock = Arc::new(Mutex::new(io::stdout()));

    // 512 KiB stacks are plenty for our workers: the only recursion is
    // `example::synthesize` descending user-authored JSON schemas, and
    // everything else is flat. Shrinking from rayon's 2 MiB default
    // saves ~150 MiB of virtual address space at --concurrency=100.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cli.concurrency)
        .stack_size(512 * 1024)
        .build()
        .context("failed to build rayon thread pool")?;

    pool.install(|| {
        files.par_iter().for_each(|path| {
            let record = pipeline.classify_file(path);
            emit(&stdout_lock, &record);
        });
    });

    Ok(())
}

/// Non-recursive listing of regular files in `dir`, sorted for determinism.
/// Dot-files and sub-directories are skipped (SPEC §5 leaves recursion
/// and filtering implementation-defined; we pick the minimal contract).
fn list_input_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let read = std::fs::read_dir(dir)
        .with_context(|| format!("failed to read input directory: {}", dir.display()))?;

    let mut out = Vec::new();
    for entry in read {
        let entry = entry.with_context(|| format!("reading {}", dir.display()))?;
        let file_type = entry.file_type()?;
        if !file_type.is_file() {
            continue;
        }
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue;
        }
        out.push(entry.path());
    }
    out.sort();
    Ok(out)
}
