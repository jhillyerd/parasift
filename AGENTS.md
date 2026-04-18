# AGENTS.md

## Project layout

- `src/` — all source files (binary crate, no `lib.rs`):
  - `main.rs` — entry point, CLI, orchestration
  - `classify.rs` — classification logic
  - `client.rs` — LLM HTTP client
  - `config.rs` — schema/config loading
  - `example.rs` — schema-driven example generation
  - `output.rs` — output formatting
- `testdata/` — test fixtures
- `examples/` — example schemas/configs

## Build & run

```sh
cargo build
cargo run -- <args>
```

## Test

```sh
cargo test
```

## Lint & format

Run these before committing; they are expected to pass cleanly:

```sh
cargo fmt --check
cargo clippy -- -D warnings
```
