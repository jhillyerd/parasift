# Parasift

A CLI for batch processing text files with LLMs, outputting structured JSON
results.  Designed for tasks like classification, extraction, and sentiment
analysis.

## Why?

I've found that smart LLMs running in agent harnesses struggle to reliably
delegate summarization tasks to cheaper language models.  Even when they do,
concurrency and throughput is often bottlenecked my the parent LLM.  Providing
the agent a CLI tool to perform this task allows for more consistency and better
throughput.

## How

Parasift provides a command-line tool for processing batches of text files
through an LLM.  It is designed to take advantage of the parallelism offered by
local LLM inference servers, such as `llama.cpp` or `vllm`, by issuing a
configurable number of concurrent requests.

Parasift provides a simple interface for constraining and validating model
output against a user-supplied JSON Schema. The same schema is used to provide
an example to the LLM to follow.  Your prompt instructs the model on how to
analyze the input file, and the schema ensures that the output is structured and
consistent.  If the model's response fails validation, Parasift will retry the
request with feedback on the validation errors, giving the model a chance to
correct its output.

See [`SPEC.md`](SPEC.md) for the full contract.

## Requirements

- An OpenAI-compatible chat completions endpoint (e.g. `llama-server`, `vllm`,
  LM Studio).  You may also specify an API key for hosted models.
- The `PARASIFT_LLM_URL` environment variable set to the server's base
  URL (e.g. `http://localhost:8080/v1`), or equivalently the
  `--llm-url` flag.

## Install

Prebuilt binaries are available on the [releases
page](https://github.com/jhillyerd/parasift/releases). You can also build from
source with the stable Rust toolchain:

```sh
cargo build --release
```

The binary is produced at `target/release/parasift`.

## Usage

```
parasift --config <config.yaml> --input <dir> [--concurrency N] [--model NAME] [--hide-filename]
```

- `-c, --config` — YAML config bundling the instructions and output JSON Schema.
- `-i, --input` — directory of files to classify (non-recursive).
- `-j, --concurrency` — in-flight requests; should match the server's slot count.
- `-m, --model` — model name sent to the endpoint (default: `local-model`).
- `--hide-filename` — don't include the basename in the prompt (useful for
  benchmarking where filenames may leak labels).

Per-file failures are reported in-band (`status: "error"`) and do not cause
a non-zero exit.

### Environment variables

It's recommended to use enviroment variables so you can change LLMs without
modifying the command line or config file.

| Variable | Flag | Description |
|---|---|---|
| `PARASIFT_LLM_URL` | `--llm-url` | Base URL of the OpenAI-compatible endpoint (required). |
| `PARASIFT_LLM_API_KEY` | `--llm-api-key` | Bearer token for the endpoint (optional, for hosted providers). |
| `PARASIFT_MODEL` | `-m` | Model name sent to the endpoint (default: `local-model`). |
| `PARASIFT_CONCURRENCY` | `-j` | In-flight request count (default: 1). |
| `RUST_LOG` | — | Logging level (e.g. `debug`, `warn`; default: `warn`). |

Setting `RUST_LOG=debug` will show the system and user prompts sent for each
file, which is useful for debugging but may produce a lot of output.

## Example

With a config like `examples/classifier.yaml`:

```yaml
instructions: |
  You are a document classifier. Read the document and produce a JSON
  object that conforms to the provided schema. Do not include any
  commentary outside the JSON.

max_retries: 2

schema:
  $schema: "https://json-schema.org/draft/2020-12/schema"
  type: object
  additionalProperties: false
  required: [category, confidence]
  properties:
    category:
      type: string
      enum: [invoice, receipt, contract, other]
    confidence:
      type: number
      minimum: 0
      maximum: 1
```

Run it:

```sh
export PARASIFT_LLM_URL=http://localhost:8080/v1
parasift -c examples/classifier.yaml -i testdata/docs -j 4 > results.jsonl
```

Results are written as JSONL on stdout, one record per file:

```json
{"source":"docs/a.txt","status":"ok","result":{"category":"invoice","confidence":0.92},"error":null}
{"source":"docs/b.txt","status":"error","result":null,"error":"validation error: missing required property `confidence`"}
```

See the [`examples/`](examples/) directory for ready-to-use configs.

## License

MIT — see [`LICENSE`](LICENSE).
