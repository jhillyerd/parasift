# parasift

Classify files in a directory using a local LLM inference server exposing
an OpenAI-compatible chat completions API. `parasift` constrains model
output against a user-supplied JSON Schema, retries on validation failure
with feedback to the model, and emits one JSONL record per input file to
stdout.

See [`SPEC.md`](SPEC.md) for the full contract.

## Requirements

- A running OpenAI-compatible chat completions endpoint (e.g.
  `llama-server`, `vllm`, LM Studio).
- The `PARASIFT_LLM_URL` environment variable set to the server's base
  URL (e.g. `http://localhost:8080/v1`), or equivalently the
  `--llm-url` flag.

## Install

```sh
cargo build --release
```

The binary is produced at `target/release/parasift`.

## Usage

```
parasift --config <config.yaml> --input <dir> [--concurrency N] [--hide-filename]
```

- `-c, --config` — YAML config bundling the instructions and output JSON Schema.
- `-i, --input` — directory of files to classify (non-recursive).
- `-j, --concurrency` — in-flight requests; should match the server's slot count.
- `--hide-filename` — don't include the basename in the prompt (useful for
  benchmarking where filenames may leak labels).

Results are written as JSONL on stdout, one record per file:

```json
{"source":"docs/a.txt","status":"ok","result":{"category":"invoice","confidence":0.92},"error":null}
```

Per-file failures are reported in-band (`status: "error"`) and do not cause
a non-zero exit.

## Example

With a config like `examples/classifier.yaml`:

```yaml
instructions: |
  You are a document classifier. Read the document and produce a JSON
  object that conforms to the provided schema. Do not include any
  commentary outside the JSON.

model: local-model
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

See the [`examples/`](examples/) directory for ready-to-use configs.

## License

MIT — see [`LICENSE`](LICENSE).
