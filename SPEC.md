# parasift — Specification

## 1. Overview

`parasift` is a command-line tool that classifies each file in a directory
using a local LLM inference server exposing an **OpenAI-compatible chat
completions API**. It parallelizes requests across the inference server's
available slots, constrains model output against a user-supplied JSON
Schema, retries on validation failure with feedback to the model, and
emits one JSONL result per input file to stdout.

This specification is intentionally tech-stack-agnostic. It fixes the
config file format, the inference endpoint contract, and the output
record shape so that independent implementations — in different
languages — remain interoperable.

---

## 2. Core responsibilities

An implementation of `parasift` MUST:

1. Read files from a user-specified input directory.
2. Load a single config file that bundles the classification
   instructions and the output JSON Schema.
3. Generate an example output from the schema (preferring schema
   `examples` / `default`, falling back to typed placeholders) and
   include that example in the prompt shown to the model.
4. Dispatch classification requests in parallel, up to a
   user-configurable concurrency limit matching the inference server's
   slot count.
5. Validate each model response against the JSON Schema.
6. On validation failure, retry up to `max_retries` times, feeding the
   previous invalid output and the validator's error detail back to the
   model. After exhausting retries, emit a failed result.
7. Emit classification results as **JSONL on stdout**, one record per
   input file.

---

## 3. Inference endpoint

- `parasift` targets an **OpenAI-compatible chat completions endpoint**
  (examples of compatible servers include `llama.cpp`'s `llama-server`,
  `vllm`, `LM Studio`, and others — the spec pins the contract, not the
  server).
- The base URL is supplied via the environment variable
  **`PARASIFT_LLM_URL`**.
  - This variable is **required**. The tool MUST exit with an error if
    it is unset or empty.
  - The value is treated as a base (for example
    `http://localhost:8080/v1`). The tool appends the standard path
    `/chat/completions`.
- Authentication, if required by the target endpoint, is
  implementation-defined (for example an optional
  `PARASIFT_LLM_API_KEY` environment variable). The spec does not
  require it.

---

## 4. Configuration file

The configuration format is **pinned** so the same file can be consumed
by any implementation.

- **Format:** YAML 1.2.
  - YAML is a superset of JSON, so a JSON Schema mapping can be pasted
    verbatim without escaping.
  - Multiline instruction prompts use YAML block scalars (`|`),
    avoiding newline escaping.
- **File extension:** `.yaml` or `.yml`.
- **Encoding:** UTF-8.

### 4.1 Top-level keys

| Key            | Type          | Required | Default | Description                                                                 |
|----------------|---------------|----------|---------|-----------------------------------------------------------------------------|
| `instructions` | string        | yes      | —       | Reusable prompt / system instructions shown to the model for every document. |
| `schema`       | mapping       | yes      | —       | JSON Schema object conforming to **JSON Schema draft 2020-12**, describing the classification output for a single document. |
| `max_retries`  | integer ≥ 0   | no       | `2`     | Retry attempts allowed after a validation failure before marking a file failed. |

Unknown top-level keys MUST be rejected. This keeps configs portable
and catches typos early.

### 4.2 Canonical example

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
    notes:
      type: string
  examples:
    - category: invoice
      confidence: 0.92
      notes: "Has invoice number and total."
```

---

## 5. Invocation contract

Abstract inputs (concrete CLI flag names are implementation-defined):

- **input directory** — path to a directory of files to classify. At
  minimum, non-recursive enumeration MUST be supported. Recursion, glob
  filters, and symlink handling are implementation-defined.
- **config file** — path to the YAML config described in §4.
- **concurrency limit** — integer ≥ 1. Matches the number of slots the
  inference server exposes.
- **model identifier** — optional string. Model name sent to the
  inference endpoint. When omitted, the implementation may send a
  sensible default (e.g. `"local-model"`).
- **`PARASIFT_LLM_URL`** — environment variable (§3).

Output:

- **JSONL on stdout**, one record per file processed, in any order.
- **Process exit status** reflects whether the run itself completed
  (config loaded, endpoint reachable, directory readable). Individual
  per-file failures do NOT cause a non-zero exit.

---

## 6. Result record shape

Every line of stdout MUST be a single JSON object with exactly these
keys:

| Key      | Type              | Notes                                                  |
|----------|-------------------|--------------------------------------------------------|
| `source` | string            | Path of the input file, as resolved by the tool.       |
| `status` | string            | `"ok"` or `"error"`.                                   |
| `result` | object \| null    | Validated classification object when `status == "ok"`, otherwise `null`. |
| `error`  | string \| null    | Human-readable error when `status == "error"`, otherwise `null`. |

Example successful line:

```json
{"source":"docs/a.pdf","status":"ok","result":{"category":"invoice","confidence":0.92},"error":null}
```

Example failed line:

```json
{"source":"docs/b.pdf","status":"error","result":null,"error":"schema validation failed after 2 retries: /confidence must be <= 1"}
```

---

## 7. Schema-driven example generation

For the prompt sent to the model, `parasift` MUST generate an example
object derived from the configured schema:

1. If the schema (or a sub-schema) defines `examples`, take the first
   entry.
2. Else if it defines `default`, use that value.
3. Else, recursively synthesize a minimal example by JSON Schema type:
   - `string` → `""`
   - `number` / `integer` → `0`
   - `boolean` → `false`
   - `array` → `[]`
   - `object` → walk `properties`, recursing per field
4. The generated example MUST itself validate against the schema. If
   it does not, the tool MUST exit with an error before processing any
   files.

---

## 8. Concurrency

- The tool keeps up to **N** in-flight requests against the inference
  server at all times until the input directory is exhausted, where
  **N** is the user-supplied concurrency limit.
- A single file's failure (read error, model error, validation error,
  retry exhaustion) MUST NOT halt the rest of the run.

---

## 9. Validation & retry loop

Per document:

1. Send a classification request with the configured `instructions`,
   the generated schema example, and the document contents.
2. Parse the model's response as JSON. A parse failure is treated as a
   validation error.
3. Validate the parsed object against `schema` using JSON Schema draft
   2020-12 semantics.
4. On failure, send a follow-up request that includes:
   - the original instructions,
   - the invalid output the model produced, and
   - the validator's error detail.
   Repeat up to `max_retries` times.
5. After the final failure, emit an `error` result record for this
   file and continue with the remaining files.

---

## 10. Non-goals

- No output destination other than JSONL on stdout.
- No built-in model hosting — `parasift` assumes an already-running
  OpenAI-compatible endpoint at `PARASIFT_LLM_URL`.
- No training, fine-tuning, or evaluation harness.
- Choice of language, HTTP client, YAML parser, and JSON Schema
  validator library are intentionally unspecified.

What **is** fixed across implementations:

- The config file format and its top-level keys (§4).
- The inference endpoint contract and the `PARASIFT_LLM_URL` env var
  (§3).
- The JSONL result record shape (§6).
- The schema-driven example generation rules (§7).
- The validate-and-retry-with-feedback loop (§9).
