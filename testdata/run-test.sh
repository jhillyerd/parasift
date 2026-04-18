#!/usr/bin/env bash
set -eo pipefail

# change to parent directory of this script
cd "$(dirname "$(dirname "$0")")"

# source .env if present
if [ -f .env ]; then
  source .env
fi

# jq or fallback to cat
if ! command -v jq &> /dev/null; then
  echo "jq not found, not formatting output" >&2
  function jq() {
    cat
  }
fi

export PARASIFT_LLM_URL PARASIFT_LLM_API_KEY PARASIFT_MODEL \
  PARASIFT_CONCURRENCY RUST_LOG

# run the test
cargo run -- --config testdata/classifier.yaml --input testdata/docs | jq
