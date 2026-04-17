//! Load and validate the `parasift` YAML config file (SPEC §4).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Parsed configuration, with the schema represented as a JSON `Value`
/// so it can be fed directly to the JSON Schema validator.
#[derive(Debug, Clone)]
pub struct Config {
    pub instructions: String,
    pub schema: Value,
    pub model: Option<String>,
    pub max_retries: u32,
}

/// Keys permitted at the top level of the config (SPEC §4.1).
const ALLOWED_KEYS: &[&str] = &["instructions", "schema", "model", "max_retries"];

pub fn load(path: &Path) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file: {}", path.display()))?;
    parse(&text)
}

pub fn parse(text: &str) -> Result<Config> {
    // Parse YAML into a generic map first so we can (a) reject unknown keys
    // and (b) convert the `schema` sub-tree into a `serde_json::Value`.
    let raw: serde_yaml::Value = serde_yaml::from_str(text).context("failed to parse YAML")?;
    let map: BTreeMap<String, serde_yaml::Value> = match raw {
        serde_yaml::Value::Mapping(m) => {
            let mut out = BTreeMap::new();
            for (k, v) in m {
                let key = match k {
                    serde_yaml::Value::String(s) => s,
                    other => bail!("top-level keys must be strings, got {:?}", other),
                };
                out.insert(key, v);
            }
            out
        }
        _ => bail!("config root must be a YAML mapping"),
    };

    // Reject unknown top-level keys.
    for k in map.keys() {
        if !ALLOWED_KEYS.contains(&k.as_str()) {
            bail!(
                "unknown top-level config key: '{}' (allowed: {})",
                k,
                ALLOWED_KEYS.join(", ")
            );
        }
    }

    // instructions: required string.
    let instructions = match map.get("instructions") {
        Some(serde_yaml::Value::String(s)) => s.clone(),
        Some(_) => bail!("'instructions' must be a string"),
        None => bail!("missing required config key: 'instructions'"),
    };

    // schema: required mapping -> serde_json::Value.
    let schema_yaml = map
        .get("schema")
        .ok_or_else(|| anyhow!("missing required config key: 'schema'"))?;
    let schema: Value =
        serde_json::to_value(schema_yaml).context("failed to convert 'schema' into JSON Value")?;
    if !schema.is_object() {
        bail!("'schema' must be a JSON Schema object (mapping)");
    }

    // model: optional string.
    let model = match map.get("model") {
        Some(serde_yaml::Value::String(s)) => Some(s.clone()),
        Some(serde_yaml::Value::Null) | None => None,
        Some(_) => bail!("'model' must be a string"),
    };

    // max_retries: optional integer >= 0, default 2.
    let max_retries = match map.get("max_retries") {
        Some(serde_yaml::Value::Number(n)) => {
            let as_i64 = n
                .as_i64()
                .ok_or_else(|| anyhow!("'max_retries' must be an integer"))?;
            if as_i64 < 0 {
                bail!("'max_retries' must be >= 0");
            }
            as_i64 as u32
        }
        Some(serde_yaml::Value::Null) | None => 2,
        Some(_) => bail!("'max_retries' must be an integer"),
    };

    Ok(Config {
        instructions,
        schema,
        model,
        max_retries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
instructions: "classify"
schema:
  type: object
"#;

    #[test]
    fn loads_minimal_config_with_defaults() {
        let cfg = parse(MINIMAL).expect("parse");
        assert_eq!(cfg.instructions, "classify");
        assert_eq!(cfg.max_retries, 2);
        assert!(cfg.model.is_none());
        assert_eq!(cfg.schema["type"], "object");
    }

    #[test]
    fn rejects_unknown_top_level_key() {
        let bad = format!("{}\nweird_key: 1\n", MINIMAL);
        let err = parse(&bad).unwrap_err().to_string();
        assert!(err.contains("unknown top-level config key"), "got: {err}");
    }

    #[test]
    fn requires_instructions() {
        let text = r#"
schema:
  type: object
"#;
        let err = parse(text).unwrap_err().to_string();
        assert!(err.contains("instructions"));
    }

    #[test]
    fn requires_schema() {
        let text = r#"
instructions: "x"
"#;
        let err = parse(text).unwrap_err().to_string();
        assert!(err.contains("schema"));
    }

    #[test]
    fn rejects_negative_max_retries() {
        let text = r#"
instructions: "x"
max_retries: -1
schema:
  type: object
"#;
        let err = parse(text).unwrap_err().to_string();
        assert!(err.contains("max_retries"));
    }

    #[test]
    fn accepts_example_from_repo() {
        let text = include_str!("../examples/classifier.yaml");
        let cfg = parse(text).expect("example classifier.yaml");
        assert_eq!(cfg.model.as_deref(), Some("local-model"));
        assert_eq!(cfg.max_retries, 2);
    }
}
