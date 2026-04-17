//! Schema-driven example generation (SPEC §7).

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

/// Generate an example output for the given JSON Schema.
///
/// Preference order, per SPEC §7:
/// 1. `examples[0]` if present
/// 2. `default` if present
/// 3. recursive type-based synthesis
pub fn generate_example(schema: &Value) -> Value {
    // 1. examples[0]
    if let Some(Value::Array(arr)) = schema.get("examples") {
        if let Some(first) = arr.first() {
            return first.clone();
        }
    }
    // 2. default
    if let Some(default) = schema.get("default") {
        return default.clone();
    }
    // 3. synthesize by type
    synthesize(schema)
}

fn synthesize(schema: &Value) -> Value {
    // const / enum take precedence if present.
    if let Some(c) = schema.get("const") {
        return c.clone();
    }
    if let Some(Value::Array(opts)) = schema.get("enum") {
        if let Some(first) = opts.first() {
            return first.clone();
        }
    }

    // Determine the effective type. `type` may be a string or an array; in
    // the array case, take the first entry.
    let ty = match schema.get("type") {
        Some(Value::String(s)) => s.as_str(),
        Some(Value::Array(arr)) => arr.first().and_then(|v| v.as_str()).unwrap_or("null"),
        _ => "object",
    };

    match ty {
        "string" => json!(""),
        "number" | "integer" => json!(0),
        "boolean" => json!(false),
        "null" => Value::Null,
        "array" => json!([]),
        "object" => synthesize_object(schema),
        _ => Value::Null,
    }
}

fn synthesize_object(schema: &Value) -> Value {
    let mut out = Map::new();
    if let Some(Value::Object(props)) = schema.get("properties") {
        for (name, sub) in props {
            out.insert(name.clone(), generate_example(sub));
        }
    }
    Value::Object(out)
}

/// Validate the generated example against the schema itself (SPEC §7.4).
pub fn validate_example_against_schema(schema: &Value, example: &Value) -> Result<()> {
    let validator = jsonschema::options()
        .with_draft(jsonschema::Draft::Draft202012)
        .build(schema)
        .map_err(|e| anyhow::anyhow!("failed to compile schema: {e}"))?;
    if let Err(errors) = collect_errors(&validator, example) {
        bail!(
            "generated example does not validate against schema: {}",
            errors.join("; ")
        );
    }
    Ok(())
}

/// Collect all validation errors as `(path, message)` pairs, flattened into
/// one string per error, returning `Err` with the list when any exist.
pub fn collect_errors(
    validator: &jsonschema::Validator,
    instance: &Value,
) -> std::result::Result<(), Vec<String>> {
    let mut errs = Vec::new();
    for e in validator.iter_errors(instance) {
        let path = e.instance_path.to_string();
        let path = if path.is_empty() {
            "/".to_string()
        } else {
            path
        };
        errs.push(format!("{path} {}", e));
    }
    if errs.is_empty() {
        Ok(())
    } else {
        Err(errs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_examples_first() {
        let schema = json!({
            "type": "object",
            "examples": [{"a": 1}]
        });
        assert_eq!(generate_example(&schema), json!({"a": 1}));
    }

    #[test]
    fn falls_back_to_default() {
        let schema = json!({"type": "integer", "default": 7});
        assert_eq!(generate_example(&schema), json!(7));
    }

    #[test]
    fn synthesizes_by_type() {
        assert_eq!(generate_example(&json!({"type": "string"})), json!(""));
        assert_eq!(generate_example(&json!({"type": "number"})), json!(0));
        assert_eq!(generate_example(&json!({"type": "integer"})), json!(0));
        assert_eq!(generate_example(&json!({"type": "boolean"})), json!(false));
        assert_eq!(generate_example(&json!({"type": "array"})), json!([]));
    }

    #[test]
    fn synthesizes_object_with_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "ok": {"type": "boolean"}
            }
        });
        let ex = generate_example(&schema);
        assert_eq!(ex, json!({"name": "", "count": 0, "ok": false}));
    }

    #[test]
    fn enum_uses_first_value() {
        let schema = json!({"type": "string", "enum": ["a", "b"]});
        assert_eq!(generate_example(&schema), json!("a"));
    }

    #[test]
    fn example_validates_against_schema_ok() {
        let schema = json!({
            "type": "object",
            "required": ["category"],
            "properties": {"category": {"type": "string", "enum": ["x", "y"]}}
        });
        let ex = generate_example(&schema);
        validate_example_against_schema(&schema, &ex).unwrap();
    }

    #[test]
    fn detects_invalid_synthesized_example() {
        // synthesized "" will not satisfy minLength: 1.
        let schema = json!({"type": "string", "minLength": 1});
        let ex = generate_example(&schema);
        let err = validate_example_against_schema(&schema, &ex).unwrap_err();
        assert!(err.to_string().contains("does not validate"));
    }
}
