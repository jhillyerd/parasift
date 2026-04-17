//! JSONL output record (SPEC §6).

use serde::Serialize;
use serde_json::Value;
use std::io::{self, Write};
use std::sync::Mutex;

#[derive(Debug, Serialize)]
pub struct ResultRecord {
    pub source: String,
    pub status: &'static str, // "ok" | "error"
    pub result: Option<Value>,
    pub error: Option<String>,
}

impl ResultRecord {
    pub fn ok(source: String, result: Value) -> Self {
        Self {
            source,
            status: "ok",
            result: Some(result),
            error: None,
        }
    }

    pub fn err(source: String, error: impl Into<String>) -> Self {
        Self {
            source,
            status: "error",
            result: None,
            error: Some(error.into()),
        }
    }
}

/// Serialize and write a record as a single JSONL line to stdout, holding
/// a mutex so concurrent writes from multiple rayon workers don't interleave.
pub fn emit(lock: &Mutex<io::Stdout>, record: &ResultRecord) {
    let line = serde_json::to_string(record).unwrap_or_else(|_| {
        // Extremely defensive fallback: should not happen for our shapes.
        format!(
            "{{\"source\":{:?},\"status\":\"error\",\"result\":null,\"error\":\"failed to serialize result record\"}}",
            record.source
        )
    });
    let mut out = lock.lock().expect("stdout mutex poisoned");
    // Ignore write errors; stdout being closed is not actionable here.
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}
