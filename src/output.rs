//! JSONL output record (SPEC §6).

use serde::Serialize;
use serde_json::Value;
use std::io::Write;
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

/// Serialize and write a record as a single JSONL line, holding a mutex so
/// concurrent writes from multiple rayon workers don't interleave.
///
/// Accepts any `Write` implementation so callers can write to stdout *or*
/// to a test buffer without coupling to `io::Stdout`.
pub fn emit(lock: &Mutex<impl Write>, record: &ResultRecord) {
    let line = serde_json::to_string(record).unwrap_or_else(|_| {
        serde_json::json!({
            "source": record.source,
            "status": "error",
            "result": null,
            "error": "failed to serialize result record"
        })
        .to_string()
    });
    let mut out = lock.lock().expect("output mutex poisoned");
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn emit_ok_record() {
        let buf = Mutex::new(Cursor::new(Vec::<u8>::new()));
        let record = ResultRecord::ok("test.txt".into(), serde_json::json!({"label": "yes"}));
        emit(&buf, &record);

        let data = buf.into_inner().unwrap().into_inner();
        let line = String::from_utf8(data).unwrap();
        let parsed: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["source"], "test.txt");
        assert_eq!(parsed["status"], "ok");
        assert_eq!(parsed["result"]["label"], "yes");
        assert!(parsed["error"].is_null());
    }

    #[test]
    fn emit_err_record() {
        let buf = Mutex::new(Cursor::new(Vec::<u8>::new()));
        let record = ResultRecord::err("bad.txt".into(), "something broke");
        emit(&buf, &record);

        let data = buf.into_inner().unwrap().into_inner();
        let line = String::from_utf8(data).unwrap();
        let parsed: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(parsed["source"], "bad.txt");
        assert_eq!(parsed["status"], "error");
        assert_eq!(parsed["error"], "something broke");
        assert!(parsed["result"].is_null());
    }
}
