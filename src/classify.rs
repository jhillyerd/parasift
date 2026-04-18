//! Per-file classification pipeline with validate-and-retry loop (SPEC §9).

use crate::client::{ChatClient, Message};
use crate::example::collect_errors;
use crate::output::ResultRecord;
use jsonschema::Validator;
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info_span, warn};

/// Shared, read-only state threaded into every worker.
pub struct Pipeline {
    pub client: ChatClient,
    pub validator: Arc<Validator>,
    pub instructions: String,
    pub example_json: String,
    pub model: Option<String>,
    pub max_retries: u32,
    /// When true, the file's basename is included in the prompt sent to
    /// the model. When false (benchmark mode), only the contents are sent.
    pub include_filename: bool,
}

/// The outcome of a single attempt at getting a valid response from the
/// model. The retry loop matches on this to decide whether to retry with
/// feedback or give up.
enum StepResult {
    /// The response validated successfully.
    Ok(Value),
    /// The response failed; include `detail` in the error record and feed
    /// `feedback` back to the model on the next attempt. `raw_assistant`
    /// is the model's verbatim output, if any (absent for transport failures).
    Failed {
        detail: String,
        raw_assistant: Option<String>,
        feedback: String,
    },
}

impl Pipeline {
    /// Classify a single file, producing a [`ResultRecord`] that is safe to
    /// emit regardless of outcome (success or error).
    pub fn classify_file(&self, path: &Path) -> ResultRecord {
        let source = path.display().to_string();
        let _span = info_span!("classify", file = %source).entered();

        // SPEC §9.1 — read the document. Binary (non-UTF-8) → error record.
        let contents = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => return ResultRecord::err(source, format!("failed to read file: {e}")),
        };
        let text = match String::from_utf8(contents) {
            Ok(s) => s,
            Err(_) => return ResultRecord::err(source, "file is not valid UTF-8 text".to_string()),
        };

        // Build the initial system + user messages.
        let system_content = format!(
            "{instructions}\n\n\
             The output MUST be a single JSON object that validates against the provided schema. \
             Do not wrap the JSON in code fences or any commentary.\n\n\
             Here is an example of a valid output:\n{example}",
            instructions = self.instructions.trim_end(),
            example = self.example_json,
        );
        let user_content = if self.include_filename {
            let basename = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            format!(
                "Classify the following document. Output ONLY the JSON object.\n\nFilename: {basename}\n\n---\n{text}\n---"
            )
        } else {
            format!(
                "Classify the following document. Output ONLY the JSON object.\n\n---\n{text}\n---"
            )
        };

        let mut messages: Vec<Message> =
            vec![Message::system(system_content), Message::user(user_content)];
        // Anything past INITIAL_MESSAGES is retry feedback that we
        // replace on each failed attempt rather than append (see
        // `requeue_feedback`). Keeps request payloads bounded regardless
        // of `max_retries`.
        const INITIAL_MESSAGES: usize = 2;

        // attempts = 1 initial + up to max_retries retries (SPEC §9.4).
        let total_attempts = 1 + self.max_retries;
        for attempt in 0..total_attempts {
            log_prompt(attempt, &messages);
            let step = self.attempt(&messages, attempt);

            match step {
                StepResult::Ok(parsed) => {
                    debug!(attempt = attempt + 1, "validation succeeded");
                    return ResultRecord::ok(source, parsed);
                }
                StepResult::Failed {
                    detail,
                    raw_assistant,
                    feedback,
                } => {
                    warn!(attempt = attempt + 1, error = %detail, "attempt failed");
                    if attempt + 1 < total_attempts {
                        requeue_feedback(
                            &mut messages,
                            INITIAL_MESSAGES,
                            raw_assistant.as_deref(),
                            &feedback,
                        );
                    } else {
                        return ResultRecord::err(source, detail);
                    }
                }
            }
        }

        unreachable!("loop always returns on the final attempt")
    }

    /// Execute a single attempt: call the model, parse JSON, validate
    /// against schema. Returns [`StepResult::Ok`] on success, or
    /// [`StepResult::Failed`] with detail and feedback for the retry loop.
    fn attempt(&self, messages: &[Message], attempt: u32) -> StepResult {
        let raw = match self.client.chat(messages, self.model.as_deref()) {
            Ok(s) => s,
            Err(e) => {
                let detail = format!("model request failed after {} retries: {e}", attempt);
                return StepResult::Failed {
                    detail,
                    raw_assistant: None,
                    feedback: format!(
                        "The previous request did not produce output ({e}). \
                         Try again and output ONLY the JSON object."
                    ),
                };
            }
        };

        debug!(attempt = attempt + 1, response = %raw, "model response");

        // Scope the borrow from strip_code_fences so `raw` is movable
        // afterward — avoids cloning the full model response on error.
        let parsed: Value = {
            let candidate = strip_code_fences(&raw);
            match serde_json::from_str(candidate) {
                Ok(v) => v,
                Err(e) => {
                    let detail = format!(
                        "JSON parse failed after {} retries: response was not valid JSON: {e}",
                        attempt
                    );
                    return StepResult::Failed {
                        detail,
                        raw_assistant: Some(raw),
                        feedback: format!(
                            "Your previous output did not validate: response was not valid JSON: {e}. \
                             Output ONLY a single JSON object that conforms to the schema."
                        ),
                    };
                }
            }
        };

        match collect_errors(&self.validator, &parsed) {
            Ok(()) => StepResult::Ok(parsed),
            Err(errs) => {
                let joined = errs.join("; ");
                let detail = format!(
                    "schema validation failed after {} retries: {joined}",
                    attempt
                );
                StepResult::Failed {
                    detail,
                    raw_assistant: Some(raw),
                    feedback: format!(
                        "Your previous output did not validate against the schema: {joined}. \
                         Fix it and output ONLY a single JSON object."
                    ),
                }
            }
        }
    }
}

/// Emit the full prompt at `debug`. On the first attempt this logs the
/// initial system + user messages; on retries it logs the appended
/// assistant/user feedback messages so you can see what the model was
/// told about its previous failure.
fn log_prompt(attempt: u32, messages: &[Message]) {
    // Enabled check avoids the allocation when the user hasn't opted in.
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    for (i, m) in messages.iter().enumerate() {
        debug!(
            attempt = attempt + 1,
            index = i,
            role = m.role,
            content = %m.content,
            "prompt message",
        );
    }
}

/// Replace any retry-feedback messages beyond the initial system+user
/// pair with the latest assistant output (if any) and the latest
/// validator/transport feedback. This keeps the request payload size
/// constant across retries regardless of `max_retries`; the model only
/// ever sees the most recent failure, which is what it actually needs
/// to fix its next response.
fn requeue_feedback(
    messages: &mut Vec<Message>,
    initial_len: usize,
    last_assistant: Option<&str>,
    feedback: &str,
) {
    messages.truncate(initial_len);
    if let Some(raw) = last_assistant {
        messages.push(Message::assistant(raw.to_string()));
    }
    messages.push(Message::user(feedback.to_string()));
}

/// If the model wrapped JSON in ```json ... ``` fences, peel them off.
/// Handles common casing variants like `` ```JSON `` and `` ```Json ``.
fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    if t.len() < 6 || !t.starts_with("```") {
        return t;
    }
    let after_ticks = &t[3..];
    // If the chars between ``` and the first newline are all alphanumeric,
    // treat them as a language tag and skip past it.
    let content = match after_ticks.find('\n') {
        Some(nl) if after_ticks[..nl].chars().all(|c| c.is_ascii_alphanumeric()) => {
            &after_ticks[nl + 1..]
        }
        _ => after_ticks,
    };
    if let Some(inner) = content.rsplit_once("```") {
        let body = inner.0.trim();
        if !body.is_empty() {
            return body;
        }
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fences_stripped() {
        assert_eq!(strip_code_fences("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```JSON\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```Json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("```\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(strip_code_fences("  {\"a\":1}  "), "{\"a\":1}");
    }

    #[test]
    fn requeue_feedback_replaces_prior_attempts() {
        let mut messages = vec![Message::system("sys"), Message::user("doc")];

        // First failed attempt: push assistant + feedback.
        requeue_feedback(&mut messages, 2, Some("bad1"), "fix1");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].content, "bad1");
        assert_eq!(messages[3].content, "fix1");

        // Second failed attempt: prior feedback is replaced, not appended.
        requeue_feedback(&mut messages, 2, Some("bad2"), "fix2");
        assert_eq!(messages.len(), 4, "retry history must not grow unboundedly");
        assert_eq!(messages[0].content, "sys");
        assert_eq!(messages[1].content, "doc");
        assert_eq!(messages[2].content, "bad2");
        assert_eq!(messages[3].content, "fix2");
    }

    #[test]
    fn requeue_feedback_without_assistant_content() {
        let mut messages = vec![Message::system("sys"), Message::user("doc")];
        // Transport failure case: no assistant output to feed back.
        requeue_feedback(&mut messages, 2, None, "try again");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].role, "user");
        assert_eq!(messages[2].content, "try again");
    }
}
