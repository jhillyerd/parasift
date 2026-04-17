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

        let mut last_error_detail = String::new();

        // attempts = 1 initial + up to max_retries retries (SPEC §9.4).
        let total_attempts = 1 + self.max_retries;
        for attempt in 0..total_attempts {
            log_prompt(attempt, &messages);
            let raw = match self.client.chat(&messages, self.model.as_deref()) {
                Ok(s) => s,
                Err(e) => {
                    // Transport / HTTP error: treat like a validation failure
                    // for retry purposes, feeding the error back to the model
                    // on the next loop.
                    last_error_detail = format!("model call failed: {e}");
                    warn!(attempt = attempt + 1, error = %last_error_detail, "model call failed");
                    // No assistant content to append; but we still want to
                    // give the model a chance to try again with feedback.
                    if attempt + 1 < total_attempts {
                        // No assistant content to show the model (request
                        // didn't even succeed), so just replace any prior
                        // feedback with a fresh "try again" nudge.
                        requeue_feedback(
                            &mut messages,
                            INITIAL_MESSAGES,
                            None,
                            &format!(
                                "The previous request did not produce output ({}). Try again and output ONLY the JSON object.",
                                last_error_detail
                            ),
                        );
                        continue;
                    } else {
                        return ResultRecord::err(
                            source,
                            format!(
                                "schema validation failed after {} retries: {}",
                                self.max_retries, last_error_detail
                            ),
                        );
                    }
                }
            };

            debug!(attempt = attempt + 1, response = %raw, "model response");
            // Strip code fences if the model emitted them anyway.
            let candidate_str = strip_code_fences(&raw);

            // Parse JSON.
            let parsed: Value = match serde_json::from_str(candidate_str) {
                Ok(v) => v,
                Err(e) => {
                    last_error_detail = format!("response was not valid JSON: {e}");
                    warn!(attempt = attempt + 1, error = %last_error_detail, "JSON parse failed");
                    if attempt + 1 < total_attempts {
                        requeue_feedback(
                            &mut messages,
                            INITIAL_MESSAGES,
                            Some(&raw),
                            &format!(
                                "Your previous output did not validate: {}. Output ONLY a single JSON object that conforms to the schema.",
                                last_error_detail
                            ),
                        );
                        continue;
                    } else {
                        return ResultRecord::err(
                            source,
                            format!(
                                "schema validation failed after {} retries: {}",
                                self.max_retries, last_error_detail
                            ),
                        );
                    }
                }
            };

            // Validate against schema.
            match collect_errors(&self.validator, &parsed) {
                Ok(()) => {
                    debug!(attempt = attempt + 1, "validation succeeded");
                    return ResultRecord::ok(source, parsed);
                }
                Err(errs) => {
                    last_error_detail = errs.join("; ");
                    warn!(attempt = attempt + 1, error = %last_error_detail, "schema validation failed");
                    if attempt + 1 < total_attempts {
                        requeue_feedback(
                            &mut messages,
                            INITIAL_MESSAGES,
                            Some(&raw),
                            &format!(
                                "Your previous output did not validate against the schema: {}. Fix it and output ONLY a single JSON object.",
                                last_error_detail
                            ),
                        );
                        continue;
                    } else {
                        return ResultRecord::err(
                            source,
                            format!(
                                "schema validation failed after {} retries: {}",
                                self.max_retries, last_error_detail
                            ),
                        );
                    }
                }
            }
        }

        // Loop invariant guarantees we return above, but provide a safety net.
        ResultRecord::err(
            source,
            format!(
                "schema validation failed after {} retries: {}",
                self.max_retries, last_error_detail
            ),
        )
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
fn strip_code_fences(s: &str) -> &str {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```json") {
        let rest = rest.trim_start_matches('\n');
        if let Some(inner) = rest.rsplit_once("```") {
            return inner.0.trim();
        }
    }
    if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.trim_start_matches('\n');
        if let Some(inner) = rest.rsplit_once("```") {
            return inner.0.trim();
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
