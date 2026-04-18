//! OpenAI-compatible chat completions client (SPEC §3).

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, SystemTime};
use tracing::warn;

/// Total HTTP attempts for transient (transport / 5xx / 429) failures:
/// one initial request plus up to this many retries. Distinct from the
/// validation-retry loop in `classify.rs` so a transient network blip
/// doesn't burn the user's `max_retries` budget.
const MAX_TRANSPORT_ATTEMPTS: u32 = 4;

/// Base for exponential backoff. Actual sleep is `rand(0, base * 2^n)`
/// per the "full jitter" AWS pattern, which maximally decorrelates
/// retry timing across concurrent workers (no thundering herd).
const BACKOFF_BASE: Duration = Duration::from_millis(250);

/// Ceiling on any individual backoff sleep. Keeps worst-case wait bounded
/// even if `Retry-After` or a high attempt index would otherwise push us
/// past a useful threshold.
const BACKOFF_CAP: Duration = Duration::from_secs(8);

#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system",
            content: content.into(),
        }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user",
            content: content.into(),
        }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant",
            content: content.into(),
        }
    }
}

/// Chat client wrapping `ureq` with a base URL + optional bearer token.
#[derive(Clone, Debug)]
pub struct ChatClient {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl ChatClient {
    /// Build a client from an explicit base URL and optional bearer
    /// token. The URL is expected to already be non-empty (validated by
    /// the CLI layer).
    ///
    /// `concurrency` is the number of worker threads that will share this
    /// client; it is used to size the HTTP keep-alive pool so workers
    /// can hand sockets back and forth instead of re-handshaking TLS on
    /// every request.
    pub fn new(base_url: &str, api_key: Option<&str>, concurrency: usize) -> Self {
        // Keep roughly 20% of the worker count as warm idle connections.
        // At N=1 this floors to 1; at N=100 it's 20. Idle sockets are cheap
        // but not free, so we don't reserve one per worker — workers mostly
        // stay busy, so we just need enough warm connections to cover the
        // churn when a few workers finish at once.
        let idle_pool = ((concurrency as f64 * 0.2).ceil() as usize).max(1);
        let agent = ureq::AgentBuilder::new()
            .max_idle_connections(idle_pool)
            .max_idle_connections_per_host(idle_pool)
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .timeout_write(Duration::from_secs(60))
            .build();
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(str::to_owned),
            agent,
        }
    }

    /// Send a chat completion request and return the `choices[0].message.content`.
    ///
    /// Transient failures (transport errors, HTTP 408/429/5xx) are
    /// retried internally with exponential full-jitter backoff, up to
    /// `MAX_TRANSPORT_ATTEMPTS` times total. A `Retry-After` header from
    /// the server is honored as a lower bound on the sleep. Non-2xx,
    /// non-retryable responses (e.g. 400, 401, 404) fail immediately.
    pub fn chat(&self, messages: &[Message], model: Option<&str>) -> Result<String> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = json!({
            "model": model.unwrap_or("local-model"),
            "messages": messages,
            "stream": false,
        });

        let mut last_err: Option<anyhow::Error> = None;
        for attempt in 0..MAX_TRANSPORT_ATTEMPTS {
            match self.chat_once(&url, &body) {
                Ok(content) => return Ok(content),
                Err(AttemptError::Fatal(e)) => return Err(e),
                Err(AttemptError::Retryable { error, retry_after }) => {
                    let is_last = attempt + 1 >= MAX_TRANSPORT_ATTEMPTS;
                    if is_last {
                        last_err = Some(error);
                        break;
                    }
                    let sleep = backoff_delay(attempt, retry_after);
                    warn!(
                        attempt = attempt + 1,
                        sleep_ms = sleep.as_millis() as u64,
                        error = %error,
                        "transient chat failure, retrying"
                    );
                    std::thread::sleep(sleep);
                    last_err = Some(error);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| anyhow!("chat request failed with no recorded error")))
    }

    /// Perform exactly one HTTP attempt. Classifies the outcome so the
    /// caller's retry loop can decide whether to sleep + try again or
    /// give up. Parsing the successful body also happens here so that
    /// malformed-but-2xx responses are treated as fatal (no point
    /// retrying a server that returns garbage).
    fn chat_once(&self, url: &str, body: &serde_json::Value) -> Result<String, AttemptError> {
        #[derive(Deserialize)]
        struct Choice {
            message: ChoiceMessage,
        }
        #[derive(Deserialize)]
        struct ChoiceMessage {
            content: Option<String>,
        }
        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        let mut req = self.agent.post(url).set("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }

        let resp = match req.send_json(body.clone()) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, resp)) => {
                // Read Retry-After before we consume the response body.
                let retry_after = parse_retry_after(resp.header("Retry-After"));
                let body_text = resp.into_string().unwrap_or_default();
                let err = anyhow!("inference server returned HTTP {code}: {body_text}");
                return if is_retryable_status(code) {
                    Err(AttemptError::Retryable {
                        error: err,
                        retry_after,
                    })
                } else {
                    Err(AttemptError::Fatal(err))
                };
            }
            Err(ureq::Error::Transport(t)) => {
                // Connection-level / timeout / TLS failures: always retryable.
                // The server may simply be busy or the TCP connection stale.
                return Err(AttemptError::Retryable {
                    error: anyhow!("request to {url} failed: {t}"),
                    retry_after: None,
                });
            }
        };

        // Defensive: ureq should only return Ok for 2xx, but double-check
        // so a future config change doesn't silently send us garbage.
        let status = resp.status();
        if !(200..300).contains(&status) {
            let text = resp.into_string().unwrap_or_default();
            return Err(AttemptError::Fatal(anyhow!(
                "inference server returned HTTP {status}: {text}"
            )));
        }

        let parsed: ChatResponse = match resp.into_json() {
            Ok(p) => p,
            Err(e) => {
                return Err(AttemptError::Fatal(
                    anyhow::Error::new(e)
                        .context("failed to parse chat completion response as JSON"),
                ));
            }
        };
        parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| {
                AttemptError::Fatal(anyhow!("chat response missing choices[0].message.content"))
            })
    }
}

/// Outcome of a single HTTP attempt. `Retryable` carries the original
/// error so we can surface it if all attempts are exhausted, plus any
/// server-provided `Retry-After` hint.
enum AttemptError {
    Fatal(anyhow::Error),
    Retryable {
        error: anyhow::Error,
        retry_after: Option<Duration>,
    },
}

/// Which HTTP status codes indicate a transient server-side condition
/// that's worth retrying. Everything else (400, 401, 403, 404, 422, …)
/// means the request itself is malformed and retrying will just repeat
/// the error.
fn is_retryable_status(code: u16) -> bool {
    matches!(code, 408 | 425 | 429 | 500 | 502 | 503 | 504)
}

/// Parse a `Retry-After` header value. Servers may send either an
/// integer seconds count or an HTTP-date; we only honor the former
/// (simpler, no chrono dep, covers the 429/503 cases that matter in
/// practice).
fn parse_retry_after(header: Option<&str>) -> Option<Duration> {
    let secs: u64 = header?.trim().parse().ok()?;
    Some(Duration::from_secs(secs))
}

/// Compute the next backoff sleep. Uses the "full jitter" pattern:
/// `sleep = rand(0, base * 2^attempt)`, capped at `BACKOFF_CAP`. When
/// the server sent a `Retry-After`, we take the max of the two so we
/// never retry sooner than the server asked — but still cap the wait.
fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    // Saturating shift so `attempt` of 30+ doesn't overflow; cap below
    // makes the upper bound bounded anyway.
    let shift = attempt.min(16);
    let ceiling = BACKOFF_BASE.saturating_mul(1u32 << shift).min(BACKOFF_CAP);
    let jittered = Duration::from_nanos(rand_range(ceiling.as_nanos() as u64));
    let base = match retry_after {
        Some(hint) => hint.max(jittered),
        None => jittered,
    };
    base.min(BACKOFF_CAP)
}

/// Return a pseudo-random `u64` in `[0, upper)`. Tiny xorshift64 seeded
/// per-call from the clock + thread id; quality is fine for backoff
/// jitter and avoids pulling in `rand`.
fn rand_range(upper: u64) -> u64 {
    if upper == 0 {
        return 0;
    }
    // Mix high-res time with a per-thread-varying id so parallel workers
    // don't all draw the same value when they retry in the same tick.
    let tid_hash = {
        let id = std::thread::current().id();
        // ThreadId's internal u64 isn't stable, but Debug is monotonic
        // enough for seeding. Cheap and we don't need cryptographic rigor.
        let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for b in format!("{id:?}").as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    };
    // `SystemTime` can go backwards across clock adjustments; for a
    // one-shot jitter seed that's acceptable (we just need entropy,
    // not monotonicity). `duration_since(UNIX_EPOCH)` gives us real
    // nanosecond-resolution entropy, unlike `Instant::now().elapsed()`
    // which always returns ~0.
    let now_nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let seed = now_nanos ^ tid_hash ^ 0x9E3779B97F4A7C15; // golden ratio, avoids seed==0

    // xorshift64 — one round is plenty for a single sample.
    let mut x = seed | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x % upper
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_status_codes() {
        for code in [408, 425, 429, 500, 502, 503, 504] {
            assert!(is_retryable_status(code), "{code} should be retryable");
        }
        for code in [200, 201, 400, 401, 403, 404, 409, 422] {
            assert!(!is_retryable_status(code), "{code} should not be retryable");
        }
    }

    #[test]
    fn retry_after_parses_seconds() {
        assert_eq!(parse_retry_after(Some("5")), Some(Duration::from_secs(5)));
        assert_eq!(
            parse_retry_after(Some(" 12 ")),
            Some(Duration::from_secs(12))
        );
        assert_eq!(parse_retry_after(Some("0")), Some(Duration::from_secs(0)));
    }

    #[test]
    fn retry_after_rejects_non_numeric() {
        // HTTP-date format is valid per RFC but we don't parse it.
        assert_eq!(
            parse_retry_after(Some("Wed, 21 Oct 2015 07:28:00 GMT")),
            None
        );
        assert_eq!(parse_retry_after(Some("")), None);
        assert_eq!(parse_retry_after(None), None);
    }

    #[test]
    fn backoff_respects_cap() {
        // Huge attempt index should still be bounded by BACKOFF_CAP.
        let d = backoff_delay(30, None);
        assert!(d <= BACKOFF_CAP, "delay {d:?} exceeded cap {BACKOFF_CAP:?}");
    }

    #[test]
    fn backoff_honors_retry_after_as_lower_bound() {
        // With Retry-After well above the jitter ceiling but below cap,
        // the result should equal the hint.
        let hint = Duration::from_secs(5);
        let d = backoff_delay(0, Some(hint));
        assert_eq!(d, hint);
    }

    #[test]
    fn backoff_retry_after_still_capped() {
        // A server that asks for an hour shouldn't stall a CLI run.
        let hint = Duration::from_secs(3600);
        let d = backoff_delay(0, Some(hint));
        assert_eq!(d, BACKOFF_CAP);
    }

    #[test]
    fn rand_range_respects_upper() {
        for _ in 0..1000 {
            assert!(rand_range(10) < 10);
        }
        assert_eq!(rand_range(0), 0);
        assert_eq!(rand_range(1), 0);
    }
}
