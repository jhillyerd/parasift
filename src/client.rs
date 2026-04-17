//! OpenAI-compatible chat completions client (SPEC §3).

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;

pub const URL_ENV: &str = "CLASSIFIER_LLM_URL";
pub const API_KEY_ENV: &str = "CLASSIFIER_LLM_API_KEY";

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
    /// Build a client from environment variables. Fails if
    /// `CLASSIFIER_LLM_URL` is unset or empty (SPEC §3).
    ///
    /// `concurrency` is the number of worker threads that will share this
    /// client; it is used to size the HTTP keep-alive pool so workers
    /// can hand sockets back and forth instead of re-handshaking TLS on
    /// every request.
    pub fn from_env(concurrency: usize) -> Result<Self> {
        let base_url = std::env::var(URL_ENV).ok().unwrap_or_default();
        if base_url.trim().is_empty() {
            bail!("required environment variable {URL_ENV} is unset or empty");
        }
        let api_key = std::env::var(API_KEY_ENV).ok().filter(|s| !s.is_empty());

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
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            agent,
        })
    }

    /// Send a chat completion request and return the `choices[0].message.content`.
    pub fn chat(&self, messages: &[Message], model: Option<&str>) -> Result<String> {
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

        let url = format!("{}/chat/completions", self.base_url);
        let body = json!({
            "model": model.unwrap_or("local-model"),
            "messages": messages,
            "stream": false,
        });

        let mut req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }

        let resp = req
            .send_json(body)
            .map_err(|e| anyhow!("request to {url} failed: {e}"))?;

        let status = resp.status();
        if !(200..300).contains(&status) {
            let text = resp.into_string().unwrap_or_default();
            bail!("inference server returned HTTP {status}: {text}");
        }

        let parsed: ChatResponse = resp
            .into_json()
            .context("failed to parse chat completion response as JSON")?;
        let content = parsed
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow!("chat response missing choices[0].message.content"))?;
        Ok(content)
    }
}
