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
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var(URL_ENV).ok().unwrap_or_default();
        if base_url.trim().is_empty() {
            bail!("required environment variable {URL_ENV} is unset or empty");
        }
        let api_key = std::env::var(API_KEY_ENV).ok().filter(|s| !s.is_empty());
        let agent = ureq::AgentBuilder::new()
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
