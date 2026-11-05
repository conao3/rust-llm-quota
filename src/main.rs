mod error;
mod grammar;
mod output;

use clap::{CommandFactory, Parser, Subcommand};
use error::{Result, other, target};
use opentelemetry::trace::{Span, TraceContextExt, Tracer, get_active_span};
use opentelemetry::{Context, KeyValue, global};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use output::{OutputFormat, Response};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG: AtomicBool = AtomicBool::new(false);

#[derive(Parser)]
#[command(
    name = "llm-quota",
    version,
    about = "Fetch Claude and Codex usage limits"
)]
struct Cli {
    /// Output format
    #[arg(long, global = true, help_heading = "Global Options", value_enum, default_value_t = OutputFormat::Json)]
    output: OutputFormat,
    /// Print the credential source and the raw response body to stderr
    #[arg(long, global = true, help_heading = "Global Options")]
    debug: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fetch the usage limits of an LLM subscription
    #[command(subcommand)]
    Usage(UsageCommand),
}

#[derive(Subcommand)]
enum UsageCommand {
    /// Fetch the Claude OAuth usage limits
    #[command(
        name = "get-claude-usage",
        after_help = "\
Environment Variables:
  ANTHROPIC_OAUTH_API_KEY  OAuth access token (overrides ~/.claude/.credentials.json)
  CLAUDE_CONFIG_DIR        Claude config directory (uses $CLAUDE_CONFIG_DIR/.credentials.json)
  ANTHROPIC_BASE_URL       API base URL [default: https://api.anthropic.com]

  OTEL_EXPORTER_OTLP_ENDPOINT  OTLP endpoint (enables tracing when set)
  OTEL_EXPORTER_OTLP_PROTOCOL  http/protobuf or grpc [default: grpc]
  OTEL_EXPORTER_OTLP_HEADERS   Auth headers (e.g. Authorization=Basic ...)"
    )]
    GetClaudeUsage,
    /// Fetch the Codex usage limits
    #[command(
        name = "get-codex-usage",
        after_help = "\
Environment Variables:
  OPENAI_OAUTH_API_KEY   OAuth access token (overrides ~/.codex/auth.json)
  CODEX_HOME             Codex config directory (uses $CODEX_HOME/auth.json)
  OPENAI_ACCOUNT_ID      Account ID (required with OPENAI_OAUTH_API_KEY)
  CHATGPT_ACCOUNT_ID     Account ID alternative to OPENAI_ACCOUNT_ID
  CHATGPT_BASE_URL       API base URL [default: https://chatgpt.com]

  OTEL_EXPORTER_OTLP_ENDPOINT  OTLP endpoint (enables tracing when set)
  OTEL_EXPORTER_OTLP_PROTOCOL  http/protobuf or grpc [default: grpc]
  OTEL_EXPORTER_OTLP_HEADERS   Auth headers (e.g. Authorization=Basic ...)"
    )]
    GetCodexUsage,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct UsageWindow {
    utilization: Option<f64>,
    resets_at: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct OAuthUsageResponse {
    five_hour: Option<UsageWindow>,
    seven_day: Option<UsageWindow>,
    seven_day_sonnet: Option<UsageWindow>,
}

#[derive(Debug, Deserialize)]
struct CodexAuthFile {
    tokens: Option<CodexTokens>,
}

#[derive(Debug, Deserialize)]
struct CodexTokens {
    access_token: Option<String>,
    account_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaudeCredentialsFile {
    #[serde(rename = "claudeAiOauth")]
    claude_ai_oauth: Option<ClaudeAiOauth>,
}

#[derive(Debug, Deserialize)]
struct ClaudeAiOauth {
    #[serde(rename = "accessToken")]
    access_token: Option<String>,
}

fn init_tracer_provider() -> Option<SdkTracerProvider> {
    if env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_err() {
        return None;
    }
    Some(
        SdkTracerProvider::builder()
            .with_resource(Resource::builder().with_service_name("llm-quota").build())
            .with_batch_exporter(match env::var("OTEL_EXPORTER_OTLP_PROTOCOL").as_deref() {
                Ok("http/protobuf") | Ok("http/json") => std::thread::spawn(|| {
                    opentelemetry_otlp::SpanExporter::builder()
                        .with_http()
                        .build()
                })
                .join()
                .ok()?
                .ok()?,
                _ => opentelemetry_otlp::SpanExporter::builder()
                    .with_tonic()
                    .build()
                    .ok()?,
            })
            .build(),
    )
}

fn left(v: Option<f64>) -> Option<f64> {
    v.map(|n| (100.0 - n).max(0.0))
}

fn expand_home_dir(path: &str) -> PathBuf {
    if path == "~" {
        return PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()));
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".to_string())).join(rest);
    }
    PathBuf::from(path)
}

fn auth_debug_enabled() -> bool {
    DEBUG.load(Ordering::Relaxed)
}

fn read_claude_oauth_token() -> std::result::Result<String, String> {
    if let Ok(v) = env::var("ANTHROPIC_OAUTH_API_KEY") {
        let token = v.trim().to_string();
        if !token.is_empty() {
            if auth_debug_enabled() {
                eprintln!("debug: claude auth source: ANTHROPIC_OAUTH_API_KEY");
            }
            return Ok(token);
        }
    }

    let mut candidates = Vec::new();
    if let Ok(config_dir) = env::var("CLAUDE_CONFIG_DIR") {
        let dir = config_dir.trim();
        if !dir.is_empty() {
            let base = expand_home_dir(dir);
            candidates.push(base.join(".credentials.json"));
        }
    }
    candidates.push(
        PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
            .join(".claude/.credentials.json"),
    );

    let credentials_path = candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            "ANTHROPIC_OAUTH_API_KEY is not set and Claude credentials file was not found (checked CLAUDE_CONFIG_DIR and ~/.claude/.credentials.json)"
                .to_string()
        })?;
    if auth_debug_enabled() {
        eprintln!("debug: claude auth source: {}", credentials_path.display());
    }

    let content = fs::read_to_string(&credentials_path)
        .map_err(|e| format!("failed to read {}: {e}", credentials_path.display()))?;
    let credentials: ClaudeCredentialsFile = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse {}: {e}", credentials_path.display()))?;

    credentials
        .claude_ai_oauth
        .and_then(|o| o.access_token)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            format!(
                "ANTHROPIC_OAUTH_API_KEY is not set and accessToken was not found in {}",
                credentials_path.display()
            )
        })
}

fn read_codex_auth() -> std::result::Result<(String, String), String> {
    if let Ok(access_token) = env::var("OPENAI_OAUTH_API_KEY") {
        let access_token = access_token.trim().to_string();
        if !access_token.is_empty() {
            if auth_debug_enabled() {
                eprintln!("debug: codex auth source: OPENAI_OAUTH_API_KEY");
            }
            let account_id = env::var("OPENAI_ACCOUNT_ID")
                .or_else(|_| env::var("CHATGPT_ACCOUNT_ID"))
                .map_err(|_| {
                    "OPENAI_OAUTH_API_KEY is set, but OPENAI_ACCOUNT_ID or CHATGPT_ACCOUNT_ID is missing"
                        .to_string()
                })?;
            let account_id = account_id.trim().to_string();
            if account_id.is_empty() {
                return Err(
                    "OPENAI_OAUTH_API_KEY is set, but OPENAI_ACCOUNT_ID/CHATGPT_ACCOUNT_ID is empty"
                        .to_string(),
                );
            }
            return Ok((access_token, account_id));
        }
    }

    let auth_path = if let Ok(codex_home) = env::var("CODEX_HOME") {
        let codex_home = codex_home.trim();
        if codex_home.is_empty() {
            PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
                .join(".codex/auth.json")
        } else {
            expand_home_dir(codex_home).join("auth.json")
        }
    } else {
        PathBuf::from(env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
            .join(".codex/auth.json")
    };

    let content = fs::read_to_string(&auth_path)
        .map_err(|e| format!("failed to read auth file {}: {e}", auth_path.display()))?;
    if auth_debug_enabled() {
        eprintln!("debug: codex auth source: {}", auth_path.display());
    }
    let auth: CodexAuthFile = serde_json::from_str(&content)
        .map_err(|e| format!("failed to parse auth file {}: {e}", auth_path.display()))?;

    let tokens = auth
        .tokens
        .ok_or_else(|| "tokens are missing in codex auth file".to_string())?;
    let access_token = tokens
        .access_token
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| "access_token is missing in codex auth file".to_string())?;
    let account_id = tokens
        .account_id
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| "account_id is missing in codex auth file".to_string())?;

    Ok((access_token, account_id))
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct ClaudeWindowSummary {
    resets_at: Option<String>,
    percent_left: Option<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct ClaudeSummary {
    five_hour: ClaudeWindowSummary,
    seven_day: ClaudeWindowSummary,
    seven_day_sonnet: ClaudeWindowSummary,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct UsageOutput<T> {
    usage: Value,
    summary: T,
}

#[derive(Serialize)]
#[serde(rename_all = "PascalCase")]
struct CodexSummary {
    five_hour: Value,
    seven_day: Value,
}

fn to_value<T: Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(|error| other(error.to_string()))
}

fn pascal_case_key(key: &str) -> String {
    key.split(['_', '-'])
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}

fn pascal_case_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(key, value)| (pascal_case_key(&key), pascal_case_keys(value)))
                .collect::<Map<_, _>>(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(pascal_case_keys).collect()),
        value => value,
    }
}

fn claude_window_summary(window: Option<&UsageWindow>) -> ClaudeWindowSummary {
    ClaudeWindowSummary {
        resets_at: window.and_then(|w| w.resets_at.clone()),
        percent_left: left(window.and_then(|w| w.utilization)),
    }
}

async fn fetch_usage(request: reqwest::RequestBuilder, url: &str) -> Result<Value> {
    let tracer = global::tracer("llm-quota");
    let mut http_span = tracer.start("http_request");
    http_span.set_attribute(KeyValue::new("http.request.method", "GET"));
    http_span.set_attribute(KeyValue::new("url.full", url.to_string()));

    let response = request
        .send()
        .await
        .map_err(|e| target(format!("request failed: {e}")))?;

    let status = response.status();
    http_span.set_attribute(KeyValue::new(
        "http.response.status_code",
        status.as_u16() as i64,
    ));
    let body_text = response
        .text()
        .await
        .map_err(|e| target(format!("failed to read response body: {e}")))?;
    http_span.add_event(
        "http.response.body",
        vec![KeyValue::new("body", body_text.clone())],
    );
    drop(http_span);
    if auth_debug_enabled() {
        eprintln!("debug: response body: {body_text}");
    }

    if !status.is_success() {
        return Err(target(format!("HTTP {}: {body_text}", status.as_u16())));
    }

    serde_json::from_str(&body_text).map_err(|e| target(format!("failed to parse JSON: {e}")))
}

fn http_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| other(format!("failed to build HTTP client: {e}")))
}

fn record_output(out: &Value) {
    get_active_span(|span| {
        span.add_event(
            "output",
            vec![KeyValue::new(
                "json",
                serde_json::to_string(out).unwrap_or_default(),
            )],
        );
    });
}

async fn get_claude_usage() -> Result<Value> {
    let tracer = global::tracer("llm-quota");
    let _root_guard = Context::current_with_span(tracer.start("run_claude")).attach();

    let base_url =
        env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".to_string());

    let api_key = {
        let _auth_guard = Context::current_with_span(tracer.start("resolve_auth")).attach();
        read_claude_oauth_token().map_err(other)?
    };

    let url = format!("{}/api/oauth/usage", base_url.trim_end_matches('/'));
    let request = http_client()?
        .get(&url)
        .header(ACCEPT, "application/json, text/plain, */*")
        .header(CONTENT_TYPE, "application/json")
        .header(USER_AGENT, "claude-code/2.0.32")
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .header("anthropic-beta", "oauth-2025-04-20");
    let usage_value = fetch_usage(request, &url).await?;

    let usage: OAuthUsageResponse =
        serde_json::from_value(usage_value.clone()).unwrap_or(OAuthUsageResponse {
            five_hour: None,
            seven_day: None,
            seven_day_sonnet: None,
        });

    let out = to_value(&UsageOutput {
        usage: pascal_case_keys(usage_value),
        summary: ClaudeSummary {
            five_hour: claude_window_summary(usage.five_hour.as_ref()),
            seven_day: claude_window_summary(usage.seven_day.as_ref()),
            seven_day_sonnet: claude_window_summary(usage.seven_day_sonnet.as_ref()),
        },
    })?;
    record_output(&out);
    Ok(out)
}

async fn get_codex_usage() -> Result<Value> {
    let tracer = global::tracer("llm-quota");
    let _root_guard = Context::current_with_span(tracer.start("run_codex")).attach();

    let (access_token, account_id) = {
        let _auth_guard = Context::current_with_span(tracer.start("resolve_auth")).attach();
        read_codex_auth().map_err(other)?
    };

    let base_url =
        env::var("CHATGPT_BASE_URL").unwrap_or_else(|_| "https://chatgpt.com".to_string());
    let url = format!("{}/backend-api/wham/usage", base_url.trim_end_matches('/'));
    let request = http_client()?
        .get(&url)
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json")
        .header(USER_AGENT, "codex-cli")
        .header(AUTHORIZATION, format!("Bearer {access_token}"))
        .header("ChatGPT-Account-Id", account_id);
    let usage_value = pascal_case_keys(fetch_usage(request, &url).await?);

    let window = |name: &str| {
        usage_value
            .get("RateLimit")
            .and_then(|v| v.get(name))
            .cloned()
            .unwrap_or(Value::Null)
    };
    let summary = CodexSummary {
        five_hour: window("PrimaryWindow"),
        seven_day: window("SecondaryWindow"),
    };
    let out = to_value(&UsageOutput {
        usage: usage_value.clone(),
        summary,
    })?;
    record_output(&out);
    Ok(out)
}

async fn run() -> Result<()> {
    let mut root = Cli::command();
    root.build();
    let argv = grammar::prepare(&root, env::args_os().collect())?;
    if argv.iter().any(|token| token == "--debug") {
        DEBUG.store(true, Ordering::Relaxed);
        eprintln!("debug: argv {:?}", &argv[1..]);
    }

    let cli = match Cli::try_parse_from(&argv) {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() { 252 } else { 0 };
            let _ = error.print();
            std::process::exit(code);
        }
    };

    let Command::Usage(command) = cli.command;
    let (name, value) = match command {
        UsageCommand::GetClaudeUsage => ("get-claude-usage", get_claude_usage().await?),
        UsageCommand::GetCodexUsage => ("get-codex-usage", get_codex_usage().await?),
    };
    print!(
        "{}",
        output::render(
            &grammar::pascal_case(name),
            Response { value, text: None },
            cli.output
        )?
    );
    Ok(())
}

fn report(result: Result<()>) -> i32 {
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("error: {error}");
            error.exit_code()
        }
    }
}

#[tokio::main]
async fn main() {
    let provider = init_tracer_provider();
    if let Some(ref p) = provider {
        global::set_tracer_provider(p.clone());
    }
    let code = report(run().await);
    if let Some(p) = provider {
        let _ = p.shutdown();
    }
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parser_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn converts_response_keys_to_pascal_case() {
        let value = json!({"rate_limit": {"primary_window": {"used_percent": 3}}, "approx_local_messages": [0, 0]});
        assert_eq!(
            pascal_case_keys(value),
            json!({"RateLimit": {"PrimaryWindow": {"UsedPercent": 3}}, "ApproxLocalMessages": [0, 0]})
        );
    }

    #[test]
    fn claude_summary_reports_the_percent_left() {
        let window = UsageWindow {
            utilization: Some(4.0),
            resets_at: Some("t".to_string()),
        };
        let summary = to_value(&claude_window_summary(Some(&window))).unwrap();
        assert_eq!(summary, json!({"ResetsAt": "t", "PercentLeft": 96.0}));
    }
}
