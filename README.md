# llm-quota

A CLI tool that fetches usage-limit information for Claude and Codex and prints JSON output.

## Features

- Fetch Claude OAuth usage with `usage get-claude-usage`
- Fetch Codex usage with `usage get-codex-usage`
- Print the raw API response under `Usage` and a normalized window summary under `Summary`

## Requirements

- Nix available in your environment
- Use `nix develop` for local Rust development tasks

## Quick start

```bash
# Show CLI help
nix run .#default -- help

# Fetch Claude usage
nix run .#default -- usage get-claude-usage

# Fetch Codex usage
nix run .#default -- usage get-codex-usage

# Print only the percentage left in the seven-day window
nix run .#default -- usage get-claude-usage | jq -r .Summary.SevenDay.PercentLeft
```

## Command line

Commands follow the AWS CLI grammar: `llm-quota [global options] <command> <subcommand> [--parameter value ...]`.

- Global options: `--output json|text|table` (default `json`), `--debug` (prints the credential source and the raw response body to stderr), `--version`.
- `help` works at every level: `llm-quota help`, `llm-quota usage help`, `llm-quota usage get-claude-usage help`.
- The usage subcommands take no parameters; credentials and endpoints come from the environment variables below.

## Authentication resolution order

### Claude

1. `ANTHROPIC_OAUTH_API_KEY`
2. `$CLAUDE_CONFIG_DIR/.credentials.json` field `claudeAiOauth.accessToken`
3. `~/.claude/.credentials.json` field `claudeAiOauth.accessToken`

Optional settings

- `ANTHROPIC_BASE_URL`
- Default is `https://api.anthropic.com`

Example

```bash
ANTHROPIC_OAUTH_API_KEY=your_token \
  nix run .#default -- usage get-claude-usage
```

### Codex

1. `OPENAI_OAUTH_API_KEY` plus `OPENAI_ACCOUNT_ID` or `CHATGPT_ACCOUNT_ID`
2. `~/.codex/auth.json` fields `tokens.access_token` and `tokens.account_id`

Optional settings

- `CHATGPT_BASE_URL`
- Default is `https://chatgpt.com`

Example

```bash
OPENAI_OAUTH_API_KEY=your_token \
OPENAI_ACCOUNT_ID=your_account_id \
  nix run .#default -- usage get-codex-usage
```

## Output

Output keys are PascalCase. The raw API response is returned under `Usage` with its keys converted to PascalCase (`rate_limit.primary_window` becomes `RateLimit.PrimaryWindow`).

- `Usage`: raw API response
- `Summary`: normalized window summary

Errors are printed to stderr as `error: <message>`; for a non-success HTTP response the message contains the status and the response body.

### Example output for `get-claude-usage` (redacted)

```json
{
  "Usage": {
    "FiveHour": {
      "Utilization": 4.0,
      "ResetsAt": "<redacted-timestamp>"
    },
    "SevenDay": {
      "Utilization": 0.0,
      "ResetsAt": "<redacted-timestamp>"
    },
    "SevenDaySonnet": {
      "Utilization": 0.0,
      "ResetsAt": "<redacted-timestamp>"
    },
    "ExtraUsage": {
      "IsEnabled": false,
      "MonthlyLimit": null,
      "UsedCredits": null,
      "Utilization": null
    }
  },
  "Summary": {
    "FiveHour": {
      "ResetsAt": "<redacted-timestamp>",
      "PercentLeft": 96.0
    },
    "SevenDay": {
      "ResetsAt": "<redacted-timestamp>",
      "PercentLeft": 100.0
    },
    "SevenDaySonnet": {
      "ResetsAt": "<redacted-timestamp>",
      "PercentLeft": 100.0
    }
  }
}
```

### Example output for `get-codex-usage` (redacted)

```json
{
  "Usage": {
    "AccountId": "<redacted-account-id>",
    "PlanType": "plus",
    "RateLimit": {
      "Allowed": true,
      "LimitReached": false,
      "PrimaryWindow": {
        "LimitWindowSeconds": 18000,
        "ResetAfterSeconds": 4266,
        "ResetAt": "<redacted-unix-timestamp>",
        "UsedPercent": 96
      },
      "SecondaryWindow": {
        "LimitWindowSeconds": 604800,
        "ResetAfterSeconds": 517980,
        "ResetAt": "<redacted-unix-timestamp>",
        "UsedPercent": 84
      }
    }
  },
  "Summary": {
    "FiveHour": {
      "LimitWindowSeconds": 18000,
      "ResetAfterSeconds": 4266,
      "ResetAt": "<redacted-unix-timestamp>",
      "UsedPercent": 96
    },
    "SevenDay": {
      "LimitWindowSeconds": 604800,
      "ResetAfterSeconds": 517980,
      "ResetAt": "<redacted-unix-timestamp>",
      "UsedPercent": 84
    }
  }
}
```

Summary fields

For `get-claude-usage`

- `FiveHour`, `SevenDay`, `SevenDaySonnet`, each with `ResetsAt` and `PercentLeft` (100 minus the utilization)

For `get-codex-usage`

- `FiveHour` from `Usage.RateLimit.PrimaryWindow`
- `SevenDay` from `Usage.RateLimit.SecondaryWindow`

## OpenTelemetry

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable OTLP trace export. When unset, tracing is disabled with zero overhead.

| Variable | Description |
|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint (required to enable tracing) |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` for HTTP or `grpc` for gRPC (default: `grpc`) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers (e.g. `Authorization=Basic ...`) |

Trace structure per command:

- `run_claude` (`get-claude-usage`) / `run_codex` (`get-codex-usage`) (root span)
  - `resolve_auth` (child span)
  - `http_request` (child span) with `http.request.method`, `url.full`, `http.response.status_code` attributes and `http.response.body` event containing the raw API response
  - `output` event containing the final JSON output

### Grafana Cloud

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-<region>.grafana.net/otlp \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic <base64(instanceID:apiToken)>" \
  nix run .#default -- usage get-claude-usage
```

### Jaeger (local)

```bash
docker run -d -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest

OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  nix run .#default -- usage get-claude-usage
```

## Exit codes

- `0`: success
- `252`: invalid command or parameter
- `254`: the API request failed, returned a non-success HTTP status, or returned a body that is not JSON
- `255`: missing or invalid credentials, or another runtime error

## Development checks

```bash
nix develop -c cargo fmt -- --check
nix develop -c cargo clippy --all-targets --all-features -- -D warnings
nix develop -c cargo test
```

## Package verification

```bash
./scripts/verify-package.sh
```
