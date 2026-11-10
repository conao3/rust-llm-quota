use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, other};

const MICRO_CENTS_PER_USD: f64 = 100_000_000.0;

pub struct Environment {
    pub opencode_db: Option<String>,
    pub xdg_data_home: Option<String>,
    pub home: Option<String>,
}

impl Environment {
    pub fn from_process() -> Self {
        Environment {
            opencode_db: std::env::var("OPENCODE_DB").ok(),
            xdg_data_home: std::env::var("XDG_DATA_HOME").ok(),
            home: std::env::var("HOME").ok(),
        }
    }
}

pub fn database_path(env: &Environment) -> Result<PathBuf> {
    let data_home = match env.xdg_data_home.as_deref() {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => match env.home.as_deref() {
            Some(home) if !home.is_empty() => Path::new(home).join(".local").join("share"),
            _ => {
                return Err(other(
                    "cannot find the OpenCode data directory: neither XDG_DATA_HOME nor HOME is set",
                ));
            }
        },
    };
    let name = env.opencode_db.as_deref().unwrap_or("opencode.db");
    if name == ":memory:" {
        return Err(other(
            "OPENCODE_DB is :memory:, so there is no database to read",
        ));
    }
    Ok(data_home.join("opencode").join(name))
}

#[derive(Debug, PartialEq)]
pub struct ConsoleCredential {
    pub label: String,
    pub access_token: String,
    pub expires_at_ms: Option<i64>,
    pub server: String,
    pub org_id: String,
}

#[derive(Deserialize)]
struct StoredValue {
    access: Option<String>,
    expires: Option<i64>,
    metadata: Option<StoredMetadata>,
}

#[derive(Deserialize)]
struct StoredMetadata {
    server: Option<String>,
    #[serde(rename = "orgID")]
    org_id: Option<String>,
}

pub fn read_console_credential(db: &Path) -> Result<ConsoleCredential> {
    if !db.is_file() {
        return Err(other(format!(
            "the OpenCode database {} was not found (set OPENCODE_DB or XDG_DATA_HOME)",
            db.display()
        )));
    }
    let connection = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| other(format!("failed to open {}: {e}", db.display())))?;
    let (label, value) = connection
        .query_row(
            "select label, value from credential where integration_id = 'opencode' and active = 1",
            [],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|e| {
            other(format!(
                "failed to read credentials from {}: {e}",
                db.display()
            ))
        })?
        .ok_or_else(|| {
            other(format!(
                "{} has no active OpenCode Console credential (run `opencode auth login opencode`)",
                db.display()
            ))
        })?;
    let missing = |what: &str| {
        other(format!(
            "the OpenCode Console credential {label:?} has no {what}"
        ))
    };
    let stored: StoredValue = serde_json::from_str(&value).map_err(|e| {
        other(format!(
            "failed to parse the OpenCode Console credential {label:?}: {e}"
        ))
    })?;
    let metadata = stored.metadata.ok_or_else(|| missing("metadata"))?;
    Ok(ConsoleCredential {
        access_token: stored
            .access
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| missing("access token"))?,
        expires_at_ms: stored.expires,
        server: metadata.server.ok_or_else(|| missing("server"))?,
        org_id: metadata
            .org_id
            .ok_or_else(|| missing("workspace (orgID)"))?,
        label,
    })
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Status {
    access: Option<Access>,
}

#[derive(Deserialize)]
struct Access {
    meters: Option<Meters>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meters {
    five_hour: Option<Meter>,
    week: Option<Meter>,
    month: Option<Meter>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Meter {
    resets_at: Option<String>,
    limit_micro_cents: String,
    used_micro_cents: String,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub struct WindowSummary {
    pub resets_at: Option<String>,
    pub percent_left: Option<f64>,
    pub used_usd: Option<f64>,
    pub limit_usd: Option<f64>,
}

#[derive(Serialize, Debug, PartialEq)]
#[serde(rename_all = "PascalCase")]
pub struct Summary {
    pub five_hour: WindowSummary,
    pub week: WindowSummary,
    pub month: WindowSummary,
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn window_summary(meter: Option<&Meter>) -> WindowSummary {
    let amounts = meter.and_then(|m| {
        Some((
            m.used_micro_cents.parse::<f64>().ok()?,
            m.limit_micro_cents.parse::<f64>().ok()?,
        ))
    });
    WindowSummary {
        resets_at: meter.and_then(|m| m.resets_at.clone()),
        percent_left: amounts
            .filter(|(_, limit)| *limit > 0.0)
            .map(|(used, limit)| round2((100.0 - used / limit * 100.0).max(0.0))),
        used_usd: amounts.map(|(used, _)| round2(used / MICRO_CENTS_PER_USD)),
        limit_usd: amounts.map(|(_, limit)| round2(limit / MICRO_CENTS_PER_USD)),
    }
}

pub fn summarize(status: &Value) -> Result<Summary> {
    let status: Status = serde_json::from_value(status.clone())
        .map_err(|e| crate::error::target(format!("unexpected Go status: {e}")))?;
    let meters = status
        .access
        .ok_or_else(|| crate::error::target("the workspace has no OpenCode Go subscription"))?
        .meters;
    let meter = |pick: fn(&Meters) -> Option<&Meter>| meters.as_ref().and_then(pick);
    Ok(Summary {
        five_hour: window_summary(meter(|m| m.five_hour.as_ref())),
        week: window_summary(meter(|m| m.week.as_ref())),
        month: window_summary(meter(|m| m.month.as_ref())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn env(opencode_db: Option<&str>, xdg: Option<&str>, home: Option<&str>) -> Environment {
        Environment {
            opencode_db: opencode_db.map(String::from),
            xdg_data_home: xdg.map(String::from),
            home: home.map(String::from),
        }
    }

    #[test]
    fn database_path_follows_opencode() {
        assert_eq!(
            database_path(&env(None, None, Some("/home/a"))).unwrap(),
            PathBuf::from("/home/a/.local/share/opencode/opencode.db")
        );
        assert_eq!(
            database_path(&env(None, Some(""), Some("/home/a"))).unwrap(),
            PathBuf::from("/home/a/.local/share/opencode/opencode.db")
        );
        assert_eq!(
            database_path(&env(None, Some("/data"), Some("/home/a"))).unwrap(),
            PathBuf::from("/data/opencode/opencode.db")
        );
        assert_eq!(
            database_path(&env(Some("/x/opencode.db"), None, Some("/home/a"))).unwrap(),
            PathBuf::from("/x/opencode.db")
        );
        assert_eq!(
            database_path(&env(Some("other.db"), Some("/data"), None)).unwrap(),
            PathBuf::from("/data/opencode/other.db")
        );
        assert!(database_path(&env(Some(":memory:"), None, Some("/home/a"))).is_err());
    }

    fn database(rows: &[(&str, bool, &str)]) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        let connection = Connection::open(file.path()).unwrap();
        connection
            .execute_batch(
                "create table credential (id text primary key, integration_id text, label text not null, value text not null, connector_id text, method_id text, active integer, time_created integer not null, time_updated integer not null);",
            )
            .unwrap();
        for (i, (label, active, value)) in rows.iter().enumerate() {
            connection
                .execute(
                    "insert into credential values (?1, 'opencode', ?2, ?3, null, null, ?4, 0, 0)",
                    rusqlite::params![i.to_string(), label, value, *active as i64],
                )
                .unwrap();
        }
        file
    }

    const VALUE: &str = r#"{"type":"oauth","access":"token-a","refresh":"r","expires":1796815655992,"metadata":{"server":"https://opencode.ai/console","orgID":"org_1","orgName":"Personal"}}"#;

    #[test]
    fn reads_the_active_console_credential() {
        let other_value = VALUE.replace("token-a", "token-b");
        let file = database(&[("Team", false, &other_value), ("Personal", true, VALUE)]);
        assert_eq!(
            read_console_credential(file.path()).unwrap(),
            ConsoleCredential {
                label: "Personal".into(),
                access_token: "token-a".into(),
                expires_at_ms: Some(1796815655992),
                server: "https://opencode.ai/console".into(),
                org_id: "org_1".into(),
            }
        );
        let inactive = database(&[("Personal", false, VALUE)]);
        assert!(read_console_credential(inactive.path()).is_err());
        assert!(read_console_credential(Path::new("/nonexistent/opencode.db")).is_err());
    }

    #[test]
    fn summarizes_the_go_meters() {
        let status = json!({"product": "go", "access": {"meters": {
            "fiveHour": {"startsAt": null, "resetsAt": null, "limitMicroCents": "1200000000", "usedMicroCents": "0"},
            "week": {"resetsAt": "2026-11-16T00:00:00.000Z", "limitMicroCents": "3000000000", "usedMicroCents": "2003178311"},
            "month": {"resetsAt": "2026-12-09T10:29:58.000Z", "limitMicroCents": "6000000000", "usedMicroCents": "2003178311"}
        }}});
        let summary = serde_json::to_value(summarize(&status).unwrap()).unwrap();
        assert_eq!(
            summary,
            json!({
                "FiveHour": {"ResetsAt": null, "PercentLeft": 100.0, "UsedUsd": 0.0, "LimitUsd": 12.0},
                "Week": {"ResetsAt": "2026-11-16T00:00:00.000Z", "PercentLeft": 33.23, "UsedUsd": 20.03, "LimitUsd": 30.0},
                "Month": {"ResetsAt": "2026-12-09T10:29:58.000Z", "PercentLeft": 66.61, "UsedUsd": 20.03, "LimitUsd": 60.0}
            })
        );
        assert!(summarize(&json!({"access": null})).is_err());
    }
}
