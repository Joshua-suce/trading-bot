# Production Runbook

This runbook is for controlled paper/demo/live operation. Live mainnet should remain disabled until demo soak tests, alert delivery, exchange reconciliation, and risk limits are verified.

## Preflight

Run the full local gate before every deployment:

```bash
python scripts/smoke_check.py
```

The smoke gate fails if total test coverage drops below 75%.

Run the secret scanner before sharing logs, building images, or opening a pull request:

```bash
python scripts/secret_scan.py
```

Run the bounded offline soak before any demo/paper deployment:

```bash
python -m src.main --mode soak --soak-iterations 2 --soak-report data/reports/soak_report.json
```

Windows PowerShell:

```powershell
.\scripts\smoke_check.ps1
```

## Required Controls

- `BINANCE_TESTNET=true` for demo/paper validation.
- `ALLOW_LIVE_TRADING=false` until mainnet approval.
- `TRADING_ENABLED=true` only when operators are ready for new entries.
- `POSITION_SCOPE=symbol` unless the strategy is explicitly approved to hold one position per symbol/timeframe.
- Use `*_FILE` secret variables where possible instead of placing secrets directly in `.env`.
- Never copy `.env`, audit databases, or log files into tickets or pull requests.
- Keep `JSON_LOGS=true` for production so logs can be parsed by external monitoring.

## Startup

```bash
python -m src.main --mode health
python -m src.main --mode admin --admin-action status
python -m src.main --mode paper
```

Docker:

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f trading-bot
```

## Health

```bash
python -m src.main --mode health
```

Health status:

- `ok`: audit DB is writable, trading is allowed, and there are no recent critical audit events.
- `degraded`: service is running but trading is paused, emergency stop is active, or recent critical events exist.
- `critical`: service health is broken, such as an unavailable audit database.

Docker uses the same health command.

## Admin Actions

Pause new entries:

```bash
python -m src.main --mode admin --admin-action pause --reason "operator pause"
```

Resume after checks:

```bash
python -m src.main --mode admin --admin-action resume --reason "checks passed"
```

Activate emergency stop:

```bash
python -m src.main --mode admin --admin-action emergency-stop --reason "manual safety stop"
```

Clear emergency stop only after exchange positions and protective orders are verified:

```bash
python -m src.main --mode admin --admin-action clear-emergency --reason "exchange reconciled"
```

## Incident Response

1. Activate emergency stop.
2. Check Binance open positions and open orders manually.
3. Verify the audit dashboard and recent critical events.
4. Confirm Telegram/Discord alert delivery.
5. If unmanaged exposure exists, flatten manually on Binance or with a controlled reduce-only order.
6. Keep emergency stop active until reconciliation is clean.

## Restart Recovery

On startup, live mode restores audited open trades, reconciles exchange positions, verifies protective stop/target order IDs, and activates emergency stop if state is unsafe. If health is degraded after restart, do not clear emergency stop until the exchange state is manually checked.

## Logs And Audit

- Text logs: `data/logs/trading_*.log`
- Error logs: `data/logs/errors_*.log`
- Structured logs: `data/logs/trading_*.jsonl`
- Audit DB: `data/audit/trading_audit.db`

Audit DB contains controls, trade records, and recent events used by health checks and the dashboard.

Secrets are redacted before standard logs and audit event payloads are written. The repository smoke check also runs `scripts/secret_scan.py` to catch accidental hardcoded credentials in source, docs, scripts, and config files.
