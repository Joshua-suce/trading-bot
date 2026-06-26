# Production Runbook

## Preflight

```powershell
.\scripts\smoke_check.ps1
.\scripts\run_bot.cmd --mode health
.\scripts\run_bot.cmd --mode admin --admin-action status
```

The smoke gate uses an isolated audit database and does not alter operator
controls or production trade records.

## Start Trading

```powershell
.\scripts\run_bot.cmd --mode supervisor
```

The configured `BINANCE_API_URL` determines demo or mainnet. The execution
workflow is identical in both environments.

The supervisor starts the trade process, monitors its atomic heartbeat, and
restarts unexpected exits or stale processes within a bounded hourly budget.
Startup reconciliation restores audited exchange positions and verifies their
protective orders before new entries are allowed.

## Rollout Gates

Promotion proceeds only through automated tests, after-cost replay,
walk-forward validation, offline soak, a minimum two-week Demo canary, and then
limited production. The rollout artifact records the first unmet gate. Mainnet
keeps separate canary risk and position-size caps until an operator explicitly
disables canary mode after reviewing production evidence.

```powershell
.\scripts\run_bot.cmd --mode rollout --rollout-evidence data/governance/rollout_evidence.json
```

The command writes the evaluated decision to `ROLLOUT_ARTIFACT_PATH` and exits
non-zero at the first unmet gate.

## Controls

```powershell
.\scripts\run_bot.cmd --mode admin --admin-action pause --reason "operator pause"
.\scripts\run_bot.cmd --mode admin --admin-action resume --reason "checks passed"
.\scripts\run_bot.cmd --mode admin --admin-action emergency-stop --reason "risk event"
.\scripts\run_bot.cmd --mode admin --admin-action clear-emergency --reason "exchange reconciled"
```

## Incident Response

1. Activate emergency stop.
2. Inspect Binance positions and open orders.
3. Verify every position has protective orders.
4. Inspect audit events and alert delivery.
5. Flatten unmanaged exposure manually if necessary.
6. Clear emergency stop only after reconciliation is clean.

## Persistent State

- Audit DB: `data/audit/trading_audit.db`
- Text logs: `data/logs/trading_*.log`
- Error logs: `data/logs/errors_*.log`
- JSON logs: `data/logs/trading_*.jsonl`

Never commit `.env`, audit databases, reports, or logs.
