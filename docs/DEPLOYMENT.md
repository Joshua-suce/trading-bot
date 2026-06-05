# Deployment Guide

## Production Deployment

### Prerequisites
- Python 3.11+ or Docker
- Binance Futures account with API key
- API key must have: **Enable Futures**, **Enable Reading**, restricted to your IP
- API key must **NOT** have withdrawal permissions

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t trading-bot:latest .

# Run with docker-compose (includes auto-restart)
docker-compose up -d

# View logs
docker logs -f trading-bot

# Stop
docker-compose down
```

### Option 2: Bare Metal

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Test with paper trading first
python -m src.main --mode paper

# Monitor with dashboard (separate terminal)
python -m src.main --mode dashboard

# Production live trading
python -m src.main --mode live
```

### Option 3: Systemd Service

Create `/etc/systemd/system/trading-bot.service`:

```ini
[Unit]
Description=AI Trading Bot
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/opt/trading-bot
EnvironmentFile=/opt/trading-bot/.env
ExecStart=/opt/trading-bot/.venv/bin/python -m src.main --mode live
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

### Option 4: Screen / tmux

```bash
screen -S trading-bot
python -m src.main --mode live
# Detach: Ctrl+A, D
# Reattach: screen -r trading-bot
```

## Configuration Checklist

Before going live, verify:

- [ ] **BINANCE_TESTNET** = `false` (set to `true` for testing)
- [ ] **ALLOW_LIVE_TRADING** = `true` only after production controls are verified
- [ ] **ALLOW_MAINNET_PAPER** = `true` only if paper mode is intentionally allowed to call mainnet account APIs
- [ ] API key IP-restricted to your server's IP
- [ ] API key has **No Withdrawal** permission
- [ ] `MAX_LEVERAGE` set appropriately (start with 1–3x)
- [ ] `RISK_PER_TRADE` conservative (start with 0.01–0.02)
- [ ] `MAX_DRAWDOWN` set to your risk tolerance
- [ ] `DAILY_LOSS_LIMIT` set to auto-stop on bad days
- [ ] Logging configured (`LOG_DIR`, `LOG_LEVEL`)
- [ ] Alerts configured (Telegram/Discord) for notifications

## Monitoring

### Dashboard
```bash
python -m src.main --mode dashboard
# http://localhost:8501
```

### Logs
```bash
# Real-time
tail -f data/logs/trading_$(date +%Y-%m-%d).log

# Error logs only
tail -f data/logs/errors_$(date +%Y-%m-%d).log
```

### Alerts
Configure `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, or `DISCORD_WEBHOOK_URL` in `.env` to receive:
- Trade entry/exit notifications
- Error alerts
- Daily PnL summaries

### Health Check
Monitor these metrics:
1. **Equity curve** — Is it trending up over time?
2. **Win rate** — Expected 40–60%
3. **Max drawdown** — Should stay below configured limit
4. **Number of trades** — Should be reasonable for the timeframe
5. **Daily PnL** — Daily loss limit should stop the bot on bad days

## Security

1. **API Key Security**
   - Never commit `.env` to version control
   - Use different API keys for testnet and mainnet
   - Restrict API keys to specific IP addresses
   - Create API keys with **No Withdrawal** permission only

2. **Server Security**
   - Run as non-root user
   - Keep the server updated
   - Use firewall rules to restrict access
   - Run behind a reverse proxy if exposing dashboard publicly

3. **Funding**
   - Only deposit what you're willing to lose
   - Start small — the strategy may need tuning for current market conditions
   - Consider running on testnet for at least 2–4 weeks before going live

## Backup & Recovery

```bash
# Backup config and data
tar -czf trading-bot-backup-$(date +%Y%m%d).tar.gz \
  .env \
  data/models/ \
  data/logs/

# Restore
tar -xzf trading-bot-backup-*.tar.gz
```

## Scaling

For multiple symbols or higher frequency:
1. Increase `--limit` for more training data
2. Consider dedicated server (2+ CPU cores, 4+ GB RAM)
3. Use a managed database for trade history (currently in-memory)
4. Run separate bot instances for different strategies
