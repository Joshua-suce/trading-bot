# Contributing

## Development Setup

```bash
# Clone the repo
git clone <repo-url>
cd trading-bot

# Create virtual environment
bash scripts/create_venv.sh
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ml.txt  # optional

# Setup pre-commit hooks
pre-commit install
```

## Code Style

This project uses:
- **Black** for formatting (line length 88)
- **Ruff** for linting (E, F, I, W, B, C, Q)
- **isort** for import sorting (black profile)
- **mypy** for type checking

```bash
# Format code
python -m black .
python -m isort src tests

# Lint
python -m ruff check src tests

# Type check
python -m mypy src

# Fix auto-fixable lint issues
python -m ruff check --fix src tests
```

## Testing

```bash
# Run all tests
python -m pytest

# With coverage
python -m pytest --cov=src --cov-report=term-missing tests/

# Run specific test
python -m pytest tests/test_backtest.py -v

# Run tests matching pattern
python -m pytest -k "indicator"
```

All new code must have tests. Test files go in `tests/` with `test_` prefix.

## Project Conventions

- **Imports**: Use absolute imports (`from src.backtest.engine import BacktestEngine`)
- **Typing**: All function signatures must have type annotations
- **Async**: Exchange calls must be async; CPU-bound work (indicators, backtesting) is sync
- **DataFrames**: Always pass DataFrames by copy; never mutate in place
- **Exceptions**: Handle exceptions at module boundaries; log with `logger` instead of print
- **Config**: New configuration values go in `src/config.py` with pydantic validation

## Adding a New Indicator

1. Add the computation method to the appropriate file in `src/indicators/` (trend, momentum, volatility, or volume)
2. Register it in `src/indicators/compute.py` → `compute_all_indicators()`
3. Add tests in `tests/test_indicators.py`
4. If used in signal logic, update `src/signals/ta_signal.py`

## Adding a New Signal Source

1. Create a new signal class (e.g., `src/signals/onchain_signal.py`)
2. Return `TASignal(direction, strength, source)` for compatibility
3. Integrate in `src/signals/aggregator.py` or create a new aggregator

## Adding a New Exchange

1. Subclass or extend `src/exchange/client.py` with exchange-specific API config
2. Update `src/config.py` with exchange selection
3. Ensure `fetch_ohlcv()`, `create_order()`, and `watch_ohlcv()` return the same contracts

## Commit Messages

Follow conventional commits:
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation
- `test:` — Tests
- `refactor:` — Code refactoring
- `style:` — Formatting
- `chore:` — Maintenance

## Pull Request Process

1. Ensure tests pass: `python -m pytest`
2. Ensure lint/format passes: `python -m ruff check src tests && python -m black --check .`
3. Ensure type checks pass: `python -m mypy src`
4. Update documentation if changing public APIs
5. Add or update tests for any functional changes
