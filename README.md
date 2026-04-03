# Trading Performance Reporter

A comprehensive Python package for generating professional trading performance reports for Hyperliquid accounts.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Report Contents](#report-contents)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Contributing](#contributing)

## Features

- **📊 AUM Tracking**: Monitor Assets Under Management over time with detailed historical data
- **💹 Performance Analysis**: Track P&L in both USD and percentage with cumulative metrics
- **🔄 Trade Analysis**: Detailed breakdown of all trades including costs, fees, and net P&L
- **💸 Funding Costs**: Comprehensive tracking of funding payments with:
  - Token price matching at funding time (UTC-aware)
  - Calculated funding verification (price × size × rate)
  - Cumulative funding costs chart
  - Cumulative funding rate chart (in basis points)
  - Local caching for efficient price data retrieval
- **📈 Visualizations**: Professional charts and graphs using matplotlib
- **📧 Email Reports**: Automated HTML email delivery with embedded visualizations
- **🎯 Spot & Perpetuals**: Full support for both spot and perpetual trading accounts
- **📅 Monthly Performance**: Monthly P&L summary in $ and % with cumulative tracking
- **✅ Unit Tests**: 73 comprehensive tests covering all core functionality

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env`:
```bash
MAILGUN_API_KEY=your_mailgun_api_key
MAILGUN_DOMAIN=your_mailgun_domain
EMAIL_FROM=your_email@example.com
```

4. Create your Hyperliquid configuration file (see Configuration section)

### Verify Installation

```bash
# Run verification script
python verify_installation.py

# Run tests to ensure everything works
python run_tests.py
```

## Quick Start

### 1. Configure Report Parameters

Edit `run_report.py`:

```python
CONFIG_FILE = "config_hyperliquid.json"  # Your config file
TESTNET = False                          # True for testnet, False for mainnet
REPORT_PERIOD = "allTime"                # "day", "week", "month", or "allTime"
LOOKBACK_DAYS = 30                       # Days to look back for trades/funding
EMAIL_TO = "your_email@example.com"      # Recipient email address
EMAIL_SUBJECT = "Trading Performance Report - {date}"
OUTPUT_DIR = "reports"                   # Directory for saving reports
```

### 2. Run the Report

```bash
python run_report.py
```

This will:
1. Load your account configuration
2. Fetch all trading data
3. Generate comprehensive analysis
4. Create visualizations
5. Generate HTML report
6. Send report via email
7. Save report locally in `reports/` directory

## Configuration

### Hyperliquid Configuration

Create a `config_hyperliquid.json` file with your account credentials:

```json
{
    "secret_key": "0x...",
    "account_address": "0x...",
    "API_address": "https://api.hyperliquid.xyz/"
}
```

**⚠️ IMPORTANT**: This file contains sensitive information and is automatically added to `.gitignore`. Never commit it to version control.

### Report Periods

Available time periods:
- `"day"`: Last 24 hours
- `"week"`: Last 7 days
- `"month"`: Last 30 days
- `"allTime"`: All available history

### Email Configuration

The package uses Mailgun for email delivery. Configure in `.env`:

```bash
MAILGUN_API_KEY=your_api_key_here
MAILGUN_DOMAIN=your_domain_here
EMAIL_FROM=your_email@example.com
```

## Usage

### Command Line

```bash
# Generate and email report
python run_report.py

# Run tests before making changes
python run_tests.py

# Verify installation
python verify_installation.py
```

### Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook example_usage.ipynb
```

The notebook demonstrates:
- Loading configuration
- Fetching account metrics
- Generating AUM data
- Analyzing performance
- Creating trade analysis
- Tracking funding costs
- Generating visualizations
- Creating HTML reports

### Python API

```python
from config import load_config
from hyperliquid_reporter.monitoring import HyperliquidMonitor
from hyperliquid_reporter.reporter import HyperliquidReporter
from datetime import datetime, timedelta

# Load configuration
config = load_config("config_hyperliquid.json", testnet=False)

# Initialize monitor
monitor = HyperliquidMonitor(
    info=config['_hyperliquid_info'],
    address=config['account_address']
)

# Initialize reporter
reporter = HyperliquidReporter(
    monitor=monitor,
    account_address=config['account_address']
)

# Generate report data
report_data = reporter.generate_report_data(
    period="allTime",
    start_time=datetime.now() - timedelta(days=30),
    end_time=datetime.now(),
    lookback_days=30
)

# Create visualizations
visualizations = reporter.create_visualizations(
    report_data=report_data,
    output_dir="reports"
)

# Generate HTML report
html_content = reporter.generate_html_report(
    report_data=report_data,
    visualizations=visualizations
)

# Save report
with open("report.html", "w") as f:
    f.write(html_content)
```

## Report Contents

### Account Summary
- Current AUM
- Net Deposits (deposits - withdrawals)
- Total P&L (USD and %)
- Spot Value
- Perpetual Value
- Unrealized P&L
- Peak AUM

### Performance Metrics
- AUM over time chart
- Cumulative P&L in USD
- Cumulative P&L in percentage
- Performance visualization with color-coded gains/losses
- Daily-resampled uniform time series

### Monthly Performance
- Monthly P&L in USD and %
- Cumulative P&L tracking across months
- Starting and ending AUM per month

### Trading Activity
- Total number of trades
- Total trading volume
- Total fees paid
- Win rate percentage
- Average win/loss amounts
- Trade distribution charts (pie chart and volume by coin)
- Recent trades table (last 20 trades)

### Funding Costs
- Total funding paid/received
- Average funding payment
- Cumulative funding costs chart (USD over time)
- Cumulative funding rate chart (basis points over time)
- Funding by coin breakdown
- Detailed funding history table with:
  - Date (UTC and EST)
  - Coin symbol
  - Funding payment (actual from exchange)
  - Position size at funding time
  - Funding rate in basis points
  - Token price at funding time (matched via UTC timestamps)
  - Calculated funding (price × |size| × rate for verification)

## API Reference

### HyperliquidMonitor

```python
monitor = HyperliquidMonitor(info, address)

# Get current metrics
metrics = monitor.get_current_metrics()
# Returns: PerformanceMetrics(account_value, unrealized_pnl, timestamp)

# Get account summary
summary = monitor.get_account_summary(lookback_days=3650)
# Returns: dict with deposits, withdrawals, current_value, total_pnl, etc.

# Get portfolio history
df = monitor.get_portfolio_dataframe(period="allTime", data_type="both")
# Returns: DataFrame with account_value and pnl columns

# Get trade history
trades = monitor.get_trade_history(
    start_time=None, 
    end_time=None, 
    coin=None,
    as_dataframe=True
)
# Returns: DataFrame with trade details

# Get funding history
funding = monitor.get_funding_history(
    start_time=None, 
    end_time=None, 
    coin=None,
    lookback=30,
    as_dataframe=True
)
# Returns: DataFrame with funding payments
```

### HyperliquidReporter

```python
reporter = HyperliquidReporter(monitor, account_address)

# Generate AUM data
aum_data = reporter.generate_aum_data(period="allTime")
# Returns: DataFrame with aum_usd column

# Generate performance data
perf_data = reporter.generate_performance_data(period="allTime")
# Returns: DataFrame with pnl_usd, pnl_pct, aum_usd columns

# Generate trade analysis
trade_analysis = reporter.generate_trade_analysis(
    start_time=None, 
    end_time=None
)
# Returns: DataFrame with coin, side, price, size, notional, fee, net_pnl

# Generate funding analysis
funding_analysis = reporter.generate_funding_analysis(
    start_time=None, 
    end_time=None,
    lookback_days=30
)
# Returns: DataFrame with coin, funding_payment, position_size, funding_rate,
#          token_price, calculated_funding

# Generate complete report
report_data = reporter.generate_report_data(
    period="allTime",
    start_time=None,
    end_time=None,
    lookback_days=30
)
# Returns: dict with all report components

# Create visualizations
visualizations = reporter.create_visualizations(
    report_data=report_data,
    output_dir="reports"
)
# Returns: dict mapping visualization names to base64-encoded images

# Generate HTML report
html = reporter.generate_html_report(
    report_data=report_data,
    visualizations=visualizations
)
# Returns: HTML string
```

## Testing

The package includes a comprehensive unit test suite with 42 tests covering all core functionality.

### Running Tests

```bash
# Run all tests (recommended)
python run_tests.py

# Or use pytest directly
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_exceptions.py

# Run with coverage report
pytest --cov=. --cov-report=html --cov-report=term
```

### Test Structure

```
tests/
├── test_exceptions.py              # 8 tests for exception classes
├── test_base_monitoring.py         # 8 tests for base monitoring
├── test_base_reporter.py           # 3 tests for base reporter
├── test_config.py                  # 13 tests for configuration
├── test_hyperliquid_reporter.py    # 12 tests for reporter implementation
├── test_funding_enhancements.py    # 1 test for funding enhancements
├── test_cumulative_rate_chart.py   # Cumulative rate chart verification
├── test_funding_timezone.py        # Timezone handling verification
├── test_price_matching.py          # Price matching verification
├── test_datetime_handling.py       # UTC/EST timezone handling
├── test_net_deposits_carryforward.py # Net deposits carry-forward logic
├── test_pnl_display_units.py       # P&L display units (% not bp)
├── test_daily_resample.py          # Daily resampling verification
├── test_monthly_performance.py     # Monthly performance aggregation
├── debug_price_data.py             # Price data structure debugging
├── conftest.py                     # Shared fixtures
└── README.md                       # Testing documentation
```

### Test Coverage

- ✅ Exception handling and inheritance
- ✅ Base class interfaces and abstract methods
- ✅ Configuration validation (TradingConfig, load_config)
- ✅ Reporter data generation (AUM, performance, trades, funding)
- ✅ Summary statistics calculation
- ✅ Visualization creation
- ✅ Funding enhancements (token prices, calculated funding)
- ✅ Price matching with UTC timezone handling
- ✅ Cumulative funding rate chart generation
- ✅ UTC-aware timestamps and US/Eastern conversion (DST-safe)
- ✅ Net deposits carry-forward (prevents phantom P&L)
- ✅ P&L display units (% columns, no double conversion)
- ✅ Daily resampling of non-uniform time series
- ✅ Monthly performance aggregation

### Testing Workflow

**Always run tests before implementing changes:**

```bash
# 1. Run tests to ensure current state is working
python run_tests.py

# 2. Make your code changes
# ... edit files ...

# 3. Run tests again to verify nothing broke
python run_tests.py

# 4. If tests fail, fix issues and repeat step 3
```

### Writing New Tests

Example test structure:

```python
"""Unit tests for my_module."""

import pytest
from my_module import MyClass

class TestMyClass:
    """Test MyClass functionality."""
    
    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return MyClass()
    
    def test_initialization(self, instance):
        """Test initialization."""
        assert instance is not None
    
    def test_method_success(self, instance):
        """Test method with valid input."""
        result = instance.my_method("valid")
        assert result == "expected"
    
    def test_method_error(self, instance):
        """Test method raises error with invalid input."""
        with pytest.raises(ValueError, match="error message"):
            instance.my_method("invalid")
```

## Project Structure

```
reporter/
├── base/                      # Base classes for extensibility
│   ├── __init__.py
│   ├── monitoring.py          # PerformanceMetrics, PortfolioMonitor
│   ├── portfolio.py           # Portfolio base class
│   └── reporter.py            # BaseReporter abstract class
│
├── hyperliquid_reporter/      # Hyperliquid-specific implementation
│   ├── __init__.py
│   ├── monitoring.py          # HyperliquidMonitor (uses token_data)
│   └── reporter.py            # HyperliquidReporter (800+ lines)
│
├── tests/                     # Unit tests
│   ├── test_*.py              # Test files
│   ├── conftest.py            # Shared fixtures
│   └── README.md              # Testing documentation
│
├── config.py                  # Configuration loader
├── exceptions.py              # Custom exceptions
├── mail_it.py                 # Email functionality
├── run_report.py              # Main script ⭐
├── run_tests.py               # Test runner
├── verify_installation.py     # Installation verification
├── example_usage.ipynb        # Example notebook
├── requirements.txt           # Dependencies
├── pytest.ini                 # Pytest configuration
└── README.md                  # This file
```

## Examples

### Generate Daily Report

```python
from datetime import datetime, timedelta

report_data = reporter.generate_report_data(
    period="day",
    start_time=datetime.now() - timedelta(days=1),
    end_time=datetime.now(),
    lookback_days=1
)
```

### Analyze Specific Coin

```python
# Get trades for specific coin
btc_trades = monitor.get_trade_history(coin="BTC", as_dataframe=True)

# Get funding for specific coin
btc_funding = monitor.get_funding_history(coin="BTC", as_dataframe=True)

# Analyze BTC performance
btc_stats = btc_trades.groupby('side').agg({
    'notional': 'sum',
    'fee': 'sum',
    'net_pnl': 'sum'
})
```

### Custom Time Range

```python
from datetime import datetime

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

trades = reporter.generate_trade_analysis(start_time=start, end_time=end)
funding = reporter.generate_funding_analysis(start_time=start, end_time=end)
```

### Get Current Metrics

```python
# Get current account metrics
metrics = monitor.get_current_metrics()
print(f"Account Value: ${metrics.account_value:,.2f}")
print(f"Unrealized P&L: ${metrics.unrealized_pnl:,.2f}")

# Get detailed account summary
summary = monitor.get_account_summary()
print(f"Total Deposits: ${summary['total_deposits']:,.2f}")
print(f"Total Withdrawals: ${summary['total_withdrawals']:,.2f}")
print(f"Net Deposits: ${summary['net_deposits']:,.2f}")
print(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['pnl_percentage']:.2f}%)")
```

## Troubleshooting

### Import Errors

If you encounter import errors, ensure you're running from the project root:

```bash
cd /path/to/reporter
source .venv/bin/activate
python run_report.py
```

### Email Not Sending

1. Verify `.env` file contains valid Mailgun credentials
2. Check that recipient email is authorized in your Mailgun account
3. Review console output for specific error messages
4. Test email function: `python mail_it.py`

### No Data Available

1. Ensure your account has trading history
2. Verify you're using the correct network (testnet vs mainnet)
3. Check that API credentials in `config_hyperliquid.json` are valid
4. Try with `TESTNET = True` first to test functionality

### Visualization Issues

If charts don't appear:
1. Ensure matplotlib is installed: `pip install matplotlib`
2. Check that you have sufficient data for the selected period
3. Try a different period (e.g., "allTime" instead of "day")
4. Check console for error messages

### Tests Failing

If tests fail:
1. Read the test failure output carefully
2. Identify which test failed and why
3. Check if your changes broke existing functionality
4. Fix the code or update the test if behavior intentionally changed
5. Re-run tests until all pass

### Pytest Not Found

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Or install all requirements
pip install -r requirements.txt
```

### Funding Analysis Issues

**Issue: Token prices showing as $0.00**
- **Cause**: Price data not available in cache or API issues
- **Solution**: First run downloads and caches price data. Check `./data/hyperliquid/perp/` directory is created with write permissions. Subsequent runs will be faster.

**Issue: Calculated funding doesn't match reported funding**
- **Cause**: Exchange may use different price source (mark price vs last price)
- **Solution**: This is expected. Small differences are normal. The calculated funding uses close price from hourly candles for verification purposes.

**Issue: Incorrect timestamps in funding table**
- **Cause**: Timezone confusion between UTC and local time
- **Solution**: All funding data is in UTC. The report displays both UTC and EST (UTC-5) columns. Verify your system time is correct.

**Issue: Price matching rate below 100%**
- **Cause**: Price data may not cover the full funding history period
- **Solution**: Price data is cached starting from the first download. Historical data before the cache was created may not have prices. This is normal for older funding entries.

## Security

### Best Practices

- **Never commit** `config_hyperliquid.json` or any file containing private keys
- Store sensitive credentials in environment variables (`.env` file)
- Use `.gitignore` to exclude configuration files (already configured)
- Regularly rotate API keys and secrets
- Use testnet for development and testing
- Review generated reports before sharing

### Protected Files

The following files are automatically excluded from version control:

- `config_hyperliquid.json`
- `config_hyperliquid_*.json`
- `*_config.json`
- `.env`

### Environment Variables

Required in `.env`:
- `MAILGUN_API_KEY`: Your Mailgun API key
- `MAILGUN_DOMAIN`: Your Mailgun domain
- `EMAIL_FROM`: Sender email address

## Dependencies

### Core Dependencies
- `pandas>=2.0.0`: Data manipulation and analysis
- `matplotlib>=3.7.0`: Visualization
- `hyperliquid-python-sdk>=0.4.0`: Hyperliquid API client
- `token_data>=0.1.0`: Enhanced Hyperliquid data access
- `requests>=2.31.0`: HTTP requests for email
- `python-dotenv>=1.0.0`: Environment variable management

### Development Dependencies
- `jupyter>=1.0.0`: Interactive notebooks
- `notebook>=7.0.0`: Jupyter notebook interface

### Testing Dependencies
- `pytest>=7.4.0`: Testing framework
- `pytest-cov>=4.1.0`: Coverage reporting
- `pytest-mock>=3.11.0`: Mocking support

## Contributing

This package follows a modular architecture:

- **Base classes** define interfaces for extensibility
- **Exchange-specific implementations** extend base classes
- **Easy to add support** for new exchanges

### Adding a New Exchange

1. Create new directory: `<exchange>_reporter/`
2. Implement `<Exchange>Monitor` extending `PortfolioMonitor`
3. Implement `<Exchange>Reporter` extending `BaseReporter`
4. Add tests in `tests/test_<exchange>_reporter.py`
5. Update documentation

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public methods
- Add unit tests for new functionality
- Run tests before committing: `python run_tests.py`

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions:
1. Check this README for documentation
2. Review `example_usage.ipynb` for code examples
3. Check console output for error messages
4. Run `python verify_installation.py` to check setup
5. Run `python run_tests.py` to verify functionality

## Changelog

### Version 1.2.0 (Current)
- ✅ Fixed datetime handling: all timestamps now UTC-aware
- ✅ Fixed EST conversion: uses US/Eastern timezone (handles DST automatically)
- ✅ Fixed net deposits bug: carry-forward prevents phantom P&L when ledger returns 0
- ✅ Fixed P&L display: removed double conversion, columns now show % instead of bp
- ✅ Fixed P&L time series: resampled to uniform daily intervals
- ✅ Added monthly performance table with P&L in $ and %
- ✅ 73 comprehensive unit tests (up from 42)

### Version 1.1.0
- ✅ Enhanced funding analysis with token price matching
- ✅ Calculated funding verification (price × size × rate)
- ✅ Cumulative funding rate chart (basis points)
- ✅ UTC timezone-aware price matching
- ✅ Local price data caching for performance
- ✅ 42 unit tests
- ✅ Improved error handling and logging

### Version 1.0.0
- ✅ Initial release
- ✅ Full Hyperliquid support
- ✅ AUM tracking with visualizations
- ✅ Performance analysis ($ and %)
- ✅ Trade cost analysis with dataframes
- ✅ Funding cost tracking with graphs
- ✅ Spot and perpetuals support
- ✅ HTML report generation
- ✅ Email delivery via Mailgun
- ✅ Jupyter notebook examples
- ✅ Professional documentation

---

**Ready to use!** Run `python run_report.py` to generate your first report. 📊
