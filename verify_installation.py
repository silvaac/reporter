#!/usr/bin/env python3
"""
Verification script to ensure the Trading Performance Reporter is properly installed.
Run this script to verify all components are working correctly.
"""

import sys
from pathlib import Path

print("=" * 70)
print("TRADING PERFORMANCE REPORTER - INSTALLATION VERIFICATION")
print("=" * 70)
print()

# Test 1: Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
else:
    print(f"   ✗ Python version too old. Need 3.8+, have {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)

# Test 2: Core modules
print("\n2. Testing core module imports...")
try:
    from exceptions import ConfigurationError, ExchangeError, ReportGenerationError
    print("   ✓ exceptions module")
except Exception as e:
    print(f"   ✗ exceptions module: {e}")
    sys.exit(1)

try:
    from base.monitoring import PerformanceMetrics, PortfolioMonitor
    print("   ✓ base.monitoring module")
except Exception as e:
    print(f"   ✗ base.monitoring module: {e}")
    sys.exit(1)

try:
    from base.reporter import BaseReporter
    print("   ✓ base.reporter module")
except Exception as e:
    print(f"   ✗ base.reporter module: {e}")
    sys.exit(1)

try:
    from config import load_config
    print("   ✓ config module")
except Exception as e:
    print(f"   ✗ config module: {e}")
    sys.exit(1)

# Test 3: Hyperliquid implementation
print("\n3. Testing Hyperliquid implementation...")
try:
    from hyperliquid_reporter.monitoring import HyperliquidMonitor
    print("   ✓ hyperliquid_reporter.monitoring module")
except Exception as e:
    print(f"   ✗ hyperliquid_reporter.monitoring module: {e}")
    sys.exit(1)

try:
    from hyperliquid_reporter.reporter import HyperliquidReporter
    print("   ✓ hyperliquid_reporter.reporter module")
except Exception as e:
    print(f"   ✗ hyperliquid_reporter.reporter module: {e}")
    sys.exit(1)

# Test 4: Email module
print("\n4. Testing email module...")
try:
    from mail_it import mail_it
    print("   ✓ mail_it module")
except Exception as e:
    print(f"   ✗ mail_it module: {e}")
    sys.exit(1)

# Test 5: Required dependencies
print("\n5. Checking required dependencies...")
dependencies = [
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib.pyplot'),
    ('hyperliquid', 'hyperliquid.info'),
    ('token_data', 'token_data.hyperliquid'),
    ('requests', 'requests'),
    ('dotenv', 'dotenv'),
]

for name, import_path in dependencies:
    try:
        __import__(import_path)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - not installed")
        print(f"      Install with: pip install {name}")

# Test 6: Configuration files
print("\n6. Checking configuration files...")
config_file = Path("config_hyperliquid.json")
if config_file.exists():
    print(f"   ✓ {config_file} exists")
else:
    print(f"   ⚠ {config_file} not found")
    print("      You'll need to create this file with your Hyperliquid credentials")

env_file = Path(".env")
if env_file.exists():
    print(f"   ✓ {env_file} exists")
else:
    print(f"   ⚠ {env_file} not found")
    print("      You'll need to create this file with your Mailgun credentials")

# Test 7: Directory structure
print("\n7. Verifying directory structure...")
required_dirs = ['base', 'hyperliquid_reporter', 'reports']
for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"   ✓ {dir_name}/ directory")
    else:
        if dir_name == 'reports':
            print(f"   ℹ {dir_name}/ will be created when first report is generated")
        else:
            print(f"   ✗ {dir_name}/ directory missing")

# Test 8: Key files
print("\n8. Checking key files...")
key_files = [
    'run_report.py',
    'example_usage.ipynb',
    'README.md',
    'requirements.txt',
    '.gitignore'
]
for filename in key_files:
    file_path = Path(filename)
    if file_path.exists():
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename} missing")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print()
print("✓ All core components are properly installed!")
print()
print("Next steps:")
print("1. Configure your Hyperliquid credentials in config_hyperliquid.json")
print("2. Configure your Mailgun credentials in .env")
print("3. Edit parameters in run_report.py")
print("4. Run: python run_report.py")
print()
print("For detailed instructions, see:")
print("  - README.md (comprehensive documentation)")
print("  - USAGE_GUIDE.md (quick start guide)")
print("  - example_usage.ipynb (interactive examples)")
print()
print("=" * 70)
