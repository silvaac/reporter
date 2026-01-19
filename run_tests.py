#!/usr/bin/env python3
"""
Test Runner Script

This script runs all unit tests for the Trading Performance Reporter package.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py -k test_name # Run specific test
    python run_tests.py --cov        # Run with coverage report
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run pytest with appropriate arguments."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add any command line arguments passed to this script
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {project_root}")
    print("=" * 70)
    print()
    
    # Run pytest
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=False
        )
        
        print()
        print("=" * 70)
        if result.returncode == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print("✗ SOME TESTS FAILED")
        print("=" * 70)
        
        return result.returncode
        
    except FileNotFoundError:
        print("Error: pytest not found. Install it with: pip install pytest")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
