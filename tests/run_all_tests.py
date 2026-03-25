#!/usr/bin/env python
"""
FP-SAN NSS Test Runner - Execute all test suites with detailed reporting.

This script runs all five verification phases:
  1. Architecture validation tests (test_architecture.py)
  2. Metrics and loss function tests (test_metrics_loss.py)
  3. Pipeline integration tests (test_pipeline.py)
  4. Phase gate validation (verify_phase1.py)
  5. Summary and reporting
"""

import subprocess
import sys
import os
from pathlib import Path


def run_pytest_suite():
    """Run pytest test suite with detailed output."""
    print("=" * 70)
    print("  RUNNING PYTEST TEST SUITE")
    print("=" * 70)
    
    test_dir = Path(__file__).parent
    cmd = [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"]
    
    result = subprocess.run(cmd, cwd=test_dir)
    return result.returncode == 0


def run_phase_gate_verification():
    """Run the Phase 1 verification suite."""
    print("\n" + "=" * 70)
    print("  RUNNING PHASE GATE VERIFICATION (verify_phase1.py)")
    print("=" * 70)
    
    repo_root = Path(__file__).parent.parent
    verify_script = repo_root / "verify_phase1.py"
    
    cmd = [sys.executable, str(verify_script)]
    result = subprocess.run(cmd, cwd=repo_root)
    return result.returncode == 0


def run_quick_tests():
    """Run quick smoke tests."""
    print("\n" + "=" * 70)
    print("  RUNNING QUICK TESTS (test_quick.py)")
    print("=" * 70)
    
    repo_root = Path(__file__).parent.parent
    quick_test = repo_root / "test_quick.py"
    
    if quick_test.exists():
        cmd = [sys.executable, str(quick_test)]
        result = subprocess.run(cmd, cwd=repo_root)
        return result.returncode == 0
    else:
        print(f"Quick test file not found: {quick_test}")
        return True


def main():
    """Execute all test phases."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FP-SAN NSS COMPREHENSIVE TEST SUITE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {}
    
    # Phase 1: Pytest suite
    print("\n[Phase 1/3] Running pytest test suite...")
    results['pytest'] = run_pytest_suite()
    
    # Phase 2: Phase gate verification
    print("\n[Phase 2/3] Running phase gate verification...")
    results['phase_gates'] = run_phase_gate_verification()
    
    # Phase 3: Quick smoke tests
    print("\n[Phase 3/3] Running quick smoke tests...")
    results['quick_tests'] = run_quick_tests()
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    for phase, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {phase.replace('_', ' ').title():.<40} {status}")
    
    all_passed = all(results.values())
    print("=" * 70)
    
    if all_passed:
        print("\n✓ ALL TEST SUITES PASSED!")
        print("  Ready to proceed with training.\n")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  Please review output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
