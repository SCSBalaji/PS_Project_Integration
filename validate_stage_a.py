#!/usr/bin/env python
"""
Stage A Validation Script

This script validates that all Stage A preparation tasks are complete.
Run this before proceeding to Stage B (block implementation).

Usage:
    python validate_stage_a.py
"""

import sys
import os
from pathlib import Path
from typing import Tuple, List


def check_pass(msg: str):
    print(f"  [PASS]: {msg}")


def check_fail(msg: str):
    print(f"  [FAIL]: {msg}")


def check_warn(msg: str):
    print(f"  [WARN]: {msg}")


def check_info(msg: str):
    print(f"  [INFO]: {msg}")


def validate_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists."""
    if Path(filepath).exists():
        check_pass(f"{description} exists: {filepath}")
        return True
    else:
        check_fail(f"{description} missing: {filepath}")
        return False


def validate_directory_exists(dirpath: str, description: str) -> bool:
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        check_pass(f"{description} exists: {dirpath}")
        return True
    else:
        check_fail(f"{description} missing: {dirpath}")
        return False


def run_validation() -> Tuple[int, int, int]:
    """
    Run all Stage A validation checks.
    
    Returns:
        Tuple of (passed, failed, warnings)
    """
    passed = 0
    failed = 0
    warnings = 0
    
    print("\n")
    print("=" * 80)
    print("  STAGE A VALIDATION CHECKLIST".center(80))
    print("  MobilePlantViT Project".center(80))
    print("=" * 80)
    print()
    
    # =========================================================================
    # CHECK 1: Configuration Files
    # =========================================================================
    print(f"\n[1/10] Configuration Files")
    print("-" * 60)
    
    if validate_file_exists("config/defaults.yaml", "Default config"):
        passed += 1
        # Validate config content
        try:
            import yaml
            with open("config/defaults.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['reproducibility', 'dataset', 'model', 'training']
            missing = [s for s in required_sections if s not in config]
            
            if not missing:
                check_pass(f"Config has all required sections: {required_sections}")
                passed += 1
            else:
                check_fail(f"Config missing sections: {missing}")
                failed += 1
            
            # Check seed
            if 'reproducibility' in config and 'seed' in config['reproducibility']:
                check_pass(f"Seed configured: {config['reproducibility']['seed']}")
                passed += 1
            else:
                check_fail("Seed not configured in reproducibility section")
                failed += 1
                
        except Exception as e:
            check_fail(f"Error reading config: {e}")
            failed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 2: Reproducibility Utilities
    # =========================================================================
    print(f"\n[2/10] Reproducibility Utilities")
    print("-" * 60)
    
    if validate_file_exists("utils/repro.py", "Reproducibility module"):
        passed += 1
        
        # Test import and functionality
        try:
            sys.path.insert(0, str(Path.cwd()))
            from utils import set_seed, load_config, verify_reproducibility
            
            check_pass("Reproducibility functions importable")
            passed += 1
            
            # Test seed setting
            set_seed(42)
            check_pass("set_seed() works correctly")
            passed += 1
            
        except ImportError as e:
            check_fail(f"Import error: {e}")
            failed += 1
        except Exception as e:
            check_fail(f"Error testing reproducibility: {e}")
            failed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 3: Experiment Tracking
    # =========================================================================
    print(f"\n[3/10] Experiment Tracking")
    print("-" * 60)
    
    if validate_file_exists("utils/logging_utils.py", "Logging utilities"):
        passed += 1
        
        try:
            from utils import ExperimentLogger
            check_pass("ExperimentLogger importable")
            passed += 1
        except ImportError as e:
            check_fail(f"ExperimentLogger import error: {e}")
            failed += 1
    else:
        failed += 1
    
    if validate_file_exists("utils/experiment.py", "Experiment manager"):
        passed += 1
        
        try:
            from utils import ExperimentManager, create_experiment
            check_pass("ExperimentManager importable")
            passed += 1
        except ImportError as e:
            check_fail(f"ExperimentManager import error: {e}")
            failed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 4: Tests Directory & Unit Tests
    # =========================================================================
    print(f"\n[4/10] Tests Directory & Unit Tests")
    print("-" * 60)
    
    if validate_directory_exists("tests", "Tests directory"):
        passed += 1
        
        # Check for required test files
        test_files = [
            ("tests/test_model_blocks.py", "Block unit tests"),
            ("tests/test_smoke.py", "Smoke tests"),
            ("tests/test_reproducibility.py", "Reproducibility tests"),
        ]
        
        for filepath, desc in test_files:
            if validate_file_exists(filepath, desc):
                passed += 1
            else:
                failed += 1
        
        # Check pytest config
        if validate_file_exists("pytest.ini", "Pytest configuration"):
            passed += 1
        else:
            check_warn("pytest.ini not found (optional)")
            warnings += 1
            
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 5: CI Configuration
    # =========================================================================
    print(f"\n[5/10] CI Configuration")
    print("-" * 60)
    
    if validate_file_exists(".github/workflows/ci.yml", "CI workflow"):
        passed += 1
        
        # Check CI content
        try:
            with open(".github/workflows/ci.yml", 'r') as f:
                ci_content = f.read()
            
            checks = [
                ("pytest", "pytest test step"),
                ("smoke", "smoke test reference"),
            ]
            
            for keyword, desc in checks:
                if keyword.lower() in ci_content.lower():
                    check_pass(f"CI includes {desc}")
                    passed += 1
                else:
                    check_warn(f"CI may be missing {desc}")
                    warnings += 1
                    
        except Exception as e:
            check_fail(f"Error reading CI config: {e}")
            failed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 6: Blocks Implementation
    # =========================================================================
    print(f"\n[6/10] Model Blocks")
    print("-" * 60)
    
    if validate_directory_exists("blocks", "Blocks directory"):
        passed += 1
        
        block_files = [
            ("blocks/ghost_conv.py", "GhostConv"),
            ("blocks/fused_ir.py", "Fused Inverted Residual"),
            ("blocks/coord_att.py", "Coordinate Attention"),
            ("blocks/patch_embed.py", "Patch Embedding"),
            ("blocks/lda.py", "Linear Differential Attention"),
            ("blocks/res_norm.py", "Residual LayerNorm"),
            ("blocks/bottleneck_ffn.py", "Bottleneck FFN"),
            ("blocks/classifier.py", "Classifier Head"),
        ]
        
        for filepath, desc in block_files:
            if validate_file_exists(filepath, desc):
                passed += 1
            else:
                failed += 1
        
        # Test imports
        try:
            from blocks import (
                GhostConv, FusedInvertedResidualBlock, CoordAtt,
                PatchEmbedding, PositionalEncoding, LinearDifferentialAttention,
                ResidualLayerNormBlock, BottleneckFFN, ClassifierHead, GlobalAveragePooling
            )
            check_pass("All blocks importable")
            passed += 1
        except ImportError as e:
            check_fail(f"Block import error: {e}")
            failed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 7: Compute Plan
    # =========================================================================
    print(f"\n[7/10] Compute Plan")
    print("-" * 60)
    
    if validate_file_exists("COMPUTE_PLAN.md", "Compute plan document"):
        passed += 1
    else:
        check_warn("COMPUTE_PLAN.md not found (recommended)")
        warnings += 1
    
    if validate_file_exists("utils/check_compute.py", "Compute checker"):
        passed += 1
    else:
        failed += 1
    
    # =========================================================================
    # CHECK 8: Documentation
    # =========================================================================
    print(f"\n[8/10] Documentation")
    print("-" * 60)
    
    docs = [
        ("README.md", "README"),
        ("CONTRIBUTING.md", "Contributing guidelines"),
        ("CODE_OF_CONDUCT.md", "Code of conduct"),
        ("LICENSE", "License file"),
    ]
    
    for filepath, desc in docs:
        if validate_file_exists(filepath, desc):
            passed += 1
        else:
            if filepath == "LICENSE":
                check_warn(f"{desc} not found (recommended)")
                warnings += 1
            else:
                failed += 1
    
    # =========================================================================
    # CHECK 9: PR/Issue Templates
    # =========================================================================
    print(f"\n[9/10] PR & Issue Templates")
    print("-" * 60)
    
    templates = [
        (".github/PULL_REQUEST_TEMPLATE.md", "PR template"),
        (".github/ISSUE_TEMPLATE/bug_report.md", "Bug report template"),
        (".github/ISSUE_TEMPLATE/feature_request.md", "Feature request template"),
    ]
    
    for filepath, desc in templates:
        if validate_file_exists(filepath, desc):
            passed += 1
        else:
            check_warn(f"{desc} not found (optional)")
            warnings += 1
    
    # =========================================================================
    # CHECK 10: Smoke Test Execution
    # =========================================================================
    print(f"\n[10/10] Smoke Test Execution")
    print("-" * 60)
    
    if validate_file_exists("run_smoke_test.py", "Smoke test script"):
        passed += 1
        
        check_info("Running smoke test (this may take a moment)...")
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "run_smoke_test.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                check_pass("Smoke test passed!")
                passed += 1
            else:
                check_fail("Smoke test failed")
                check_info(f"Error output: {result.stderr[:500] if result.stderr else 'None'}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            check_fail("Smoke test timed out (>120s)")
            failed += 1
        except Exception as e:
            check_fail(f"Error running smoke test: {e}")
            failed += 1
    else:
        failed += 1
    
    return passed, failed, warnings


def print_summary(passed: int, failed: int, warnings: int):
    """Print validation summary."""
    total = passed + failed
    
    print("\n")
    print("=" * 80)
    print("  STAGE A VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Warnings: {warnings}")
    print()
    
    if failed == 0:
        print("  *** STAGE A COMPLETE! ***")
        print()
        print("  All critical checks passed. You are ready to proceed to Stage B!")
        print()
        print("  Next steps:")
        print("    1. Commit any remaining changes")
        print("    2. Push to feature/upgraded-arch branch")
        print("    3. Begin Stage B: Full model integration")
        print()
    else:
        print("  *** STAGE A INCOMPLETE ***")
        print()
        print(f"  {failed} check(s) failed. Please fix the issues above before proceeding.")
        print()
        print("  Common fixes:")
        print("    - Missing files: Create the required files")
        print("    - Import errors: Check __init__.py exports")
        print("    - Test failures: Run pytest to see detailed errors")
        print()
    
    if warnings > 0:
        print(f"  Note: {warnings} warning(s) found. These are optional but recommended.")
        print()
    
    print("=" * 80)


def main():
    """Main entry point."""
    # Change to project root
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Run validation
    passed, failed, warnings = run_validation()
    
    # Print summary
    print_summary(passed, failed, warnings)
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()