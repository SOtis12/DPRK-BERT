#!/usr/bin/env python3
"""
Verification script to test codebase fixes
Tests import handling and basic functionality
"""

import sys
from pathlib import Path

print("=" * 70)
print("DPRK-BERT Codebase Verification")
print("=" * 70)
print()

# Test 1: Check Python version
print("1. Python Version Check")
print(f"   Version: {sys.version}")
if sys.version_info < (3, 7):
    print("   ❌ FAIL: Python 3.7+ required")
    sys.exit(1)
print("   ✅ PASS")
print()

# Test 2: Core dependencies
print("2. Core Dependencies Check")
required_imports = {
    "json": "json",
    "pathlib": "pathlib",
    "csv": "csv",
    "re": "re"
}

all_core_passed = True
for name, module in required_imports.items():
    try:
        __import__(module)
        print(f"   ✅ {name}")
    except ImportError as e:
        print(f"   ❌ {name}: {e}")
        all_core_passed = False

if not all_core_passed:
    print("   ❌ FAIL: Core dependencies missing")
    sys.exit(1)
print()

# Test 3: Optional dependencies (graceful failure expected)
print("3. Optional Dependencies Check (warnings OK)")
optional_imports = {
    "torch": "PyTorch for model training",
    "transformers": "HuggingFace Transformers",
    "requests": "HTTP requests (for scraping)",
    "bs4": "BeautifulSoup (for parsing)",
}

for module, description in optional_imports.items():
    try:
        __import__(module)
        print(f"   ✅ {module}: {description}")
    except ImportError:
        print(f"   ⚠️  {module}: Not installed - {description}")
print()

# Test 4: TPU libraries (should fail gracefully)
print("4. TPU Libraries Check (expected to fail on Mac)")
try:
    import torch_xla.core.xla_model as xm
    print("   ✅ torch_xla: Available (running on TPU?)")
except ImportError:
    print("   ⚠️  torch_xla: Not installed (expected on local machine)")
print()

# Test 5: Project structure
print("5. Project Structure Check")
project_root = Path(__file__).parent
required_paths = {
    "Resources": "Data resources directory",
    "DPRK-BERT-master": "Original DPRK-BERT code",
    "train_with_local_data.py": "Local training script",
    "TASK.md": "Task definition",
}

structure_ok = True
for path_name, description in required_paths.items():
    path = project_root / path_name
    if path.exists():
        print(f"   ✅ {path_name}: {description}")
    else:
        print(f"   ❌ {path_name}: Missing - {description}")
        structure_ok = False

if not structure_ok:
    print("   ⚠️  WARNING: Some project files missing")
print()

# Test 6: Import main scripts
print("6. Script Import Check")
scripts_to_test = [
    "train_with_local_data",
]

import_ok = True
for script_name in scripts_to_test:
    script_path = project_root / f"{script_name}.py"
    if not script_path.exists():
        print(f"   ⚠️  {script_name}.py: Not found")
        continue
    
    try:
        # Try to compile the script
        with open(script_path) as f:
            code = f.read()
        compile(code, script_path, 'exec')
        print(f"   ✅ {script_name}.py: Compiles successfully")
    except SyntaxError as e:
        print(f"   ❌ {script_name}.py: Syntax error - {e}")
        import_ok = False
    except Exception as e:
        print(f"   ⚠️  {script_name}.py: {type(e).__name__}: {e}")
print()

# Test 7: Error handling test
print("7. Error Handling Test")
try:
    # Test urllib3 import handling
    try:
        from urllib3.exceptions import InsecureRequestWarning
        print("   ✅ urllib3 modern import works")
    except ImportError:
        try:
            from requests.packages.urllib3.exceptions import InsecureRequestWarning
            print("   ✅ urllib3 legacy import works (fallback)")
        except ImportError:
            print("   ⚠️  urllib3 not available (requests not installed)")
except Exception as e:
    print(f"   ❌ Error handling test failed: {e}")
    import_ok = False
print()

# Summary
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
if all_core_passed and import_ok:
    print("✅ All critical tests passed!")
    print()
    print("Next steps:")
    print("1. Install optional dependencies if needed:")
    print("   pip install -r requirements.txt")
    print()
    print("2. Run data processing:")
    print("   python3 train_with_local_data.py")
    print()
    print("3. For TPU training, human authorization required per TASK.md")
    sys.exit(0)
else:
    print("⚠️  Some tests failed. Review output above.")
    print()
    print("Common fixes:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Check Python version: python3 --version")
    print("3. Ensure project structure is complete")
    sys.exit(1)
