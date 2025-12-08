#!/usr/bin/env python
"""
Simple import test for the backend
"""
import sys
import traceback

print("=" * 60)
print("Testing SynQc Backend Imports")
print("=" * 60)

try:
    print("\n[1/5] Checking Python version...")
    print(f"      Python {sys.version}")
    
    print("\n[2/5] Testing standard library imports...")
    import json
    import asyncio
    import threading
    print("      ✓ Standard library OK")
    
    print("\n[3/5] Testing third-party imports...")
    import numpy
    from fastapi import FastAPI
    from pydantic import BaseModel
    from dotenv import load_dotenv
    print("      ✓ Third-party libraries OK")
    
    print("\n[4/5] Attempting backend import...")
    import synqc_tds_super_backend as backend
    print("      ✓ Backend module imported successfully")
    
    print("\n[5/5] Verifying key components...")
    assert hasattr(backend, 'app'), "Missing 'app'"
    assert hasattr(backend, 'store'), "Missing 'store'"
    assert hasattr(backend, 'engine'), "Missing 'engine'"
    assert hasattr(backend, 'SynQcEngine'), "Missing 'SynQcEngine'"
    assert hasattr(backend, 'ChatRequest'), "Missing 'ChatRequest'"
    assert hasattr(backend, 'ChatResponse'), "Missing 'ChatResponse'"
    assert hasattr(backend, 'RunConfiguration'), "Missing 'RunConfiguration'"
    assert hasattr(backend, 'KpiBundle'), "Missing 'KpiBundle'"
    print("      ✓ All key components present")
    
    print("\n" + "=" * 60)
    print("✓ All tests PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Test FAILED!")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
