#!/usr/bin/env python3
"""
SynQc Backend Test Results
===========================
Quick validation that all backend components are properly configured.
"""

import sys
import json
from datetime import datetime

def main():
    print("\n" + "="*80)
    print(" SynQc Backend Test Results")
    print("="*80 + "\n")
    
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }
    
    # Test 1: Module import
    print("[1/6] Testing backend module import...")
    try:
        import synqc_tds_super_backend as backend
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Backend Import",
            "status": "PASS",
            "message": "synqc_tds_super_backend imported successfully"
        })
        print("      ✓ PASS\n")
    except Exception as e:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Backend Import",
            "status": "FAIL",
            "message": str(e)
        })
        print(f"      ✗ FAIL: {e}\n")
        return results
    
    # Test 2: Component check
    print("[2/6] Checking core components...")
    components = [
        "app", "store", "engine", "SynQcEngine", "ChatRequest", "ChatResponse",
        "RunConfiguration", "RunRecord", "KpiBundle", "_synthesize_measurements"
    ]
    all_present = True
    for comp in components:
        if hasattr(backend, comp) or (comp.startswith("_") and hasattr(backend.SynQcEngine, comp)):
            print(f"      ✓ {comp}")
        else:
            print(f"      ✗ {comp} - MISSING")
            all_present = False
    
    if all_present:
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Core Components",
            "status": "PASS",
            "message": f"All {len(components)} components present"
        })
        print("      ✓ PASS\n")
    else:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Core Components",
            "status": "FAIL",
            "message": "Some components missing"
        })
        print("      ✗ FAIL\n")
    
    # Test 3: Model creation
    print("[3/6] Testing model instantiation...")
    try:
        cfg = backend.RunConfiguration(
            hardware_target=backend.HardwareTarget.SIM_LOCAL,
            hardware_preset=backend.HardwarePreset.TRANSMON_DEFAULT,
            drive_envelope=backend.DriveEnvelope.GAUSSIAN,
        )
        
        kpi = backend.KpiBundle.from_raw(
            fidelity=0.99, latency_us=15.0, backaction=0.1,
            shots_used=1000, shot_limit=50000
        )
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Model Instantiation",
            "status": "PASS",
            "message": "RunConfiguration and KpiBundle created successfully"
        })
        print("      ✓ PASS\n")
    except Exception as e:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Model Instantiation",
            "status": "FAIL",
            "message": str(e)
        })
        print(f"      ✗ FAIL: {e}\n")
    
    # Test 4: Measurement synthesis
    print("[4/6] Testing measurement synthesis...")
    try:
        engine = backend.SynQcEngine()
        measurements = engine._synthesize_measurements(cfg, fidelity=0.95)
        
        assert len(measurements) == 4, f"Expected 4 measurements, got {len(measurements)}"
        for m in measurements:
            assert "qubit" in m and "p0" in m and "p1" in m and "last" in m
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Measurement Synthesis",
            "status": "PASS",
            "message": f"Generated {len(measurements)} qubit measurements with proper structure"
        })
        print(f"      ✓ PASS (generated {len(measurements)} measurements)\n")
    except Exception as e:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Measurement Synthesis",
            "status": "FAIL",
            "message": str(e)
        })
        print(f"      ✗ FAIL: {e}\n")
    
    # Test 5: Chat models
    print("[5/6] Testing chat models...")
    try:
        chat_msg = backend.ChatMessage(role="user", content="Test")
        chat_req = backend.ChatRequest(message="Hello", session_id="test-1")
        chat_resp = backend.ChatResponse(reply="Hi", session_id="test-1")
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "Chat Models",
            "status": "PASS",
            "message": "ChatMessage, ChatRequest, and ChatResponse created successfully"
        })
        print("      ✓ PASS\n")
    except Exception as e:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "Chat Models",
            "status": "FAIL",
            "message": str(e)
        })
        print(f"      ✗ FAIL: {e}\n")
    
    # Test 6: FastAPI app
    print("[6/6] Testing FastAPI app initialization...")
    try:
        from fastapi.testclient import TestClient
        client = TestClient(backend.app)
        response = client.get("/api/v1/synqc/health")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok"
        
        results["tests_run"] += 1
        results["tests_passed"] += 1
        results["details"].append({
            "test": "FastAPI Health Check",
            "status": "PASS",
            "message": f"Health endpoint responded with status=ok (v{data.get('version')})"
        })
        print(f"      ✓ PASS (v{data.get('version')})\n")
    except Exception as e:
        results["tests_run"] += 1
        results["tests_failed"] += 1
        results["details"].append({
            "test": "FastAPI Health Check",
            "status": "FAIL",
            "message": str(e)
        })
        print(f"      ✗ FAIL: {e}\n")
    
    # Print summary
    print("="*80)
    print(f" Summary: {results['tests_passed']} PASS, {results['tests_failed']} FAIL")
    print("="*80)
    
    for detail in results["details"]:
        status_icon = "✓" if detail["status"] == "PASS" else "✗"
        print(f"{status_icon} {detail['test']:30} {detail['message']}")
    
    print("="*80 + "\n")
    
    # Write results to file
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to test_results.json\n")
    
    return results


if __name__ == "__main__":
    results = main()
    success = results["tests_failed"] == 0
    sys.exit(0 if success else 1)
