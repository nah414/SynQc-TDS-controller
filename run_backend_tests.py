#!/usr/bin/env python
"""
SynQc Backend Test Runner
=========================
Comprehensive test suite for the backend API and core components.
"""
import sys
import json
from datetime import datetime

def run_tests():
    print("=" * 70)
    print(" SynQc Backend Test Suite")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: Import synqc_tds_super_backend
    print("\n[TEST 1] Import Backend Module")
    print("-" * 70)
    try:
        import synqc_tds_super_backend as backend
        print("✓ PASS: Backend module imported successfully")
        test_results.append(("Backend Import", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("Backend Import", False, str(e)))
        return test_results
    
    # Test 2: Check core components exist
    print("\n[TEST 2] Verify Core Components")
    print("-" * 70)
    components = ['app', 'store', 'engine', 'SynQcEngine', 'ChatRequest', 'ChatResponse', 
                  'RunConfiguration', 'KpiBundle', 'RunRecord']
    all_ok = True
    for comp in components:
        if hasattr(backend, comp):
            print(f"  ✓ {comp}")
        else:
            print(f"  ✗ {comp} - MISSING")
            all_ok = False
    if all_ok:
        test_results.append(("Components Exist", True, None))
    else:
        test_results.append(("Components Exist", False, "Missing components"))
    
    # Test 3: Enum values
    print("\n[TEST 3] Verify Enum Values")
    print("-" * 70)
    try:
        assert backend.HardwareTarget.SIM_LOCAL.value == "sim-local"
        assert backend.RunMode.RUN.value == "run"
        assert backend.SessionStatus.IDLE.value == "idle"
        print("✓ PASS: Enum values correct")
        test_results.append(("Enum Values", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("Enum Values", False, str(e)))
    
    # Test 4: RunConfiguration model validation
    print("\n[TEST 4] RunConfiguration Model")
    print("-" * 70)
    try:
        cfg = backend.RunConfiguration(
            hardware_target=backend.HardwareTarget.SIM_LOCAL,
            hardware_preset=backend.HardwarePreset.TRANSMON_DEFAULT,
            drive_envelope=backend.DriveEnvelope.GAUSSIAN,
            probe_strength=0.25,
            probe_duration_ns=150,
        )
        assert cfg.probe_strength == 0.25
        assert cfg.probe_duration_ns == 150
        print(f"✓ PASS: Created config with strength={cfg.probe_strength}, duration={cfg.probe_duration_ns}ns")
        test_results.append(("RunConfiguration", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("RunConfiguration", False, str(e)))
    
    # Test 5: KpiBundle model
    print("\n[TEST 5] KpiBundle Model")
    print("-" * 70)
    try:
        kpi = backend.KpiBundle.from_raw(
            fidelity=0.985,
            latency_us=20.5,
            backaction=0.12,
            shots_used=2500,
            shot_limit=50000,
        )
        assert kpi.fidelity == 0.985
        assert kpi.shots_used_fraction == 0.05
        print(f"✓ PASS: Created KPI with fidelity={kpi.fidelity}, shots_used_fraction={kpi.shots_used_fraction}")
        test_results.append(("KpiBundle", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("KpiBundle", False, str(e)))
    
    # Test 6: Engine measurement synthesis
    print("\n[TEST 6] Engine Measurement Synthesis")
    print("-" * 70)
    try:
        engine = backend.SynQcEngine()
        measurements = engine._synthesize_measurements(
            backend.RunConfiguration(
                hardware_target=backend.HardwareTarget.SIM_LOCAL,
                hardware_preset=backend.HardwarePreset.TRANSMON_DEFAULT,
                drive_envelope=backend.DriveEnvelope.GAUSSIAN,
            ),
            fidelity=0.95
        )
        assert len(measurements) == 4, f"Expected 4 qubits, got {len(measurements)}"
        assert all(isinstance(m, dict) for m in measurements), "Measurements should be dicts"
        required_keys = {'qubit', 'p0', 'p1', 'last'}
        for m in measurements:
            assert required_keys.issubset(m.keys()), f"Missing keys in {m}"
        print(f"✓ PASS: Generated {len(measurements)} measurements")
        for i, m in enumerate(measurements):
            print(f"  Qubit {m['qubit']}: P0={m['p0']:.4f}, P1={m['p1']:.4f}, Last={m['last']}")
        test_results.append(("Measurement Synthesis", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("Measurement Synthesis", False, str(e)))
    
    # Test 7: Chat models
    print("\n[TEST 7] Chat Models")
    print("-" * 70)
    try:
        chat_msg = backend.ChatMessage(role="user", content="Hello")
        assert chat_msg.role == "user"
        
        chat_req = backend.ChatRequest(
            session_id="test-session-1",
            message="What is the optimal probe strength?",
            history=[chat_msg]
        )
        assert chat_req.session_id == "test-session-1"
        assert len(chat_req.history) == 1
        
        chat_resp = backend.ChatResponse(
            reply="The optimal probe strength is around 0.2.",
            session_id="test-session-1"
        )
        assert "optimal" in chat_resp.reply.lower()
        
        print("✓ PASS: Chat models validated")
        test_results.append(("Chat Models", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("Chat Models", False, str(e)))
    
    # Test 8: FastAPI app initialization
    print("\n[TEST 8] FastAPI App Initialization")
    print("-" * 70)
    try:
        from fastapi.testclient import TestClient
        client = TestClient(backend.app)
        response = client.get("/api/v1/synqc/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data['status'] == 'ok'
        print(f"✓ PASS: Health check returned status=ok")
        print(f"  Version: {data.get('version')}")
        print(f"  API Prefix: {data.get('api_prefix')}")
        test_results.append(("FastAPI Health Check", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("FastAPI Health Check", False, str(e)))
    
    # Test 9: Session CRUD
    print("\n[TEST 9] Session CRUD Operations")
    print("-" * 70)
    try:
        from fastapi.testclient import TestClient
        client = TestClient(backend.app)
        
        # List sessions (should be empty initially)
        resp = client.get("/api/v1/synqc/sessions")
        assert resp.status_code == 200
        sessions = resp.json()
        print(f"✓ List sessions: {len(sessions)} sessions")
        
        # Create a session
        create_payload = {
            "session_id": "test-session-001",
            "config": {
                "hardware_target": "sim-local",
                "hardware_preset": "transmon-default",
                "drive_envelope": "gaussian",
                "probe_strength": 0.2,
                "probe_duration_ns": 120,
                "adaptive_rule": "none",
                "objective": "maximize-fidelity",
            }
        }
        resp = client.post("/api/v1/synqc/sessions", json=create_payload)
        assert resp.status_code == 200, f"Create failed with {resp.status_code}: {resp.text}"
        session = resp.json()
        assert session['session_id'] == "test-session-001"
        print(f"✓ Create session: {session['session_id']}")
        
        # Get the session
        resp = client.get(f"/api/v1/synqc/sessions/{session['session_id']}")
        assert resp.status_code == 200
        retrieved = resp.json()
        assert retrieved['session_id'] == session['session_id']
        print(f"✓ Get session: {retrieved['session_id']} (status={retrieved['status']})")
        
        test_results.append(("Session CRUD", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        test_results.append(("Session CRUD", False, str(e)))
    
    # Test 10: Run endpoint
    print("\n[TEST 10] Run Execution")
    print("-" * 70)
    try:
        from fastapi.testclient import TestClient
        client = TestClient(backend.app)
        
        # First create a session
        create_payload = {
            "config": {
                "hardware_target": "sim-local",
                "hardware_preset": "transmon-default",
                "drive_envelope": "gaussian",
                "probe_strength": 0.2,
                "probe_duration_ns": 120,
                "adaptive_rule": "none",
                "objective": "maximize-fidelity",
            }
        }
        resp = client.post("/api/v1/synqc/sessions", json=create_payload)
        session_id = resp.json()['session_id']
        
        # Launch a run
        run_payload = {"mode": "run"}
        resp = client.post(f"/api/v1/synqc/sessions/{session_id}/run", json=run_payload)
        assert resp.status_code == 200, f"Run failed with {resp.status_code}: {resp.text}"
        result = resp.json()
        
        run = result['run']
        assert 'run_id' in run
        assert 'kpis' in run
        assert 'measurements' in run
        assert len(run['measurements']) > 0
        
        print(f"✓ Run executed: {run['run_id']}")
        print(f"  Fidelity: {run['kpis']['fidelity']:.4f}")
        print(f"  Latency: {run['kpis']['latency_us']:.2f} µs")
        print(f"  Measurements: {len(run['measurements'])} qubits")
        
        test_results.append(("Run Execution", True, None))
    except Exception as e:
        print(f"✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        test_results.append(("Run Execution", False, str(e)))
    
    # Summary
    print("\n" + "=" * 70)
    print(" Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in test_results if success)
    failed = sum(1 for _, success, _ in test_results if not success)
    
    for name, success, error in test_results:
        status = "✓ PASS" if success else f"✗ FAIL"
        print(f"{status:8} {name:30} {error or ''}")
    
    print("=" * 70)
    print(f"Total: {passed} passed, {failed} failed out of {len(test_results)} tests")
    print("=" * 70)
    
    return test_results


if __name__ == "__main__":
    results = run_tests()
    failed = sum(1 for _, success, _ in results if not success)
    sys.exit(0 if failed == 0 else 1)
