"""
Quick tests for SynQc backend
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime

# Import the app
import sys
from pathlib import Path

# We'll test basic imports and structure
def test_backend_imports():
    """Test that the backend can be imported without errors"""
    try:
        import synqc_tds_super_backend as backend
        assert hasattr(backend, 'app')
        assert hasattr(backend, 'store')
        assert hasattr(backend, 'engine')
        print("✓ All backend imports OK")
    except Exception as e:
        pytest.fail(f"Backend import failed: {e}")


def test_app_models():
    """Test that Pydantic models are correctly defined"""
    try:
        import synqc_tds_super_backend as backend
        
        # Test RunConfiguration model
        cfg = backend.RunConfiguration(
            hardware_target=backend.HardwareTarget.SIM_LOCAL,
            hardware_preset=backend.HardwarePreset.TRANSMON_DEFAULT,
            drive_envelope=backend.DriveEnvelope.GAUSSIAN,
        )
        assert cfg.probe_strength == 0.2
        print("✓ RunConfiguration model OK")
        
        # Test KpiBundle model
        kpi = backend.KpiBundle.from_raw(
            fidelity=0.99,
            latency_us=15.0,
            backaction=0.1,
            shots_used=1000,
            shot_limit=50000,
        )
        assert kpi.fidelity == 0.99
        assert kpi.shots_used_fraction == 0.02
        print("✓ KpiBundle model OK")
        
    except Exception as e:
        pytest.fail(f"Model test failed: {e}")


def test_engine_synthesis():
    """Test that the engine can synthesize measurements"""
    try:
        import synqc_tds_super_backend as backend
        
        engine = backend.SynQcEngine()
        cfg = backend.RunConfiguration(
            hardware_target=backend.HardwareTarget.SIM_LOCAL,
            hardware_preset=backend.HardwarePreset.TRANSMON_DEFAULT,
            drive_envelope=backend.DriveEnvelope.GAUSSIAN,
        )
        
        measurements = engine._synthesize_measurements(cfg, fidelity=0.95)
        assert len(measurements) == 4
        assert all(isinstance(m, dict) for m in measurements)
        assert all('qubit' in m and 'p0' in m and 'p1' in m and 'last' in m for m in measurements)
        print(f"✓ Measurements synthesis OK: {measurements}")
        
    except Exception as e:
        pytest.fail(f"Engine synthesis test failed: {e}")


def test_fast_api_startup():
    """Test that FastAPI app can start"""
    try:
        import synqc_tds_super_backend as backend
        from fastapi.testclient import TestClient
        
        client = TestClient(backend.app)
        response = client.get("/api/v1/synqc/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'
        print(f"✓ FastAPI health check OK: {data}")
        
    except Exception as e:
        pytest.fail(f"FastAPI startup test failed: {e}")


def test_chat_models():
    """Test ChatRequest and ChatResponse models"""
    try:
        import synqc_tds_super_backend as backend
        
        chat_req = backend.ChatRequest(
            session_id="test-123",
            message="Hello"
        )
        assert chat_req.message == "Hello"
        print("✓ ChatRequest model OK")
        
        chat_resp = backend.ChatResponse(
            reply="Hi there",
            session_id="test-123"
        )
        assert chat_resp.reply == "Hi there"
        print("✓ ChatResponse model OK")
        
    except Exception as e:
        pytest.fail(f"Chat models test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
