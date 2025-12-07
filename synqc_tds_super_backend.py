"""
Compatibility module so existing docs/commands using
`uvicorn synqc_tds_super_backend:app` still work.
"""

from synqc_tds.api import app  # noqa: F401
