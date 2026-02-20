#!/usr/bin/env python3
"""Uvicorn server startup script for Quila API"""

import uvicorn
import sys
import os

# Add bindings to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bindings'))

if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
