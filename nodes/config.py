"""
Centralized endpoint configuration for AnimoFlow nodes.

Reads from environment variables. Defaults match the ports in .env / docker-compose.yml.
To override, set the env var before launching ComfyUI (start.sh does this automatically).
"""
import os

MDM_ENDPOINT         = os.environ.get("MDM_ENDPOINT",         "http://localhost:8001")
PRIORMDM_ENDPOINT    = os.environ.get("PRIORMDM_ENDPOINT",    "http://localhost:8002")
MOMASK_ENDPOINT      = os.environ.get("MOMASK_ENDPOINT",      "http://localhost:8003")
KIMODO_ENDPOINT      = os.environ.get("KIMODO_ENDPOINT",      "http://localhost:8005")
