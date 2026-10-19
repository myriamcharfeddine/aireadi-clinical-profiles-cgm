"""Shared fixtures: tests needing the AI-READI data are skipped when AIREADI_DATA_DIR is not set."""
import os

import pytest

needs_data = pytest.mark.skipif(not os.environ.get("AIREADI_DATA_DIR"), reason="AIREADI_DATA_DIR not set (data under the AI-READI licence)")
