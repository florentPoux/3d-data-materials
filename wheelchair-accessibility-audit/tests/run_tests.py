# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-automate-3d-deliverables-accessibility-map-from-3d-scan-3ecef96a5a6f
# Dataset:  https://learngeodata.eu/materials/wheelchair-accessibility-audit/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""Run the test suite from a script (python run_tests.py), for the resource gate."""
import sys
from pathlib import Path
import pytest
sys.exit(pytest.main([str(Path(__file__).resolve().parent), "-q", "-p", "no:cacheprovider"]))
