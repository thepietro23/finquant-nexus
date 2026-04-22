"""pytest configuration — adds project root to sys.path so src.* imports work."""

import sys
import os

# Add repo root (fqn1/) to path so tests can import src.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
