#!/usr/bin/env python3
"""
Root wrapper for scripts/validate_submission.py
"""
import sys
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
scripts_dir = os.path.join(this_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from validate_submission import main, validate_submission

if __name__ == "__main__":
    main()
