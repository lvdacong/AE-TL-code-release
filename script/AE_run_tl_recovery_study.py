"""
AE_run_tl_recovery_study.py
============================
Entry point for TL reconstruction error recovery study.
Focus: DR scenario, difficulty=12, N=400.

Investigates why TL val_loss (0.001245) doesn't recover to pretrain
baseline (0.001148), and systematically tests LR schedule, optimizer,
and training duration fixes.

Usage:
    cd script && python AE_run_tl_recovery_study.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from AE_tl_recovery_study_auxiliary import run_recovery_study

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        choices=["all", "phase1", "phase2", "phase3", "phase3_only",
                                 "phase4", "phase4_only", "phase5", "phase5_only"])
    args = parser.parse_args()
    run_recovery_study(phase=args.phase)
