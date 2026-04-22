#!/bin/bash
# run_all.sh — Execute the full BDA pipeline end-to-end
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source env.sh

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     BDA Project: Flight Delay Prediction Pipeline    ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

run_step() {
    local step="$1"
    local script="$2"
    echo ""
    echo "─────────────────────────────────────────────────────"
    echo "  Running step $step: $script"
    echo "─────────────────────────────────────────────────────"
    python3 "src/$script"
    echo "  ✓ Step $step complete"
}

run_step 1 "01_ingest.py"
run_step 2 "02_preprocess.py"
run_step 3 "03_features.py"
run_step 4 "04_train.py"
run_step 5 "05_evaluate.py"
run_step 6 "06_report.py"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              Pipeline Complete!                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Outputs:"
ls -lh output/*.json output/*.png output/*.csv 2>/dev/null || true
echo ""
echo "Report:"
ls -lh report/*.pdf 2>/dev/null || true
echo ""
