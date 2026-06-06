#!/usr/bin/env bash
# Wrapper to run any Python script with SOFA available.
# Usage: ./sofa/run_sofa.sh sofa/simulate_cell.py
#        ./sofa/run_sofa.sh sofa/simulate_cell.py arg1 arg2 ...
#
# Uses Homebrew Python 3.12 (the only Python that can load the SOFA macOS binary).
# The kgnn_mac JAX pipeline calls this as a subprocess to get simulation results.

export SOFA_ROOT="$HOME/sofa/v25.12/SOFA_v25.12.00_MacOS"
export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH"
export DYLD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$DYLD_LIBRARY_PATH"

exec /opt/homebrew/bin/python3.12 "$@"
