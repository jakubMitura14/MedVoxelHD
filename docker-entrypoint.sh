#!/usr/bin/env bash



python3 /workspaces/Hausdorff_morphological/benchmarking.py


# set -ex

# script_dir=$(cd $(dirname $0) || exit 1; pwd)

# ################################################################################
# # Set up headless environment
# source $script_dir/start-xorg.sh



export SHELL=/bin/bash



exec "$@"