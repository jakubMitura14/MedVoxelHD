#!/usr/bin/env bash


cd /home/sliceruser/code &&\
python3 benchmarking.py


# set -ex

# script_dir=$(cd $(dirname $0) || exit 1; pwd)

# ################################################################################
# # Set up headless environment
# source $script_dir/start-xorg.sh



export SHELL=/bin/bash



exec "$@"