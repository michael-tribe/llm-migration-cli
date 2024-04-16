#!/bin/bash
# shellcheck disable=SC1091,SC2086,SC2128

if [ "$0" = "${BASH_SOURCE}" ]; then
  echo "This script must be sourced, not executed: 'source scripts/activate.sh' or '. scripts/activate.sh'"
else
  CONDA_ENV_NAME=$(head -n 1 environment.yaml | cut -d' ' -f2)
  echo "Activating conda environment ${CONDA_ENV_NAME}"
  source "$(conda info -q --base)/etc/profile.d/conda.sh" || echo "Conda not found"
  conda activate ${CONDA_ENV_NAME} || echo "Conda environment ${CONDA_ENV_NAME} not found"
fi
