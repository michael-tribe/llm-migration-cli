#!/bin/bash
# shellcheck disable=SC1091

set -eo pipefail

source scripts/check_cwd.sh

if ! command -v conda &> /dev/null; then
  echo "conda could not be found, please install it first."
  exit
fi

CONDA_ENV_NAME=$(head -n 1 environment.yaml | cut -d' ' -f2)
if conda env list | grep -q "${CONDA_ENV_NAME}"; then
  read -p "Conda environment ${CONDA_ENV_NAME} already exists. Do you want to delete it? [y/N] " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting conda environment..."
    conda env remove -n "${CONDA_ENV_NAME}"
  else
    echo "Aborting setup."
    exit
  fi
fi

echo "Creating conda environment..."
conda env create -f environment.yaml

echo "Activating conda environment..."
source "$(conda info -q --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

echo "Installing pip dependencies..."
pip install -r requirements.txt
