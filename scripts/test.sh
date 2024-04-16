#!/bin/bash
# shellcheck disable=SC1091

set -euo pipefail

source vars.env

export ENV=test
export MIN_COVERAGE=${MIN_COVERAGE:-0}

pytest --cov=src --cov-fail-under="${MIN_COVERAGE}" tests/
