#!/usr/bin/env bash
# Trains Akira's three Core brains one after another. Safe to rerun: each run resumes.
set -e
cd "$(dirname "$0")/.."
python3 -W ignore -m akira.native.train --brain core-standard --minutes "${STANDARD_MIN:-45}"
python3 -W ignore -m akira.native.train --brain core-pro --minutes "${PRO_MIN:-90}"
python3 -W ignore -m akira.native.train --brain core-ultra --minutes "${ULTRA_MIN:-180}"
