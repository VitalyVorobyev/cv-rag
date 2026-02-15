#!/usr/bin/env bash
set -euo pipefail

GROBID_DIR="${GROBID_DIR:-$HOME/grobid}"
CVRAG_DIR="${CVRAG_DIR:-$HOME/cv-rag}"

if [[ ! -d "$GROBID_DIR" ]]; then
  echo "GROBID directory not found: $GROBID_DIR" >&2
  exit 1
fi

if [[ ! -f "$GROBID_DIR/gradlew" ]]; then
  echo "gradlew not found in: $GROBID_DIR" >&2
  exit 1
fi

cd "$GROBID_DIR"
./gradlew run
cd "$CVRAG_DIR"
