#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

# Subset for A2 (15 issues, 1975-1993). Extend this array toward the
# full 176-issue byte-magazine-YYYY-MM set for later milestones.
ISSUES=(
  "byte-magazine-1975-09"
  "byte-magazine-1978-07"
  "byte-magazine-1980-07"
  "byte-magazine-1981-05"
  "byte-magazine-1982-04"
  "byte-magazine-1983-11"
  "byte-magazine-1983-12"
  "byte-magazine-1984-01"
  "byte-magazine-1985-06"
  "byte-magazine-1986-08"
  "byte-magazine-1987-09"
  "byte-magazine-1988-05"
  "byte-magazine-1989-04"
  "byte-magazine-1992-03"
  "byte-magazine-1993-06"
)

echo "Fetching ${#ISSUES[@]} BYTE issues into $RAW_DIR ..."

for id in "${ISSUES[@]}"; do
  echo "-> $id"
  ia download "$id" --glob="*.pdf" --destdir="$RAW_DIR" --no-directories
done

echo "Done. Contents of $RAW_DIR:"
ls -la "$RAW_DIR"
