#!/bin/bash
# Print the best PQ and final line of each run's training log.
ROOT=<PROJECT_ROOT>
cd "$ROOT" || exit 1
for r in "$@"; do
  f="runs/$r/train.log"
  [ -f "$f" ] || { echo "$r: no log"; continue; }
  best=$(grep -o "best PQ=[0-9.]*" "$f" 2>/dev/null | tail -n 1)
  done_=$(grep -c "=== done" "$f" 2>/dev/null)
  last=$(grep -E "^ep " "$f" 2>/dev/null | tail -n 1)
  echo "$r | ${best:-no-best} | done=$done_ | ${last:0:52}"
done
