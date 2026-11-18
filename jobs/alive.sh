#!/bin/bash
# Check whether a run is actually progressing: log size, last epoch, process age.
ROOT=<PROJECT_ROOT>
cd "$ROOT" || exit 1
for r in "$@"; do
  echo "=== $r"
  f="runs/$r/train.log"
  if [ -f "$f" ]; then
    echo "  log bytes: $(wc -c < "$f")"
    grep -E "^ep |=== res start|params" "$f" 2>/dev/null | tail -n 3
  else
    echo "  no train.log yet"
  fi
  n=$(pgrep -f "out runs/$r" 2>/dev/null | wc -l)
  echo "  processes: $n"
done
