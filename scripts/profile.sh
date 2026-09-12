#!/usr/bin/env bash
# Usage: scripts/profile.sh [env assignments...]   e.g. scripts/profile.sh N=200 MS=5000
# Builds the bench for node, records a CPU profile, and prints the hottest functions
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(mktemp -d)
bun build scripts/bench.ts --target=node --outfile "$out/bench.js" >/dev/null
env "$@" node --cpu-prof --cpu-prof-dir="$out" "$out/bench.js"
node scripts/summarizeProfile.mjs "$out"/*.cpuprofile "${TOP:-25}"
