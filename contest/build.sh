#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker buildx build --network=none --platform linux/amd64 --load \
  -t ftsaki-contest:model0-1118 -f contest/Dockerfile \
  --build-context baseline_a=baselines/A \
  --build-context baseline_b=baselines/B \
  --build-context baseline_c=baselines/C .
