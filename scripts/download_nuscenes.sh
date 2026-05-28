#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
nuScenes requires authenticated downloads, so this script is a checklist.

1. Download these files from https://www.nuscenes.org/download:
   - v1.0-mini.tgz
   - nuScenes-lidarseg-mini-v1.0.tar.bz2

2. Place both archives in data/archives/.

3. Extract into data/nuscenes:
   mkdir -p data/nuscenes
   tar -xzf data/archives/v1.0-mini.tgz -C data/nuscenes
   tar -xjf data/archives/nuScenes-lidarseg-mini-v1.0.tar.bz2 -C data/nuscenes

4. Verify:
   python scripts/preprocess.py --dataroot data/nuscenes --split mini_val --summary-only
EOF
