#!/usr/bin/env bash
# Fetch the Franka Emika Panda model (MuJoCo Menagerie, Apache-2.0) into third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=370c43965323c0944b433b202aac5058b5f907a9
mkdir -p third_party
if [ ! -d third_party/mujoco_menagerie ]; then
  git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
fi
cd third_party/mujoco_menagerie
git sparse-checkout set franka_emika_panda
git checkout -q "$REV"
