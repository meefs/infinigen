#!/bin/bash
#SBATCH --job-name=house_tour_build
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --no-requeue

# usage: SCENE_OFFSET=600 sbatch --array=1-N examples/house_tour/sbatch_build.sh
# CPU only: plans each tour once into seed####/scene.blend; rejected seeds write no blend

[ -f /n/fs/pvl-renders/${USER}/uv/env ] && source /n/fs/pvl-renders/${USER}/uv/env

: "${SLURM_ARRAY_TASK_ID:?must be set (slurm env, or e.g. SLURM_ARRAY_TASK_ID=1)}"
N_FRAMES="${N_FRAMES:-288}"
SCENE_OFFSET="${SCENE_OFFSET:-0}"
RUN="${RUN:-house_tour}"
OUTBASE="${OUTBASE:-/n/fs/scratch/${USER}/house_tour}"

SEED=$((SLURM_ARRAY_TASK_ID - 1 + SCENE_OFFSET))
printf -v SEED_ID "%04d" "${SEED}"
SCENE_DIR="${OUTBASE}/${RUN}/seed${SEED_ID}"
mkdir -p outputs/renderjobs "${SCENE_DIR}"

if [ -f "${SCENE_DIR}/scene.blend" ]; then
    echo "already built: ${SCENE_DIR}/scene.blend"
    exit 0
fi

echo "SEED=${SEED} frames 0-$((N_FRAMES - 1)) -> ${SCENE_DIR}/scene.blend"
uv run --no-sync python examples/house_tour/render.py \
    --seed "${SEED}" --frames 0 $((N_FRAMES - 1)) \
    --output "${SCENE_DIR}" --save_blend "${SCENE_DIR}/scene.blend" ${BUILD_ARGS}
