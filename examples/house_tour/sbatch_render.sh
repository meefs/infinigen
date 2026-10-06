#!/bin/bash
#SBATCH --job-name=house_tour
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --no-requeue

# usage: SEEDS="608 610 ..." sbatch --array=1-$((N_SEEDS*N_SHARDS)) examples/house_tour/sbatch_render.sh
# renders frame slices of the seed####/scene.blend files written by sbatch_build.sh

[ -f /n/fs/pvl-renders/${USER}/uv/env ] && source /n/fs/pvl-renders/${USER}/uv/env

: "${SLURM_ARRAY_TASK_ID:?must be set (slurm env, or e.g. SLURM_ARRAY_TASK_ID=1)}"
: "${SEEDS:?set SEEDS to the space-separated seeds that sbatch_build.sh built}"
read -r -a SEED_LIST <<< "${SEEDS}"
N_FRAMES="${N_FRAMES:-288}"
N_SHARDS="${N_SHARDS:-24}"
RUN="${RUN:-house_tour}"
OUTBASE="${OUTBASE:-/n/fs/scratch/${USER}/house_tour}"
RENDER_ARGS="${RENDER_ARGS:---resolution 1920 1080 --samples 1024}"

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SHARD=$((IDX % N_SHARDS))
SEED="${SEED_LIST[$((IDX / N_SHARDS))]}"
PER_SHARD=$(((N_FRAMES + N_SHARDS - 1) / N_SHARDS))
START=$((SHARD * PER_SHARD))
END=$((START + PER_SHARD - 1))
[ "${END}" -lt "${N_FRAMES}" ] || END=$((N_FRAMES - 1))

printf -v SEED_ID "%04d" "${SEED}"
printf -v SHARD_ID "%02d" "${SHARD}"
SCENE_DIR="${OUTBASE}/${RUN}/seed${SEED_ID}"
OUTDIR="${SCENE_DIR}/shard${SHARD_ID}"
JOBNAME="${SLURM_JOB_NAME:-house_tour}_${SLURM_ARRAY_JOB_ID:-local}"
LOG_BASE="outputs/renderjobs"
mkdir -p "${LOG_BASE}" "${OUTDIR}"
START_TIME=$(date -Iseconds)

emit_state() {
    echo "${START_TIME} $(date -Iseconds) ${SLURM_NODELIST:-none} ${CUDA_VISIBLE_DEVICES:-none} ${OUTDIR} house seed${SEED_ID} $1" >> "${LOG_BASE}/${JOBNAME}_state.log"
}
trap 'emit_state killed; exit 1' TERM

[ -f "${SCENE_DIR}/scene.blend" ] || { echo "ERROR: no ${SCENE_DIR}/scene.blend; run sbatch_build.sh first"; emit_state "missing_blend"; exit 1; }
if [ -f "${OUTDIR}/metadata.json" ]; then
    echo "already done: ${OUTDIR}"
    emit_state "skipped"
    exit 0
fi

echo "SEED=${SEED} frames ${START}-${END} of 0-$((N_FRAMES - 1)) -> ${OUTDIR}"
if ! uv run --no-sync python examples/house_tour/render.py \
    --seed "${SEED}" --load_blend "${SCENE_DIR}/scene.blend" \
    --frames 0 $((N_FRAMES - 1)) --render_frames "${START}" "${END}" \
    --output "${OUTDIR}" ${RENDER_ARGS}; then
    echo "ERROR: render failed for ${OUTDIR}"
    emit_state "crashed"
    exit 1
fi
[ -f "${OUTDIR}/metadata.json" ] || { echo "ERROR: no ${OUTDIR}/metadata.json"; emit_state "crashed"; exit 1; }
trap - TERM
emit_state "completed"
