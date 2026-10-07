#!/bin/bash
#SBATCH --job-name=clay_orbit
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=outputs/renderjobs/%x_%A_%a.out
#SBATCH --error=outputs/renderjobs/%x_%A_%a.err
#SBATCH --no-requeue

# usage: SCENETYPE=bedroom SEED_BASE=200 sbatch --array=1-$((N_SCENES*N_SHARDS)) sbatch.sh

[ -f /n/fs/pvl-renders/${USER}/uv/env ] && source /n/fs/pvl-renders/${USER}/uv/env

: "${SCENETYPE:?set SCENETYPE (livingroom diningroom bedroom bathroom kitchen indoor_space)}"
: "${SEED_BASE:?set SEED_BASE, the first seed of a disjoint 100-seed block for this scenetype}"
: "${SLURM_ARRAY_TASK_ID:?must be set (slurm env, or e.g. SLURM_ARRAY_TASK_ID=1)}"
N_FRAMES="${N_FRAMES:-72}"
N_SHARDS="${N_SHARDS:-12}"
SCENE_OFFSET="${SCENE_OFFSET:-0}"
SEED_SALT="${SEED_SALT:-0}"
RUN="${RUN:-clay_orbit}"
OUTBASE="${OUTBASE:-/n/fs/scratch/${USER}/clay_orbit}"

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SHARD=$((IDX % N_SHARDS))
SCENE_IDX=$((IDX / N_SHARDS + SCENE_OFFSET))
[ "${SCENE_IDX}" -lt 100 ] || { echo "ERROR: scene ${SCENE_IDX} overflows the ${SCENETYPE} seed block"; exit 1; }
SEED=$((SEED_SALT + SEED_BASE + SCENE_IDX))
PER_SHARD=$(((N_FRAMES + N_SHARDS - 1) / N_SHARDS))
START=$((SHARD * PER_SHARD))
END=$((START + PER_SHARD - 1))
[ "${END}" -lt "${N_FRAMES}" ] || END=$((N_FRAMES - 1))

printf -v SEED_ID "%04d" "${SEED}"
printf -v SHARD_ID "%02d" "${SHARD}"
OUTDIR="${OUTBASE}/${RUN}/${SCENETYPE}/seed${SEED_ID}/shard${SHARD_ID}"
JOBNAME="${SLURM_JOB_NAME:-clay_orbit}_${SLURM_ARRAY_JOB_ID:-local}"
LOG_BASE="outputs/renderjobs"
mkdir -p "${LOG_BASE}" "${OUTDIR}"
START_TIME=$(date -Iseconds)

emit_state() {
    echo "${START_TIME} $(date -Iseconds) ${SLURM_NODELIST:-none} ${CUDA_VISIBLE_DEVICES:-none} ${OUTDIR} ${SCENETYPE} seed${SEED_ID} $1" >> "${LOG_BASE}/${JOBNAME}_state.log"
}
trap 'emit_state killed; exit 1' TERM

if [ -f "${OUTDIR}/metadata.json" ]; then
    echo "already done: ${OUTDIR}"
    emit_state "skipped"
    exit 0
fi

echo "SCENETYPE=${SCENETYPE} SEED=${SEED} frames ${START}-${END} of 0-$((N_FRAMES - 1)) -> ${OUTDIR}"
if ! uv run --no-sync python examples/clay_orbit/render.py \
    --scene_type "${SCENETYPE}" --seed "${SEED}" \
    --frames 0 $((N_FRAMES - 1)) --render_frames "${START}" "${END}" \
    --output "${OUTDIR}" --skip_gt ${RENDER_ARGS:---rgb_samples 2048}; then
    echo "ERROR: render failed for ${OUTDIR}"
    emit_state "crashed"
    exit 1
fi
[ -f "${OUTDIR}/metadata.json" ] || { echo "ERROR: no ${OUTDIR}/metadata.json"; emit_state "crashed"; exit 1; }
trap - TERM
emit_state "completed"
