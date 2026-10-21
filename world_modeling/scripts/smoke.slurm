#!/bin/bash
#SBATCH --job-name=priming-world-smoke
#SBATCH --account=kempner_sitanc_lab
#SBATCH --partition=kempner_h100_priority
#SBATCH --qos=kemp_gpu32_id50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=01:00:00
set -euo pipefail
: "${WORLD_MODEL_PYTHON:?Set WORLD_MODEL_PYTHON to the environment python}"
: "${WORLD_MODEL_STORAGE:?Set WORLD_MODEL_STORAGE to the read-only asset root}"
: "${DISTMATCH_SOURCE:?Set DISTMATCH_SOURCE to the recovered package root or src directory}"
: "${WORLD_MODEL_RUN:?Set WORLD_MODEL_RUN to a new output directory}"
: "${WORLD_MODEL_REFERENCE:?Set WORLD_MODEL_REFERENCE to the extracted reference root}"
cd "${SLURM_SUBMIT_DIR}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
mkdir -p "$WORLD_MODEL_RUN"
if [[ ! -f "$WORLD_MODEL_RUN/source-parity.json" ]]; then
    "$WORLD_MODEL_PYTHON" -m world_modeling.scripts.validate_source --reference "$WORLD_MODEL_REFERENCE" --storage-root "$WORLD_MODEL_STORAGE" --output "$WORLD_MODEL_RUN/source-parity.json"
fi
if [[ ! -f "$WORLD_MODEL_RUN/feature-parity.json" ]]; then
    "$WORLD_MODEL_PYTHON" -m world_modeling.scripts.validate_features --reference "$WORLD_MODEL_REFERENCE" --storage-root "$WORLD_MODEL_STORAGE" --output "$WORLD_MODEL_RUN/feature-parity.json"
fi
run_stage() {
    "$WORLD_MODEL_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 -m "world_modeling.$1" "${@:2}"
}
for dataset in rt1 bridge; do
    pre="$WORLD_MODEL_RUN/$dataset/pretrain"
    post="$WORLD_MODEL_RUN/$dataset/posttrain"
    if [[ ! -f "$pre/ckpt_save/ckpt-step3.pth" ]]; then
        if [[ ! -f "$pre/ckpt_save/ckpt-step2.pth" ]]; then
            run_stage pretrain --config "world_modeling/configs/pretrain/$dataset.yaml" --output "$pre" --max-steps 2 --set dataloader.num_workers=2
        fi
        run_stage pretrain --config "world_modeling/configs/pretrain/$dataset.yaml" --output "$pre" --resume "$pre/ckpt_save/ckpt-step2.pth" --max-steps 3 --set dataloader.num_workers=2
    fi
    if [[ ! -f "$post/ckpt_save/ckpt-step3.pth" ]]; then
        if [[ ! -f "$post/ckpt_save/ckpt-step2.pth" ]]; then
            run_stage posttrain --config "world_modeling/configs/posttrain/$dataset.yaml" --output "$post" --init-checkpoint "$pre/ckpt_save/ckpt-step3.pth" --max-steps 2 --set dataloader.num_workers=2 --set distmatch.verify_replay=true
        fi
        run_stage posttrain --config "world_modeling/configs/posttrain/$dataset.yaml" --output "$post" --resume "$post/ckpt_save/ckpt-step2.pth" --max-steps 3 --set dataloader.num_workers=2 --set distmatch.verify_replay=true
    fi
    run_stage eval --config "world_modeling/configs/eval/$dataset.yaml" --output "$WORLD_MODEL_RUN/$dataset/eval" --checkpoint "$post/ckpt_save/ckpt-step3.pth" --max-videos-per-shard 2 --set dataloader.num_workers=2 --set evaluation.save_videos=1
 done
