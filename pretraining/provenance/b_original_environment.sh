#!/usr/bin/env bash
export STUDY=/n/home10/jaykim/woobin/OST/isolated_experiments/ost_b_pretrain_pair_20260912
export PYTHON=/n/home10/jaykim/.conda/envs/fdloss/bin/python
export PATH=/n/home10/jaykim/.conda/envs/fdloss/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
export HF_HOME=/n/netscratch/sham_lab/Lab/workdir/huggingface
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/n/home10/jaykim/woobin/OST/tests/mmd/runs/torch_cache
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=4
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((23000 + SLURM_JOB_ID % 15000))
run_ranks() {
    srun --kill-on-bad-exit=1 --export=ALL bash -c '
        exec "$PYTHON" -m torch.distributed.run --nnodes=4 --nproc_per_node=2 \
          --node_rank="$SLURM_NODEID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$@"
    ' bash "$@"
}
# wait can return early on SIGUSR1; keep waiting until srun and every rank exit.
wait_for_ranks() {
    local pid=$1 code
    while true; do
        wait "$pid"; code=$?
        if kill -0 "$pid" 2>/dev/null; then continue; fi
        return "$code"
    done
}

export CUBLAS_WORKSPACE_CONFIG=:4096:8 NCCL_ALGO=Ring NCCL_PROTO=Simple
