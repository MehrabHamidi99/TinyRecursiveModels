#!/bin/bash
#SBATCH --job-name=pretrain_mlp_t_sudoku_lqr
# SBATCH --account=def-ioannism
#SBATCH --time=3:59:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:4


module load httpproxy


cd /home/m/mehrab/projects/TinyRecursiveModels

# run_name="pretrain_mlp_t_sudoku_lqr"

# uv run torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain_lqr.py arch=trm \
# data_paths="[data/sudoku-extreme-1k-aug-1000]" \
# evaluators="[]" \
# epochs=50000 eval_interval=5000 \
# lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
# arch.mlp_t=True arch.pos_encodings=none \
# arch.L_layers=2 \
# arch.H_cycles=3 arch.L_cycles=6 \
# +run_name=${run_name} ema=True use_lqr=True

run_name="pretrain_mlp_t_sudoku_muon"


