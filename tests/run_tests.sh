#!/bin/bash
#SBATCH --job-name=mood-bench-tests
#SBATCH --partition=main
#SBATCH --qos=high
#SBATCH --gres=gpu:A100-SXM4-80GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=tests/slurm-%j.out
#SBATCH --error=tests/slurm-%j.err

echo "Node: $(hostname)"
nvidia-smi -L | head -5
echo "---"

cd /nas/ucb/dfeng/code/mood-bench

uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo "--- torch check done ---"

uv run pytest tests/ -v --tb=short -x
PYTEST_EXIT=$?
echo "--- pytest done, exit=$PYTEST_EXIT ---"
exit $PYTEST_EXIT
