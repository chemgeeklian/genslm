#!/bin/bash
set -euo pipefail

# conda activate /lus/eagle/projects/FoundEpidem/xlian/conda/envs/genslm_agent

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTA_DIR="${SCRIPT_DIR}/tpc26data"

for fasta in "${FASTA_DIR}"/*.fasta; do
    echo "Running: $(basename "${fasta}")"
    python "${SCRIPT_DIR}/generate_from_prompt.py" \
        --fasta      "${fasta}" \
        --n          90 \
        --num_seqs   10 \
        --max_length 700
done