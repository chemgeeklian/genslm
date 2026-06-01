#!/usr/bin/env python
# coding: utf-8
"""
Generate sequences from a DNA prompt using GenSLM and compare to the original.

Usage:
    python generate_from_prompt.py --fasta <path_to_fasta> --n <prompt_length_nt> [options]

Example:
python generate_from_prompt.py --fasta tpc26data/1ESF.fasta --n 90 --num_seqs 10 --max_length 700

Output: a CSV file saved next to the input FASTA (e.g. 1ESF_results.csv).
"""

import argparse
import csv
import os
import torch
from genslm import GenSLM
from Bio import SeqIO
from Bio.Seq import Seq
from difflib import SequenceMatcher


# ── Similarity metrics ────────────────────────────────────────────────────────

def nucleotide_identity(seq1: str, seq2: str) -> float:
    length = min(len(seq1), len(seq2))
    if length == 0:
        return 0.0
    return sum(a == b for a, b in zip(seq1[:length], seq2[:length])) / length


def sequence_similarity(seq1: str, seq2: str) -> float:
    return SequenceMatcher(None, seq1, seq2).ratio()


def protein_identity(prot1: str, prot2: str) -> float:
    length = min(len(prot1), len(prot2))
    if length == 0:
        return 0.0
    return sum(a == b for a, b in zip(prot1[:length], prot2[:length])) / length


# ── Translation helper ────────────────────────────────────────────────────────

def translate(dna: str) -> str:
    return str(Seq(dna.replace(" ", "").upper()).translate(to_stop=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GenSLM sequence generation + similarity scoring")
    parser.add_argument("--fasta",       required=True,  help="Path to FASTA file (first sequence is used)")
    parser.add_argument("--n",           required=True,  type=int, help="First N nucleotides to use as prompt")
    parser.add_argument("--model",       default="genslm_2.5B_patric", help="GenSLM model name")
    parser.add_argument("--model_dir",   default="/lus/grand/projects/FRAME-IDP/xlian/checkpoints/genslm_models/2.5B",
                        help="Path to model weights cache")
    parser.add_argument("--num_seqs",    default=4,   type=int,   help="Number of sequences to generate")
    parser.add_argument("--max_length",  default=400, type=int,   help="Max generation length (in codons)")
    parser.add_argument("--min_length",  default=330, type=int,   help="Min generation length (in codons)")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature")
    parser.add_argument("--top_k",       default=50,  type=int)
    parser.add_argument("--top_p",       default=0.95,type=float)
    args = parser.parse_args()

    # ── Validate & prepare sequence ───────────────────────────────────────────
    record = next(SeqIO.parse(args.fasta, "fasta"))
    seq_id   = record.id
    full_dna = str(record.seq).upper().replace(" ", "")
    if len(full_dna) % 3 != 0:
        raise ValueError(f"Sequence length ({len(full_dna)}) is not divisible by 3")

    n = args.n
    if n > len(full_dna):
        raise ValueError(f"--n ({n}) exceeds sequence length ({len(full_dna)})")
    if n % 3 != 0:
        n = (n // 3) * 3

    prompt_dna        = full_dna[:n]
    prompt_codons     = [prompt_dna[i:i+3] for i in range(0, len(prompt_dna), 3)]
    prompt_len_codons = len(prompt_codons)

    # ── Load model ────────────────────────────────────────────────────────────
    model = GenSLM(args.model, model_cache_dir=args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ── Encode prompt ─────────────────────────────────────────────────────────
    prompt_tokens = model.tokenizer.encode(
        ' '.join(prompt_codons), return_tensors="pt"
    ).to(device)

    # ── Generate ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        tokens = model.model.generate(
            prompt_tokens,
            max_length           = args.max_length - prompt_len_codons,
            min_length           = args.min_length - prompt_len_codons,
            do_sample            = True,
            top_k                = args.top_k,
            top_p                = args.top_p,
            num_return_sequences = args.num_seqs,
            remove_invalid_values= True,
            use_cache            = True,
            pad_token_id         = model.tokenizer.encode("[PAD]")[0],
            temperature          = args.temperature,
        )

    generated_codon_seqs = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    generated_dnas       = [s.replace(" ", "").upper() for s in generated_codon_seqs]

    # ── Reference translations ────────────────────────────────────────────────
    original_protein = translate(full_dna)
    prompt_protein   = translate(prompt_dna)

    # ── Collect rows ──────────────────────────────────────────────────────────
    rows = []
    for i, gen_dna in enumerate(generated_dnas):
        gen_protein = translate(gen_dna)
        rows.append({
            "seq_id":               seq_id,
            "gen_idx":              i + 1,
            "prompt_nt":            n,
            "original_dna_len":     len(full_dna),
            "original_protein":     original_protein,
            "prompt_protein":       prompt_protein,
            "gen_dna":              gen_dna,
            "gen_dna_len":          len(gen_dna),
            "gen_protein":          gen_protein,
            "gen_protein_len":      len(gen_protein),
            "nt_identity":          round(nucleotide_identity(full_dna, gen_dna), 4),
            "nt_similarity":        round(sequence_similarity(full_dna, gen_dna), 4),
            "aa_identity":          round(protein_identity(original_protein, gen_protein), 4),
            "aa_similarity":        round(sequence_similarity(original_protein, gen_protein), 4),
        })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    fasta_stem = os.path.splitext(os.path.basename(args.fasta))[0]
    csv_path   = os.path.join(os.path.dirname(os.path.abspath(args.fasta)),
                              f"{fasta_stem}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
