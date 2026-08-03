#!/usr/bin/env python3
"""
extract_genomic_seqs_local.py
Extract genomic sequences (gene body ±10kb) from local genome FASTA + GTF.
No internet required. Run as SLURM job after download_genome_gtf.py.

Usage:
    .venv/bin/python3 scripts/extract_genomic_seqs_local.py
    .venv/bin/python3 scripts/extract_genomic_seqs_local.py --species homo_sapiens
"""
import argparse, gzip, os, re, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure pyfaidx available
try:
    import pyfaidx
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyfaidx"], check=True)
    import pyfaidx

BASE       = Path("/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT")
GENOME_DIR = BASE / "data/genomes"
ESM3_DIR   = BASE / "data/main/gene_embs"
OUT_DIR    = BASE / "data/main/genomic_seqs"
FLANK      = 10_000
MAX_SEQ    = 131_072

SPECIES = [
    "homo_sapiens", "mus_musculus", "arabidopsis_thaliana", "bos_taurus",
    "caenorhabditis_elegans", "callithrix_jacchus", "danio_rerio",
    "drosophila_melanogaster", "gallus_gallus", "heterocephalus_glaber_male",
    "macaca_mulatta", "oryctolagus_cuniculus", "ovis_aries", "pan_troglodytes",
    "sus_scrofa", "zea_mays",
]


def find_file(sp, ftype):
    cap = sp.split("_")[0].capitalize()
    suffix = {"genome": ".fa.gz", "gtf": ".gtf.gz"}[ftype]
    candidates = [f for f in GENOME_DIR.glob(f"*.{suffix.lstrip('.')}") 
                  if f.name.startswith(cap)]
    if not candidates:
        # fallback: any file containing species prefix
        candidates = [f for f in GENOME_DIR.iterdir()
                      if f.suffix in (".gz",) and cap.lower() in f.name.lower()
                      and (("dna" in f.name) if ftype == "genome" else ("gtf" in f.name))]
    return candidates[0] if candidates else None


def parse_gtf(gtf_path, gene_ids):
    """Parse GTF → {gene_id: (chrom, start, end, strand)} for requested gene IDs."""
    coords = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#") or "\tgene\t" not in line:
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, start, end, strand = fields[0], int(fields[3])-1, int(fields[4]), fields[6]
            attrs = fields[8]
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if not m:
                continue
            gid = m.group(1).split(".")[0]  # strip version
            if gid in gene_ids:
                coords[gid] = (chrom, start, end, strand)
    return coords


def build_fai(fa_gz_path):
    """Decompress genome to TMPDIR and build pyfaidx index. Returns Fasta object."""
    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    fa_out = tmpdir / fa_gz_path.stem  # removes .gz
    if not fa_out.exists():
        print(f"  Decompressing {fa_gz_path.name}...", flush=True)
        subprocess.run(["gunzip", "-c", str(fa_gz_path)],
                       stdout=open(fa_out, "wb"), check=True)
    print(f"  Building pyfaidx index...", flush=True)
    fa = pyfaidx.Fasta(str(fa_out), rebuild=True, key_function=lambda x: x.split()[0])
    return fa


def extract_seq(fa, chrom, start, end, strand):
    """Extract sequence using pyfaidx. Returns sequence string."""
    try:
        # pyfaidx uses 0-based half-open intervals
        seq = str(fa[chrom][start:end]).upper()
    except (KeyError, ValueError):
        return None
    if not seq:
        return None
    if strand == "-":
        seq = seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    return seq


def process_species(sp):
    out_path = OUT_DIR / f"{sp.split('_')[0]}_emb.parquet"
    # reuse naming convention: short_emb.parquet like ESM3
    out_path = OUT_DIR / f"{sp}.parquet"
    if out_path.exists():
        print(f"  [SKIP] {sp}")
        return

    # Load gene IDs from ESM3 parquets
    short = "mouse" if sp == "mus_musculus" else sp.split("_")[0]
    pq = list(ESM3_DIR.glob(f"{short}_emb.parquet"))
    if not pq:
        print(f"  [SKIP] {sp} — no ESM3 parquet", file=sys.stderr)
        return
    gene_ids = set(pd.read_parquet(pq[0]).index.tolist())
    print(f"  {sp}: {len(gene_ids)} gene IDs", flush=True)

    # Find files
    fa_gz = find_file(sp, "genome")
    gtf   = find_file(sp, "gtf")
    if not fa_gz or not gtf:
        print(f"  ERROR: missing genome/GTF for {sp}", file=sys.stderr)
        return

    # Parse GTF
    print(f"  Parsing GTF {gtf.name}...", flush=True)
    coords = parse_gtf(gtf, gene_ids)
    print(f"  Found coords for {len(coords)}/{len(gene_ids)} genes", flush=True)

    # Decompress genome + build pyfaidx index (once per species)
    fa = build_fai(fa_gz)

    # Extract sequences
    records = []
    for i, (gid, (chrom, start, end, strand)) in enumerate(coords.items()):
        gs  = max(0, start - FLANK)
        ge  = end + FLANK
        seq = extract_seq(fa, chrom, gs, ge, strand)
        if seq is None:
            continue
        gene_start_in_seq = FLANK
        gene_end_in_seq   = FLANK + (end - start)
        # Truncate if too long
        if len(seq) > MAX_SEQ:
            mid  = (gene_start_in_seq + gene_end_in_seq) // 2
            off  = max(0, mid - MAX_SEQ // 2)
            seq  = seq[off: off + MAX_SEQ]
            gene_start_in_seq = max(0, gene_start_in_seq - off)
            gene_end_in_seq   = min(len(seq), gene_end_in_seq - off)
        records.append({"gene_id": gid, "sequence": seq,
                        "gene_start": gene_start_in_seq,
                        "gene_end":   gene_end_in_seq})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(coords)} sequences extracted", flush=True)

    if not records:
        print(f"  ERROR: no sequences for {sp}", file=sys.stderr)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records).set_index("gene_id")
    df.to_parquet(out_path)
    print(f"  ✓ {len(df)} sequences → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="*", default=None)
    args = parser.parse_args()
    slist = args.species or SPECIES

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in slist:
        print(f"\n{'='*50}\n{sp}\n{'='*50}", flush=True)
        try:
            process_species(sp)
        except Exception as e:
            print(f"  ERROR {sp}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
    print("\nAll done.")


if __name__ == "__main__":
    main()
