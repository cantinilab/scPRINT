#!/usr/bin/env python3
# Override TORCH_HOME *before* any torch/esm imports so models load from GPFS
import os
os.environ["TORCH_HOME"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.cache/torch"
os.environ["HF_HOME"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub"

"""
generate_all_embeddings.py
Generate gene embeddings for all scPRINT species using ESM2 or GENA-LM.

Usage:
    python scripts/generate_all_embeddings.py --embedder esm2
    python scripts/generate_all_embeddings.py --embedder gena_lm

Run from the scPRINT project root. Output goes to:
    data/main/gene_embs_esm2/   (ESM2)
    data/main/gene_embs_gena_lm/ (GENA-LM)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import AdaptiveAvgPool1d

# ---------------------------------------------------------------------------
# Species list — mirrors gene_embs/ existing ESM3 parquets
# Key: Ensembl organism name,  Value: NCBITaxon ID (for scdataloader.load_genes)
# ---------------------------------------------------------------------------
SPECIES = {
    "homo_sapiens":            "NCBITaxon:9606",
    "mus_musculus":            "NCBITaxon:10090",
    "arabidopsis_thaliana":    "NCBITaxon:3702",
    "bos_taurus":              "NCBITaxon:9913",
    "caenorhabditis_elegans":  "NCBITaxon:6239",
    "callithrix_jacchus":      "NCBITaxon:9483",
    "danio_rerio":             "NCBITaxon:7955",
    "drosophila_melanogaster": "NCBITaxon:7227",
    "gallus_gallus":           "NCBITaxon:9031",
    "heterocephalus_glaber_male": "NCBITaxon:10181",
    "macaca_mulatta":          "NCBITaxon:9544",
    "oryctolagus_cuniculus":   "NCBITaxon:9986",
    "ovis_aries":              "NCBITaxon:9940",
    "pan_troglodytes":         "NCBITaxon:9598",
    "sus_scrofa":              "NCBITaxon:9823",
    "zea_mays":                "NCBITaxon:4577",
}

# Fasta cache on GPFS — pre-downloaded by scripts/download_fastas.py on login node
# Compute nodes have no internet access!
FASTA_PATH = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/fasta/"

# Output sizes
EMBEDDING_SIZE = {
    "esm2":    1152,
    "gena_lm": 1024,
}


FASTA_SUFFIX = {
    "pep":  ".pep.all.fa.gz",
    "cdna": ".cdna.all.fa.gz",
}


def find_fasta(species: str, fasta_type: str) -> Path:
    """
    Locate a pre-downloaded FASTA file for this species/type.
    Avoids calling load_fasta_species (which uses FTP, unavailable on compute nodes).
    """
    suffix = FASTA_SUFFIX[fasta_type]
    cap = species.split("_")[0].capitalize()
    candidates = [f for f in Path(FASTA_PATH).glob("*.fa.gz")
                  if f.name.endswith(suffix) and f.name.startswith(cap)]
    if not candidates:
        raise FileNotFoundError(
            f"No {fasta_type} FASTA for {species!r} in {FASTA_PATH}. "
            f"Run scripts/download_fastas.py on the login node first."
        )
    return candidates[0]


def get_output_dir(embedder: str) -> Path:
    d = Path(f"data/main/gene_embs_{embedder}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_species(organism: str, embedder: str, output_dir: Path, cuda: bool = True) -> None:
    from scprint import utils
    from scprint.tokenizers.protein_embedder import ESM2, GenaLM

    device = "cuda" if cuda and torch.cuda.is_available() else "cpu"
    embedding_size = EMBEDDING_SIZE[embedder]

    print(f"\n{'='*60}")
    print(f"Processing: {organism}  (embedder={embedder}, device={device})")
    print(f"{'='*60}")

    # ---- load FASTA --------------------------------------------------------
    # Per-job scratch dir for unzipped/subset fastas (node-local SSD, fast I/O)
    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "fasta_work"
    scratch.mkdir(parents=True, exist_ok=True)

    if embedder in ("esm2",):
        # Protein sequences — pre-downloaded, no internet needed
        gz_path = find_fasta(organism, "pep")
        unzipped = scratch / gz_path.stem  # .fa (drop last .gz)
        if not unzipped.exists():
            print(f"  Decompressing {gz_path.name}...")
            import subprocess
            subprocess.run(["gunzip", "-c", str(gz_path)], stdout=open(unzipped, "wb"), check=True)

        subset_fa = str(scratch / f"subset_{organism}.fa")
        _, naming_df = utils.subset_fasta(
            gene_tosubset=None,
            subfasta_path=subset_fa,
            fasta_path=str(unzipped),
            drop_unknown_seq=True,
            subset_protein_coding=True,
        )
        embedder_obj = ESM2(batch_size=16)
        emb_raw = embedder_obj(subset_fa, device=device)

    elif embedder == "gena_lm":
        # cDNA sequences — pre-downloaded, no internet needed
        gz_path = find_fasta(organism, "cdna")
        unzipped = scratch / gz_path.stem
        if not unzipped.exists():
            print(f"  Decompressing {gz_path.name}...")
            import subprocess
            subprocess.run(["gunzip", "-c", str(gz_path)], stdout=open(unzipped, "wb"), check=True)

        subset_fa = str(scratch / f"subset_cdna_{organism}.fa")
        _, naming_df = utils.subset_fasta(
            gene_tosubset=None,
            subfasta_path=subset_fa,
            fasta_path=str(unzipped),
            drop_unknown_seq=False,
            subset_protein_coding=False,
        )
        embedder_obj = GenaLM(batch_size=8)
        emb_raw = embedder_obj(subset_fa, device=device)

    else:
        raise ValueError(f"Unknown embedder: {embedder}")

    # ---- AdaptiveAvgPool to target size ------------------------------------
    m = AdaptiveAvgPool1d(embedding_size)
    emb = pd.DataFrame(
        data=m(torch.tensor(emb_raw.values.astype(float))).numpy(),
        index=emb_raw.index,
    )
    print(f"  Embeddings shape: {emb.shape}")

    # ---- Save --------------------------------------------------------------
    short = organism.split("_")[0]
    out_path = output_dir / f"{short}_emb.parquet"
    emb.to_parquet(out_path)
    print(f"  Saved → {out_path}")


def concatenate_all(embedder: str, output_dir: Path) -> None:
    parts = list(output_dir.glob("*_emb.parquet"))
    if not parts:
        print("No parquet files found to concatenate.")
        return
    dfs = [pd.read_parquet(p) for p in sorted(parts)]
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    out = output_dir / "gene_embeddings.parquet"
    combined.to_parquet(out)
    print(f"\nCombined {len(parts)} species → {combined.shape}  saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["esm2", "gena_lm"], required=True)
    parser.add_argument("--species", nargs="*", default=None,
                        help="Subset of species to run (default: all)")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    cuda = not args.no_cuda
    output_dir = get_output_dir(args.embedder)

    species_to_run = args.species if args.species else list(SPECIES.keys())

    for organism in species_to_run:
        short = organism.split("_")[0]
        out_path = output_dir / f"{short}_emb.parquet"
        if out_path.exists():
            print(f"Skipping {organism} — {out_path} already exists")
            continue
        try:
            run_species(organism, args.embedder, output_dir, cuda=cuda)
        except Exception as e:
            print(f"ERROR processing {organism}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            print("Continuing with next species...")

    concatenate_all(args.embedder, output_dir)
    print("\nAll done.")


if __name__ == "__main__":
    main()
