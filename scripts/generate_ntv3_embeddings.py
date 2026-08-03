#!/usr/bin/env python3
# Override cache paths before any imports (compute nodes have no internet)
import os
os.environ["TORCH_HOME"]           = "/lustre/fswork/projects/rech/xeg/uat95fg/.cache/torch"
os.environ["HF_HOME"]              = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub"
os.environ["TRANSFORMERS_CACHE"]   = "/lustre/fswork/projects/rech/xeg/uat95fg/.hf/hub"

"""
generate_ntv3_embeddings.py
Generate gene embeddings for all scPRINT species using NTv3_100M_post.

Each gene: genomic sequence (gene body ±10kb) → NTv3 encoder
           → mean pool over gene body nucleotide positions → embedding vector

Usage (from scPRINT project root):
    .venv/bin/python3 scripts/generate_ntv3_embeddings.py
    .venv/bin/python3 scripts/generate_ntv3_embeddings.py --species homo_sapiens mus_musculus
    .venv/bin/python3 scripts/generate_ntv3_embeddings.py --no-cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import AdaptiveAvgPool1d

# ---------------------------------------------------------------------------
# Species list + output dir
# ---------------------------------------------------------------------------
SPECIES = [
    "homo_sapiens",
    "mus_musculus",
    "arabidopsis_thaliana",
    "bos_taurus",
    "caenorhabditis_elegans",
    "callithrix_jacchus",
    "danio_rerio",
    "drosophila_melanogaster",
    "gallus_gallus",
    "heterocephalus_glaber_male",
    "macaca_mulatta",
    "oryctolagus_cuniculus",
    "ovis_aries",
    "pan_troglodytes",
    "sus_scrofa",
    "zea_mays",
]

EMBEDDING_SIZE = 512   # final AdaptiveAvgPool target (NTv3_100M hidden=512)
OUTPUT_DIR     = Path("data/main/gene_embs_ntv3")

# Gene data from scdataloader (for gene list per species)
NCBITAXON = {
    "homo_sapiens":               "NCBITaxon:9606",
    "mus_musculus":               "NCBITaxon:10090",
    "arabidopsis_thaliana":       "NCBITaxon:3702",
    "bos_taurus":                 "NCBITaxon:9913",
    "caenorhabditis_elegans":     "NCBITaxon:6239",
    "callithrix_jacchus":         "NCBITaxon:9483",
    "danio_rerio":                "NCBITaxon:7955",
    "drosophila_melanogaster":    "NCBITaxon:7227",
    "gallus_gallus":              "NCBITaxon:9031",
    "heterocephalus_glaber_male": "NCBITaxon:10181",
    "macaca_mulatta":             "NCBITaxon:9544",
    "oryctolagus_cuniculus":      "NCBITaxon:9986",
    "ovis_aries":                 "NCBITaxon:9940",
    "pan_troglodytes":            "NCBITaxon:9598",
    "sus_scrofa":                 "NCBITaxon:9823",
    "zea_mays":                   "NCBITaxon:4577",
}


SEQ_CACHE = Path("data/main/genomic_seqs")


def get_sequences(species: str) -> pd.DataFrame:
    """
    Load pre-fetched genomic sequences from GPFS cache.
    Returns DataFrame with columns: sequence, gene_start, gene_end.
    Raises FileNotFoundError if not pre-fetched yet.
    """
    path = SEQ_CACHE / f"{species}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No pre-fetched sequences for {species}.\n"
            f"Run scripts/prefetch_genomic_seqs.py on the LOGIN NODE first."
        )
    return pd.read_parquet(path)


def run_species(species: str, device: str) -> pd.DataFrame:
    """Embed all genes for one species. Returns DataFrame (genes × hidden_dim)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from ntv3_embedder_v2 import NTv3EmbedderFromCache

    print(f"\n{'='*60}")
    print(f"Processing: {species}  (device={device})")
    print(f"{'='*60}")

    seq_df = get_sequences(species)
    print(f"  Loaded {len(seq_df)} pre-fetched sequences")

    import os as _os
    bs = int(_os.environ.get("NTV3_BATCH_SIZE", "2"))
    embedder = NTv3EmbedderFromCache(
        model_name="InstaDeepAI/NTv3_100M_post",
        batch_size=bs,
        use_gene_body_only=True,
    )
    emb_raw = embedder(seq_df, species=species, device=device)
    print(f"  Raw embeddings: {emb_raw.shape}")

    # AdaptiveAvgPool to standard EMBEDDING_SIZE
    m = AdaptiveAvgPool1d(EMBEDDING_SIZE)
    emb = pd.DataFrame(
        data=m(torch.tensor(emb_raw.values.astype(float))).numpy(),
        index=emb_raw.index,
    )
    print(f"  Final embeddings: {emb.shape}")
    return emb


def concatenate_all(output_dir: Path) -> None:
    parts = sorted(output_dir.glob("*_emb.parquet"))
    if not parts:
        print("No parquet files to concatenate.")
        return
    dfs = [pd.read_parquet(p) for p in parts]
    combined = pd.concat(dfs)
    combined = combined[~combined.index.duplicated(keep="first")]
    out = output_dir / "gene_embeddings.parquet"
    combined.to_parquet(out)
    print(f"\nCombined {len(parts)} species → {combined.shape}  saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    cuda   = not args.no_cuda and torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    species_list = args.species or SPECIES

    for sp in species_list:
        short    = sp.split("_")[0]
        out_path = OUTPUT_DIR / f"{short}_emb.parquet"
        if out_path.exists():
            print(f"Skipping {sp} — {out_path} already exists")
            continue
        try:
            emb = run_species(sp, device)
            emb.to_parquet(out_path)
            print(f"  Saved → {out_path}")
        except Exception as e:
            print(f"ERROR {sp}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            print("Continuing...")

    concatenate_all(OUTPUT_DIR)
    print("\nAll done.")


if __name__ == "__main__":
    main()
