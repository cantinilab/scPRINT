#!/usr/bin/env python3
"""
umap_embeddings.py
UMAP comparison of ESM2 / GENA-LM / ESM3 gene embeddings, coloured by species.
Saves figures to data/main/umap_*.png
Run from scPRINT project root.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

BASE = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT"

SPECIES_MAP = {
    "arabidopsis": "A. thaliana",
    "bos":         "B. taurus",
    "caenorhabditis": "C. elegans",
    "callithrix":  "C. jacchus",
    "danio":       "D. rerio",
    "drosophila":  "D. melanogaster",
    "gallus":      "G. gallus",
    "heterocephalus": "H. glaber",
    "homo":        "H. sapiens",
    "macaca":      "M. mulatta",
    "mouse":       "M. musculus",
    "oryctolagus": "O. cuniculus",
    "ovis":        "O. aries",
    "pan":         "P. troglodytes",
    "sus":         "S. scrofa",
    "zea":         "Z. mays",
}

EMBEDDERS = {
    "ESM3":    f"{BASE}/data/main/gene_embs",
    "ESM2":    f"{BASE}/data/main/gene_embs_esm2",
    "GENA-LM": f"{BASE}/data/main/gene_embs_gena_lm",
}

# 20 visually distinct colours
PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4",
    "#469990","#dcbeff","#9A6324","#fffac8","#800000",
    "#aaffc3",
]

N_PER_SPECIES = 3000   # subsample per species for UMAP speed
UMAP_PARAMS   = dict(n_neighbors=30, min_dist=0.1, metric="cosine",
                     n_components=2, random_state=42, low_memory=True)


def load_embedder(folder: str) -> pd.DataFrame:
    """Load per-species parquets, add 'species' column, subsample."""
    parts = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith("_emb.parquet"):
            continue
        key = fname.replace("_emb.parquet", "")
        label = SPECIES_MAP.get(key, key)
        df = pd.read_parquet(os.path.join(folder, fname))
        df = df.sample(min(N_PER_SPECIES, len(df)), random_state=42)
        df["species"] = label
        parts.append(df)
    combined = pd.concat(parts, ignore_index=True)
    print(f"  Loaded {len(combined):,} genes, {combined.shape[1]-1} dims")
    return combined


def run_umap(df: pd.DataFrame) -> np.ndarray:
    X = df.drop(columns=["species"]).values.astype(np.float32)
    reducer = umap.UMAP(**UMAP_PARAMS)
    return reducer.fit_transform(X)


# ── Main ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("Gene embedding UMAP — coloured by species", fontsize=14, y=1.01)

for ax, (name, folder) in zip(axes, EMBEDDERS.items()):
    print(f"\n[{name}] loading...")
    df = load_embedder(folder)

    species_list = sorted(df["species"].unique())
    colour_map   = {sp: PALETTE[i % len(PALETTE)] for i, sp in enumerate(species_list)}
    colours      = df["species"].map(colour_map).values

    print(f"[{name}] running UMAP...")
    xy = run_umap(df)

    ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=1.5, alpha=0.5, linewidths=0, rasterized=True)
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Shared legend (right of last panel)
patches = [mpatches.Patch(color=PALETTE[i % len(PALETTE)], label=sp)
           for i, sp in enumerate(sorted(SPECIES_MAP.values()))]
axes[-1].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc="upper left",
                fontsize=7.5, frameon=False, markerscale=2)

plt.tight_layout()
out = f"{BASE}/data/main/umap_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out}")

# Also save per-embedder for closer inspection
for ax, (name, folder) in zip(axes, EMBEDDERS.items()):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"{BASE}/data/main/umap_{name.lower().replace('-','_')}.png",
                dpi=180, bbox_inches=extent.expanded(1.1, 1.2))

print("All done.")
