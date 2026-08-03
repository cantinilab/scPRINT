#!/usr/bin/env python3
# Set IDRIS proxy (needed on prepost/compute nodes)
import os
os.environ["http_proxy"] = "http://prodprox.idris.fr:3128")
os.environ["https_proxy"] = "http://prodprox.idris.fr:3128")
import os
os.environ["http_proxy"] =  "http://prodprox.idris.fr:3128")
os.environ["https_proxy"] = "http://prodprox.idris.fr:3128")
os.environ["HTTP_PROXY"] =  "http://prodprox.idris.fr:3128")
os.environ["HTTPS_PROXY"] = "http://prodprox.idris.fr:3128")

"""
prefetch_genomic_seqs.py
Pre-fetch genomic sequences (gene body ±10kb) from Ensembl REST API.
Run on Jean Zay LOGIN NODE (has internet). Compute nodes do NOT.

Saves per-gene sequences to:
  data/main/genomic_seqs/<species>/<gene_id>.fa.gz  (individual, too many files)
  → Better: data/main/genomic_seqs/<species>.parquet  (gene_id → sequence + coords)

Usage:
    python3 scripts/prefetch_genomic_seqs.py
    python3 scripts/prefetch_genomic_seqs.py --species homo_sapiens mus_musculus
    python3 scripts/prefetch_genomic_seqs.py --resume   # skip already-done species
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE        = Path("/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT")
ESM3_DIR    = BASE / "data/main/gene_embs"        # source of gene ID lists
OUTPUT_DIR  = BASE / "data/main/genomic_seqs"
FLANK       = 10_000
BATCH       = 80         # genes per REST batch (Ensembl limit ~100)
SLEEP       = 0.07       # ~14 req/s, under 15 req/s limit
MAX_SEQ     = 131_072    # truncate sequences above this (multiple of 128)

ENSEMBL_MAIN   = "https://rest.ensembl.org"
ENSEMBL_GENOME = "https://rest.ensemblgenomes.org"
PLANT_SP  = {"arabidopsis_thaliana", "zea_mays"}
META_SP   = {"caenorhabditis_elegans", "drosophila_melanogaster"}

SPECIES_LIST = [
    "homo_sapiens", "mus_musculus", "arabidopsis_thaliana", "bos_taurus",
    "caenorhabditis_elegans", "callithrix_jacchus", "danio_rerio",
    "drosophila_melanogaster", "gallus_gallus", "heterocephalus_glaber_male",
    "macaca_mulatta", "oryctolagus_cuniculus", "ovis_aries", "pan_troglodytes",
    "sus_scrofa", "zea_mays",
]


def rest_base(sp):
    if sp in PLANT_SP or sp in META_SP:
        return ENSEMBL_GENOME
    return ENSEMBL_MAIN


def batch_lookup(ids, base, session):
    url  = f"{base}/lookup/id"
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    for attempt in range(4):
        r = session.post(url, data=json.dumps({"ids": ids}), headers=hdrs, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            print(f"    rate-limited, waiting {wait}s...", flush=True)
            time.sleep(wait)
        else:
            print(f"    lookup error {r.status_code}", flush=True)
            break
    return {}


def fetch_sequence(gid, start, end, strand, base, session, flank=FLANK):
    """
    Fetch genomic sequence for a gene using region endpoint (no retry on gene ID).
    Returns (sequence, gene_start_in_seq, gene_end_in_seq) or (None, 0, 0).
    """
    # Use /sequence/region for direct coordinate-based fetching
    # expand upstream/downstream based on strand
    url = f"{base}/sequence/id/{gid}?type=genomic&expand_5prime={flank}&expand_3prime={flank}"
    hdrs = {"Accept": "text/plain"}
    for attempt in range(3):
        r = session.get(url, headers=hdrs, timeout=60)
        if r.status_code == 200:
            seq = r.text.strip().upper()
            # gene body is at flank : flank + (end - start + 1)
            gene_len = abs(end - start) + 1
            gs = flank
            ge = flank + gene_len
            # Truncate if too long
            if len(seq) > MAX_SEQ:
                mid = (gs + ge) // 2
                half = MAX_SEQ // 2
                off = max(0, mid - half)
                seq = seq[off: off + MAX_SEQ]
                gs  = max(0, gs - off)
                ge  = min(len(seq), ge - off)
            return seq, gs, ge
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 2 ** attempt))
            time.sleep(wait)
        else:
            break
    return None, 0, 0


def process_species(sp: str, session: requests.Session) -> None:
    out_path = OUTPUT_DIR / f"{sp}.parquet"
    if out_path.exists():
        print(f"  [SKIP] {sp} — already done")
        return

    # Load gene IDs from ESM3 parquets
    short    = sp.split("_")[0]
    pq_files = list(ESM3_DIR.glob(f"{short}_emb.parquet"))
    if not pq_files:
        # try 'mouse' for mus
        alt = "mouse" if sp == "mus_musculus" else short
        pq_files = list(ESM3_DIR.glob(f"{alt}_emb.parquet"))
    if not pq_files:
        print(f"  [SKIP] {sp} — no ESM3 parquet found in {ESM3_DIR}")
        return

    gene_ids = pd.read_parquet(pq_files[0]).index.tolist()
    base     = rest_base(sp)
    print(f"  {sp}: {len(gene_ids)} genes, base={base}", flush=True)

    # Step 1: batch lookup coordinates
    print(f"  Fetching coordinates...", flush=True)
    coords = {}
    for i in range(0, len(gene_ids), BATCH):
        batch = gene_ids[i : i + BATCH]
        result = batch_lookup(batch, base, session)
        coords.update({k: v for k, v in result.items() if v is not None})
        time.sleep(SLEEP)
        if i % 1000 == 0:
            print(f"    {i}/{len(gene_ids)} coords fetched", flush=True)
    print(f"  Found coords for {len(coords)}/{len(gene_ids)} genes", flush=True)

    # Step 2: fetch sequences — save in chunks of 2000 to avoid OOM
    CHUNK = 2000
    chunk_dir = OUTPUT_DIR / f"{sp}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    records = []
    chunk_idx = 0
    found_ids = [g for g in gene_ids if g in coords]

    for i, gid in enumerate(found_ids):
        info = coords[gid]
        seq, gs, ge = fetch_sequence(
            gid, info.get("start", 0), info.get("end", 0),
            info.get("strand", 1), base, session
        )
        time.sleep(SLEEP)
        if seq is None:
            continue
        records.append({"gene_id": gid, "sequence": seq,
                        "gene_start": gs, "gene_end": ge})

        if len(records) >= CHUNK:
            chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
            pd.DataFrame(records).set_index("gene_id").to_parquet(chunk_path)
            print(f"    {i+1}/{len(found_ids)} seqs — saved chunk {chunk_idx}", flush=True)
            records = []
            chunk_idx += 1

    # Save remaining
    if records:
        chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.parquet"
        pd.DataFrame(records).set_index("gene_id").to_parquet(chunk_path)
        print(f"    {len(found_ids)} seqs — saved final chunk {chunk_idx}", flush=True)

    # Merge chunks
    chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    if not chunks:
        print(f"  ERROR: no sequences for {sp}", file=sys.stderr)
        return
    df = pd.concat([pd.read_parquet(c) for c in chunks])
    df.to_parquet(out_path)
    # Clean up chunks
    import shutil; shutil.rmtree(chunk_dir)
    print(f"  ✓ Saved {len(df)} sequences → {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="*", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    species = args.species or SPECIES_LIST
    PROXY = "http://prodprox.idris.fr:3128"
    session = requests.Session()
    session.proxies.update({"http": PROXY, "https": PROXY})
    session.headers["User-Agent"] = "scPRINT-NTv3-prefetch/1.0"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Prefetching genomic sequences (±{FLANK//1000}kb) for {len(species)} species")
    print(f"Output: {OUTPUT_DIR}\n")

    for sp in species:
        try:
            process_species(sp, session)
        except Exception as e:
            print(f"ERROR {sp}: {e}", file=sys.stderr)
            import traceback; traceback.print_exc()
            print("Continuing...\n")

    print("\nAll done.")


if __name__ == "__main__":
    main()
