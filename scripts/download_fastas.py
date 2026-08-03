#!/usr/bin/env python3
"""
download_fastas.py — pre-download Ensembl FASTA files via HTTPS.
Run on Jean Zay login node (internet access). Compute nodes do NOT have internet.

Usage:
    python3 scripts/download_fastas.py                 # all species, pep + cdna
    python3 scripts/download_fastas.py --only pep      # ESM2 only
    python3 scripts/download_fastas.py --only cdna     # GENA-LM only
"""
import argparse
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---- GPFS persistent cache (accessible from compute nodes) ------------------
FASTA_PATH = Path("/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/fasta")

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

PLANTS   = {"arabidopsis_thaliana", "zea_mays"}
METAZOA  = {"caenorhabditis_elegans", "drosophila_melanogaster"}

# Ensembl base URLs (HTTPS)
BASE_ANIMAL  = "https://ftp.ensembl.org/pub/release-110/fasta"
BASE_PLANT   = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60/fasta"
BASE_METAZOA = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/metazoa/release-60/fasta"

FASTA_SUFFIX = {
    "pep":  ".all.fa.gz",
    "cdna": ".cdna.all.fa.gz",
    "ncrna": ".ncrna.fa.gz",
}


def species_base(species: str) -> str:
    if species in PLANTS:
        return f"{BASE_PLANT}/{species}"
    elif species in METAZOA:
        return f"{BASE_METAZOA}/{species}"
    else:
        return f"{BASE_ANIMAL}/{species}"


def list_dir(url: str):
    """Return list of hrefs from an Ensembl HTTPS directory listing."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as f:
            html = f.read().decode(errors="replace")
        return re.findall(r'href="([^"/][^"]*\.fa\.gz)"', html)
    except Exception as e:
        raise RuntimeError(f"Failed to list {url}: {e}")


def download_file(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream-download url to dest with a progress indicator."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            pct = 100 * downloaded // total if total else 0
            print(f"\r    {pct:3d}%  {downloaded/1e6:.0f}/{total/1e6:.0f} MB  {dest.name}", end="", flush=True)
    print()  # newline after progress


def download_one(species: str, fasta_type: str) -> Path:
    """Download one FASTA file; return local path. Skip if already present."""
    FASTA_PATH.mkdir(parents=True, exist_ok=True)
    suffix = FASTA_SUFFIX[fasta_type]
    base = species_base(species)

    # Check if already cached
    existing = [f for f in FASTA_PATH.iterdir() if f.name.endswith(suffix)
                and species.split("_")[0].lower() in f.name.lower()]
    if existing:
        print(f"  [SKIP] {species} {fasta_type} — {existing[0].name}")
        return existing[0]

    dir_url = f"{base}/{fasta_type}/"
    print(f"  Listing {dir_url}", flush=True)
    files = list_dir(dir_url)
    matches = [f for f in files if f.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"No {suffix} file found at {dir_url}. Got: {files[:5]}")
    fname = matches[0]
    dest = FASTA_PATH / fname

    if dest.exists():
        print(f"  [SKIP] {dest.name}")
        return dest

    file_url = dir_url + fname
    print(f"  Downloading {fname}...", flush=True)
    download_file(file_url, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"  ✓ {dest.name}  ({size_mb:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["pep", "cdna", "both"], default="both")
    parser.add_argument("--species", nargs="*", default=None)
    args = parser.parse_args()

    ftypes = {"both": ["pep", "cdna"], "pep": ["pep"], "cdna": ["cdna"]}[args.only]
    species_list = args.species or SPECIES

    print(f"Downloading {ftypes} FASTAs for {len(species_list)} species")
    print(f"Output: {FASTA_PATH}\n")

    errors = []
    for sp in species_list:
        for ft in ftypes:
            try:
                download_one(sp, ft)
            except Exception as e:
                print(f"\n  ERROR {sp} {ft}: {e}", file=sys.stderr)
                errors.append((sp, ft, str(e)))

    if errors:
        print(f"\n{len(errors)} errors:")
        for sp, ft, err in errors:
            print(f"  {sp} {ft}: {err}")
        sys.exit(1)
    else:
        print(f"\nAll done. Files in {FASTA_PATH}")
        print(f"Total files: {len(list(FASTA_PATH.iterdir()))}")


if __name__ == "__main__":
    main()
