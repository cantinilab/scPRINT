#!/usr/bin/env python3
"""
download_genome_gtf.py — Download genome FASTA + GTF per species via HTTPS.
Run on Jean Zay LOGIN NODE (internet). Compute nodes do NOT have internet.
Same approach as download_fastas.py (pep/cdna).

Usage:
    python3 scripts/download_genome_gtf.py
    python3 scripts/download_genome_gtf.py --species homo_sapiens mus_musculus
"""
import argparse, os, re, sys, urllib.request
from pathlib import Path

os.environ["http_proxy"]  = "http://prodprox.idris.fr:3128"
os.environ["https_proxy"] = "http://prodprox.idris.fr:3128"

GENOME_PATH = Path("/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/genomes")

SPECIES = [
    "homo_sapiens", "mus_musculus", "arabidopsis_thaliana", "bos_taurus",
    "caenorhabditis_elegans", "callithrix_jacchus", "danio_rerio",
    "drosophila_melanogaster", "gallus_gallus", "heterocephalus_glaber_male",
    "macaca_mulatta", "oryctolagus_cuniculus", "ovis_aries", "pan_troglodytes",
    "sus_scrofa", "zea_mays",
]

PLANTS  = {"arabidopsis_thaliana", "zea_mays"}
METAZOA = {"caenorhabditis_elegans", "drosophila_melanogaster"}

BASE_ANIMAL  = "https://ftp.ensembl.org/pub/release-110"
BASE_PLANT   = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/plants/release-60"
BASE_METAZOA = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/metazoa/release-60"


def species_base(sp):
    if sp in PLANTS:   return BASE_PLANT
    if sp in METAZOA:  return BASE_METAZOA
    return BASE_ANIMAL


def list_dir(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as f:
        html = f.read().decode(errors="replace")
    return re.findall(r'href="([^"/][^"]*\.(?:fa|gtf)\.gz)"', html)


def download_file(url, dest, chunk=1 << 20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            data = r.read(chunk)
            if not data: break
            f.write(data); done += len(data)
            pct = 100 * done // total if total else 0
            print(f"\r  {pct:3d}%  {done/1e6:6.0f}/{total/1e6:.0f} MB  {dest.name}", end="", flush=True)
    print()


def download_one(sp, ftype):
    """ftype: 'genome' or 'gtf'"""
    GENOME_PATH.mkdir(parents=True, exist_ok=True)
    base = species_base(sp)

    if ftype == "genome":
        subdir = f"fasta/{sp}/dna"
        suffix = "dna_sm.primary_assembly.fa.gz"
        suffix2 = "dna_sm.toplevel.fa.gz"  # fallback
    else:  # gtf
        subdir = f"gtf/{sp}"
        suffix = ".gtf.gz"
        suffix2 = None

    # Check cached
    existing = [f for f in GENOME_PATH.glob(f"*{suffix}") 
                if sp.split("_")[0].lower() in f.name.lower()]
    if existing:
        print(f"  [SKIP] {sp} {ftype} — {existing[0].name}")
        return existing[0]

    dir_url = f"{base}/{subdir}/"
    print(f"  Listing {dir_url}", flush=True)
    files = list_dir(dir_url)

    # prefer primary_assembly, fall back to toplevel
    BAD = ["abinitio","chr","patch","scaff","hapl"]
    matches = [f for f in files if suffix in f and not any(b in f for b in BAD)]
    if not matches and suffix2:
        matches = [f for f in files if suffix2 in f]
    if not matches:
        # for GTF, just find any .gtf.gz that isn't chr or abinitio
        matches = [f for f in files if f.endswith(".gtf.gz") 
                   and "abinitio" not in f and "chr." not in f]
    if not matches:
        print(f"  ERROR: no {ftype} file for {sp} at {dir_url}", file=sys.stderr)
        return None

    fname = min(matches, key=len)
    dest  = GENOME_PATH / fname
    if dest.exists():
        print(f"  [SKIP] {dest.name}")
        return dest

    print(f"  Downloading {fname}...", flush=True)
    download_file(f"{dir_url}{fname}", dest)
    print(f"  ✓ {dest.name}  ({dest.stat().st_size/1e6:.0f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", nargs="*", default=None)
    args = parser.parse_args()
    slist = args.species or SPECIES

    GENOME_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Downloading genome + GTF for {len(slist)} species → {GENOME_PATH}\n")

    errors = []
    for sp in slist:
        for ftype in ("genome", "gtf"):
            try:
                download_one(sp, ftype)
            except Exception as e:
                print(f"\n  ERROR {sp} {ftype}: {e}", file=sys.stderr)
                errors.append((sp, ftype, str(e)))

    if errors:
        print(f"\n{len(errors)} errors:")
        for sp, ft, err in errors: print(f"  {sp} {ft}: {err}")
    else:
        print(f"\nAll done. Files in {GENOME_PATH}")
        print(f"Total: {len(list(GENOME_PATH.iterdir()))} files")


if __name__ == "__main__":
    main()
