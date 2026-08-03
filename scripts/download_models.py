#!/usr/bin/env python3
"""
download_models.py — pre-download ESM2 and GENA-LM models on the login node.
Compute nodes have no internet. Run this on Jean Zay login before submitting jobs.

Usage:
    python3 scripts/download_models.py
    python3 scripts/download_models.py --only esm2
    python3 scripts/download_models.py --only gena_lm
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

# ---- writable GPFS paths -----------------------------------------------
TORCH_CACHE  = Path("/lustre/fswork/projects/rech/xeg/uat95fg/.cache/torch")
HF_HOME      = Path("/lustre/fswork/projects/rech/xeg/uat95fg/.hf")

ESM2_MODEL   = "esm2_t33_650M_UR50D"
ESM2_FILES   = {
    f"{ESM2_MODEL}.pt":
        f"https://dl.fbaipublicfiles.com/fair-esm/models/{ESM2_MODEL}.pt",
    f"{ESM2_MODEL}-contact-regression.pt":
        f"https://dl.fbaipublicfiles.com/fair-esm/regression/{ESM2_MODEL}-contact-regression.pt",
}
GENA_REPO    = "AIRI-Institute/gena-lm-bert-large-t2t"


def download_url(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Stream-download with progress."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            pct = 100 * downloaded // total if total else 0
            print(f"\r  {pct:3d}%  {downloaded/1e6:6.0f}/{total/1e6:.0f} MB  {dest.name}", end="", flush=True)
    print()


def download_esm2():
    hub_dir = TORCH_CACHE / "hub" / "checkpoints"
    hub_dir.mkdir(parents=True, exist_ok=True)
    print(f"ESM2 model cache: {hub_dir}")
    for fname, url in ESM2_FILES.items():
        dest = hub_dir / fname
        if dest.exists():
            print(f"  [SKIP] {fname} ({dest.stat().st_size/1e6:.0f} MB)")
            continue
        print(f"  Downloading {fname}...")
        download_url(url, dest)
        print(f"  ✓ {fname}  ({dest.stat().st_size/1e6:.0f} MB)")
    print("ESM2 models ready.\n")


def download_gena_lm():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not found; installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    cache_dir = HF_HOME / "hub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"GENA-LM cache: {cache_dir}")

    # Check if already cached
    repo_slug = GENA_REPO.replace("/", "--")
    existing = [d for d in cache_dir.glob(f"models--{repo_slug}*")]
    if existing:
        print(f"  [SKIP] Already cached: {existing[0].name}")
        return

    print(f"  Downloading {GENA_REPO} from HuggingFace...")
    path = snapshot_download(
        repo_id=GENA_REPO,
        cache_dir=str(cache_dir),
    )
    print(f"  ✓ Saved to {path}")
    print("GENA-LM model ready.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["esm2", "gena_lm", "both"], default="both")
    args = parser.parse_args()

    if args.only in ("esm2", "both"):
        download_esm2()
    if args.only in ("gena_lm", "both"):
        download_gena_lm()

    print("All models downloaded.")


if __name__ == "__main__":
    main()
