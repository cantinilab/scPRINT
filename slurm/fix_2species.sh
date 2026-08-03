#!/bin/bash
#SBATCH --job-name=fix_ovis_het
#SBATCH --output=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%j.out
#SBATCH --error=/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/slurm/slurm-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=prepost
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH -A wbg@v100
set -e
export http_proxy=http://prodprox.idris.fr:3128
export https_proxy=http://prodprox.idris.fr:3128
cd /lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
mkdir -p data/genomes/_archive
echo "=== Move old wrong builds to archive (idempotent) ==="
mv -f data/genomes/Ovis_aries.Oar_v3.1.* data/genomes/_archive/ 2>/dev/null || true
mv -f data/genomes/Heterocephalus_glaber_male.* data/genomes/_archive/ 2>/dev/null || true
echo "=== Download Rambouillet (ovis) ==="
BASE=https://ftp.ensembl.org/pub/release-110
cd data/genomes
curl -fsSL --retry 3 -O $BASE/gtf/ovis_aries_rambouillet/Ovis_aries_rambouillet.Oar_rambouillet_v1.0.110.chr.gtf.gz
curl -fsSL --retry 3 -O $BASE/fasta/ovis_aries_rambouillet/dna/Ovis_aries_rambouillet.Oar_rambouillet_v1.0.dna_sm.toplevel.fa.gz
echo "=== Download het_female ==="
curl -fsSL --retry 3 -O $BASE/gtf/heterocephalus_glaber_female/Heterocephalus_glaber_female.Naked_mole-rat_maternal.110.chr.gtf.gz
curl -fsSL --retry 3 -O $BASE/fasta/heterocephalus_glaber_female/dna/Heterocephalus_glaber_female.Naked_mole-rat_maternal.dna_sm.toplevel.fa.gz
ls -lh Ovis_aries_rambouillet*.gz Heterocephalus_glaber_female*.gz
cd /lustre/fswork/projects/rech/xeg/uat95fg/scPRINT
echo "=== Verify gene_id match ==="
echo "-- Ovis Rambouillet first gene_ids --"
zcat data/genomes/Ovis_aries_rambouillet.Oar_rambouillet_v1.0.110.chr.gtf.gz | awk -F"\t" "\$3==\"gene\"" | head -3 | grep -oE "gene_id \"[^\"]+\""
echo "-- Het female first gene_ids --"
zcat data/genomes/Heterocephalus_glaber_female.Naked_mole-rat_maternal.110.chr.gtf.gz | awk -F"\t" "\$3==\"gene\"" | head -3 | grep -oE "gene_id \"[^\"]+\""
echo "=== Re-run extraction ==="
.venv/bin/python3 scripts/extract_genomic_seqs_local.py --species ovis_aries heterocephalus_glaber_male
echo "=== Done. Parquets: ==="
ls data/main/genomic_seqs/*.parquet | wc -l
