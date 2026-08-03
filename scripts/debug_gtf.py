import gzip, re, pandas as pd
parquet = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/main/gene_embs/ovis_emb.parquet"
df = pd.read_parquet(parquet)
print("ESM3:", df.index[:3].tolist())
gtf = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/genomes/Ovis_aries.Oar_v3.1.110.gtf.gz"
count = 0
with gzip.open(gtf, "rt") as f:
    for line in f:
        if "#" in line[:1] or "\tgene\t" not in line: continue
        m = re.search('gene_id "([^"]+)"', line)
        if m:
            print("GTF:", m.group(1))
            count += 1
        if count >= 3: break
