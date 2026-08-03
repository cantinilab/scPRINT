import gzip, re, pandas as pd, glob
BASE = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT"
for sp, short in [("ovis_aries","ovis"), ("heterocephalus_glaber_male","heterocephalus")]:
    print(f"\n=== {sp} ===")
    df = pd.read_parquet(f"{BASE}/data/main/gene_embs/{short}_emb.parquet")
    print(f"ESM3 IDs: {df.index[:3].tolist()}")
    gtfs = glob.glob(f"{BASE}/data/genomes/*{sp.split('_')[0].capitalize()}*.gtf.gz")
    print(f"GTF found: {gtfs}")
    if not gtfs: continue
    count = 0
    with gzip.open(gtfs[0], "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            if "\tgene\t" in line:
                print("gene attrs:", line.split("\t")[8][:150] if len(line.split("\t")) > 8 else "")
                count += 1
            if count >= 2: break
