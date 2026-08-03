import os, numpy as np, pandas as pd

N = 1500
BASE = "/lustre/fswork/projects/rech/xeg/uat95fg/scPRINT/data/main"
EMBEDDERS = {
    "esm3":    f"{BASE}/gene_embs",
    "esm2":    f"{BASE}/gene_embs_esm2",
    "gena_lm": f"{BASE}/gene_embs_gena_lm",
}
for name, folder in EMBEDDERS.items():
    Xs, labels = [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith("_emb.parquet"): continue
        sp = fname.replace("_emb.parquet", "")
        df = pd.read_parquet(f"{folder}/{fname}")
        s = df.sample(min(N, len(df)), random_state=42)
        Xs.append(s.values.astype("float32"))
        labels += [sp] * len(s)
    X = np.vstack(Xs)
    out = f"{BASE}/subsample_{name}.npz"
    np.savez_compressed(out, X=X, labels=np.array(labels))
    print(f"{name}: {X.shape} saved ({os.path.getsize(out)//1024}KB)")
