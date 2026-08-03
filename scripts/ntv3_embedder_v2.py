"""
ntv3_embedder_v2.py — NTv3 embedder reading from pre-fetched sequence cache.
No internet access needed; all sequences are loaded from GPFS parquet files.
"""
import math
import numpy as np
import pandas as pd
import torch

from ntv3_embedder import SPECIES_TO_NTV3  # reuse mapping


class NTv3EmbedderFromCache:
    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_100M_post",
        batch_size: int = 2,  # can be overridden via NTV3_BATCH_SIZE env
        use_gene_body_only: bool = True,
    ):
        self.model_name         = model_name
        self.batch_size         = batch_size
        self.use_gene_body_only = use_gene_body_only

    def __call__(
        self,
        seq_df: pd.DataFrame,
        species: str,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Embed genes using pre-fetched sequences.

        Args:
            seq_df  : DataFrame with index=gene_id,
                      columns: sequence (str), gene_start (int), gene_end (int)
            species : Ensembl organism name (for NTv3 conditioning)
            device  : "cuda" or "cpu"

        Returns:
            pd.DataFrame (n_genes × hidden_dim)
        """
        from transformers import AutoTokenizer, AutoModel
        from tqdm import tqdm

        print(f"  Loading {self.model_name}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device)
        model.eval()

        ntv3_sp = SPECIES_TO_NTV3.get(species, "human")
        print(f"  Species: {species} → '{ntv3_sp}'", flush=True)

        # Sort by sequence length for efficient padding (shorter batches together)
        seq_df = seq_df.assign(_len=seq_df["sequence"].str.len()).sort_values("_len")
        gene_ids = seq_df.index.tolist()
        all_embs, all_names = [], []

        def _process_batch(batch_ids_local):
            seqs, gss, ges = [], [], []
            for gid in batch_ids_local:
                row = seq_df.loc[gid]
                seq = str(row["sequence"]).upper()
                gs  = int(row["gene_start"])
                ge  = int(row["gene_end"])
                rem = len(seq) % 128
                if rem != 0:
                    seq += "N" * (128 - rem)
                seqs.append(seq); gss.append(gs); ges.append(ge)
            if not seqs:
                return
            max_len = max(len(s) for s in seqs)
            max_len = math.ceil(max_len / 128) * 128
            seqs_padded = [s.ljust(max_len, "N") for s in seqs]
            enc = tokenizer(
                seqs_padded, add_special_tokens=False, padding="max_length",
                max_length=max_len, truncation=True, pad_to_multiple_of=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            sp_ids_raw = model.encode_species([ntv3_sp] * len(seqs))
            if isinstance(sp_ids_raw, (list, tuple)):
                sp_ids = [t.to(device) if hasattr(t, 'to') else t for t in sp_ids_raw]
            elif hasattr(sp_ids_raw, 'to'):
                sp_ids = sp_ids_raw.to(device)
            else:
                sp_ids = sp_ids_raw
            out = model(input_ids=input_ids, species_ids=sp_ids)
            emb_mat = out.embedding
            for j in range(len(seqs)):
                gs, ge = gss[j], ges[j]
                if self.use_gene_body_only and 0 <= gs < ge <= emb_mat.shape[1]:
                    gene_emb = emb_mat[j, gs:ge, :].mean(dim=0)
                else:
                    gene_emb = emb_mat[j].mean(dim=0)
                all_embs.append(gene_emb.cpu().float().numpy())
                all_names.append(batch_ids_local[j])
            del input_ids, out, emb_mat, sp_ids

        def _process_with_oom_retry(batch_ids_local):
            try:
                _process_batch(batch_ids_local)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch_ids_local) == 1:
                    print(f"  [OOM-SKIP] gene {batch_ids_local[0]} too large even at b=1", flush=True)
                    return
                mid = len(batch_ids_local) // 2
                print(f"  [OOM-SPLIT] {len(batch_ids_local)} -> {mid}+{len(batch_ids_local)-mid}", flush=True)
                _process_with_oom_retry(batch_ids_local[:mid])
                _process_with_oom_retry(batch_ids_local[mid:])

        with torch.no_grad():
            for i in tqdm(range(0, len(gene_ids), self.batch_size)):
                batch_ids = gene_ids[i : i + self.batch_size]
                _process_with_oom_retry(batch_ids)
                if (i // self.batch_size) % 50 == 0:
                    torch.cuda.empty_cache()

        if not all_embs:
            raise RuntimeError(f"No embeddings generated for {species}")

        return pd.DataFrame(data=np.array(all_embs), index=all_names)
