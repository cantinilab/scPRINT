"""
NTv3Embedder — Gene embeddings via Nucleotide Transformer v3 (NTv3_100M_post).

Input  : per-species gene DataFrames with Ensembl gene IDs
Process: fetch genomic sequences (gene body ±10kb) from Ensembl REST API
         → NTv3 encoder → mean pool over gene body nucleotide embeddings
Output : pd.DataFrame (genes × hidden_dim), same API as ESM2/GenaLM
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Species name mapping: Ensembl organism name → NTv3 conditioning label
# For unsupported species, fall back to the closest relative.
# ---------------------------------------------------------------------------
# NTv3_100M_post supports only this list of organism conditioning ids.
# We map each scPRINT species to the closest available one.
SPECIES_TO_NTV3 = {
    "homo_sapiens":               "human",
    "mus_musculus":               "mouse",
    "arabidopsis_thaliana":       "arabidopsis_thaliana",
    "bos_taurus":                 "bison_bison_bison",       # closest bovine
    "caenorhabditis_elegans":     "caenorhabditis_elegans",
    "callithrix_jacchus":         "macaca_nemestrina",        # primate non-humain plus proche
    "danio_rerio":                "danio_rerio",
    "drosophila_melanogaster":    "drosophila_melanogaster",
    "gallus_gallus":              "gallus_gallus",
    "heterocephalus_glaber_male": "chinchilla_lanigera",      # closest rodent
    "macaca_mulatta":             "macaca_nemestrina",
    "oryctolagus_cuniculus":      "rattus_norvegicus",         # lagomorphe -> murin plus proche
    "ovis_aries":                 "bison_bison_bison",        # closest bovine for sheep
    "pan_troglodytes":            "gorilla_gorilla",
    "sus_scrofa":                 "bison_bison_bison",        # closest large mammal
    "zea_mays":                   "zea_mays",
}

ENSEMBL_REST     = "https://rest.ensembl.org"
ENSEMBL_GENOMES  = "https://rest.ensemblgenomes.org"   # plants, metazoa
PLANT_SPECIES    = {"arabidopsis_thaliana", "zea_mays", "oryza_sativa",
                    "solanum_lycopersicum"}
METAZOA_SPECIES  = {"caenorhabditis_elegans", "drosophila_melanogaster"}

FLANK            = 10_000     # ±10kb around gene body
MAX_SEQ_LEN      = 131_072    # pad/truncate ceiling (128kb, multiple of 128)
BATCH_REST       = 50         # gene IDs per Ensembl batch lookup
RATE_LIMIT_SLEEP = 0.08       # ~12 req/s, well under Ensembl 15 req/s limit


def _rest_base(species: str) -> str:
    if species in PLANT_SPECIES:
        return ENSEMBL_GENOMES
    if species in METAZOA_SPECIES:
        return ENSEMBL_GENOMES
    return ENSEMBL_REST


def _batch_lookup(gene_ids: List[str], base: str, session) -> Dict:
    """POST /lookup/id for up to 50 IDs → {gene_id: {start, end, strand, ...}}"""
    import json
    url  = f"{base}/lookup/id"
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    body = json.dumps({"ids": gene_ids})
    for attempt in range(3):
        r = session.post(url, data=body, headers=hdrs, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            break
    return {}


def _fetch_sequence(gene_id: str, base: str, session,
                    flank: int = FLANK) -> Tuple[Optional[str], int, int]:
    """
    Fetch genomic sequence ±flank around gene body.
    Returns (sequence, gene_start_in_seq, gene_end_in_seq) or (None, 0, 0).
    """
    url  = (f"{base}/sequence/id/{gene_id}"
            f"?type=genomic&expand_5prime={flank}&expand_3prime={flank}")
    hdrs = {"Content-Type": "text/plain"}
    for attempt in range(3):
        r = session.get(url, headers=hdrs, timeout=60)
        if r.status_code == 200:
            seq = r.text.strip()
            return seq, flank, len(seq) - flank
        if r.status_code == 429:
            time.sleep(2 ** attempt)
        else:
            break
    return None, 0, 0


def _to_ntv3_input(seq: str, gene_start: int, gene_end: int,
                   pad_multiple: int = 128) -> Tuple[str, int, int]:
    """Truncate or pad sequence to fit NTv3 requirements; adjust gene coords."""
    seq = seq.upper().replace(" ", "").replace("\n", "")

    # Truncate if needed (keep as much gene body as possible)
    if len(seq) > MAX_SEQ_LEN:
        # Centre the gene body
        mid       = (gene_start + gene_end) // 2
        new_start = max(0, mid - MAX_SEQ_LEN // 2)
        new_end   = new_start + MAX_SEQ_LEN
        if new_end > len(seq):
            new_end   = len(seq)
            new_start = max(0, new_end - MAX_SEQ_LEN)
        gene_start -= new_start
        gene_end   -= new_start
        seq         = seq[new_start:new_end]

    gene_start = max(0, gene_start)
    gene_end   = min(len(seq), gene_end)

    # Pad to next multiple of 128
    remainder = len(seq) % pad_multiple
    if remainder != 0:
        seq += "N" * (pad_multiple - remainder)

    return seq, gene_start, gene_end


class NTv3Embedder:
    def __init__(
        self,
        model_name: str = "InstaDeepAI/NTv3_100M_post",
        batch_size: int = 4,
        use_gene_body_only: bool = True,
    ):
        """
        NTv3 gene embedder.

        Args:
            model_name: HuggingFace model ID. Default: NTv3_100M_post.
            batch_size: Sequences per GPU batch (keep low for large sequences).
            use_gene_body_only: If True, mean pool only gene body positions
                                (excludes ±10kb flanking from the embedding).
        """
        self.model_name          = model_name
        self.batch_size          = batch_size
        self.use_gene_body_only  = use_gene_body_only

    def __call__(
        self,
        genedf: pd.DataFrame,
        species: str,
        device: str = "cuda",
    ) -> pd.DataFrame:
        """
        Embed all genes in genedf for a given species.

        Args:
            genedf : DataFrame with Ensembl gene IDs as index.
            species: Ensembl organism name (e.g. "homo_sapiens").
            device : "cuda" or "cpu".

        Returns:
            pd.DataFrame of shape (n_genes, hidden_dim), indexed by gene ID.
        """
        import requests
        from transformers import AutoTokenizer, AutoModel
        from tqdm import tqdm

        # ------ Load model ------
        print(f"  Loading {self.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True
        ).to(device)
        model.eval()

        ntv3_species = SPECIES_TO_NTV3.get(species, "human")
        print(f"  Species conditioning: {species} → '{ntv3_species}'")

        base = _rest_base(species)
        gene_ids = genedf.index.tolist()

        # ------ Batch lookup gene coordinates ------
        print(f"  Fetching coordinates for {len(gene_ids)} genes...")
        coords = {}
        session = requests.Session()
        session.headers.update({"User-Agent": "scPRINT-NTv3/1.0"})
        for i in range(0, len(gene_ids), BATCH_REST):
            batch = gene_ids[i : i + BATCH_REST]
            result = _batch_lookup(batch, base, session)
            coords.update(result)
            time.sleep(RATE_LIMIT_SLEEP)

        found = [g for g in gene_ids if g in coords and coords[g] is not None]
        print(f"  Found coordinates for {len(found)}/{len(gene_ids)} genes")

        # ------ Fetch sequences + embed ------
        all_embs = []
        all_names = []

        for i in tqdm(range(0, len(found), self.batch_size)):
            batch_ids = found[i : i + self.batch_size]
            seqs, gene_starts, gene_ends = [], [], []

            for gid in batch_ids:
                seq, gs, ge = _fetch_sequence(gid, base, session)
                time.sleep(RATE_LIMIT_SLEEP)
                if seq is None or len(seq) < 200:
                    continue
                seq, gs, ge = _to_ntv3_input(seq, gs, ge)
                seqs.append(seq)
                gene_starts.append(gs)
                gene_ends.append(ge)
                all_names.append(gid)

            if not seqs:
                continue

            # Tokenise (character-level, pad to same length)
            max_len = max(len(s) for s in seqs)
            # Ensure max_len is a multiple of 128
            max_len = math.ceil(max_len / 128) * 128
            seqs_padded = [s.ljust(max_len, "N") for s in seqs]

            enc = tokenizer(
                seqs_padded,
                add_special_tokens=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                pad_to_multiple_of=128,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            sp_ids = model.encode_species([ntv3_species] * len(seqs))

            with torch.no_grad():
                out = model(input_ids=input_ids, species_ids=sp_ids)

            # out.embedding: (B, L, hidden_dim) at nucleotide resolution
            emb_mat = out.embedding  # (B, L, D)

            for j in range(len(seqs)):
                gs, ge = gene_starts[j], gene_ends[j]
                if self.use_gene_body_only and gs < ge:
                    gene_emb = emb_mat[j, gs:ge, :].mean(dim=0)
                else:
                    gene_emb = emb_mat[j, :, :].mean(dim=0)
                all_embs.append(gene_emb.cpu().float().numpy())

        if not all_embs:
            raise RuntimeError(f"No embeddings generated for {species}")

        return pd.DataFrame(data=np.array(all_embs), index=all_names)
