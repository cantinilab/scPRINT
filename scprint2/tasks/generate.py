import os
from typing import Any, Dict, List, Optional

import bionty as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData, concat
from scdataloader import Collator, Preprocessor
from scdataloader.data import SimpleAnnDataset
from scdataloader.utils import get_descendants, random_str
from scib_metrics.benchmark import Benchmarker
from scipy.stats import spearmanr
from simpler_flash import FlashTransformer
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE_LOC = os.path.dirname(os.path.realpath(__file__))


class Generate:
    def __init__(
        self,
        genelist: List[str],
        batch_size: int = 64,
        embedding_to_use: List[str] = [
            "all",
        ],
    ):
        """
        Embedder a class to embed and annotate cells using a model

        Args:
            genelist (List[str]): The list of genes for which to generate expression data.
            batch_size (int, optional): The size of the batches to be used in the DataLoader. Defaults to 64.
            embedding_to_use (List[str], optional): The list of embeddings to be used for generating expression. Defaults to ["all"].
        """
        self.batch_size = batch_size
        self.embedding_to_use = embedding_to_use
        self.genelist = genelist if genelist is not None else []

    def __call__(self, model: torch.nn.Module, adata: AnnData) -> tuple[AnnData, List[str], np.ndarray, dict]:
        """
        __call__ function to call the embedding

        Args:
            model (torch.nn.Module): The scPRINT model to be used for embedding and annotation.
            adata (AnnData): The annotated data matrix of shape n_obs x n_vars. Rows correspond to cells and columns to genes.

        Raises:
            ValueError: If the model does not have a logger attribute.
            ValueError: If the model does not have a global_step attribute.

        Returns:
            AnnData: The annotated data matrix with embedded cell representations.
            List[str]: List of gene names used in the embedding.
            np.ndarray: The predicted expression values if sample"none".
            dict: Additional metrics and information from the embedding process.
        """
        # one of "all" "sample" "none"
        model.predict_mode = "none"
        model.eval()
        model.on_predict_epoch_start()
        device = model.device.type
        dtype = (
            torch.float16
            if isinstance(model.transformer, FlashTransformer)
            else model.dtype
        )
        if self.embedding_to_use == ["all"]:
            use = [
                i
                for i in adata.obsm.keys()
                if i.startswith("scprint_emb_") and i != "scprint_emb_other"
            ]
        else:
            use = self.embedding_to_use
        res = []
        with (
            torch.no_grad(),
            torch.autocast(device_type=device, dtype=dtype),
        ):
            gene_pos = torch.tensor(
                [model.genes.index(g) for g in self.genelist],
            ).to(device=device)
            gene_pos = gene_pos.unsqueeze(0).repeat_interleave(self.batch_size, 0)
            req_depth = torch.tensor(adata.X.sum(1)).squeeze(-1).to(device=device)

            for batch in tqdm(range(adata.shape[0] // self.batch_size + 1)):
                embeddings = []
                start = batch * self.batch_size
                end = min((batch + 1) * self.batch_size, adata.shape[0])
                for emb in use:
                    embeddings.append(
                        torch.tensor(adata.obsm[emb][start:end]).unsqueeze(1)
                    )
                embeddings = torch.concat(embeddings, dim=1).to(device=device)

                output = model._generate(
                    gene_pos=gene_pos[0 : end - start, :],
                    cell_embs=embeddings,
                    depth_mult=req_depth[start:end],
                    req_depth=req_depth[start:end],
                    metacell_token=None,
                )
                res.append(
                    torch.concat(
                        [
                            output["mean"].detach().cpu().unsqueeze(0),
                            output["disp"].detach().cpu().unsqueeze(0),
                            output["zero_logits"].detach().cpu().unsqueeze(0),
                        ]
                    )
                    if "disp" in output
                    else output["mean"].detach().cpu().unsqueeze(0)
                )
                torch.cuda.empty_cache()
        res = torch.concat(res, dim=1)
        pred_adata = AnnData(
            X=res[0, :, :].numpy(),
            obs=adata.obs.copy(),
            var=pd.DataFrame(
                index=pd.Index(self.genelist),
            ),
            layers=None
            if res.shape[1] == 1
            else {
                "disp": res[1, :, :].numpy(),
                "zero_logits": res[2, :, :].numpy(),
            },
        )
        return pred_adata
