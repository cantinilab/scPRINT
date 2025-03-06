import os

import pandas as pd
import torch

# from RNABERT import RNABERT
from torch.nn import AdaptiveAvgPool1d

from scprint import utils

from .protein_embedder import ESM2
from tqdm import tqdm
import numpy as np


def protein_embeddings_generator(
    genedf: pd.DataFrame = None,
    organism: str = "homo_sapiens",  # mus_musculus,
    cache: bool = True,
    fasta_path: str = "/tmp/data/fasta/",
    embedding_size: int = 512,
    embedder: str = "esm3",  # or glm2
    cuda: bool = True,
):
    """
    protein_embeddings_generator embed a set of genes using fasta file and LLMs

    Args:
        genedf (pd.DataFrame): A DataFrame containing gene information.
        organism (str, optional): The organism to which the genes belong. Defaults to "homo_sapiens".
        cache (bool, optional): If True, the function will use cached data if available. Defaults to True.
        fasta_path (str, optional): The path to the directory where the fasta files are stored. Defaults to "/tmp/data/fasta/".
        embedding_size (int, optional): The size of the embeddings to be generated. Defaults to 512.
    Returns:
        pd.DataFrame: Returns a DataFrame containing the protein embeddings.
        pd.DataFrame: Returns the naming dataframe.
    """
    # given a gene file and organism
    # load the organism fasta if not already done
    fasta_path_pep, fasta_path_ncrna = utils.load_fasta_species(
        species=organism, output_path=fasta_path, cache=cache
    )
    # subset the fasta
    fasta_name = fasta_path_pep.split("/")[-1]
    utils.utils.run_command(["gunzip", fasta_path_pep])
    protgenedf = (
        genedf[genedf["biotype"] == "protein_coding"] if genedf is not None else None
    )
    found, naming_df = utils.subset_fasta(
        protgenedf.index.tolist() if protgenedf is not None else None,
        subfasta_path=fasta_path + "subset.fa",
        fasta_path=fasta_path + fasta_name[:-3],
        drop_unknown_seq=True,
    )
    if embedder == "esm2":
        prot_embedder = ESM2()
        prot_embeddings = prot_embedder(
            fasta_path + "subset.fa", output_folder=fasta_path + "esm_out/", cache=cache
        )
    elif embedder == "esm3":
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        from Bio import SeqIO

        prot_embeddings = []
        names = []
        client = ESMC.from_pretrained("esmc_600m").to("cuda" if cuda else "cpu")
        conf = LogitsConfig(sequence=True, return_embeddings=True)
        with (
            open(fasta_path + "subset.fa", "r") as fasta,
        ):
            for record in tqdm(SeqIO.parse(fasta, "fasta")):
                protein = ESMProtein(sequence=str(record.seq))
                protein_tensor = client.encode(protein)
                logits_output = client.logits(protein_tensor, conf)
                prot_embeddings.append(
                    logits_output.embeddings[0].mean(0).cpu().numpy().tolist()
                )
                names.append(record.id)
    else:
        raise ValueError(f"Embedder {embedder} not supported")
    # load the data and erase / zip the rest
    # utils.utils.run_command(["gzip", fasta_path + fasta_name[:-3]])
    # return the embedding and gene file
    # TODO: to redebug
    # do the same for RNA
    # rnagenedf = genedf[genedf["biotype"] != "protein_coding"]
    # fasta_file = next(
    #    file for file in os.listdir(fasta_path) if file.endswith(".ncrna.fa.gz")
    # )
    # utils.utils.run_command(["gunzip", fasta_path + fasta_file])
    # utils.subset_fasta(
    #    rnagenedf["ensembl_gene_id"].tolist(),
    #    subfasta_path=fasta_path + "subset.ncrna.fa",
    #    fasta_path=fasta_path + fasta_file[:-3],
    #    drop_unknown_seq=True,
    # )
    # rna_embedder = RNABERT()
    # rna_embeddings = rna_embedder(fasta_path + "subset.ncrna.fa")
    ## Check if the sizes of the cembeddings are not the same
    # utils.utils.run_command(["gzip", fasta_path + fasta_file[:-3]])
    #
    m = AdaptiveAvgPool1d(embedding_size)
    prot_embeddings = pd.DataFrame(
        data=m(torch.tensor(np.array(prot_embeddings))), index=names
    )
    # rna_embeddings = pd.DataFrame(
    #    data=m(torch.tensor(rna_embeddings.values)), index=rna_embeddings.index
    # )
    # Concatenate the embeddings
    return prot_embeddings, naming_df  # pd.concat([prot_embeddings, rna_embeddings])
