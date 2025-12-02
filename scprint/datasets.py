EMBEDDING_DATASETS = {
    "lung": "https://figshare.com/ndownloader/files/24539942",
    "pancreas": "https://figshare.com/ndownloader/files/24539828",
    "kidney": "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",
    # "gtex": "https://datasets.cellxgene.cziscience.com/661d5ec2-ca57-413c-8374-f49b0054ddba.h5ad",
    "bone_marrow_5batch": "https://datasets.cellxgene.cziscience.com/b2eca8f3-b461-45fd-8639-890bbbf050aa.h5ad",
}

DENOISE_DATASETS = {
    "intestine": "https://datasets.cellxgene.cziscience.com/d9a99b4a-3755-47c4-8eb5-09821ffbde17.h5ad",  # R4ZHoQegxXdSFNFY5LGe in my case # R4ZHoQegxXdSFNFYMaIQ
    "retina": "https://datasets.cellxgene.cziscience.com/53bd4177-79c6-40c8-b84d-ff300dcf1b5b.h5ad",  # gNNpgpo6gATjuxTE7CCp in my case # gNNpgpo6gATjuxTEW2mj
    "kidney": "https://datasets.cellxgene.cziscience.com/01bc7039-961f-4c24-b407-d535a2a7ba2c.h5ad",  # in my case d0JqVUfPuonxM3K3USwN
    "glio_smart_highdepth": "https://datasets.cellxgene.cziscience.com/6ec440b4-542a-4022-ac01-56f812e25593.h5ad",  # in my case s8x0Idi587LQtXCo0Pif
    "lung_smart": "https://datasets.cellxgene.cziscience.com/6ebba0e0-a159-406f-8095-451115673a2c.h5ad",  # in my case NwFOBmT6emiJaeyrEc66
}


def download_datasets():
    """
    Download the required test datasets in case you are in a server without internet access.
    """
    import scanpy as sc

    for dataset_name, url in EMBEDDING_DATASETS.items():
        sc.read(
            "../data/"
            + dataset_name
            + (".h5ad" if not dataset_name.endswith(".h5ad") else ""),
            backup_url=url,
        )
    for dataset_name, url in DENOISE_DATASETS.items():
        sc.read(
            "../data/"
            + dataset_name
            + (".h5ad" if not dataset_name.endswith(".h5ad") else ""),
            backup_url=url,
        )
