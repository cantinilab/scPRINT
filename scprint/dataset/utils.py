import os
import lamindb as ln
import bionty as bt
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import median_abs_deviation


def validate(adata, lb, organism):
    """
    validate _summary_

    Args:
        adata (_type_): _description_
        lb (_type_): _description_
        organism (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    organism = lb.Organism.filter(ontology_id=organism).one().name
    lb.settings.organism = organism

    if adata.var.index.duplicated().any():
        raise ValueError("Duplicate gene names found in adata.var.index")
    if adata.obs.index.duplicated().any():
        raise ValueError("Duplicate cell names found in adata.obs.index")
    for val in [
        "self_reported_ethnicity_ontology_term_id",
        "organism_ontology_term_id",
        "disease_ontology_term_id",
        "cell_type_ontology_term_id",
        "development_stage_ontology_term_id",
        "tissue_ontology_term_id",
        "assay_ontology_term_id",
    ]:
        if val not in adata.obs.columns:
            raise ValueError(
                f"Column '{val}' is missing in the provided anndata object."
            )
    bionty_source = lb.BiontySource.filter(
        entity="DevelopmentalStage", organism=organism
    ).one()

    if not lb.Ethnicity.validate(
        adata.obs["self_reported_ethnicity_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid ethnicity ontology term id found")
    if not lb.Organism.validate(
        adata.obs["organism_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid organism ontology term id found")
    if not lb.Phenotype.validate(
        adata.obs["sex_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid sex ontology term id found")
    if not lb.Disease.validate(
        adata.obs["disease_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid disease ontology term id found")
    if not lb.CellType.validate(
        adata.obs["cell_type_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid cell type ontology term id found")
    if (
        not lb.DevelopmentalStage.filter(bionty_source=bionty_source)
        .validate(adata.obs["development_stage_ontology_term_id"], field="ontology_id")
        .all()
    ):
        raise ValueError("Invalid dev stage ontology term id found")
    if not lb.Tissue.validate(
        adata.obs["tissue_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid tissue ontology term id found")
    if not lb.ExperimentalFactor.validate(
        adata.obs["assay_ontology_term_id"], field="ontology_id"
    ).all():
        raise ValueError("Invalid assay ontology term id found")
    if (
        not lb.Gene.filter(organism=lb.settings.organism)
        .validate(adata.var.index, field="ensembl_gene_id")
        .all()
    ):
        raise ValueError("Invalid gene ensembl id found")
    return True


def get_all_ancestors(val, df):
    if val not in df.index:
        return set()
    parents = df.loc[val].parents__ontology_id
    if parents is None or len(parents) == 0:
        return set()
    else:
        return set.union(set(parents), *[get_all_ancestors(val, df) for val in parents])


def get_ancestry_mapping(all_elem, onto_df):
    ancestors = {}
    full_ancestors = set()
    for val in all_elem:
        ancestors[val] = get_all_ancestors(val, onto_df) - set([val])

    for val in ancestors.values():
        full_ancestors |= set(val)
    # removing ancestors that are not in our datasets
    full_ancestors = full_ancestors & set(ancestors.keys())
    leafs = set(all_elem) - full_ancestors
    full_ancestors = full_ancestors - leafs

    groupings = {}
    for val in full_ancestors:
        groupings[val] = set()
    for leaf in leafs:
        for ancestor in ancestors[leaf]:
            if ancestor in full_ancestors:
                groupings[ancestor].add(leaf)

    return groupings, full_ancestors, leafs


def load_dataset_local(
    lb, remote_dataset, download_folder, name, description, use_cache=True, only=None
):
    saved_files = []
    default_storage = ln.Storage.filter(root=ln.settings.storage.as_posix()).one()
    files = (
        remote_dataset.files.all()
        if not only
        else remote_dataset.files.all()[only[0] : only[1]]
    )

    for file in files:
        organism = list(set([i.ontology_id for i in file.organism.all()]))
        if len(organism) > 1:
            print(organism)
            print("Multiple organisms detected")
            continue
        if len(organism) == 0:
            print("No organism detected")
            continue
        lb.settings.organism = organism[0]
        file.save()
        # if location already has a file, don't save again
        if use_cache and os.path.exists(os.path.expanduser(download_folder + file.key)):
            print(f"File {file.key} already exists in storage")
        else:
            file.path.download_to(download_folder + file.key)
        file.storage = default_storage
        file.save()
        saved_files.append(file)
    dataset = ln.Dataset(saved_files, name=name, description=description)
    dataset.save()
    return dataset


def populate_my_ontology(
    lb,
    organisms=["NCBITaxon:10090", "NCBITaxon:9606"],
    sex=["PATO:0000384", "PATO:0000383"],
    celltypes=[],
    ethnicities=[],
    assays=[],
    tissues=[],
    diseases=[],
    dev_stages=[],
):
    """
    creates a local version of the lamin ontologies and add the required missing values in base ontologies

    run this function just one for each new lamin storage

    erase everything with lb.$ontology.filter().delete()

    add whatever value you need afterward like it is done here with lb.$ontology(name="ddd", ontology_id="ddddd").save()
    # df["assay_ontology_term_id"].unique()
    """
    names = bt.CellType().df().index if not celltypes else celltypes
    records = lb.CellType.from_values(names, field=lb.CellType.ontology_id)
    ln.save(records)
    lb.CellType(name="unknown", ontology_id="unknown").save()
    # Organism
    names = bt.Organism().df().index if not organisms else organisms
    records = lb.Organism.from_values(names, field=lb.Organism.ontology_id)
    ln.save(records)
    lb.Organism(name="unknown", ontology_id="unknown").save()
    # Phenotype
    name = bt.Phenotype().df().index if not sex else sex
    records = lb.Phenotype.from_values(name, field=lb.Phenotype.ontology_id)
    ln.save(records)
    lb.Phenotype(name="unknown", ontology_id="unknown").save()
    # ethnicity
    names = bt.Ethnicity().df().index if not ethnicities else ethnicities
    records = lb.Ethnicity.from_values(names, field=lb.Ethnicity.ontology_id)
    ln.save(records)
    lb.Ethnicity(
        name="unknown", ontology_id="unknown"
    ).save()  # multi ethnic will have to get renamed
    # ExperimentalFactor
    names = bt.ExperimentalFactor().df().index if not assays else assays
    records = lb.ExperimentalFactor.from_values(
        names, field=lb.ExperimentalFactor.ontology_id
    )
    ln.save(records)
    lb.ExperimentalFactor(name="SMART-Seq v4", ontology_id="EFO:0700016").save()
    lb.ExperimentalFactor(name="unknown", ontology_id="unknown").save()
    lookup = lb.ExperimentalFactor.lookup()
    lookup.smart_seq_v4.parents.add(lookup.smart_like)
    # Tissue
    names = bt.Tissue().df().index if not tissues else tissues
    records = lb.Tissue.from_values(names, field=lb.Tissue.ontology_id)
    ln.save(records)
    lb.Tissue(name="unknown", ontology_id="unknown").save()
    # DevelopmentalStage
    names = bt.DevelopmentalStage().df().index if not dev_stages else dev_stages
    records = lb.DevelopmentalStage.from_values(
        names, field=lb.DevelopmentalStage.ontology_id
    )
    ln.save(records)
    lb.DevelopmentalStage(name="unknown", ontology_id="unknown").save()
    # Disease
    names = bt.Disease().df().index if not diseases else diseases
    records = lb.Disease.from_values(names, field=lb.Disease.ontology_id)
    ln.save(records)
    lb.Disease(name="normal", ontology_id="PATO:0000461").save()
    lb.Disease(name="unknown", ontology_id="unknown").save()
    # genes
    for organism in organisms:
        # convert onto to name
        organism = lb.Organism.filter(ontology_id=organism).one().name
        names = bt.Gene(organism=organism).df()["ensembl_gene_id"]
        records = lb.Gene.from_values(names, field="ensembl_gene_id")
        ln.save(records)


def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier


def length_normalize(adata, gene_lengths):
    adata.X = csr_matrix((adata.X.T / gene_lengths).T)
    return adata
