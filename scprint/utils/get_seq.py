import ftplib
import logging
import os
from typing import List, Optional, Union

import numpy as np
from Bio import SeqIO
import pandas as pd

# Constants
from gget.constants import ENSEMBL_REST_API, UNIPROT_REST_API
from gget.gget_info import info

# Custom functions
from gget.utils import get_uniprot_seqs, rest_query

# Add and format time stamp in logging messages
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%c",
)
# Mute numexpr threads info
logging.getLogger("numexpr").setLevel(logging.WARNING)


def list_files(ftp, match=""):
    files = ftp.nlst()
    return [file for file in files if file.endswith(match)]


def load_fasta_species(
    species: str = "homo_sapiens",
    output_path: str = "/tmp/data/fasta/",
    cache: bool = True,
) -> None:
    """
    Downloads and caches FASTA files for a given species from the Ensembl FTP server.

    Args:
        species (str, optional): The species name for which to download FASTA files. Defaults to "homo_sapiens".
        output_path (str, optional): The local directory path where the FASTA files will be saved. Defaults to "/tmp/data/fasta/".
        cache (bool, optional): If True, use cached files if they exist. If False, re-download the files. Defaults to True.
    """
    ftp = ftplib.FTP("ftp.ensembl.org")
    ftp.login()
    try:
        ftp.cwd("/pub/release-110/fasta/" + species + "/pep/")
        types = "animals"
    except ftplib.error_perm:
        try:
            ftp = ftplib.FTP("ftp.ensemblgenomes.ebi.ac.uk")
            ftp.login()
            ftp.cwd("/pub/plants/release-60/fasta/" + species + "/pep/")
            types = "plants"
        except ftplib.error_perm:
            try:
                ftp.cwd("/pub/metazoa/release-60/fasta/" + species + "/pep/")
                types = "metazoa"
            except ftplib.error_perm:
                raise ValueError(
                    f"Species {species} not found in Ensembl or Ensembl Genomes."
                )
    file = list_files(ftp, ".all.fa.gz")[0]
    local_file_path = output_path + file
    if not os.path.exists(local_file_path) or not cache:
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        with open(local_file_path, "wb") as local_file:
            ftp.retrbinary("RETR " + file, local_file.write)
    if types == "animals":
        ftp.cwd("/pub/release-110/fasta/" + species + "/ncrna/")
    elif types == "plants":
        ftp.cwd("/pub/plants/release-60/fasta/" + species + "/ncrna/")
    file = list_files(ftp, ".ncrna.fa.gz")[0]
    local_file_path2 = output_path + file
    if not os.path.exists(local_file_path2) or not cache:
        with open(local_file_path2, "wb") as local_file:
            ftp.retrbinary("RETR " + file, local_file.write)
    ftp.quit()
    return local_file_path, local_file_path2


def subset_fasta(
    gene_tosubset: set = None,
    fasta_path: str = None,
    subfasta_path: str = "./data/fasta/subset.fa",
    drop_unknown_seq: bool = True,
    subset_protein_coding: bool = True,
) -> set:
    """
    subset_fasta: creates a new fasta file with only the sequence which names contain one of gene_names

    Args:
        gene_tosubset (set): A set of gene names to subset from the original FASTA file.
        fasta_path (str): The path to the original FASTA file.
        subfasta_path (str, optional): The path to save the subsetted FASTA file. Defaults to "./data/fasta/subset.fa".
        drop_unknown_seq (bool, optional): If True, drop sequences containing unknown amino acids (denoted by '*'). Defaults to True.
        subset_protein_coding (bool, optional): If True, subset only protein coding genes. Defaults to True.
    Returns:
        set: A set of gene names that were found and included in the subsetted FASTA file.

    Raises:
        ValueError: If a gene name does not start with "ENS".
    """
    dup = set()
    weird = 0
    genes_found = set()
    gene_tosubset = set(gene_tosubset) if gene_tosubset else []
    names = []
    with (
        open(fasta_path, "r") as original_fasta,
        open(subfasta_path, "w") as subset_fasta,
    ):
        for record in SeqIO.parse(original_fasta, "fasta"):
            gene_name = (
                record.description.split(" gene:")[1].split(" ")[0].split(".")[0]
            )
            gene_biotype = record.description.split("gene_biotype:")[1].split(" ")[0]
            if "gene_symbol:" not in record.description:
                gene_symbol = gene_name
            else:
                gene_symbol = record.description.split("gene_symbol:")[1].split(" ")[0]
            if "description:" not in record.description:
                description = ""
            else:
                description = record.description.split("description:")[1]
            names.append([gene_name, gene_biotype, record.id, gene_symbol, description])
            if subset_protein_coding and gene_biotype != "protein_coding":
                continue
            if len(gene_tosubset) == 0 or gene_name in gene_tosubset:
                if drop_unknown_seq:
                    if "*" in record.seq:
                        weird += 1

                        continue
                if gene_name in genes_found:
                    dup.add(gene_name)
                    continue
                record.description = ""
                record.id = gene_name
                SeqIO.write(record, subset_fasta, "fasta")
                genes_found.add(gene_name)
    print(len(dup), " genes had duplicates")
    print("dropped", weird, "weird sequences")
    return genes_found, pd.DataFrame(
        names, columns=["name", "biotype", "ensembl_id", "gene_symbol", "description"]
    )


def seq(
    ens_ids: Union[str, List[str]],
    translate: bool = False,
    isoforms: bool = False,
    parallel: bool = True,
    save: bool = False,
    transcribe: Optional[bool] = None,
    seqtype: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Fetch nucleotide or amino acid sequence (FASTA) of a gene (and all its isoforms) or transcript by Ensembl, WormBase, or FlyBase ID.

    Args:
        ens_ids (Union[str, List[str]]): One or more Ensembl IDs (passed as string or list of strings).
                                         Also supports WormBase and FlyBase IDs.
        translate (bool, optional): Defines whether nucleotide or amino acid sequences are returned.
                                    Defaults to False (returns nucleotide sequences).
                                    Nucleotide sequences are fetched from the Ensembl REST API server.
                                    Amino acid sequences are fetched from the UniProt REST API server.
        isoforms (bool, optional): If True, returns the sequences of all known transcripts. Defaults to False.
                                   (Only for gene IDs.)
        parallel (bool, optional): If True, fetches sequences in parallel. Defaults to True.
        save (bool, optional): If True, saves output FASTA to current directory. Defaults to False.
        transcribe (bool, optional): Deprecated. Use 'translate' instead.
        seqtype (str, optional): Deprecated. Use 'translate' instead.
        verbose (bool, optional): If True, prints progress information. Defaults to True.

    Returns:
        List[str]: A list containing the requested sequences, or a FASTA file if 'save' is True.

    Raises:
        ValueError: If an invalid Ensembl ID is provided.
    """
    # Handle deprecated arguments
    if seqtype:
        logging.error(
            "'seqtype' argument deprecated! Please use True/False argument 'translate' instead."
        )
        return
    if transcribe:
        translate = transcribe

    ## Clean up arguments
    # Clean up Ensembl IDs
    # If single Ensembl ID passed as string, convert to list
    if type(ens_ids) is str:
        ens_ids = [ens_ids]
    # Remove Ensembl ID version if passed
    ens_ids_clean = []
    temp = 0
    for ensembl_ID in ens_ids:
        # But only for Ensembl ID (and not for flybase/wormbase IDs)
        if ensembl_ID.startswith("ENS"):
            ens_ids_clean.append(ensembl_ID.split(".")[0])

            if "." in ensembl_ID and temp == 0:
                if verbose:
                    logging.info(
                        "We noticed that you may have passed a version number with your Ensembl ID.\n"
                        "Please note that gget seq will return information linked to the latest Ensembl ID version."
                    )
                temp = +1

        else:
            ens_ids_clean.append(ensembl_ID)

    # Initiate empty 'fasta'
    fasta = []

    ## Fetch nucleotide sequece
    if translate is False:
        # Define Ensembl REST API server
        server = ENSEMBL_REST_API
        # Define type of returned content from REST
        content_type = "application/json"

        # Initiate dictionary to save results for all IDs in
        master_dict = {}

        # Query REST APIs from https://rest.ensembl.org/
        for ensembl_ID in ens_ids_clean:
            # Create dict to save query results
            results_dict = {ensembl_ID: {}}

            # If isoforms False, just fetch sequences of passed Ensembl ID
            if isoforms is False:
                # sequence/id/ query: Request sequence by stable identifier
                query = "sequence/id/" + ensembl_ID + "?"

                # Try if query valid
                try:
                    # Submit query; this will throw RuntimeError if ID not found
                    df_temp = rest_query(server, query, content_type)

                    # Delete superfluous entries
                    keys_to_delete = ["query", "id", "version", "molecule"]
                    for key in keys_to_delete:
                        # Pop keys, None -> do not raise an error if key to delete not found
                        df_temp.pop(key, None)

                    # Add results to main dict
                    results_dict[ensembl_ID].update({"seq": df_temp})

                    if verbose:
                        logging.info(
                            f"Requesting nucleotide sequence of {ensembl_ID} from Ensembl."
                        )

                except RuntimeError:
                    logging.error(
                        f"ID {ensembl_ID} not found. Please double-check spelling/arguments and try again."
                    )

            # If isoforms true, fetch sequences of isoforms instead
            if isoforms is True:
                # Get ID type (gene, transcript, ...) using gget info
                info_df = info(
                    ensembl_ID, verbose=False, pdb=False, ncbi=False, uniprot=False
                )

                # Check if Ensembl ID was found
                if isinstance(info_df, type(None)):
                    logging.warning(
                        f"ID '{ensembl_ID}' not found. Please double-check spelling/arguments and try again."
                    )
                    continue

                ens_ID_type = info_df.loc[ensembl_ID]["object_type"]

                # If the ID is a gene, get the IDs of all its transcripts
                if ens_ID_type == "Gene":
                    if verbose:
                        logging.info(
                            f"Requesting nucleotide sequences of all transcripts of {ensembl_ID} from Ensembl."
                        )

                    for transcipt_id in info_df.loc[ensembl_ID]["all_transcripts"]:
                        # Remove version number for Ensembl IDs (not for flybase/wormbase IDs)
                        if transcipt_id.startswith("ENS"):
                            transcipt_id = transcipt_id.split(".")[0]

                        # Try if query is valid
                        try:
                            # Define the REST query
                            query = "sequence/id/" + transcipt_id + "?"
                            # Submit query
                            df_temp = rest_query(server, query, content_type)

                            # Delete superfluous entries
                            keys_to_delete = ["query", "version", "molecule"]
                            for key in keys_to_delete:
                                # Pop keys, None -> do not raise an error if key to delete not found
                                df_temp.pop(key, None)

                            # Add results to main dict
                            results_dict[ensembl_ID].update(
                                {f"{transcipt_id}": df_temp}
                            )

                        except RuntimeError:
                            logging.error(
                                f"ID {transcipt_id} not found. "
                                "Please double-check spelling/arguments and try again."
                            )

                # If isoform true, but ID is not a gene; ignore the isoform parameter
                else:
                    # Try if query is valid
                    try:
                        # Define the REST query
                        query = "sequence/id/" + ensembl_ID + "?"

                        # Submit query
                        df_temp = rest_query(server, query, content_type)

                        # Delete superfluous entries
                        keys_to_delete = ["query", "id", "version", "molecule"]
                        for key in keys_to_delete:
                            # Pop keys, None -> do not raise an error if key to delete not found
                            df_temp.pop(key, None)

                        # Add results to main dict
                        results_dict[ensembl_ID].update({"seq": df_temp})

                        logging.info(
                            f"Requesting nucleotide sequence of {ensembl_ID} from Ensembl."
                        )
                        logging.warning("The isoform option only applies to gene IDs.")

                    except RuntimeError:
                        logging.error(
                            f"ID {ensembl_ID} not found. "
                            "Please double-check spelling/arguments and try again."
                        )

            # Add results to master dict
            master_dict.update(results_dict)

        # Build FASTA file
        for ens_ID in master_dict:
            for key in master_dict[ens_ID].keys():
                if key == "seq":
                    fasta.append(">" + ens_ID + " " + master_dict[ens_ID][key]["desc"])
                    fasta.append(master_dict[ens_ID][key]["seq"])
                else:
                    fasta.append(
                        ">"
                        + master_dict[ens_ID][key]["id"]
                        + " "
                        + master_dict[ens_ID][key]["desc"]
                    )
                    fasta.append(master_dict[ens_ID][key]["seq"])

    ## Fetch amino acid sequences from UniProt
    if translate is True:
        if isoforms is False:
            # List to collect transcript IDs
            trans_ids = []

            # Get ID type (gene, transcript, ...) using gget info
            info_df = info(
                ens_ids_clean, verbose=False, pdb=False, ncbi=False, uniprot=False
            )

            # Check that Ensembl ID was found
            missing = set(ens_ids_clean) - set(info_df.index.values)
            if len(missing) > 0:
                logging.warning(
                    f"{str(missing)} IDs not found. Please double-check spelling/arguments."
                )

            ens_ID_type = info_df.loc[ens_ids_clean[0]]["object_type"]

            # If the ID is a gene, use the ID of its canonical transcript
            if ens_ID_type == "Gene":
                # Get ID of canonical transcript
                for ensembl_ID in info_df.index.values:
                    can_trans = info_df.loc[ensembl_ID]["canonical_transcript"]

                    if ensembl_ID.startswith("ENS"):
                        # Remove Ensembl ID version from transcript IDs and append to transcript IDs list
                        temp_trans_id = can_trans.split(".")[0]
                        trans_ids.append(temp_trans_id)

                    elif ensembl_ID.startswith("WB"):
                        # Remove added "." at the end of transcript IDs
                        temp_trans_id = ".".join(can_trans.split(".")[:-1])
                        # # For WormBase transcript IDs, also remove the version number for submission to UniProt API
                        # temp_trans_id = ".".join(temp_trans_id1.split(".")[:-1])
                        trans_ids.append(temp_trans_id)

                    else:
                        # Remove added "." at the end of other transcript IDs
                        temp_trans_id = ".".join(can_trans.split(".")[:-1])
                        trans_ids.append(temp_trans_id)

                    if verbose:
                        logging.info(
                            f"Requesting amino acid sequence of the canonical transcript {temp_trans_id} of gene {ensembl_ID} from UniProt."
                        )

            # If the ID is a transcript, append the ID directly
            elif ens_ID_type == "Transcript":
                # # For WormBase transcript IDs, remove the version number for submission to UniProt API
                # if ensembl_ID.startswith("T"):
                #     trans_ids.append(".".join(ensembl_ID.split(".")[:-1]))
                # else:
                trans_ids = ens_ids_clean

                if verbose:
                    logging.info(
                        f"Requesting amino acid sequence of {trans_ids} from UniProt."
                    )

            else:
                logging.warning(
                    "ensembl_IDs not recognized as either a gene or transcript ID. It will not be included in the UniProt query."
                )

            # Fetch the amino acid sequences of the transcript Ensembl IDs
            df_uniprot = get_uniprot_seqs(UNIPROT_REST_API, trans_ids)
            # Add info_df.loc[ensembl_ID] to df_uniprot by joining on "canonical_transcript" / "gene_name" respectively
            info_df.set_index("canonical_transcript", inplace=True)

            df_uniprot.loc[:, "gene_id"] = info_df.loc[
                df_uniprot["query"], "gene_name"
            ].values

        if isoforms is True:
            # List to collect transcript IDs
            trans_ids = []

            for ensembl_ID in ens_ids_clean:
                # Get ID type (gene, transcript, ...) using gget info
                info_df = info(
                    ensembl_ID, verbose=False, pdb=False, ncbi=False, uniprot=False
                )

                # Check that Ensembl ID was found
                if isinstance(info_df, type(None)):
                    logging.warning(
                        f"ID '{ensembl_ID}' not found. Please double-check spelling/arguments."
                    )
                    continue

                ens_ID_type = info_df.loc[ensembl_ID]["object_type"]

                # If the ID is a gene, get the IDs of all isoforms
                if ens_ID_type == "Gene":
                    # Get the IDs of all transcripts from the gget info results
                    for transcipt_id in info_df.loc[ensembl_ID]["all_transcripts"]:
                        if ensembl_ID.startswith("ENS"):
                            # Append transcript ID (without Ensembl version number) to list of transcripts to fetch
                            trans_ids.append(transcipt_id.split(".")[0])

                        # elif ensembl_ID.startswith("WB"):
                        #     # For WormBase transcript IDs, remove the version number for submission to UniProt API
                        #     temp_trans_id = ".".join(transcipt_id.split(".")[:-1])
                        #     trans_ids.append(temp_trans_id)

                        else:
                            # Note: No need to remove the added "." at the end of unversioned transcripts here, because "all_transcripts" are returned without it
                            trans_ids.append(transcipt_id)

                    if verbose:
                        logging.info(
                            f"Requesting amino acid sequences of all transcripts of gene {ensembl_ID} from UniProt."
                        )

                elif ens_ID_type == "Transcript":
                    # # For WormBase transcript IDs, remove the version number for submission to UniProt API
                    # if ensembl_ID.startswith("T"):
                    #     trans_ids.append(".".join(ensembl_ID.split(".")[:-1]))

                    # else:
                    trans_ids.append(ensembl_ID)

                    if verbose:
                        logging.info(
                            f"Requesting amino acid sequence of {ensembl_ID} from UniProt."
                        )
                    logging.warning("The isoform option only applies to gene IDs.")

                else:
                    logging.warning(
                        f"{ensembl_ID} not recognized as either a gene or transcript ID. It will not be included in the UniProt query."
                    )

            # Fetch amino acid sequences of all isoforms from the UniProt REST API
            df_uniprot = get_uniprot_seqs(UNIPROT_REST_API, trans_ids)

        # Check if any results were found
        if len(df_uniprot) < 1:
            logging.error("No UniProt amino acid sequences were found for these ID(s).")

        else:
            # Build FASTA file from UniProt results
            for (
                uniprot_id,
                query_ensembl_id,
                gene_name,
                organism,
                sequence_length,
                uniprot_seq,
            ) in zip(
                df_uniprot["uniprot_id"].values,
                df_uniprot["query"].values,
                df_uniprot["gene_name"].values,
                df_uniprot["gene_id"].values,
                df_uniprot["organism"].values,
                df_uniprot["sequence_length"].values,
                df_uniprot["sequence"].values,
            ):
                fasta.append(
                    ">"
                    + str(query_ensembl_id)
                    + " uniprot_id: "
                    + str(uniprot_id)
                    + " ensembl_id: "
                    + str(query_ensembl_id)
                    + " gene_name: "
                    + str(gene_name)
                    + " organism: "
                    + str(organism)
                    + " sequence_length: "
                    + str(sequence_length)
                )
                fasta.append(str(uniprot_seq))

    # Save
    if save:
        file = open("gget_seq_results.fa", "w")
        for element in fasta:
            file.write(element + "\n")
        file.close()
        # missed samples
        return (set(trans_ids) - set(df_uniprot["query"].values)) | set(missing)

    return fasta
