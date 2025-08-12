import functools
import io
import json
import os
import random
import subprocess
import urllib.request
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import bionty as bt
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from biomart import BiomartServer
from IPython import get_ipython


def run_command(command: str, **kwargs):
    """
    run_command runs a command in the shell and prints the output.

    Args:
        command (str): The command to be executed in the shell.

    Returns:
        int: The return code of the command executed.
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, **kwargs)
    while True:
        if process.poll() is not None:
            break
        output = process.stdout.readline()
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def fileToList(filename: str, strconv: callable = lambda x: x) -> list:
    """
    loads an input file with a\\n b\\n.. into a list [a,b,..]

    Args:
        input_str (str): The input string to be completed.

    Returns:
        str: The completed string with 'complete' appended.
    """
    with open(filename) as f:
        return [strconv(val[:-1]) for val in f.readlines()]


def listToFile(
    li: List[str], filename: str, strconv: callable = lambda x: str(x)
) -> None:
    """
    listToFile loads a list with [a,b,..] into an input file a\\n b\\n..

    Args:
        l (list): The list of elements to be written to the file.
        filename (str): The name of the file where the list will be written.
        strconv (callable, optional): A function to convert each element of the list to a string. Defaults to str.

    Returns:
        None
    """
    with open(filename, "w") as f:
        for item in li:
            f.write("%s\n" % strconv(item))


def set_seed(seed: int = 42):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def createFoldersFor(filepath):
    """
    will recursively create folders if needed until having all the folders required to save the file in this filepath
    """
    prevval = ""
    for val in os.path.expanduser(filepath).split("/")[:-1]:
        prevval += val + "/"
        if not os.path.exists(prevval):
            os.mkdir(prevval)


def category_str2int(category_strs: List[str]) -> List[int]:
    """
    category_str2int converts a list of category strings to a list of category integers.

    Args:
        category_strs (List[str]): A list of category strings to be converted.

    Returns:
        List[int]: A list of integers corresponding to the input category strings.
    """
    set_category_strs = set(category_strs)
    name2id = {name: i for i, name in enumerate(set_category_strs)}
    return [name2id[name] for name in category_strs]


def isnotebook() -> bool:
    """check whether excuting in jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def get_free_gpu():
    """
    get_free_gpu finds the GPU with the most free memory using nvidia-smi.

    Returns:
        int: The index of the GPU with the most free memory.
    """
    import subprocess
    import sys
    from io import StringIO

    gpu_stats = subprocess.check_output(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=memory.used,memory.free",
        ]
    ).decode("utf-8")
    gpu_df = pd.read_csv(
        StringIO(gpu_stats), names=["memory.used", "memory.free"], skiprows=1
    )
    print("GPU usage:\n{}".format(gpu_df))
    gpu_df["memory.free"] = gpu_df["memory.free"].map(lambda x: int(x.rstrip(" [MiB]")))
    idx = gpu_df["memory.free"].idxmax()
    print(
        "Find free GPU{} with {} free MiB".format(idx, gpu_df.iloc[idx]["memory.free"])
    )

    return idx


def get_git_commit():
    """
    get_git_commit gets the current git commit hash.

    Returns:
        str: The current git commit
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids
