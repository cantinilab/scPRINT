import json
import os
import random
import subprocess
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import List

import bionty as bt
import colorcet as cc
import lamindb as ln
import numpy as np
import pandas as pd
import torch
from bokeh.io import export_svg, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, TextInput
from bokeh.models.annotations import LabelSet
from bokeh.palettes import Category10, Category20
from bokeh.plotting import figure, save
from IPython import get_ipython

# What pops up on hover?
from matplotlib import pyplot as plt


def run_command(command: str, **kwargs) -> int:
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
        filename (str): The path to the file to be loaded.
        strconv (callable): Function to convert each line. Defaults to identity function.

    Returns:
        list: The list of converted values from the file.
    """
    with open(filename) as f:
        return [strconv(val[:-1]) for val in f.readlines()]


def listToFile(
    li: List[str], filename: str, strconv: callable = lambda x: str(x)
) -> None:
    """
    listToFile loads a list with [a,b,..] into an input file a\\n b\\n..

    Args:
        li (list): The list of elements to be written to the file.
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


def get_free_gpu() -> int:
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


def get_git_commit() -> str:
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


"""Helper functions related to subsetting AnnData objects based on the file format 
specifications in the .config.vsh.yaml and slot mapping overrides."""


# create new anndata objects according to api spec
def subset_h5ad_by_format(adata, config, arg_name, field_rename_dict={}):
    """Create new anndata object according to slot info specifications.

    Arguments:
    adata -- An AnnData object to subset (required)
    config -- A Viash config object as read by openproblems.project.read_viash_config (required)
    arg_name -- The name of the argument in the config file that specifies the output format (required)
    field_rename_dict -- A mapping between the slots of the source h5ad and the slots of the destination h5ad.
      Example of slot_mapping:
        ```
        slot_mapping = {
          "layers": {
            "counts": par["layer_counts"],
          },
          "obs": {
            "cell_type": par["obs_cell_type"],
            "batch": par["obs_batch"],
          }
        }
    """
    import anndata as ad
    import pandas as pd

    assert isinstance(adata, ad.AnnData), "adata must be an AnnData object"
    assert isinstance(config, dict), "config must be a dictionary"

    # find argument
    arg = next(
        (x for x in config["all_arguments"] if x["clean_name"] == arg_name), None
    )
    assert arg, f"Argument '{arg_name}' not found in config"

    # find file format
    file_format = (arg.get("info") or {}).get("format")
    assert file_format, f"Argument '{arg_name}' has no .info.format"

    # find file format type
    file_format_type = file_format.get("type")
    assert file_format_type == "h5ad", "format must be a h5ad type"

    structs = ["layers", "obs", "var", "uns", "obsp", "obsm", "varp", "varm"]
    kwargs = {}

    for struct in structs:
        struct_format = file_format.get(struct, {})
        struct_rename = field_rename_dict.get(struct, {})

        # fetch data from adata
        data = {}
        for field_format in struct_format:
            dest_name = field_format["name"]
            # where to find the data. if the dest_name is in the rename dict, use the renamed name
            # as the source name, otherwise use the dest_name as the source name
            src_name = struct_rename.get(dest_name, dest_name)
            data[dest_name] = getattr(adata, struct)[src_name]

        if len(data) > 0:
            if struct in ["obs", "var"]:
                data = pd.concat(data, axis=1)
            kwargs[struct] = data
        elif struct in ["obs", "var"]:
            # if no columns need to be copied, we still need an 'obs' and a 'var'
            # to help determine the shape of the adata
            kwargs[struct] = getattr(adata, struct).iloc[:, []]

    return ad.AnnData(**kwargs)


def volcano(
    data,
    folder="",
    tohighlight=None,
    tooltips=[("gene", "@gene_id")],
    title="volcano plot",
    xlabel="log-fold change",
    ylabel="-log(Q)",
    maxvalue=100,
    searchbox=False,
    logfoldtohighlight=0.15,
    pvaltohighlight=0.1,
    showlabels=False,
):
    """
    Make an interactive volcano plot from Differential Expression analysis tools outputs

    Args:
    -----
        data: a df with rows genes and cols [log2FoldChange, pvalue, gene_id]
        folder: str of location where to save the plot, won't save if empty
        tohighlight: list[str] of genes to highlight in the plot
        tooltips: list[tuples(str,str)] if user wants tot specify another bokeh tooltip
        title: str plot title
        xlabel: str if user wants to specify the title of the x axis
        ylabel: str if user wants tot specify the title of the y axis
        maxvalue: float the max -log2(pvalue authorized usefull when managing inf vals)
        searchbox: bool whether or not to add a searchBox to interactively highlight genes
        logfoldtohighlight: float min logfoldchange when to diplay points
        pvaltohighlight: float min pvalue when to diplay points
        showlabels: bool whether or not to show a text above each datapoint with its label information

    Returns:
    --------
        The bokeh object
    """
    to_plot_not, to_plot_yes = selector(
        data,
        tohighlight if tohighlight is not None else [],
        logfoldtohighlight,
        pvaltohighlight,
    )
    hover = HoverTool(tooltips=tooltips, name="circles")

    # Create figure
    p = figure(title=title, width=650, height=450)

    p.xgrid.grid_line_color = "white"
    p.ygrid.grid_line_color = "white"
    p.xaxis.axis_label = xlabel
    p.yaxis.axis_label = ylabel

    # Add the hover tool
    p.add_tools(hover)
    p, source1 = add_points(
        p, to_plot_not, "log2FoldChange", "pvalue", color="#1a9641", maxvalue=maxvalue
    )
    p, source2 = add_points(
        p,
        to_plot_yes,
        "log2FoldChange",
        "pvalue",
        color="#fc8d59",
        alpha=0.6,
        outline=True,
        maxvalue=maxvalue,
    )
    if showlabels:
        labels = LabelSet(
            x="log2FoldChange",
            y="transformed_q",
            text_font_size="7pt",
            text="gene_id",
            level="glyph",
            x_offset=5,
            y_offset=5,
            source=source2,
            # renderers="canvas",
        )
        p.add_layout(labels)
    if searchbox:
        text = TextInput(title="text", value="gene")
        text.js_on_change(
            "value",
            CustomJS(
                args=dict(source=source1),
                code="""
                var data = source.data
                var value = cb_obj.value
                var gene_id = data.gene_id
                var a = -1
                for (let i=0; i < gene_id.length; i++) {
                    if ( gene_id[i]===value ) { a=i; console.log(i); data.size[i]=7; data.alpha[i]=1; data.color[i]='#fc8d59' }
                }
                source.data = data
                console.log(source)
                console.log(cb_obj)
                source.change.emit()
                console.log(source)
                """,
            ),
        )
        p = column(text, p)
    p.output_backend = "svg"
    if folder:
        save(p, folder + title.replace(" ", "_") + "_volcano.html")
        try:
            p.output_backend = "svg"
            export_svg(p, filename=folder + title.replace(" ", "_") + "_volcano.svg")
        except (RuntimeError, Exception) as e:
            print(f"Could not save SVG: {e}")
    try:
        show(p)
    except Exception as e:
        print(f"Could not show plot: {e}")
    return p


def add_points(p, df1, x, y, color="blue", alpha=0.2, outline=False, maxvalue=100):
    """parts of volcano plot"""
    # Define colors in a dictionary to access them with
    # the key from the pandas groupby funciton.
    df = df1.copy()
    transformed_q = -df[y].apply(np.log10).values
    transformed_q[transformed_q == np.inf] = maxvalue
    transformed_q[transformed_q > maxvalue] = maxvalue
    df["transformed_q"] = transformed_q
    df["color"] = color
    df["alpha"] = alpha
    df["size"] = 7
    source1 = ColumnDataSource(df)

    # Specify data source
    p.scatter(
        x=x,
        y="transformed_q",
        size="size",
        alpha="alpha",
        source=source1,
        color="color",
        name="circles",
    )
    if outline:
        p.scatter(
            x=x,
            y="transformed_q",
            size=7,
            alpha=1,
            source=source1,
            color="black",
            fill_color=None,
            name="outlines",
        )

    # prettify
    p.background_fill_color = "#DFDFE5"
    p.background_fill_alpha = 0.5
    return p, source1


def selector(
    df,
    valtoextract=[],
    logfoldtohighlight=0.15,
    pvaltohighlight=0.1,
    minlogfold=0.15,
    minpval=0.1,
):
    """Part of Volcano plot: A function to separate tfs from everything else"""
    toshow = (df.pvalue < minpval) & (abs(df.log2FoldChange) > minlogfold)
    df = df[toshow]
    sig = (df.pvalue < pvaltohighlight) & (abs(df.log2FoldChange) > logfoldtohighlight)
    if valtoextract:
        not_tf = ~df.gene_id.isin(valtoextract)
        is_tf = df.gene_id.isin(valtoextract)
        to_plot_not = df[~sig | not_tf]
        to_plot_yes = df[sig & is_tf]
    else:
        to_plot_not = df[~sig]
        to_plot_yes = df[sig]
    return to_plot_not, to_plot_yes


def correlationMatrix(
    data,
    names,
    colors=None,
    pvals=None,
    maxokpval=10**-9,
    other=None,
    title="correlation Matrix",
    dataIsCorr=False,
    invert=False,
    size=40,
    folder="",
    interactive=False,
    maxval=None,
    minval=None,
):
    """
    Make an interactive correlation matrix from an array using bokeh

    Args:
    -----
      data: arrayLike of int / float/ bool of size(names*val) or (names*names)
      names: list[str] of names for each rows
      colors: list[int] of size(names) a color for each names (good to display clusters)
      pvals: arraylike of int / float/ bool of size(names*val) or (names*names) with the corresponding pvalues
      maxokpval: float threshold when pvalue is considered good. otherwise lowers the size of the square
        until 10**-3 when it disappears
      other: arrayLike of int / float/ bool of size(names*val) or (names*names), an additional information
        matrix that you want ot display with opacity whereas correlations willl be displayed with
      title: str the plot title
      dataIsCorr: bool if not true, we will compute the corrcoef of the data array
      invert: bool whether or not to invert the matrix before running corrcoef
      size: int the plot size
      folder: str of folder location where to save the plot, won't save if empty
      interactive: bool whether or not to make the plot interactive (else will use matplotlib)
      maxval: float clamping coloring up to maxval
      minval: float clamping coloring down to minval

    Returns:
    -------
      the bokeh object if interactive else None

    """
    if not dataIsCorr:
        print("computing correlations")
        data = np.corrcoef(np.array(data) if not invert else np.array(data).T)
    else:
        data = np.array(data)
    regdata = data.copy()
    if minval is not None:
        data[data < minval] = minval
    if maxval is not None:
        data[data > maxval] = maxval
    data = data / data.max()
    TOOLS = (
        "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,save"
    )
    xname = []
    yname = []
    color = []
    alpha = []
    height = []
    width = []
    if type(colors) is list:
        print("we are assuming you want to display clusters with colors")
    elif other is not None:
        print(
            "we are assuming you want to display the other of your correlation with opacity"
        )
    if pvals is not None:
        print(
            "we are assuming you want to display the pvals of your correlation with size"
        )
        regpvals = pvals.copy()
        u = pvals < maxokpval
        pvals[~u] = np.log10(1 / pvals[~u])
        pvals = pvals / pvals.max()
        pvals[u] = 1
    if interactive:
        xname = []
        yname = []
        color = []
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                xname.append(name1)
                yname.append(name2)
                if pvals is not None:
                    height.append(max(0.1, min(0.9, pvals[i, j])))
                    color.append(cc.coolwarm[int((data[i, j] * 127) + 127)])
                    alpha.append(min(abs(data[i, j]), 0.9))
                elif other is not None:
                    color.append(cc.coolwarm[int((data[i, j] * 127) + 127)])
                    alpha.append(
                        max(min(other[i, j], 0.9), 0.1) if other[i, j] != 0 else 0
                    )
                else:
                    alpha.append(min(abs(data[i, j]), 0.9))
                if colors is not None:
                    if type(colors) is list:
                        if colors[i] == colors[j]:
                            color.append(Category10[10][colors[i]])
                        else:
                            color.append("lightgrey")

                elif pvals is None and other is None:
                    color.append("grey" if data[i, j] > 0 else Category20[3][2])
        print(regdata.max())
        if pvals is not None:
            width = height.copy()
            data = dict(
                xname=xname,
                yname=yname,
                colors=color,
                alphas=alpha,
                data=regdata.ravel(),
                pvals=regpvals.ravel(),
                width=width,
                height=height,
            )
        else:
            data = dict(
                xname=xname, yname=yname, colors=color, alphas=alpha, data=data.ravel()
            )
        tt = [("names", "@yname, @xname"), ("value", "@data")]
        if pvals is not None:
            tt.append(("pvals", "@pvals"))
        p = figure(
            title=title if title is not None else "Correlation Matrix",
            x_axis_location="above",
            tools=TOOLS,
            x_range=list(reversed(names)),
            y_range=names,
            tooltips=tt,
        )

        p.width = 800
        p.height = 800
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "5pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi / 3
        p.output_backend = "svg"
        p.rect(
            "xname",
            "yname",
            width=0.9 if not width else "width",
            height=0.9 if not height else "height",
            source=data,
            color="colors",
            alpha="alphas",
            line_color=None,
            hover_line_color="black",
            hover_color="colors",
        )
        save(p, folder + title.replace(" ", "_") + "_correlation.html")
        try:
            p.output_backend = "svg"
            export_svg(
                p, filename=folder + title.replace(" ", "_") + "_correlation.svg"
            )
        except (RuntimeError, Exception) as e:
            print(f"Could not save SVG: {e}")
        try:
            show(p)
        except Exception as e:
            print(f"Could not show plot: {e}")
        return p  # show the plot
    else:
        plt.figure(figsize=(size, 200))
        plt.title("the correlation matrix")
        plt.imshow(data)
        plt.savefig(title + "_correlation.pdf")
        plt.show()


def heatmap(
    data,
    colors=None,
    title="correlation Matrix",
    size=40,
    other=None,
    folder="",
    interactive=False,
    pvals=None,
    maxokpval=10**-9,
    maxval=None,
    minval=None,
):
    """
    Make an interactive heatmap from a dataframe using bokeh

    Args:
    -----
      data: dataframe of int / float/ bool of size(names1*names2)
      colors: list[int] of size(names) a color for each names (good to display clusters)
      pvals: arraylike of int / float/ bool of size(names*val) or (names*names) with the corresponding pvalues
      maxokpval: float threshold when pvalue is considered good. otherwise lowers the size of the square
        until 10**-3 when it disappears
      title: str the plot title
      size: int the plot size
      folder: str of folder location where to save the plot, won't save if empty
      interactive: bool whether or not to make the plot interactive (else will use matplotlib)
      maxval: float clamping coloring up to maxval
      minval: float clamping coloring down to minval

    Returns:
    -------
      the bokeh object if interactive else None

    """
    regdata = data.copy()
    if minval is not None:
        data[data < minval] = minval
    if maxval is not None:
        data[data > maxval] = maxval
    data = data / data.max()
    data = data.values
    TOOLS = (
        "hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,save"
    )
    xname = []
    yname = []
    color = []
    alpha = []
    height = []
    width = []
    if pvals is not None:
        print(
            "we are assuming you want to display the pvals of your correlation with size"
        )
        regpvals = pvals.copy()
        u = pvals < maxokpval
        pvals[~u] = np.log10(1 / pvals[~u])
        pvals = pvals / pvals.max()
        pvals[u] = 1
    if interactive:
        xname = []
        yname = []
        color = []
        for i, name1 in enumerate(regdata.index):
            for j, name2 in enumerate(regdata.columns):
                xname.append(name2)
                yname.append(name1)
                if pvals is not None:
                    # import pdb;pdb.set_trace()
                    height.append(max(0.1, min(0.9, pvals.loc[name1][name2])))
                    color.append(cc.coolwarm[int((data[i, j] * 128) + 127)])
                    alpha.append(min(abs(data[i, j]), 0.9))
                elif other is not None:
                    color.append(cc.coolwarm[int((data[i, j] * 128) + 127)])
                    alpha.append(
                        max(min(other[i, j], 0.9), 0.1) if other[i, j] != 0 else 0
                    )
                else:
                    alpha.append(min(abs(data[i, j]), 0.9))
                if colors is not None:
                    if type(colors) is list:
                        if colors[i] == colors[j]:
                            color.append(Category10[10][colors[i]])
                        else:
                            color.append("lightgrey")

                elif pvals is None and other is None:
                    color.append("grey" if data[i, j] > 0 else Category20[3][2])
        if pvals is not None:
            width = height.copy()
            data = dict(
                xname=xname,
                yname=yname,
                colors=color,
                alphas=alpha,
                data=regdata.values.ravel(),
                pvals=regpvals.values.ravel(),
                width=width,
                height=height,
            )
        else:
            data = dict(
                xname=xname, yname=yname, colors=color, alphas=alpha, data=data.ravel()
            )
        tt = [("names", "@yname, @xname"), ("value", "@data")]
        if pvals is not None:
            tt.append(("pvals", "@pvals"))
        p = figure(
            title=title if title is not None else "Heatmap",
            x_axis_location="above",
            tools=TOOLS,
            x_range=list(reversed(regdata.columns.astype(str).tolist())),
            y_range=regdata.index.tolist(),
            tooltips=tt,
        )

        p.width = 800
        p.height = 800
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "5pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi / 3
        p.output_backend = "svg"
        p.rect(
            "xname",
            "yname",
            width=0.9 if not width else "width",
            height=0.9 if not height else "height",
            source=data,
            color="colors",
            alpha="alphas",
            line_color=None,
            hover_line_color="black",
            hover_color="colors",
        )
        save(p, folder + title.replace(" ", "_") + "_heatmap.html")
        try:
            p.output_backend = "svg"
            export_svg(
                p, filename=folder + title.replace(" ", "_") + "_correlation.svg"
            )
        except (RuntimeError, Exception) as e:
            print(f"Could not save SVG: {e}")
        try:
            show(p)
        except Exception as e:
            print(f"Could not show plot: {e}")
        return p  # show the plot
    else:
        plt.figure(figsize=size)
        plt.title("the correlation matrix")
        plt.imshow(data)
        plt.savefig(title + "_correlation.pdf")
        plt.show()
