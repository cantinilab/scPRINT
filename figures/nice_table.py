import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib import path as mpath
from matplotlib.colors import TwoSlopeNorm


def _draw_curly_brace(ax, y0, y1, x, width=0.6, lw=1.5):
    """
    Draw a vertical curly brace from y0 to y1 at x (to the left of heatmap).
    """
    Path = mpath.Path
    mid = (y0 + y1) / 2.0
    dx = width
    # Four cubic Bezier segments: top curl, top stem -> mid, mid -> bottom stem, bottom curl
    verts = [
        (x, y1),
        (x - dx, y1),
        (x - dx, y1),
        (x - dx, mid + 0.25 * (y1 - y0)),  # top curl
        (x - dx, mid + 0.1 * (y1 - y0)),
        (x, mid + 0.1 * (y1 - y0)),
        (x, mid),  # to mid
        (x, mid - 0.1 * (y1 - y0)),
        (x - dx, mid - 0.1 * (y1 - y0)),
        (x - dx, mid - 0.25 * (y1 - y0)),  # down stem
        (x - dx, y0),
        (x - dx, y0),
        (x, y0),  # bottom curl
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
    ]
    brace = patches.PathPatch(Path(verts, codes), fill=False, lw=lw, capstyle="round")
    ax.add_patch(brace)


def plot_grouped_heatmap(
    values: pd.DataFrame,
    row_groups: pd.Series | pd.DataFrame,
    col_groups: pd.Series | pd.DataFrame | None = None,
    *,
    center: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".1f",
    annotate: bool = True,
    annot_df: pd.DataFrame | None = None,
    group_pad: float = 0.6,
    figsize=(10, 6),
):
    """
    values: DataFrame shaped (rows: methods/items, cols: metrics/sub-metrics).
            Columns can be regular Index or MultiIndex.
    row_groups: Series/DataFrame with index matching values.index and one column 'group'
                (or name set) giving the row group for each row.
    col_groups: Series/DataFrame mapping each column (2nd level if MultiIndex) to a top group.
                If values.columns is MultiIndex, this can be omitted and top level will be used.
    """
    # --- Normalize mapping inputs
    if isinstance(row_groups, pd.DataFrame):
        assert row_groups.shape[1] == 1, "row_groups DataFrame must have one column"
        row_groups = row_groups.iloc[:, 0]
    row_groups = row_groups.loc[values.index]

    # Columns/group structure
    if isinstance(values.columns, pd.MultiIndex) and col_groups is None:
        # Use top level as big groups, second level as displayed labels
        top = values.columns.get_level_values(0)
        sub = values.columns.get_level_values(1)
        col_group_series = pd.Series(top.values, index=values.columns)
        col_labels = sub
    else:
        if col_groups is None:
            # Single group containing all columns
            col_group_series = pd.Series("Group", index=values.columns)
        else:
            if isinstance(col_groups, pd.DataFrame):
                assert col_groups.shape[1] == 1, (
                    "col_groups DataFrame must have one column"
                )
                col_groups = col_groups.iloc[:, 0]
            col_group_series = pd.Series(
                col_groups.loc[values.columns].values, index=values.columns
            )
        col_labels = pd.Index([str(c) for c in values.columns])

    # --- Reorder by groups to make blocks contiguous
    row_order = (
        row_groups.reset_index()
        .sort_values(
            [
                row_groups.name if row_groups.name else 0,
                row_groups.index.name or "index",
            ]
        )
        .set_index("index")
        .index
    )
    values = values.loc[row_order]
    row_groups = row_groups.loc[row_order]

    col_order_df = col_group_series.reset_index()
    # Get the last column name (the original series values) and all but the last (the index levels)
    series_col = col_order_df.columns[-1]  # The series values column
    index_cols = col_order_df.columns[:-1].tolist()  # The index level columns
    col_order = (
        col_order_df.sort_values([series_col] + index_cols).set_index(index_cols).index
    )
    values = values.loc[:, col_order]
    col_group_series = col_group_series.loc[col_order]
    col_labels = pd.Index(
        [str(c[1]) if isinstance(c, tuple) else str(c) for c in values.columns]
    )

    # --- Figure & heatmap
    fig, ax = plt.subplots(figsize=figsize)

    # Color normalization centered at `center`
    if vmin is None:
        vmin = np.nanmin(values.values)
    if vmax is None:
        vmax = np.nanmax(values.values)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    im = ax.imshow(values.values, aspect="auto", norm=norm)  # default colormap

    # Add subtle background shading for alternating row groups
    y = 0
    row_group_list = list(values.groupby(row_groups, sort=False))
    for i, (g, block) in enumerate(row_group_list):
        h = len(block)
        if i % 2 == 1:  # Alternate groups
            ax.axhspan(y - 0.5, y + h - 0.5, alpha=0.1, color="gray", zorder=0)
        y += h

    # Gridlines between groups
    # Row separators - make them more prominent
    y = 0
    row_sep_positions = []
    row_group_list = list(values.groupby(row_groups, sort=False))
    for i, (g, block) in enumerate(row_group_list):
        h = len(block)
        y += h
        row_sep_positions.append((g, y))
        # Don't draw line after the last group
        if i < len(row_group_list) - 1:
            ax.axhline(y - 0.5, color="white", lw=3.0)

    # Column separators
    x = 0
    col_sep_positions = []
    col_group_list = list(values.T.groupby(col_group_series, sort=False))
    for i, (g, block) in enumerate(col_group_list):
        w = len(block)
        x += w
        col_sep_positions.append((g, x))
        # Don't draw line after the last group
        if i < len(col_group_list) - 1:
            ax.axvline(x - 0.5, color="white", lw=3.0)

    # Ticks
    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels(col_labels, rotation=90, ha="center", va="top")
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(values.index)

    # Annotations
    if annotate:
        shown = annot_df if annot_df is not None else values
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = shown.iat[i, j]
                if pd.isna(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:{fmt}}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                )

    # Top-level column group labels
    x0 = 0
    for g, block in values.T.groupby(col_group_series, sort=False):
        w = len(block)
        xc = x0 + (w - 1) / 2.0
        ax.text(xc, -group_pad, str(g), ha="center", va="bottom", fontsize=11)
        x0 += w

    # Left curly braces + row group labels
    y0 = 0
    x_left = -group_pad * 1.8
    for g, block in values.groupby(row_groups, sort=False):
        h = len(block)
        _draw_curly_brace(ax, y0 - 0.45, y0 + h - 0.55, x_left, width=0.5, lw=1.6)
        ax.text(
            x_left - 0.55,
            y0 + (h - 1) / 2.0,
            str(g),
            ha="right",
            va="center",
            fontsize=11,
        )
        y0 += h

    # Cosmetics
    ax.spines[:].set_visible(False)
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)  # y-axis downward

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Score (centered at {:.1f})".format(center))

    plt.tight_layout()
    return fig, ax


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Fake data shaped like "methods x metrics", with top-level column groups
    methods = [
        "PIDC",
        "GENIE3",
        "GRNBOOST2",
        "PPCOR",
        "SCODE",
        "GRISLI",
        "SINGE",
        "SCNS",
        "LEAP",
        "SINCERITIES",
        "GRNVBEM",
        "SCRIBE",
    ]
    metrics = pd.MultiIndex.from_product(
        [
            ["AUC No Rate", "Early Precision Rate", "EPR Activation", "EPR Inhibition"],
            ["mCAD", "VSC", "HSC", "GSD"],
        ],
        names=["Panel", "Dataset"],
    )
    rng = np.random.default_rng(0)
    vals = pd.DataFrame(
        0.8 + 2.6 * rng.random((len(methods), len(metrics))),
        index=methods,
        columns=metrics,
    )
    # Make some ~random-predictorish values around 1.0
    vals.iloc[:, ::4] = 1.0 + 0.05 * rng.standard_normal(
        (len(methods), vals.shape[1] // 4)
    )

    # Row group map (like "STRING", "Non-specific ChIP-Seq", etc.)
    row_grp = pd.Series(
        ["Set A"] * 4 + ["Set B"] * 4 + ["Set C"] * 4, index=methods, name="RowGroup"
    )
    # (Optional) If you don't have MultiIndex columns, provide a Series mapping columns->group.

    fig, ax = plot_grouped_heatmap(
        vals, row_grp, center=1.0, fmt=".1f", annotate=True, figsize=(11, 6)
    )
    plt.show()
    plt.savefig("fig.png")
