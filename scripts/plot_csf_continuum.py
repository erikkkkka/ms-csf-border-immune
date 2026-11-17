"""Draw the four quantitative panels of Figure 1 using the CSF trajectory results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


WIDTH, HEIGHT = 518.74, 350
INK, SECONDARY, LEADER = "#20282D", "#495761", "#687982"
ENTRY, RESPONSE, BAM = "#466F99", "#AE681B", "#237D6C"
COHORTS = ["GSE138266", "GSE194078", "GSE286068"]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.65,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "legend.frameon": False,
}


def axes(fig, x, y, w, h, **kwargs):
    return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT], **kwargs)


def put(fig, x, y, label, **kwargs):
    return fig.text(x / WIDTH, 1 - y / HEIGHT, label, **kwargs)


def heading(fig, x, y, letter, title):
    put(fig, x, y, letter, size=8.6, weight="bold")
    put(fig, x + 13, y, title, size=8.6, weight="bold")


def quantitative(ax):
    ax.tick_params(length=2.6, pad=2.4, width=0.55)
    ax.set_axisbelow(True)


def enrichment(fig, df):
    heading(fig, 17, 21, "a", "CSF enrichment")
    ax = axes(fig, 47, 43, 151, 111)
    assert len(df) == 4
    assert np.allclose(df.n_BAM_like / df.n_cells, df.BAM_like_fraction)
    df = df.pivot(index="dataset", columns="tissue", values="BAM_like_fraction")
    for cohort, color, marker in zip(COHORTS[:2], [ENTRY, BAM], ["o", "s"]):
        values = 100 * df.loc[cohort, ["PBMC", "CSF"]].to_numpy(float)
        ax.plot([0, 1], values, color=color, linewidth=1.25)
        ax.plot([0, 1], values, ls="none", color=color, marker=marker,
                markersize=5.4 if marker == "o" else 3.25,
                markerfacecolor="none" if marker == "o" else color,
                markeredgewidth=0.9, zorder=5 if marker == "o" else 4, clip_on=False)
        ax.text(1.07, values[1], cohort, size=6.2, va="center", color=color)
    ax.set(xlim=(-0.13, 1.73), ylim=(-1, 59), ylabel="BAM-like cells among APCs (%)")
    ax.set_xticks([0, 1], ["PBMC", "CSF"])
    ax.set_yticks(np.arange(0, 60, 10))
    quantitative(ax)


def continuum(fig, cells):
    heading(fig, 226, 21, "b", "Monocyte-to-BAM-like continuum")
    lo_x, hi_x, lo_y, hi_y = -6.45, 12.9, 4.47, 14.8
    x0, y0, w = 244, 36, 239
    scale = w / (hi_x - lo_x)
    ax = axes(fig, x0, y0, w, scale * (hi_y - lo_y))
    ax.set(xlim=(lo_x, hi_x), ylim=(lo_y, hi_y), aspect="equal")
    ax.set_axis_off()
    assert len(cells) == 6360 and set(cells.display_anchor.value_counts()) == {2150, 307, 3903}
    assert cells.UMAP1.between(lo_x, hi_x).all() and cells.UMAP2.between(lo_y, hi_y).all()
    assert cells.consensus_pseudotime.between(0, 1).all()
    plotted = cells.sort_values("consensus_pseudotime", kind="stable")
    viridis = mpl.colormaps["viridis"]
    cmap = LinearSegmentedColormap.from_list("ordered_pseudotime", viridis(np.linspace(0.12, 0.86, 256)))
    sc = ax.scatter(plotted.UMAP1, plotted.UMAP2, c=plotted.consensus_pseudotime,
                    cmap=cmap, norm=Normalize(0, 1), s=1.65, alpha=0.76, linewidths=0)
    assert np.array_equal(sc.get_offsets(), plotted[["UMAP1", "UMAP2"]].to_numpy())
    assert np.array_equal(sc.get_array(), plotted.consensus_pseudotime.to_numpy())

    callouts = [
        ("PBMC monocytes", "PBMC monocyte root", (6.8, 13.98), (430, 35), (414, 40)),
        ("BAM-like endpoint", "C1Q/APOE-high BAM-like pole", (-2.7, 5.24), (280, 161), (286, 155)),
        ("CSF monocyte entry", "CSF monocyte entry", (2.1, 8.46), (379, 150), (365, 144)),
    ]
    for label, group, target, (tx, ty), (sx, sy) in callouts:
        selected = cells.loc[cells.display_anchor.eq(group), ["UMAP1", "UMAP2"]].to_numpy()
        px, py = selected[np.argmin(np.sum((selected - target) ** 2, axis=1))]
        point_x, point_y = x0 + (px - lo_x) * scale, y0 + (hi_y - py) * scale
        fig.add_artist(Line2D([sx / WIDTH, point_x / WIDTH], [1 - sy / HEIGHT, 1 - point_y / HEIGHT],
                              transform=fig.transFigure, color=LEADER, lw=0.55))
        put(fig, tx, ty, label, size=6.1, va="center", ha="center", weight="bold")

    cax = axes(fig, 404, 165, 81, 3.7)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1],
                      format=mpl.ticker.FormatStrFormatter("%g"))
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(LEADER)
    cb.ax.tick_params(length=2, width=0.45, pad=1.6, labelsize=5.5)
    put(fig, 399, 167, "Pseudotime", size=6.1, ha="right", va="center", color=SECONDARY)


def programs(fig, df):
    heading(fig, 17, 204, "c", "Programs by annotation confidence")
    ax = axes(fig, 47, 238, 156, 78)
    stages = ["CSF_classical_monocyte", "CSF_BAM_intermediate", "CSF_BAM_terminal"]
    groups = [
        ("Monocyte entry", "Monocyte entry", ENTRY, "o"),
        ("GM-CSF response", "CSF1/GM-CSF", RESPONSE, "^"),
        ("BAM identity", "BAM-like", BAM, "s"),
    ]
    x = np.arange(3)
    assert set(df.dataset) == set(COHORTS) and len(df) == 27
    handles = []
    for key, label, color, marker in groups:
        values = df.loc[df.module.eq(key)].pivot(index="dataset", columns="stage",
                                                  values="scaled_median")[stages]
        assert np.isfinite(values.to_numpy()).all()
        for cohort in values.to_numpy():
            ax.plot(x, cohort, color=color, lw=0.6, alpha=0.25)
        line, = ax.plot(x, values.mean(axis=0).to_numpy(), color=color, lw=1.45,
                        marker=marker, markersize=3.25, label=label)
        handles.append(line)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(35 / WIDTH, 1 - 221 / HEIGHT),
               fontsize=5.6, handlelength=1.55, handletextpad=0.42, columnspacing=1.05, ncol=3)
    ax.set(xlim=(-0.12, 2.12), ylim=(-0.06, 1.07), ylabel="Within-cohort scaled median")
    ax.set_xticks(x, ["CSF entry", "BAM-like\nnon-core", "BAM-like\ncore"])
    ax.set_yticks(np.linspace(0, 1, 5), ["0", "0.25", "0.5", "0.75", "1"])
    quantitative(ax)


def robustness(fig, df):
    heading(fig, 226, 204, "d", "Donor-omission robustness")
    ax = axes(fig, 278, 238, 207, 78)
    df = df.set_index("dataset").loc[COHORTS]
    assert df.n_leave_one_donor_runs.tolist() == [12, 11, 15]
    assert df.representation.eq("primary").all()
    rows = np.arange(len(df))[::-1]
    ax.axvline(0.5, color=LEADER, ls=(0, (3, 2)), lw=0.65)
    for y, row in zip(rows, df.itertuples()):
        ax.plot([row.jackknife_min, row.jackknife_max], [y - 0.13] * 2,
                color="#B4C2CB", lw=2.0, solid_capstyle="round", zorder=2)
        ax.scatter(row.jackknife_median, y - 0.13, s=15, color=ENTRY, zorder=3)
        ax.scatter(row.full_auc, y + 0.13, s=17, marker="D", color=INK, zorder=3)
    handles = [
        Line2D([], [], color=INK, marker="D", markersize=3.8, lw=0, label="Full model"),
        Line2D([], [], color="#B4C2CB", lw=2, marker="o", markersize=3.6,
               markerfacecolor=ENTRY, markeredgecolor=ENTRY, label="Donor jackknife"),
    ]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(273 / WIDTH, 1 - 221 / HEIGHT),
               fontsize=6, handlelength=1.6, handletextpad=0.5, columnspacing=1.5, ncol=2)
    ax.set(xlim=(0.45, 1.025), ylim=(-0.41, 2.42), xlabel="Endpoint vs root AUC")
    ax.set_yticks(rows, df.index)
    ax.set_xticks(np.linspace(0.5, 1, 6))
    quantitative(ax)


def build_figure(enrichment_data, continuum_data, program_data, robustness_data):
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor="white")
        enrichment(fig, enrichment_data)
        continuum(fig, continuum_data)
        programs(fig, program_data)
        robustness(fig, robustness_data)
    return fig
