"""Plots for the paper from the validation curves in figures/data (exported from W&B).

Run from the paper folder: python scripts/make_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "figures" / "data"
OUT = ROOT / "figures"

# Okabe-Ito, plus distinct line styles so the curves survive grayscale printing.
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#777777"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{name}.csv")
    df["kstep"] = df["step"] / 1000
    return df


def anatomy() -> None:
    runs = [
        ("grid64_l1_f1_moe", "healthy", BLUE, "-"),
        ("specls_ls", "position code", RED, "--"),
        ("fixedgsd_smax600", "dimensional", ORANGE, "-."),
    ]
    panels = [
        ("val_loss", "val. loss $\\mathcal{L}_{\\mathrm{cm}}$", None),
        ("target_rank", "within-tile rank", None),
        ("vicreg_cov", "VICReg cov. $\\mathcal{C}$", "log"),
        ("G", "cross-sample gap $\\Delta$", "log"),
        ("S", "sample-agnostic share $S$", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.9, 1.6))
    for ax, (key, title, yscale) in zip(axes, panels):
        for name, label, color, ls in runs:
            df = load(name)
            ax.plot(df["kstep"], df[key], color=color, ls=ls, label=label)
        ax.set_title(title, pad=3)
        ax.set_xlabel("step (k)", labelpad=1)
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].axhspan(0.9, 1.02, color=RED, alpha=0.08, lw=0)
    axes[-1].set_ylim(0, 1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.6,
               bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "anatomy.pdf")
    plt.close(fig)


def regularizers() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.45))
    hinge = [
        ("gsweep_g64", "hinge $\\gamma{=}1$", RED, "-"),
        ("gsweep_g64_predgam03", "hinge $\\gamma{=}0.3$", ORANGE, "-."),
        ("gsweep_g64_predvar0", "no hinge on $\\hat z$", BLUE, "--"),
    ]
    for name, label, color, ls in hinge:
        df = load(name)
        axes[0].plot(df["kstep"], df["pred_std"], color=color, ls=ls, label=label)
        axes[1].plot(df["kstep"], df["G"], color=color, ls=ls, label=label)
    axes[0].set_title("prediction std (per feature)", pad=3)
    axes[1].set_title("cross-sample gap $\\Delta$", pad=3)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, loc="center right", handlelength=2.2)
    for name, label, color, ls in [("cov0.005", "$w_c{=}0.005$", BLUE, "-"), ("cov0.05", "$w_c{=}0.05$", RED, "--")]:
        df = load(name)
        axes[2].plot(df["kstep"], df["S"], color=color, ls=ls, label=label, marker="o", ms=1.8)
    axes[2].set_title("share $S$, covariance weight", pad=3)
    axes[2].set_ylim(0, 1.0)
    axes[2].legend(frameon=False, loc="lower right", handlelength=2.2)
    for ax in axes:
        ax.set_xlabel("step (k)", labelpad=1)
        ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3, w_pad=0.8)
    fig.savefig(OUT / "regularizers.pdf")
    plt.close(fig)


if __name__ == "__main__":
    anatomy()
    regularizers()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
