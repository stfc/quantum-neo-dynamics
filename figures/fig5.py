import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants & config
RNG_SEED = 31415
CSV_PATH = Path("../results/noisy_aer_simulations_random_seeds_raw.csv")
JSON_PATH = Path("../results/statevector_data.json")
PLOT_STYLE = "proton.mplstyle"
FIGSIZE = (7, 3)
NOISE_LIMIT = (0, 4)
FIT_POINTS = 20


# Define models
def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x ** 2 + b * x + c


def draw_brace(ax, xspan, yy, text, upsidedown=False):
    """Draws an annotated brace on the axes."""
    # shamelessly copied from https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    if upsidedown:
        y = np.concatenate((y_half_brace[-2::-1], y_half_brace))
    else:
        y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    line = ax.plot(x, y, color='black', lw=1)

    if upsidedown:
        text = ax.text((xmax+xmin)/2., yy+-.05*yspan, text, ha='center', va='bottom',fontsize="small")
    else:
        text = ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom',fontsize="small")
    return line, text


def load_data(csv_path: Path, json_path: Path) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load CSV into DataFrame and JSON into dict of noiseless values."""
    rng = np.random.default_rng(RNG_SEED)
    # (Seed assignment corrected)
    df = pd.read_csv(csv_path, dtype={
        "argparse.label": str,
        "lambda_noise": float,
        "expectation_value": float,
        "LSFJOBID": int,
        "num_shots": int,
    })
    with open(json_path, "r") as f:
        data = json.load(f)["aqc-low"]
    return df, data



def plot_panels(df: pd.DataFrame, noiseless: Dict[str, float]) -> None:
    """Create a 1×3 panel of scatter, errorbar, fit, and annotations."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    grp = df.groupby(["argparse.label", "lambda_noise"])["expectation_value"]
    means = grp.mean().reset_index().sort_values("lambda_noise")
    stds = grp.std().reset_index().sort_values("lambda_noise")

    offset = -258.9475070402643 + 259.2063702444815
    labels = ['300', '030']

    for i, lbl in enumerate(labels):
        ax = axes[i]
        sub = df[df["argparse.label"] == lbl]
        ax.scatter(sub["lambda_noise"], sub["expectation_value"],
                   marker='_', s=4, alpha=0.6, color="tab:green",
                   linewidth=0.2, label="Sampled circuits")
        m = means[means["argparse.label"] == lbl]
        s = stds[stds["argparse.label"] == lbl]
        ax.errorbar(m["lambda_noise"], m["expectation_value"],
                    yerr=s["expectation_value"],
                    ls='none', marker='.', ms=10, color="tab:red", mec="w", mew=0.4, label="Average")
        # fit quadratic
        popt, pcov = curve_fit(
            quadratic,
            m["lambda_noise"].to_numpy(),
            m["expectation_value"].to_numpy(),
            sigma=1 / (s["expectation_value"].to_numpy() ** 2)
        )
        perr = np.sqrt(np.diag(pcov))
        # plot intercept at x=0
        ax.errorbar([0], [popt[-1]], yerr=perr[-1],
                    ms=5, ls='none', marker="^", mfc='#ffa600', c="grey", linewidth=2, label="Zero-noise limit")

        # plot smooth fit curve
        x_fit = np.linspace(*NOISE_LIMIT, FIT_POINTS)
        y_fit = quadratic(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='tab:red', lw=1)
        print(f"{lbl} intercept: {popt[-1]:.6f} ± {perr[-1]:.6f}")

        ax.axhline(noiseless[lbl], ls=':', label="AQC-low (noiseless)")
        ax.set(xlabel="Noise factor", ylabel="Energy [Ha]")
        ax.axvline(1, ls=':', alpha=0.2)

    # Panel 3: difference (030 - 300) minus offset
    ax3 = axes[2]

    lambdas = []
    diff_mean = []
    diff_std = []
    for noise in df["lambda_noise"].unique():
        sub_030 = df[(df["argparse.label"] == "030") & (df["lambda_noise"] == noise)]["expectation_value"].to_numpy()
        sub_300 = df[(df["argparse.label"] == "300") & (df["lambda_noise"] == noise)]["expectation_value"].to_numpy()
        diff = sub_030 - sub_300 - offset

        ax3.scatter([noise] * len(diff), diff,
                   marker='_', s=4, alpha=0.6, color="tab:green",
                   linewidth=0.2)

        lambdas.append(noise)
        diff_mean.append(np.mean(diff))
        diff_std.append(np.std(diff))
        axes[2].errorbar([noise], np.mean(diff), yerr=np.std(diff),
                         marker=".", linestyle="none", color="tab:red", mec="w", mew=0.4, ms=15)

    popt, pcov = curve_fit(
        linear,
        np.asarray(lambdas),
        np.asarray(diff_mean),
        sigma=1 / (np.asarray(diff_std) ** 2)
    )
    perr = np.sqrt(np.diag(pcov))
    # plot intercept at x=0
    ax3.errorbar([0], [popt[-1]], yerr=perr[-1],
                ms=5, ls='none', marker="v", mfc='#ffa600', c="grey", linewidth=2, label="ZNE (diff first)", zorder=11)

    zne_fit_first_mean = -6.317742 + 6.600325 - offset
    zne_fit_first_std = np.sqrt(0.010315 ** 2 + 0.006512 ** 2)
    print(f"Fit first intercept: {zne_fit_first_mean * 1e3:.4f} ± {zne_fit_first_std * 1e3:.4f} mHa")
    ax3.errorbar([-0.4], [zne_fit_first_mean], yerr=zne_fit_first_std,
                 ms=5, ls='none', marker="^", mfc='#ffa600', c="grey", linewidth=2, label="ZNE (fit first)", zorder=11)

    # plot smooth fit curve
    x_fit = np.linspace(*NOISE_LIMIT, FIT_POINTS)
    y_fit = linear(x_fit, *popt)
    ax3.plot(x_fit, y_fit, color='tab:red', lw=1)
    print(f"Diff first intercept: {popt[-1] * 1e3:.4f} ± {perr[-1] * 1e3:.4f} mHa")

    ax3.set(xlabel="Noise factor", ylabel="Energy barrier [Ha]", ylim=(-0.26, 0.07))
    ax3.axvline(1, ls=':', alpha=0.2)

    # Legends & annotations
    lg = axes[0].legend(loc="upper left", title="Left (300)")
    lg.legend_handles[0]._sizes = [40]
    axes[1].legend(handles=[lg.legend_handles[-1]], loc="upper left", title="Middle (030)")

    axes[0].annotate(
        "Quadratic extrapolation\n$100$ randomized foldings\n$1000$ shots per circuit",
        xy=(4, -6.6), xycoords="data", ha="right", va="bottom", fontsize="small"
    )
    axes[2].annotate(
        "Linear extrapolation",
        xy=(4, 0.06), xycoords="data", ha="right", va="top", fontsize="small"
    )
    draw_brace(axes[1], (1.3, 4), -6.2, 'Gate-folded', upsidedown=True)

    axes[2].set_xlabel("Noise factor")
    axes[2].set_ylim(-0.26, 0.07)
    axes[2].axhline(noiseless['030'] - noiseless['300'] - offset, ls=':', label="AQC-low (noiseless)")
    axes[2].set_ylabel("Energy barrier [Ha]")
    axes[2].axhline(0.011857048493202349, ls='-', label="CASCI")
    axes[2].legend(loc="lower left")


if __name__ == "__main__":
    df, noiseless = load_data(CSV_PATH, JSON_PATH)
    plot_panels(df, noiseless)
    plt.savefig("fig5.pdf", bbox_inches="tight")
    plt.show()
