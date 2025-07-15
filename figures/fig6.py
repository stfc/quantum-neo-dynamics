import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


offset = -258.9475070402643 + 259.2063702444815

if __name__ == "__main__":

    labels = ['300', '210', '120', '030', '021', '012', '003']
    label_positions = np.arange(7) + 0.5


    with open("../results/statevector_data.json", "r") as f:
        file = json.load(f)
        aqc_high = np.asarray(list(file["aqc-high"].values()))
        aqc_low = np.asarray(list(file["aqc-low"].values()))
        hf = np.asarray(list(file["product"].values()))
        casci = np.asarray(list(file["casci"].values()))
        vqe_shallow = np.asarray(list(file["adapt-vqe-low"].values()))
        vqe_deep = np.asarray(list(file["adapt-vqe-high"].values()))


    plt.style.use("proton.mplstyle")
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.0, 3.2), constrained_layout=True)

    x = np.arange(8)
    ax0.hlines(casci, x[:-1], x[1:], colors='#008000', label="CASCI")

    for i, (y, xmin, xmax) in enumerate(zip(casci, x[:-1], x[1:])):
        ax0.fill_between([xmin, xmax], y, y2=np.min(casci), alpha=0.15 if i % 2 else 0.05, fc='k', ec="none")

    x = np.arange(7)
    dx = 1 / 7
    ax0.scatter(x + 1 * dx, hf, marker="o", fc="#008000", s=50, ec="w", lw=0.6, label="HF-product")
    ax0.scatter(np.array([0, 3, 6]) + 2 * dx, vqe_deep, marker="D", fc="#003f5c", s=50, ec="w", lw=0.6, label="VQE-deep")
    ax0.scatter(x + 3 * dx, vqe_shallow, marker="s", fc="#003f5c", s=50, ec="w", lw=0.6, label="VQE-shallow")
    ax0.scatter(np.array([0, 3]) + 4 * dx, aqc_high, marker=">", fc="#bc5090", s=50, ec="w", lw=0.6,label="AQC-high")
    ax0.scatter(x + 5 * dx, aqc_low, marker="<", fc="#bc5090", s=50, ec="w", lw=0.6, label="AQC-low")

    # zne_300_mean = -6.606763850950848
    # zne_300_upper = 0.03647268682671623
    # zne_300_lower = 0.03688050785988217
    # zne_030_mean = -6.323201063453085
    # zne_030_upper = 0.037899901381925005
    # zne_030_lower = 0.039590060370863434
    zne_300_mean = -6.600325
    zne_300_upper = 0.006512
    zne_300_lower = 0.006512
    zne_030_mean = -6.317742
    zne_030_upper = 0.010315
    zne_030_lower = 0.010315

    ax0.errorbar([0 + 6 * dx], [zne_300_mean], yerr=([zne_300_upper], [zne_300_lower]), marker="none", c="#ffa600", zorder=10)
    ax0.scatter([0 + 6 * dx], [zne_300_mean], marker="^", fc='#ffa600', ec="grey", linewidth=0.5, zorder=11)
    ax0.errorbar([3 + 6 * dx], [zne_030_mean], yerr=([zne_030_upper], [zne_030_lower]), marker="none", c="#ffa600", zorder=10)
    ax0.scatter([3 + 6 * dx], [zne_030_mean], marker="^", fc='#ffa600', ec="grey", linewidth=0.5, label="ZNE (fit first)", zorder=11)

    ax0.set_xlabel("Proton state")
    ax0.set_xticks(label_positions)
    ax0.set_xticklabels(list(labels))

    ax0.legend(loc="upper left")

    x = np.arange(8)
    ax1.hlines(casci - np.min(casci), x[:-1], x[1:], colors='#008000')

    for i, (y, xmin, xmax) in enumerate(zip(casci - np.min(casci), x[:-1], x[1:])):
        ax1.fill_between([xmin, xmax], y, y2=0, alpha=0.15 if i % 2 else 0.05, fc='k', ec="none", zorder=0)

    x = np.arange(7)
    dx = 1 / 6

    ax1.scatter(x + 1 * dx, hf - np.min(hf), marker="o", s=50, fc="#008000", ec="w", lw=0.6, zorder=1)
    ax1.scatter(np.array([0, 3, 6]) + 2 * dx, vqe_deep - np.min(vqe_deep), marker="D", s=50, fc="#003f5c", ec="w", lw=0.6, zorder=2)
    ax1.scatter(x + 3 * dx, vqe_shallow - np.min(vqe_shallow), marker="s", s=50, fc="#003f5c", ec="w", lw=0.6, zorder=2)
    ax1.scatter(np.array([0, 3]) + 4 * dx, aqc_high - np.min(aqc_high), marker=">", s=50, fc="#bc5090", ec="w", lw=0.6, zorder=3)
    ax1.scatter(x + 5 * dx, aqc_low - np.min(aqc_low), marker="<", s=50, fc="#bc5090", ec="w", lw=0.6, label="AQC-low", zorder=4)

    # zne_diff_first_mean = 0.017103257521407733 + offset
    # zne_diff_first_upper = 0.02203559676696859
    # zne_diff_first_lower = 0.022181237586102576
    # zne_fit_first_mean = 0.024699583280599846 + offset
    # zne_fit_first_upper = 0.05359855454692115
    # zne_fit_first_lower = 0.055988800311644674

    zne_diff_first_mean = 0.0176721 + offset
    zne_diff_first_upper = 0.0033033
    zne_diff_first_lower = 0.0033033
    zne_fit_first_mean = 0.0237198 + offset
    zne_fit_first_upper = 0.0121986
    zne_fit_first_lower = 0.0121986

    ax1.errorbar([3 + 1. * dx], [zne_diff_first_mean], yerr=([zne_diff_first_lower], [zne_diff_first_upper]), marker="none", c="#ffa600", zorder=10)
    ax1.scatter([3 + 1. * dx], [zne_diff_first_mean], marker="v", fc='#ffa600', ec="grey", linewidth=0.5, label="ZNE (diff first)", zorder=11)
    ax1.errorbar([3 + 2. * dx], [zne_fit_first_mean], yerr=([zne_fit_first_lower], [zne_fit_first_upper]), marker="none", c="#ffa600", zorder=10)
    ax1.scatter([3 + 2. * dx], [zne_fit_first_mean], marker="^", fc='#ffa600', ec="grey", linewidth=0.5, label="ZNE (fit first)", zorder=11)

    ax1.legend(loc="upper left")

    ax1.set_xlabel("Proton state")
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(list(labels))

    # ax0.text(0.5, np.min(casci), "Barrier", va="bottom", ha="center", fontsize="x-large",
    #          transform=mtransforms.blended_transform_factory(ax0.transAxes, ax0.transData))
    # ax0.text(0.97, 1.02, f"{fit_method.title()} fit\n{num_shots:d} shots", va="bottom", ha="right", transform=ax0.transAxes)
    # ax1.text(0.5, 0, "Barrier", va="bottom", ha="center", fontsize="x-large",
    #          transform=mtransforms.blended_transform_factory(ax1.transAxes, ax1.transData))

    ax0.set_ylabel("Energy [Ha]")
    ax1.set_ylabel(r"$\Delta E = E_{030} - E_{300}$ [Ha]")
    plt.savefig("fig6.pdf")
    plt.show()
